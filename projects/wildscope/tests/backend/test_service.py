import time
from io import BytesIO

from PIL import Image

from wildscope.contracts import ModelPrediction, ObservationPhoto, WildlifeFeed
from wildscope.service import WildlifeService
from wildscope.storage import WildlifeStore


class Client:
    def __init__(self):
        self.since_calls = []

    def fetch_recent(self, feed, *, hours):
        return (
            ObservationPhoto(
                1, 1, "2026-08-18", "2026-08-18T10:00:00Z", 10,
                "Panthera onca", "Jaguar", "https://static.inaturalist.org/1.jpg",
                "cc-by", "Observer", "research",
            ),
            ObservationPhoto(
                2, 2, "2026-08-18", "2026-08-18T11:00:00Z", 11,
                "Panthera onca", "Jaguar", "https://static.inaturalist.org/2.jpg",
                "cc-by", "Observer", "research",
            ),
        )

    def fetch_since(self, feed, *, since):
        self.since_calls.append(since)
        return self.fetch_recent(feed, hours=24)


class StaticModel:
    def predict(self, images, *, country):
        assert country == "ECU"
        return {
            photo_id: ModelPrediction("felidae", 0.6 + photo_id / 10, "speciesnet-test")
            for photo_id in images
        }


class VisualModel:
    def __init__(self):
        self.calls = []

    def predict(self, image, candidates, *, trained_at):
        self.calls.append((image, candidates, trained_at))
        return ModelPrediction(
            "Panthera onca",
            0.08,
            "bioclip-vit-b16-selective-margin-0.075",
            trained_at,
        )


def wait(job):
    deadline = time.monotonic() + 2
    while job.state in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.005)
    assert job.state == "completed", job.error


def test_sync_then_train_uses_only_data_newer_than_watermark(tmp_path, monkeypatch) -> None:
    feed = WildlifeFeed("yasuni", "Yasuni", 68650, "ECU", "rainforest")
    store = WildlifeStore(tmp_path / "store.sqlite3")
    service = WildlifeService(
        (feed,), store, tmp_path / "cache", client=Client(), static_model=StaticModel()
    )

    def download(feed_id, photo_id, url):
        path = tmp_path / f"{photo_id}.jpg"
        Image.new("RGB", (64, 48), (photo_id * 20, 70, 40)).save(path)
        return path, f"{photo_id:064x}"

    monkeypatch.setattr(service, "_download_photo", download)

    sync = service.start_sync("yasuni")
    wait(sync)
    training = service.start_training("yasuni")
    wait(training)
    model = store.adaptive_model("yasuni")
    dashboard = service.training_dashboard("yasuni")

    assert sync.details["static_predictions"] == 2
    assert training.details["new_samples"] == 2
    assert training.details["training_samples"] == 2
    assert training.details["evaluated_model_id"] == "SpeciesNet baseline"
    assert training.details["trained_model_id"] == training.details["model_id"]
    assert training.details["baseline_accuracy"] == 0.0
    assert training.details["deployed_accuracy"] is None
    assert training.details["training_agreement"] == 1.0
    assert training.details["baseline_mean_confidence"] == 0.75
    assert training.details["baseline_coverage"] == 1.0
    assert training.details["target_count"] == 1
    assert [stage["id"] for stage in training.details["pipeline"]] == [
        "fetch",
        "download",
        "preprocess",
        "baseline-inference",
        "evaluate",
        "train",
        "persist",
    ]
    assert all(stage["state"] == "completed" for stage in training.details["pipeline"])
    assert training.details["duration_seconds"] >= 0
    assert model["payload"]["training_photo_ids"] == [1, 2]
    assert model["payload"]["protocol_version"] == (
        "test-then-train-v3-bioclip-selective"
    )
    assert [sample["photo_id"] for sample in training.details["samples"]] == [1, 2]
    assert training.details["samples"][0]["deployed_margin"] is None
    assert dashboard["baseline_model"]["name"] == "SpeciesNet"
    assert dashboard["baseline_model"]["engine_version"] == "speciesnet-5.0.5"
    assert dashboard["dataset"]["total_observations"] == 2
    assert dashboard["dataset"]["eligible_labels"] == 2
    assert dashboard["dataset"]["target_distribution"] == {"Panthera onca": 2}
    assert model["watermark"] == "2026-08-18T11:00:00Z"
    assert store.frames("yasuni", page=1)["items"][0]["adaptive_label"] == "Panthera onca"

    second = service.start_training("yasuni")
    deadline = time.monotonic() + 2
    while second.state in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.005)
    assert second.state == "failed"
    assert "no new labeled" in second.error
    assert service.client.since_calls == ["2026-08-18T11:00:00Z"]


def test_multi_species_model_uses_visual_inference_and_exposes_margin(tmp_path) -> None:
    feed = WildlifeFeed("yasuni", "Yasuni", 68650, "ECU", "rainforest")
    store = WildlifeStore(tmp_path / "store.sqlite3")
    visual_model = VisualModel()
    service = WildlifeService(
        (feed,),
        store,
        tmp_path / "cache",
        client=Client(),
        static_model=StaticModel(),
        visual_model=visual_model,
    )
    image_path = tmp_path / "jaguar.jpg"
    Image.new("RGB", (64, 48), "green").save(image_path)
    photo = Client().fetch_recent(feed, hours=24)[0]
    store.upsert_observation(
        "yasuni", photo, sha256="1" * 64, cached_path=str(image_path)
    )
    store.save_prediction(1, "static", ModelPrediction("animal", 0.8, "speciesnet"))
    payload = {
        "protocol_version": "test-then-train-v3-bioclip-selective",
        "target_catalog": {
            "Leopardus pardalis": {
                "taxon_id": 41977,
                "scientific_name": "Leopardus pardalis",
                "common_name": "Ocelot",
            },
            "Panthera onca": {
                "taxon_id": 41970,
                "scientific_name": "Panthera onca",
                "common_name": "Jaguar",
            },
        },
        "visual_model": {
            "name": "hf-hub:imageomics/bioclip",
            "decision_metric": "top-1 minus top-2 cosine similarity",
            "margin_threshold": 0.075,
        },
    }
    store.save_adaptive_model(
        "yasuni", "visual-v1", "2026-08-18T13:00:00Z", "2026-08-18T11:00:00Z", payload
    )

    service._apply_adaptive("yasuni")
    enriched = service.frames("yasuni", page=1)["items"][0]

    assert len(visual_model.calls) == 1
    assert visual_model.calls[0][1] == (
        "Leopardus pardalis",
        "Panthera onca",
    )
    assert enriched["adaptive_label"] == "Panthera onca"
    assert enriched["adaptive_confidence"] is None
    assert enriched["adaptive_margin"] == 0.08
    assert enriched["adaptive_identification"]["common_name"] == "Jaguar"
    assert enriched["adaptive_identification"]["decision_metric"] == (
        "top-1 minus top-2 cosine similarity"
    )


class DownloadResponse:
    def __init__(self, payload=b"jpeg", *, status_error=None):
        self.payload = payload
        self.status_error = status_error
        self.headers = {"Content-Type": "image/jpeg"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, size):
        yield from BytesIO(self.payload)


def test_download_falls_back_to_large_when_original_is_unavailable(tmp_path, monkeypatch) -> None:
    feed = WildlifeFeed("yasuni", "Yasuni", 68650, "ECU", "rainforest")
    service = WildlifeService(
        (feed,), WildlifeStore(tmp_path / "store.sqlite3"), tmp_path / "cache",
        client=Client(), static_model=StaticModel(),
    )
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if "/original." in url:
            return DownloadResponse(status_error=RuntimeError("not found"))
        return DownloadResponse(b"fallback-jpeg")

    monkeypatch.setattr(service._http, "get", get)

    path, _ = service._download_photo(
        "yasuni", 1, "https://static.inaturalist.org/photos/1/original.jpg"
    )

    assert path.read_bytes() == b"fallback-jpeg"
    assert calls == [
        "https://static.inaturalist.org/photos/1/original.jpg",
        "https://static.inaturalist.org/photos/1/large.jpg",
    ]


def test_legacy_model_bootstraps_test_then_train_from_existing_labels(
    tmp_path, monkeypatch
) -> None:
    feed = WildlifeFeed("yasuni", "Yasuni", 68650, "ECU", "rainforest")
    store = WildlifeStore(tmp_path / "store.sqlite3")
    service = WildlifeService(
        (feed,), store, tmp_path / "cache", client=Client(), static_model=StaticModel()
    )

    def download(feed_id, photo_id, url):
        path = tmp_path / f"legacy-{photo_id}.jpg"
        Image.new("RGB", (64, 48), "green").save(path)
        return path, f"{photo_id:064x}"

    monkeypatch.setattr(service, "_download_photo", download)
    sync = service.start_sync("yasuni")
    wait(sync)
    store.save_adaptive_model(
        "yasuni",
        "legacy-model",
        "2026-08-18T12:00:00Z",
        "2026-08-18T11:00:00Z",
        {
            "counts": {"felidae": {"Panthera onca": 2}},
            "sample_count": 2,
            "protocol_version": "test-then-train-v1",
        },
    )

    dashboard = service.training_dashboard("yasuni")

    assert dashboard["live_batch"]["status"] == "bootstrap-ready"
    assert dashboard["live_batch"]["bootstrap_migration"] is True
    assert dashboard["live_batch"]["eligible_samples"] == 2
    assert len(dashboard["live_batch"]["samples"]) == 2

    rebuild = service.start_training("yasuni")
    wait(rebuild)
    model = store.adaptive_model("yasuni")

    assert rebuild.details["bootstrap_migration"] is True
    assert rebuild.details["new_samples"] == 0
    assert rebuild.details["training_samples"] == 2
    assert rebuild.details["evaluated_model_id"] == "SpeciesNet baseline"
    assert model["payload"]["protocol_version"] == (
        "test-then-train-v3-bioclip-selective"
    )
    assert model["payload"]["target_catalog"]["Panthera onca"]["common_name"] == (
        "Jaguar"
    )


def test_protocol_bootstrap_rebuilds_catalog_from_pre_watermark_labels(tmp_path) -> None:
    feed = WildlifeFeed("yasuni", "Yasuni", 68650, "ECU", "rainforest")
    store = WildlifeStore(tmp_path / "store.sqlite3")
    service = WildlifeService(
        (feed,), store, tmp_path / "cache", client=Client(), static_model=StaticModel()
    )
    for photo in Client().fetch_recent(feed, hours=24):
        store.upsert_observation("yasuni", photo)
        store.save_prediction(
            photo.photo_id,
            "static",
            ModelPrediction("animal", 0.8, "speciesnet"),
        )
    store.save_adaptive_model(
        "yasuni",
        "v2-model",
        "2026-08-18T10:30:00Z",
        "2026-08-18T10:30:00Z",
        {"protocol_version": "test-then-train-v2-canonical-targets"},
    )

    rows, bootstrap = service._pending_training_rows(
        "yasuni", store.adaptive_model("yasuni")
    )

    assert bootstrap is True
    assert [row["photo_id"] for row in rows] == [1, 2]
