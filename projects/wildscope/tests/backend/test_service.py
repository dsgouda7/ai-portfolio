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

    assert sync.details["static_predictions"] == 2
    assert training.details["new_samples"] == 2
    assert training.details["training_samples"] == 2
    assert training.details["evaluated_model_id"] == "SpeciesNet baseline"
    assert training.details["trained_model_id"] == training.details["model_id"]
    assert training.details["baseline_accuracy"] == 0.0
    assert training.details["deployed_accuracy"] is None
    assert training.details["training_agreement"] == 1.0
    assert training.details["duration_seconds"] >= 0
    assert model["payload"]["training_photo_ids"] == [1, 2]
    assert model["payload"]["protocol_version"] == "test-then-train-v1"
    assert [sample["photo_id"] for sample in training.details["samples"]] == [1, 2]
    assert model["watermark"] == "2026-08-18T11:00:00Z"
    assert store.frames("yasuni", page=1)["items"][0]["adaptive_label"] == "Panthera onca"

    second = service.start_training("yasuni")
    deadline = time.monotonic() + 2
    while second.state in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.005)
    assert second.state == "failed"
    assert "no new labeled" in second.error
    assert service.client.since_calls == ["2026-08-18T11:00:00Z"]


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
        {"counts": {"felidae": {"Panthera onca": 2}}, "sample_count": 2},
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
    assert model["payload"]["protocol_version"] == "test-then-train-v1"
