from pathlib import Path
from types import SimpleNamespace

from wildscope.web.app import create_app


class Service:
    feeds = {"yasuni": object()}

    def __init__(self, image_path: Path):
        self.image = image_path

    def list_feeds(self):
        return [
            {
                "feed_id": "yasuni",
                "name": "Yasuni",
                "place_id": 68650,
                "country": "ECU",
                "habitat": "rainforest",
                "adaptive_model": None,
            }
        ]

    def start_sync(self, feed_id, *, hours):
        return SimpleNamespace(public_dict=lambda: {"job_id": "sync-1", "state": "running"})

    def start_training(self, feed_id):
        return SimpleNamespace(public_dict=lambda: {"job_id": "train-1", "state": "running"})

    def job(self, job_id):
        return SimpleNamespace(public_dict=lambda: {"job_id": job_id, "state": "completed"})

    def frames(self, feed_id, page):
        return {
            "items": [
                {
                    "photo_id": 1,
                    "common_name": "Jaguar",
                    "scientific_name": "Panthera onca",
                    "static_label": "Panthera onca",
                    "static_confidence": 0.9,
                    "static_identification": {
                        "scientific_name": "Panthera onca",
                        "common_name": "Jaguar",
                        "source_label": "Panthera onca",
                        "ambiguous": False,
                    },
                    "adaptive_label": "Panthera onca",
                    "adaptive_confidence": 0.95,
                    "adaptive_identification": {
                        "scientific_name": "Panthera onca",
                        "common_name": "Jaguar",
                        "source_label": "felidae",
                        "candidate_count": 2,
                        "ambiguous": True,
                    },
                    "cached_path": "private",
                    "sha256": "private",
                    "latitude": -0.7,
                    "longitude": -78.3,
                }
            ],
            "page": page,
            "pages": 1,
            "total": 1,
        }

    def locations(self, feed_id):
        return [
            {
                "anchor_photo_id": 1,
                "latitude": -0.7,
                "longitude": -78.3,
                "photo_count": 1,
                "common_name": "Jaguar",
            }
        ]

    def location_frames(self, feed_id, anchor_photo_id):
        return self.frames(feed_id, 1)["items"]

    def frame_detail(self, photo_id):
        return {
            **self.frames("yasuni", 1)["items"][0],
            "cached_width": 1600,
            "cached_height": 1000,
            "enhancement_method": "original-resolution-passthrough",
            "enhancement_applied": 0,
            "normalized_path": "private-normalized",
            "enhanced_path": "private-enhanced",
            "model_input_path": "private-input",
            "static_model_version": "speciesnet-test",
        }

    def training_dashboard(self, feed_id):
        return {
            "model": None,
            "live_batch": {
                "status": "bootstrap-ready",
                "evaluated_model_id": "SpeciesNet baseline",
                "window_from": None,
                "window_to": "2026-08-18T10:00:00Z",
                "eligible_samples": 1,
                "bootstrap_migration": True,
                "baseline": {"samples": 1, "correct": 1, "accuracy": 1.0},
                "deployed": {"samples": 1, "correct": None, "accuracy": None},
                "samples": [{"photo_id": 1}],
            },
            "confidence": {"sample_count": 1, "confidence_delta": None},
            "runs": [],
        }

    def image_path(self, photo_id, stage="source"):
        return self.image


def test_wildscope_routes_are_operational_and_private(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    app = create_app(service=Service(image))
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/training").status_code == 200
    assert client.get("/api/status").get_json()["service"] == "WildScope"
    assert client.get("/api/feeds").get_json()["feeds"][0]["feed_id"] == "yasuni"
    assert client.post("/api/feeds/yasuni/sync", json={"hours": 24}).status_code == 202
    assert client.post("/api/feeds/yasuni/train", json={}).status_code == 202
    frames = client.get("/api/feeds/yasuni/frames?page=1").get_json()
    assert frames["items"][0]["image_url"] == "/api/images/1"
    assert frames["items"][0]["obtained_identification"] == {
        "source": "iNaturalist community identification",
        "scientific_name": "Panthera onca",
        "common_name": "Jaguar",
        "quality_grade": None,
        "research_grade": False,
    }
    assert frames["items"][0]["static_match"] is True
    assert frames["items"][0]["adaptive_identification"]["common_name"] == "Jaguar"
    assert frames["items"][0]["adaptive_identification"]["ambiguous"] is True
    assert "cached_path" not in str(frames)
    assert "sha256" not in str(frames)
    assert client.get("/api/images/1").status_code == 200
    locations = client.get("/api/feeds/yasuni/locations").get_json()
    assert locations["locations"][0]["thumbnail_url"] == "/api/images/1"
    location_frames = client.get("/api/feeds/yasuni/locations/1/frames").get_json()
    assert location_frames["items"][0]["photo_id"] == 1
    detail = client.get("/api/frames/1").get_json()
    assert [stage["id"] for stage in detail["stages"]] == [
        "source",
        "normalized",
        "enhanced",
        "classification",
    ]
    assert detail["stages"][-1]["obtained"]["scientific_name"] == "Panthera onca"
    assert detail["stages"][-1]["static"]["matches_obtained"] is True
    assert detail["stages"][-1]["adaptive"]["identification"]["scientific_name"] == (
        "Panthera onca"
    )
    assert "cached_path" not in str(detail)
    dashboard = client.get("/api/feeds/yasuni/training").get_json()
    assert dashboard["confidence"]["sample_count"] == 1
    assert dashboard["live_batch"]["eligible_samples"] == 1
    assert dashboard["live_batch"]["samples"][0]["photo_id"] == 1
