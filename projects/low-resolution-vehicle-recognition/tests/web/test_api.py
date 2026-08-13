from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from roadid.contracts import CameraSource, FramePacket, PipelineEvent, StageStatus
from roadid.settings import Paths, Settings, WebSettings
from roadid.web import create_app
from roadid.web.dependencies import RecorderHub

SOURCE = CameraSource(
    source_id="replay",
    name="Replay",
    adapter_type="replay",
    enabled=True,
    attribution="fixture",
    refresh_seconds=0.01,
    options={"source_url": "https://private.invalid/frame", "token": "secret"},
)


class Adapter:
    def __init__(self, gate=None):
        self.gate = gate

    def frames(self, run_id, stop_event):
        if self.gate is not None:
            self.gate.wait(1)
        if stop_event.is_set():
            return
        now = datetime.now(UTC)
        yield FramePacket(run_id, "replay", 1, now, now, np.zeros((2, 2, 3), np.uint8))

    def close(self):
        if self.gate is not None:
            self.gate.set()


class Registry:
    def __init__(self, gate=None):
        self.adapter = Adapter(gate)
        self.opens = 0

    def list_sources(self):
        return [SOURCE]

    def open(self, source_id):
        self.opens += 1
        return self.adapter


class Recorder:
    def __init__(self):
        self.events = []

    def events_after(self, run_id, sequence_id):
        return [
            event
            for event in self.events
            if event.run_id == run_id and event.sequence_id > sequence_id
        ]


class Pipeline:
    def __init__(self, run_id, recorder, fail=False):
        self.run_id = run_id
        self.recorder = recorder
        self.fail = fail

    def process_frame(self, frame):
        if self.fail:
            raise RuntimeError("source URL https://private.invalid")
        now = datetime.now(UTC)
        self.recorder.events.extend(
            [
                PipelineEvent(
                    "event-2",
                    2,
                    self.run_id,
                    "frame_validation",
                    StageStatus.COMPLETED,
                    now,
                    duration_ms=1,
                ),
                PipelineEvent(
                    "event-1",
                    1,
                    self.run_id,
                    "source_acquisition",
                    StageStatus.COMPLETED,
                    now,
                    duration_ms=1,
                ),
            ]
        )
        return {
            "frame_jpeg": b"\xff\xd8\xff\xd9",
            "tracks": [{"track_id": "track-1", "decision": "ACCEPT_BODY_ONLY"}],
            "report": {"model_version": "fixture", "credential": "hidden"},
        }

    def close(self):
        pass


class Factory:
    def __init__(self, recorder, fail=False):
        self.recorder = recorder
        self.fail = fail
        self.creates = 0
        self.options = None

    def create(self, run_id, source, options):
        self.creates += 1
        self.options = options
        return Pipeline(run_id, self.recorder, self.fail)


@pytest.fixture
def app(tmp_path):
    recorder = Recorder()
    registry = Registry()
    factory = Factory(recorder)
    settings = Settings(
        web=WebSettings(),
        paths=Paths(tmp_path, tmp_path / "inference.yaml", tmp_path / "sources.yaml", None),
        inference={"runtime": {"heartbeat_seconds": 0.01}},
    )
    app = create_app(
        source_registry=registry,
        pipeline_factory=factory,
        recorder=recorder,
        settings=settings,
    )
    app.config.update(TESTING=True)
    yield app
    app.extensions["roadid"]["shutdown"]()


def wait_terminal(client, run_id):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        payload = client.get(f"/api/runs/{run_id}").get_json()["run"]
        if payload["state"] in {"completed", "failed", "stopped"}:
            return payload
        time.sleep(0.005)
    pytest.fail("run did not terminate")


def test_status_sources_and_run_endpoint_schemas_are_private(app):
    client = app.test_client()
    assert client.get("/").status_code == 200
    status = client.get("/api/status").get_json()
    assert status["service"] == "CarFace" and status["ready"] is True
    assert status["worker"] == {"ready": True, "active_runs": 0, "total_runs": 0}
    assert set(status["model"]) == {"ready", "model_id", "version", "bundle_configured"}

    sources = client.get("/api/sources").get_json()
    assert list(sources) == ["sources"]
    assert "options" not in sources["sources"][0]
    assert "private.invalid" not in str(sources)
    assert "secret" not in str(sources)

    response = client.post(
        "/api/runs",
        json={"source_id": "replay", "processing_fps": 2, "confidence_profile": "strict"},
    )
    assert response.status_code == 201
    run_id = response.get_json()["run"]["run_id"]
    terminal = wait_terminal(client, run_id)
    assert terminal["state"] == "completed"
    assert set(terminal) == {
        "run_id",
        "source_id",
        "state",
        "created_at",
        "started_at",
        "finished_at",
        "processed_frames",
        "latest_frame_id",
        "track_count",
        "error",
    }
    assert app.extensions["roadid"]["manager"].get_run(run_id).options == {
        "processing_fps": 2.0,
        "confidence_profile": "strict",
    }


def test_track_frame_and_report_endpoints(app):
    client = app.test_client()
    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]
    wait_terminal(client, run_id)

    tracks = client.get(f"/api/runs/{run_id}/tracks").get_json()
    assert tracks["tracks"][0]["track_id"] == "track-1"
    assert client.get("/api/tracks/track-1").get_json()["track"]["decision"] == "ACCEPT_BODY_ONLY"
    frame_response = client.get(f"/api/runs/{run_id}/frame")
    assert frame_response.status_code == 200 and frame_response.mimetype == "image/jpeg"
    assert frame_response.headers["X-Frame-ID"] == "1"
    report = client.get(f"/api/runs/{run_id}/report").get_json()
    assert report["report"]["schema_version"] == 1
    assert "credential" not in str(report)


def test_unknown_ids_invalid_inputs_and_transitions_are_typed(app):
    client = app.test_client()
    assert client.get("/api/runs/missing").get_json()["error"]["code"] == "RUN_NOT_FOUND"
    assert client.get("/api/tracks/missing").get_json()["error"]["code"] == "TRACK_NOT_FOUND"
    assert client.post("/api/runs", json={"source_url": "https://example.com"}).status_code == 400
    assert client.post("/api/runs", json={"source_id": "missing"}).status_code == 404

    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]
    wait_terminal(client, run_id)
    response = client.post(f"/api/runs/{run_id}/pause")
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "INVALID_RUN_TRANSITION"
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 200
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 200


def test_sse_uses_recorder_order_and_last_event_id(app):
    client = app.test_client()
    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]
    wait_terminal(client, run_id)

    response = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "1"})
    body = response.get_data(as_text=True)
    assert "id: 1" not in body
    assert body.index("id: 2") >= 0
    assert "event: pipeline" in body


def test_sse_accepts_shared_event_recorder_batches(tmp_path):
    from roadid.telemetry.recorder import EventRecorder

    gate = threading.Event()
    recorder = RecorderHub(EventRecorder, capacity=20)
    registry = Registry(gate)
    factory = Factory(recorder)
    settings = Settings(
        WebSettings(),
        Paths(tmp_path, tmp_path / "i", tmp_path / "s", None),
        {"runtime": {"heartbeat_seconds": 0.01}},
    )
    app = create_app(
        source_registry=registry,
        pipeline_factory=factory,
        recorder=recorder,
        settings=settings,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]
    run_recorder = recorder.for_run(run_id)
    run_recorder.start("source_acquisition", frame_id=1)
    run_recorder.complete("source_acquisition", frame_id=1, duration_ms=1.0)

    body = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "1"}).get_data(
        as_text=True
    )

    assert "id: 1" not in body
    assert "id: 2" in body
    app.extensions["roadid"]["shutdown"]()


def test_worker_failure_is_visible_as_sanitized_terminal_run(tmp_path):
    recorder = Recorder()
    settings = Settings(
        WebSettings(),
        Paths(tmp_path, tmp_path / "i", tmp_path / "s", None),
        {"runtime": {"heartbeat_seconds": 0.01}},
    )
    app = create_app(
        source_registry=Registry(),
        pipeline_factory=Factory(recorder, fail=True),
        recorder=recorder,
        settings=settings,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]

    terminal = wait_terminal(client, run_id)

    assert terminal["state"] == "failed"
    assert terminal["error"] == {"code": "WORKER_FAILED", "message": "Inference worker failed."}
    assert "private.invalid" not in str(terminal)
    app.extensions["roadid"]["shutdown"]()


def test_duplicate_sse_subscribers_do_not_create_workers_and_receive_heartbeats(tmp_path):
    gate = threading.Event()
    recorder = Recorder()
    registry = Registry(gate)
    factory = Factory(recorder)
    settings = Settings(
        WebSettings(),
        Paths(tmp_path, tmp_path / "i", tmp_path / "s", None),
        {"runtime": {"heartbeat_seconds": 0.01}},
    )
    app = create_app(
        source_registry=registry,
        pipeline_factory=factory,
        recorder=recorder,
        settings=settings,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    run_id = client.post("/api/runs", json={"source_id": "replay"}).get_json()["run"]["run_id"]

    first = client.get(f"/api/runs/{run_id}/events").get_data(as_text=True)
    second = client.get(f"/api/runs/{run_id}/events").get_data(as_text=True)
    assert first.count(": heartbeat") == second.count(": heartbeat") == 3
    assert registry.opens == factory.creates == 1
    app.extensions["roadid"]["shutdown"]()


def test_duplicate_active_run_is_a_typed_conflict(tmp_path):
    gate = threading.Event()
    recorder = Recorder()
    settings = Settings(
        WebSettings(),
        Paths(tmp_path, tmp_path / "i", tmp_path / "s", None),
        {"runtime": {"heartbeat_seconds": 0.01}},
    )
    app = create_app(
        source_registry=Registry(gate),
        pipeline_factory=Factory(recorder),
        recorder=recorder,
        settings=settings,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    first = client.post("/api/runs", json={"source_id": "replay"})

    second = client.post("/api/runs", json={"source_id": "replay"})

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.get_json()["error"]["code"] == "SOURCE_BUSY"
    app.extensions["roadid"]["shutdown"]()


def test_route_map_has_no_training_or_remote_url_surface(app):
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert not any("train" in rule for rule in rules)
    assert not any("url" in rule for rule in rules)


def test_default_builder_runs_offline_replay_profile():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    status = client.get("/api/status").get_json()
    response = client.post("/api/runs", json={"source_id": "replay-demo"})

    assert status["components"]["sources"] is True
    assert status["components"]["telemetry"] is True
    assert status["components"]["inference"] is True
    assert status["model_version"] == "deterministic-replay-v1"
    assert response.status_code == 201
    run_id = response.get_json()["run"]["run_id"]
    assert wait_terminal(client, run_id)["state"] == "completed"
    assert client.get(f"/api/runs/{run_id}/frame").status_code == 200

    restarted = client.post("/api/runs", json={"source_id": "replay-demo"})
    restarted_id = restarted.get_json()["run"]["run_id"]
    assert restarted.status_code == 201
    assert wait_terminal(client, restarted_id)["processed_frames"] == 30
    app.extensions["roadid"]["shutdown"]()
