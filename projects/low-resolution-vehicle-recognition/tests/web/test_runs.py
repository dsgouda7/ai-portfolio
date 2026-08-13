from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from roadid.contracts import CameraSource, FramePacket, RunState
from roadid.web.runs import RunManager

SOURCE = CameraSource(
    source_id="replay",
    name="Replay",
    adapter_type="replay",
    enabled=True,
    attribution="fixture",
    refresh_seconds=0.01,
)


def frame(frame_id: int) -> FramePacket:
    now = datetime.now(UTC)
    return FramePacket(
        "ignored",
        SOURCE.source_id,
        frame_id,
        now,
        now,
        np.zeros((2, 2, 3), np.uint8),
    )


class Adapter:
    def __init__(self, frames, gate: threading.Event | None = None):
        self._frames = frames
        self._gate = gate
        self.closed = False

    def frames(self, run_id, stop_event):
        for item in self._frames:
            if self._gate is not None:
                self._gate.wait(1)
            if stop_event.is_set():
                return
            yield item

    def close(self):
        self.closed = True
        if self._gate is not None:
            self._gate.set()


class Registry:
    def __init__(self, adapter):
        self.adapter = adapter
        self.open_count = 0

    def list_sources(self):
        return [SOURCE]

    def open(self, source_id):
        self.open_count += 1
        return self.adapter


class Pipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.processed = []

    def process_frame(self, item):
        if self.fail:
            raise RuntimeError("private worker detail")
        self.processed.append(item.frame_id)
        return {"tracks": [{"track_id": "track-1", "usable_frames": len(self.processed)}]}

    def close(self):
        pass


class Factory:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.create_count = 0

    def create(self, run_id, source, options):
        self.create_count += 1
        return self.pipeline


def wait_state(run, expected):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if run.state is expected:
            return
        time.sleep(0.005)
    pytest.fail(f"run never reached {expected}: {run.state}")


def test_run_completes_with_one_worker_and_track_snapshot():
    registry = Registry(Adapter([frame(1), frame(2)]))
    factory = Factory(Pipeline())
    manager = RunManager(registry, factory)

    run = manager.create_run("replay")
    wait_state(run, RunState.COMPLETED)

    assert registry.open_count == factory.create_count == 1
    assert run.processed_frames == 2
    assert manager.tracks(run.run_id) == [{"track_id": "track-1", "usable_frames": 2}]


def test_pause_resume_and_idempotent_stop_are_contract_valid():
    gate = threading.Event()
    pipeline = Pipeline()
    manager = RunManager(Registry(Adapter([frame(1), frame(2)], gate)), Factory(pipeline))
    run = manager.create_run("replay")
    wait_state(run, RunState.RUNNING)

    manager.pause(run.run_id)
    assert run.state is RunState.PAUSED
    manager.resume(run.run_id)
    gate.set()
    wait_state(run, RunState.COMPLETED)

    assert manager.stop(run.run_id).state is RunState.COMPLETED
    assert manager.stop(run.run_id).state is RunState.COMPLETED


def test_pause_retains_at_most_the_inflight_frame_and_active_stop_is_idempotent():
    gate = threading.Event()
    pipeline = Pipeline()
    manager = RunManager(Registry(Adapter([frame(1), frame(2)], gate)), Factory(pipeline))
    run = manager.create_run("replay")
    wait_state(run, RunState.RUNNING)

    manager.pause(run.run_id)
    gate.set()
    deadline = time.monotonic() + 1
    while run.pending_frame is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert run.pending_frame is not None
    assert run.pending_frame.frame_id == 1
    assert pipeline.processed == []
    assert manager.stop(run.run_id).state is RunState.STOPPED
    assert manager.stop(run.run_id).state is RunState.STOPPED
    assert run.pending_frame is None


def test_one_source_cannot_start_duplicate_workers():
    gate = threading.Event()
    manager = RunManager(Registry(Adapter([frame(1)], gate)), Factory(Pipeline()))
    run = manager.create_run("replay")
    wait_state(run, RunState.RUNNING)

    with pytest.raises(RuntimeError, match="active run"):
        manager.create_run("replay")

    manager.stop(run.run_id)


def test_invalid_transition_and_unknown_ids_fail_closed():
    manager = RunManager(Registry(Adapter([])), Factory(Pipeline()))
    with pytest.raises(LookupError):
        manager.create_run("missing")
    with pytest.raises(LookupError):
        manager.get_run("missing")

    run = manager.create_run("replay")
    wait_state(run, RunState.COMPLETED)
    with pytest.raises(ValueError, match="invalid run transition"):
        manager.pause(run.run_id)


def test_worker_failure_is_terminal_and_private():
    manager = RunManager(Registry(Adapter([frame(1)])), Factory(Pipeline(fail=True)))
    run = manager.create_run("replay")
    wait_state(run, RunState.FAILED)

    assert run.error_code == "WORKER_FAILED"
    assert run.error_message == "Inference worker failed."
    assert "private" not in str(run.public_dict())
