from datetime import timedelta

import numpy as np
import pytest

from roadid.contracts import CameraSource
from roadid.sources import ReplaySource, SourceExhaustedError


def _source() -> CameraSource:
    return CameraSource(
        source_id="replay-test",
        name="Replay",
        adapter_type="replay",
        enabled=True,
        attribution="synthetic",
        refresh_seconds=0.25,
        options={"frame_count": 30, "width": 160, "height": 96},
    )


def test_replay_produces_deterministic_frames_ids_and_timestamps() -> None:
    first = list(ReplaySource(_source()).frames("run-a"))
    second = list(ReplaySource(_source()).frames("run-b"))

    assert [frame.frame_id for frame in first] == list(range(30))
    assert [frame.captured_at for frame in first] == [
        first[0].captured_at + timedelta(seconds=index * 0.25) for index in range(30)
    ]
    assert [frame.received_at for frame in first] == [frame.captured_at for frame in first]
    assert all(
        np.array_equal(left.image_bgr, right.image_bgr)
        for left, right in zip(first, second, strict=True)
    )


def test_replay_has_one_vehicle_track_only_on_frames_3_through_24() -> None:
    frames = list(ReplaySource(_source()).frames("run-a"))
    active = [frame for frame in frames if frame.source_metadata["vehicle_track_active"]]

    assert [frame.frame_id for frame in active] == list(range(3, 25))
    assert {frame.source_metadata["track_id"] for frame in active} == {"replay-test-track-0001"}
    assert all(frame.image_bgr.dtype == np.uint8 for frame in frames)
    assert all(frame.image_bgr.shape == (96, 160, 3) for frame in frames)


def test_replay_exhaustion_is_typed_and_resettable() -> None:
    replay = ReplaySource(_source())
    list(replay.frames("run-a"))

    with pytest.raises(SourceExhaustedError):
        replay.capture("run-a")

    replay.reset()
    assert replay.capture("run-b").frame_id == 0
