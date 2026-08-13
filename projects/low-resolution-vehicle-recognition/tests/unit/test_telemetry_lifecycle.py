from types import MappingProxyType
from typing import Any, cast

import pytest

from roadid.contracts import StageStatus
from roadid.telemetry import PIPELINE_STAGES, STAGE_ORDER, EventRecorder


def test_stage_order_is_frozen() -> None:
    assert PIPELINE_STAGES == (
        "source_acquisition",
        "frame_validation",
        "vehicle_detection",
        "track_association",
        "crop_quality",
        "frame_classification",
        "track_fusion",
        "calibration",
        "hierarchy_decision",
        "privacy_render",
    )
    assert isinstance(STAGE_ORDER, MappingProxyType)
    with pytest.raises(TypeError):
        cast(Any, STAGE_ORDER)["source_acquisition"] = 99


def test_terminal_event_requires_running_and_has_duration() -> None:
    recorder = EventRecorder("run-1")

    try:
        recorder.complete_stage("vehicle_detection")
    except ValueError as error:
        assert "must be running" in str(error)
    else:
        raise AssertionError("completion without a running event must fail")

    running = recorder.start_stage("vehicle_detection", input_summary={"detections": 0})
    completed = recorder.complete_stage(
        "vehicle_detection", output_summary={"detections": 2}, duration_ms=3.5
    )

    assert running.status is StageStatus.RUNNING
    assert completed.status is StageStatus.COMPLETED
    assert (running.sequence_id, completed.sequence_id) == (1, 2)
    assert completed.duration_ms == 3.5
    assert completed.input_summary == {"detections": 0}
    assert completed.output_summary == {"detections": 2}


def test_non_finite_duration_is_rejected_without_closing_active_stage() -> None:
    recorder = EventRecorder("run-duration")
    recorder.start("calibration")

    with pytest.raises(ValueError, match="finite"):
        recorder.complete("calibration", duration_ms=float("nan"))

    completed = recorder.complete("calibration", duration_ms=1.0)
    assert completed.duration_ms == 1.0
