from roadid.contracts import StageStatus
from roadid.telemetry import EventRecorder


def test_failure_and_skip_helpers_are_terminal_and_ordered() -> None:
    recorder = EventRecorder("run-failure")
    recorder.start("vehicle_detection", frame_id=7)

    terminal_events = recorder.fail_stage_and_skip_remaining(
        "vehicle_detection",
        frame_id=7,
        error_code="DETECTOR_UNAVAILABLE",
        warning="detector did not initialize",
        duration_ms=12.0,
    )

    assert terminal_events[0].status is StageStatus.FAILED
    assert terminal_events[0].error_code == "DETECTOR_UNAVAILABLE"
    assert all(event.status is StageStatus.SKIPPED for event in terminal_events[1:])
    assert all(event.duration_ms is not None for event in terminal_events)
    assert [event.stage for event in terminal_events] == [
        "vehicle_detection",
        "track_association",
        "crop_quality",
        "frame_classification",
        "track_fusion",
        "calibration",
        "hierarchy_decision",
        "privacy_render",
    ]


def test_warning_and_skip_require_running() -> None:
    recorder = EventRecorder("run-warning")
    recorder.start("frame_validation")
    warning = recorder.warn("frame_validation", warning="stale frame", duration_ms=1.0)
    recorder.start("vehicle_detection")
    skipped = recorder.skip("vehicle_detection", reason="no fresh frame", duration_ms=0.0)

    assert warning.status is StageStatus.WARNING
    assert skipped.status is StageStatus.SKIPPED
