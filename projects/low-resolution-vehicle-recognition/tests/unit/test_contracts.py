from datetime import UTC, datetime

import pytest

from roadid.contracts import (
    Decision,
    LabelPrediction,
    PipelineEvent,
    RunState,
    StageStatus,
    VehiclePrediction,
    assert_run_transition,
)


def test_hierarchy_acceptance_is_consistent() -> None:
    prediction = VehiclePrediction(
        track_id="track-1",
        body_type=LabelPrediction("suv", 0.95, True),
        make=LabelPrediction("toyota", 0.88, True),
        model_family=LabelPrediction("rav4", 0.61, False),
        decision=Decision.ACCEPT_BODY_MAKE,
        usable_frames=8,
        disagreement=0.12,
        model_version="fixture",
    )
    prediction.validate_hierarchy({"rav4": "toyota"})


def test_invalid_run_transition_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid run transition"):
        assert_run_transition(RunState.STOPPED, RunState.RUNNING)


def test_terminal_event_requires_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        PipelineEvent(
            event_id="event-1",
            sequence_id=1,
            run_id="run-1",
            stage="vehicle_detection",
            status=StageStatus.COMPLETED,
            started_at=datetime.now(UTC),
        )
