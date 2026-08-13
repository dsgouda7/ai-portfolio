import numpy as np
import pytest

from roadid.contracts import CalibrationContract, CropQuality, Decision
from roadid.inference.calibration import (
    CalibrationMismatchError,
    HierarchicalDecisionEngine,
    TemperatureCalibrator,
)
from roadid.inference.classifier import HierarchicalScores, LabelSpace
from roadid.inference.fusion import TrackFuser
from roadid.inference.privacy import (
    DeterministicNoPIIRedactor,
    PrivacyGuard,
    PrivacyRedactionError,
)

SPACE = LabelSpace(
    body_types=("sedan", "suv"),
    makes=("ford", "toyota"),
    model_families=("focus", "rav4"),
)
CONTRACT = CalibrationContract(
    method="temperature_scaling",
    dataset_manifest_sha256="1" * 64,
    validation_split_sha256="2" * 64,
    test_split_sha256="3" * 64,
    body_threshold=0.7,
    make_threshold=0.7,
    model_threshold=0.8,
)


def _quality(weight: float) -> CropQuality:
    return CropQuality(24, 0.1, 0.1, 0.0, True, (), weight)


def test_quality_weighting_is_order_invariant_and_replayable() -> None:
    strong = HierarchicalScores((0.05, 0.95), (0.05, 0.95), (0.05, 0.95))
    weak = HierarchicalScores((0.8, 0.2), (0.8, 0.2), (0.8, 0.2))

    first = TrackFuser(SPACE)
    first.add(
        track_id="track-1",
        crop_id="crop-a",
        frame_id=1,
        quality=_quality(0.9),
        scores=strong,
    )
    fused = first.add(
        track_id="track-1",
        crop_id="crop-b",
        frame_id=2,
        quality=_quality(0.1),
        scores=weak,
    )
    replayed = first.replay(first.evidence("track-1"))

    second = TrackFuser(SPACE)
    second.add(
        track_id="track-1",
        crop_id="crop-b",
        frame_id=2,
        quality=_quality(0.1),
        scores=weak,
    )
    reversed_result = second.add(
        track_id="track-1",
        crop_id="crop-a",
        frame_id=1,
        quality=_quality(0.9),
        scores=strong,
    )

    assert fused.scores == replayed.scores == reversed_result.scores
    assert fused.scores.make[1] > 0.85
    assert fused.disagreement == pytest.approx(replayed.disagreement)
    assert 0.0 < fused.disagreement < 1.0


def test_calibration_identity_mismatch_fails_closed() -> None:
    with pytest.raises(CalibrationMismatchError, match="dataset"):
        TemperatureCalibrator(
            CONTRACT,
            dataset_manifest_sha256="9" * 64,
            validation_split_sha256="2" * 64,
        )


def test_hierarchy_accepts_make_but_abstains_model_below_threshold() -> None:
    engine = HierarchicalDecisionEngine(
        label_space=SPACE,
        calibration=CONTRACT,
        model_version="fixture",
        model_to_make={"focus": "ford", "rav4": "toyota"},
        make_to_body={"ford": "sedan", "toyota": "suv"},
    )
    prediction = engine.decide(
        track_id="track-1",
        scores=HierarchicalScores((0.05, 0.95), (0.1, 0.9), (0.25, 0.75)),
        usable_frames=3,
        disagreement=0.1,
    )

    assert prediction.decision is Decision.ACCEPT_BODY_MAKE
    assert prediction.make.accepted
    assert not prediction.model_family.accepted


def test_insufficient_frames_abstains_at_every_level() -> None:
    engine = HierarchicalDecisionEngine(
        label_space=SPACE,
        calibration=CONTRACT,
        model_version="fixture",
        model_to_make={"focus": "ford", "rav4": "toyota"},
        minimum_usable_frames=2,
    )
    prediction = engine.decide(
        track_id="track-1",
        scores=HierarchicalScores((0.01, 0.99), (0.01, 0.99), (0.01, 0.99)),
        usable_frames=1,
        disagreement=0.0,
    )

    assert prediction.decision is Decision.INSUFFICIENT_VISUAL_EVIDENCE
    assert not prediction.body_type.accepted


def test_public_privacy_fails_closed_and_replay_emits_no_text() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(PrivacyRedactionError, match="public-source"):
        PrivacyGuard(None).protect(image, frame_id=4, public_source=True)

    protected = PrivacyGuard(DeterministicNoPIIRedactor()).protect(
        image, frame_id=4, public_source=False
    )

    assert protected.result.safe_for_display
    assert protected.result.face_masks == protected.result.plate_masks == ()
    assert not hasattr(protected.result, "text")
