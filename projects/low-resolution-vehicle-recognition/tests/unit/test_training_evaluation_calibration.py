import hashlib

import numpy as np
import pytest

from roadid.contracts import Decision
from roadid.training.calibration import create_calibration_contract, select_threshold
from roadid.training.evaluation import (
    FrameResult,
    evaluate_frame_and_tracks,
    fuse_logits,
    hierarchy_decision,
)
from roadid.training.hierarchy import LabelHierarchy


def test_quality_weighted_fusion_is_order_invariant_and_improves_fixture() -> None:
    frames = [
        FrameResult(
            "001",
            "track-1",
            {level: (4.0, 0.0) for level in ("body_type", "make", "model_family")},
            {level: 0 for level in ("body_type", "make", "model_family")},
            0.9,
            24,
            0.2,
            False,
            "fixture",
        ),
        FrameResult(
            "002",
            "track-1",
            {level: (0.0, 5.0) for level in ("body_type", "make", "model_family")},
            {level: 0 for level in ("body_type", "make", "model_family")},
            0.1,
            12,
            0.8,
            False,
            "fixture",
        ),
    ]
    forward = fuse_logits([(4.0, 0.0), (0.0, 5.0)], [0.9, 0.1])
    reverse = fuse_logits([(0.0, 5.0), (4.0, 0.0)], [0.1, 0.9])
    report = evaluate_frame_and_tracks(frames)

    np.testing.assert_allclose(forward, reverse)
    assert report["track_fusion"]["quality_weighted"]["make"] == 1.0
    assert report["track_fusion"]["latest"]["make"] == 0.0
    assert report["slices"]["synthetic"]["count"] == 0


def test_calibration_contract_is_bound_and_rejects_overlap() -> None:
    dataset_hash = hashlib.sha256(b"manifest").hexdigest()
    contract = create_calibration_contract(
        method="temperature_scaling",
        dataset_manifest_sha256=dataset_hash,
        validation_identities={"v-1", "v-2"},
        test_identities={"t-1"},
        thresholds={"body_type": 0.6, "make": 0.7, "model_family": 0.8},
    )
    assert contract.dataset_manifest_sha256 == dataset_hash
    assert contract.validation_split_sha256 != contract.test_split_sha256
    with pytest.raises(ValueError, match="overlap"):
        create_calibration_contract(
            method="temperature_scaling",
            dataset_manifest_sha256=dataset_hash,
            validation_identities={"same"},
            test_identities={"same"},
            thresholds={"body_type": 0.6, "make": 0.7, "model_family": 0.8},
        )


def test_selective_threshold_maximizes_coverage_at_precision() -> None:
    threshold = select_threshold(
        [0.95, 0.8, 0.7, 0.6], [True, True, False, True], target_precision=1.0
    )
    assert threshold == 0.8


def test_hierarchy_decision_conditions_children_on_accepted_parent() -> None:
    hierarchy = LabelHierarchy(
        body_types=("sedan", "suv"),
        makes=("honda", "toyota"),
        model_families=("civic", "rav4"),
        make_to_body={"honda": "sedan", "toyota": "suv"},
        model_to_make={"civic": "honda", "rav4": "toyota"},
    )
    prediction = hierarchy_decision(
        track_id="track-1",
        logits={"body_type": (0.0, 6.0), "make": (9.0, 3.0), "model_family": (8.0, 2.0)},
        hierarchy=hierarchy,
        thresholds={"body_type": 0.6, "make": 0.002, "model_family": 0.002},
        usable_frames=3,
        disagreement=0.1,
        model_version="fixture",
    )
    assert prediction.body_type.label == "suv"
    assert prediction.make.label == "toyota"
    assert prediction.model_family.label == "rav4"
    assert prediction.decision is Decision.ACCEPT_BODY_MAKE_MODEL
