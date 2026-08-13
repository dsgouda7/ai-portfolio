"""Calibration identity checks and hierarchical acceptance/abstention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from roadid.contracts import (
    CalibrationContract,
    Decision,
    LabelPrediction,
    VehiclePrediction,
)
from roadid.inference.classifier import HierarchicalScores, LabelSpace


class CalibrationMismatchError(RuntimeError):
    """Raised when calibration is bound to different data identities."""


@dataclass(frozen=True, slots=True)
class CalibrationTemperatures:
    body_type: float = 1.0
    make: float = 1.0
    model_family: float = 1.0

    def __post_init__(self) -> None:
        if min(self.body_type, self.make, self.model_family) <= 0:
            raise ValueError("calibration temperatures must be positive")


class TemperatureCalibrator:
    def __init__(
        self,
        contract: CalibrationContract,
        *,
        dataset_manifest_sha256: str,
        validation_split_sha256: str,
        temperatures: CalibrationTemperatures | None = None,
    ) -> None:
        validate_calibration_identity(
            contract,
            dataset_manifest_sha256=dataset_manifest_sha256,
            validation_split_sha256=validation_split_sha256,
        )
        self.contract = contract
        self.temperatures = temperatures or CalibrationTemperatures()

    def calibrate(self, scores: HierarchicalScores) -> HierarchicalScores:
        return HierarchicalScores(
            body_type=_apply_temperature(scores.body_type, self.temperatures.body_type),
            make=_apply_temperature(scores.make, self.temperatures.make),
            model_family=_apply_temperature(scores.model_family, self.temperatures.model_family),
        )


class HierarchicalDecisionEngine:
    def __init__(
        self,
        *,
        label_space: LabelSpace,
        calibration: CalibrationContract,
        model_version: str,
        model_to_make: Mapping[str, str],
        make_to_body: Mapping[str, str] | None = None,
        minimum_usable_frames: int = 1,
    ) -> None:
        if minimum_usable_frames <= 0:
            raise ValueError("minimum_usable_frames must be positive")
        self.label_space = label_space
        self.calibration = calibration
        self.model_version = model_version
        self.model_to_make = dict(model_to_make)
        self.make_to_body = dict(make_to_body or {})
        self.minimum_usable_frames = minimum_usable_frames

    def decide(
        self,
        *,
        track_id: str,
        scores: HierarchicalScores,
        usable_frames: int,
        disagreement: float,
    ) -> VehiclePrediction:
        if tuple(len(level) for level in scores.levels) != self.label_space.widths:
            raise ValueError("fused scores do not match configured label space")
        body_label, body_confidence = _winner(self.label_space.body_types, scores.body_type)
        make_label, make_confidence = _winner(self.label_space.makes, scores.make)
        model_label, model_confidence = _winner(
            self.label_space.model_families, scores.model_family
        )

        enough_evidence = usable_frames >= self.minimum_usable_frames
        body_accepted = enough_evidence and body_confidence >= self.calibration.body_threshold
        make_compatible = not self.make_to_body or self.make_to_body.get(make_label) == body_label
        make_accepted = (
            body_accepted and make_compatible and make_confidence >= self.calibration.make_threshold
        )
        model_accepted = (
            make_accepted
            and self.model_to_make.get(model_label) == make_label
            and model_confidence >= self.calibration.model_threshold
        )
        if model_accepted:
            decision = Decision.ACCEPT_BODY_MAKE_MODEL
        elif make_accepted:
            decision = Decision.ACCEPT_BODY_MAKE
        elif body_accepted:
            decision = Decision.ACCEPT_BODY_ONLY
        else:
            decision = Decision.INSUFFICIENT_VISUAL_EVIDENCE

        prediction = VehiclePrediction(
            track_id=track_id,
            body_type=LabelPrediction(body_label, body_confidence, body_accepted),
            make=LabelPrediction(make_label, make_confidence, make_accepted),
            model_family=LabelPrediction(model_label, model_confidence, model_accepted),
            decision=decision,
            usable_frames=usable_frames,
            disagreement=disagreement,
            model_version=self.model_version,
        )
        prediction.validate_hierarchy(self.model_to_make)
        return prediction


CalibratedHierarchicalDecision = HierarchicalDecisionEngine


def validate_calibration_identity(
    contract: CalibrationContract,
    *,
    dataset_manifest_sha256: str,
    validation_split_sha256: str,
) -> None:
    if contract.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise CalibrationMismatchError("calibration dataset manifest identity mismatch")
    if contract.validation_split_sha256 != validation_split_sha256:
        raise CalibrationMismatchError("calibration validation split identity mismatch")


def _apply_temperature(probabilities: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    exponentials = np.exp(logits - np.max(logits))
    return tuple((exponentials / exponentials.sum()).tolist())


def _winner(labels: tuple[str, ...], probabilities: tuple[float, ...]) -> tuple[str, float]:
    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index])
