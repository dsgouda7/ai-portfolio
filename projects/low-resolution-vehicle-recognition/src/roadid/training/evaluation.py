"""Frame metrics, quality-weighted track fusion, and selective decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from roadid.contracts import Decision, LabelPrediction, VehiclePrediction
from roadid.training.calibration import softmax
from roadid.training.hierarchy import LEVELS, LabelHierarchy


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_id: str
    track_id: str
    logits: Mapping[str, tuple[float, ...]]
    target_indices: Mapping[str, int]
    quality_weight: float
    apparent_height_px: int
    blur_score: float
    synthetic: bool
    source_id: str


def fuse_logits(
    frame_logits: Sequence[Sequence[float]], weights: Sequence[float] | None = None
) -> np.ndarray:
    values = np.asarray(frame_logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("frame logits must be a non-empty finite matrix")
    fusion_weights = (
        np.ones(values.shape[0], dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if fusion_weights.shape != (values.shape[0],) or np.any(fusion_weights < 0):
        raise ValueError("fusion weights must be aligned and non-negative")
    if fusion_weights.sum() <= 0:
        raise ValueError("at least one fusion weight must be positive")
    return np.average(values, axis=0, weights=fusion_weights)


def evaluate_frame_and_tracks(frames: Sequence[FrameResult]) -> dict[str, object]:
    if not frames:
        raise ValueError("evaluation requires frame results")
    report: dict[str, object] = {
        "frame": _frame_accuracy(frames),
        "track_fusion": _track_fusion_accuracy(frames),
        "slices": {
            "real": _frame_accuracy([frame for frame in frames if not frame.synthetic]),
            "synthetic": _frame_accuracy([frame for frame in frames if frame.synthetic]),
            "apparent_height": _binned_accuracy(frames, "height"),
            "blur": _binned_accuracy(frames, "blur"),
        },
    }
    return report


def hierarchy_decision(
    *,
    track_id: str,
    logits: Mapping[str, Sequence[float]],
    hierarchy: LabelHierarchy,
    thresholds: Mapping[str, float],
    temperatures: Mapping[str, float] | None = None,
    usable_frames: int,
    disagreement: float,
    model_version: str,
) -> VehiclePrediction:
    labels = {
        "body_type": hierarchy.body_types,
        "make": hierarchy.makes,
        "model_family": hierarchy.model_families,
    }
    temperatures = temperatures or {}
    probabilities = {
        level: softmax(
            np.asarray(logits[level], dtype=np.float64),
            temperature=float(temperatures.get(level, 1.0)),
        )
        for level in LEVELS
    }

    body_index = int(np.argmax(probabilities["body_type"]))
    body_label = labels["body_type"][body_index]
    body_confidence = float(probabilities["body_type"][body_index])
    body_accepted = body_confidence >= thresholds["body_type"]

    make_indices = [
        index
        for index, label in enumerate(labels["make"])
        if hierarchy.make_to_body[label] == body_label
    ]
    make_index = _best_allowed(probabilities["make"], make_indices)
    make_label = labels["make"][make_index]
    make_confidence = float(probabilities["make"][make_index])
    make_accepted = body_accepted and make_confidence >= thresholds["make"]

    model_indices = [
        index
        for index, label in enumerate(labels["model_family"])
        if hierarchy.model_to_make[label] == make_label
    ]
    model_index = _best_allowed(probabilities["model_family"], model_indices)
    model_label = labels["model_family"][model_index]
    model_confidence = float(probabilities["model_family"][model_index])
    model_accepted = make_accepted and model_confidence >= thresholds["model_family"]

    decision = (
        Decision.ACCEPT_BODY_MAKE_MODEL
        if model_accepted
        else Decision.ACCEPT_BODY_MAKE
        if make_accepted
        else Decision.ACCEPT_BODY_ONLY
        if body_accepted
        else Decision.INSUFFICIENT_VISUAL_EVIDENCE
    )
    prediction = VehiclePrediction(
        track_id=track_id,
        body_type=LabelPrediction(body_label, body_confidence, body_accepted),
        make=LabelPrediction(make_label, make_confidence, make_accepted),
        model_family=LabelPrediction(model_label, model_confidence, model_accepted),
        decision=decision,
        usable_frames=usable_frames,
        disagreement=disagreement,
        model_version=model_version,
    )
    prediction.validate_hierarchy(hierarchy.model_to_make)
    return prediction


def _best_allowed(probabilities: np.ndarray, indices: Sequence[int]) -> int:
    if not indices:
        raise ValueError("hierarchy parent has no child labels")
    return max(indices, key=lambda index: float(probabilities[index]))


def _frame_accuracy(frames: Sequence[FrameResult]) -> dict[str, object]:
    result: dict[str, object] = {"count": len(frames)}
    for level in LEVELS:
        labeled = [frame for frame in frames if frame.target_indices[level] >= 0]
        correct = sum(
            int(np.argmax(frame.logits[level]) == frame.target_indices[level]) for frame in labeled
        )
        result[level] = {
            "labeled": len(labeled),
            "top1_accuracy": correct / len(labeled) if labeled else None,
        }
    return result


def _track_fusion_accuracy(frames: Sequence[FrameResult]) -> dict[str, object]:
    tracks: dict[str, list[FrameResult]] = {}
    for frame in frames:
        tracks.setdefault(frame.track_id, []).append(frame)
    methods = ("latest", "best_quality", "uniform", "quality_weighted")
    correct = {method: {level: 0 for level in LEVELS} for method in methods}
    labeled = {level: 0 for level in LEVELS}
    for track_frames in tracks.values():
        ordered = sorted(track_frames, key=lambda frame: frame.frame_id)
        target = ordered[0].target_indices
        for level in LEVELS:
            if target[level] < 0:
                continue
            labeled[level] += 1
            matrices = [frame.logits[level] for frame in ordered]
            predictions = {
                "latest": int(np.argmax(matrices[-1])),
                "best_quality": int(
                    np.argmax(max(ordered, key=lambda frame: frame.quality_weight).logits[level])
                ),
                "uniform": int(np.argmax(fuse_logits(matrices))),
                "quality_weighted": int(
                    np.argmax(fuse_logits(matrices, [frame.quality_weight for frame in ordered]))
                ),
            }
            for method, prediction in predictions.items():
                correct[method][level] += int(prediction == target[level])
    return {
        "track_count": len(tracks),
        **{
            method: {
                level: correct[method][level] / labeled[level] if labeled[level] else None
                for level in LEVELS
            }
            for method in methods
        },
    }


def _binned_accuracy(frames: Sequence[FrameResult], kind: str) -> dict[str, object]:
    if kind == "height":
        bins = ((0, 16), (16, 32), (32, 64), (64, 10_000))

        def value(frame):
            return float(frame.apparent_height_px)
    else:
        bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))

        def value(frame):
            return frame.blur_score

    return {
        f"{lower:g}-{upper:g}": _frame_accuracy(
            [frame for frame in frames if lower <= value(frame) < upper]
        )
        for lower, upper in bins
    }
