"""Order-invariant quality-weighted fusion with bounded replayable evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from roadid.contracts import CropQuality, EvidenceItem, TrackEvidence
from roadid.inference.classifier import HierarchicalScores, LabelSpace


class InsufficientEvidenceError(RuntimeError):
    """Raised when a track has no positive-weight usable evidence."""


@dataclass(frozen=True, slots=True)
class FusionResult:
    scores: HierarchicalScores
    disagreement: float
    total_weight: float


class EvidenceLedger:
    """Keep the strongest deterministic evidence subset for one track."""

    def __init__(self, track_id: str, max_items: int = 16) -> None:
        if not track_id or max_items <= 0:
            raise ValueError("evidence ledger requires a track ID and positive capacity")
        self.track_id = track_id
        self.max_items = max_items
        self._items: dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> None:
        existing = self._items.get(item.crop_id)
        if existing is not None and existing != item:
            raise ValueError(f"crop ID already has different evidence: {item.crop_id}")
        self._items[item.crop_id] = item
        ranked = sorted(
            self._items.values(),
            key=lambda value: (-value.fusion_weight, value.frame_id, value.crop_id),
        )
        self._items = {value.crop_id: value for value in ranked[: self.max_items]}

    def snapshot(self) -> TrackEvidence:
        items = tuple(
            sorted(self._items.values(), key=lambda value: (value.frame_id, value.crop_id))
        )
        return TrackEvidence(track_id=self.track_id, items=items)


class TrackFuser:
    def __init__(self, label_space: LabelSpace, max_evidence_per_track: int = 16) -> None:
        self.label_space = label_space
        self.max_evidence_per_track = max_evidence_per_track
        self._ledgers: dict[str, EvidenceLedger] = {}

    def add(
        self,
        *,
        track_id: str,
        crop_id: str,
        frame_id: int,
        quality: CropQuality,
        scores: HierarchicalScores,
    ) -> FusionResult:
        if tuple(len(level) for level in scores.levels) != self.label_space.widths:
            raise ValueError("frame prediction does not match configured label space")
        ledger = self._ledgers.setdefault(
            track_id, EvidenceLedger(track_id, self.max_evidence_per_track)
        )
        ledger.add(
            EvidenceItem(
                crop_id=crop_id,
                frame_id=frame_id,
                quality=quality,
                fusion_weight=quality.fusion_weight,
                frame_prediction=scores.flatten(),
            )
        )
        return replay_evidence(ledger.snapshot(), self.label_space)

    def evidence(self, track_id: str) -> TrackEvidence:
        if track_id not in self._ledgers:
            return TrackEvidence(track_id=track_id, items=())
        return self._ledgers[track_id].snapshot()

    def replay(self, evidence: TrackEvidence) -> FusionResult:
        return replay_evidence(evidence, self.label_space)


QualityWeightedFusion = TrackFuser


def replay_evidence(evidence: TrackEvidence, label_space: LabelSpace) -> FusionResult:
    positive = [item for item in evidence.items if item.quality.usable and item.fusion_weight > 0]
    if not positive:
        raise InsufficientEvidenceError(f"track {evidence.track_id} has no usable evidence")
    ordered = sorted(positive, key=lambda item: (item.frame_id, item.crop_id))
    weights = np.asarray([item.fusion_weight for item in ordered], dtype=float)
    matrix = np.asarray([item.frame_prediction for item in ordered], dtype=float)
    if matrix.shape[1] != label_space.size:
        raise ValueError("evidence prediction width does not match configured label space")
    fused = np.average(matrix, axis=0, weights=weights)
    scores = HierarchicalScores.from_flat(fused, label_space)
    disagreement = _disagreement(matrix, weights, scores, label_space)
    return FusionResult(scores=scores, disagreement=disagreement, total_weight=float(weights.sum()))


def quality_weighted_fusion(
    items: Iterable[EvidenceItem], label_space: LabelSpace, track_id: str = "track"
) -> FusionResult:
    return replay_evidence(TrackEvidence(track_id=track_id, items=tuple(items)), label_space)


def _disagreement(
    matrix: np.ndarray,
    weights: np.ndarray,
    fused: HierarchicalScores,
    space: LabelSpace,
) -> float:
    starts = np.cumsum((0, *space.widths))
    fused_flat = np.asarray(fused.flatten())
    row_disagreement = []
    for row in matrix:
        level_distances = [
            0.5
            * np.abs(
                row[starts[index] : starts[index + 1]]
                - fused_flat[starts[index] : starts[index + 1]]
            ).sum()
            for index in range(3)
        ]
        row_disagreement.append(float(np.mean(level_distances)))
    return float(max(0.0, min(1.0, np.average(row_disagreement, weights=weights))))
