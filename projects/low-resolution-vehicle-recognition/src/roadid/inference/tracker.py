"""Production ByteTrack adapter and deterministic test/demo association."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from roadid.contracts import Detection


class TrackerUnavailableError(RuntimeError):
    """Raised when the optional production tracker is unavailable."""


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    track_id: str
    detection: Detection


@dataclass(slots=True)
class _TrackState:
    track_id: str
    bbox: tuple[int, int, int, int]
    last_seen_frame: int


class DeterministicTracker:
    """Deterministic IoU/centroid tracker for tests and demo replay only."""

    DEFAULT_PROFILE_LABEL = "deterministic-centroid-iou-test-demo-only"

    def __init__(
        self,
        *,
        lost_track_buffer: int = 20,
        matching_iou: float = 0.2,
        maximum_centroid_distance: float = 80.0,
        profile_label: str = DEFAULT_PROFILE_LABEL,
    ) -> None:
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be non-negative")
        if not 0.0 <= matching_iou <= 1.0:
            raise ValueError("matching_iou must be in [0, 1]")
        if not profile_label:
            raise ValueError("deterministic tracker requires an explicit profile label")
        self.lost_track_buffer = lost_track_buffer
        self.matching_iou = matching_iou
        self.maximum_centroid_distance = maximum_centroid_distance
        self.profile_label = profile_label
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    def update(
        self, detections: Sequence[Detection], frame_id: int | None = None
    ) -> tuple[TrackedDetection, ...]:
        resolved_frame_id = _resolve_frame_id(detections, frame_id)
        self._expire(resolved_frame_id)
        available_tracks = set(self._tracks)
        assigned: list[TrackedDetection] = []
        for detection in sorted(detections, key=lambda item: item.bbox_xyxy):
            track_id = self._best_match(detection, available_tracks)
            if track_id is None:
                track_id = f"track-{self._next_track_number:05d}"
                self._next_track_number += 1
            else:
                available_tracks.remove(track_id)
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox_xyxy,
                last_seen_frame=resolved_frame_id,
            )
            assigned.append(TrackedDetection(track_id, detection))
        return tuple(assigned)

    def _expire(self, frame_id: int) -> None:
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if frame_id - state.last_seen_frame > self.lost_track_buffer
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _best_match(self, detection: Detection, available: set[str]) -> str | None:
        candidates = []
        for track_id in sorted(available):
            state = self._tracks[track_id]
            overlap = bbox_iou(detection.bbox_xyxy, state.bbox)
            distance = centroid_distance(detection.bbox_xyxy, state.bbox)
            if overlap >= self.matching_iou or distance <= self.maximum_centroid_distance:
                candidates.append((-overlap, distance, track_id))
        return min(candidates)[2] if candidates else None


DeterministicCentroidIoUTracker = DeterministicTracker


class ByteTrackTracker:
    """Lazy adapter for supervision.ByteTrack used by production profiles."""

    profile_label = "production-bytetrack-supervision"

    def __init__(
        self,
        *,
        activation_threshold: float = 0.25,
        matching_threshold: float = 0.8,
        lost_track_buffer: int = 20,
        frame_rate: int = 30,
        tracker: Any | None = None,
    ) -> None:
        self.activation_threshold = activation_threshold
        self.matching_threshold = matching_threshold
        self.lost_track_buffer = lost_track_buffer
        self.frame_rate = frame_rate
        self._tracker = tracker
        self._supervision: Any | None = None

    def update(
        self, detections: Sequence[Detection], frame_id: int | None = None
    ) -> tuple[TrackedDetection, ...]:
        _resolve_frame_id(detections, frame_id)
        self._ensure_tracker()
        if not detections:
            empty = self._supervision.Detections.empty()
            self._tracker.update_with_detections(empty)
            return ()
        xyxy = np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32)
        confidence = np.asarray([item.confidence for item in detections], dtype=np.float32)
        class_id = np.arange(len(detections), dtype=int)
        native = self._supervision.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self._tracker.update_with_detections(native)
        results = []
        tracker_ids = getattr(tracked, "tracker_id", None)
        source_ids = getattr(tracked, "class_id", None)
        if tracker_ids is None or source_ids is None:
            return ()
        for tracker_id, source_id in zip(tracker_ids, source_ids, strict=True):
            if tracker_id is None:
                continue
            source_index = int(source_id)
            if not 0 <= source_index < len(detections):
                continue
            results.append(
                TrackedDetection(
                    track_id=f"track-{int(tracker_id):05d}",
                    detection=detections[source_index],
                )
            )
        return tuple(results)

    def _ensure_tracker(self) -> None:
        if self._tracker is not None and self._supervision is not None:
            return
        try:
            import supervision as supervision
        except ImportError as error:
            raise TrackerUnavailableError(
                "ByteTrack requires the optional 'tracking' dependency: install roadid[tracking]"
            ) from error
        self._supervision = supervision
        if self._tracker is None:
            self._tracker = supervision.ByteTrack(
                track_activation_threshold=self.activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.matching_threshold,
                frame_rate=self.frame_rate,
            )


SupervisionByteTrackTracker = ByteTrackTracker


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def centroid_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    return float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))


def _resolve_frame_id(detections: Sequence[Detection], frame_id: int | None) -> int:
    ids = {item.frame_id for item in detections}
    if frame_id is None:
        if len(ids) != 1:
            raise ValueError("frame_id is required for an empty or mixed-frame detection batch")
        return next(iter(ids))
    if ids and ids != {frame_id}:
        raise ValueError("detection frame IDs do not match tracker frame_id")
    return frame_id
