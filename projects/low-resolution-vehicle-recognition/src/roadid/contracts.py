"""Shared contracts for training, inference, telemetry, and the Flask boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias

import numpy as np

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
BBox: TypeAlias = tuple[int, int, int, int]


class Decision(StrEnum):
    ACCEPT_BODY_MAKE_MODEL = "ACCEPT_BODY_MAKE_MODEL"
    ACCEPT_BODY_MAKE = "ACCEPT_BODY_MAKE"
    ACCEPT_BODY_ONLY = "ACCEPT_BODY_ONLY"
    INSUFFICIENT_VISUAL_EVIDENCE = "INSUFFICIENT_VISUAL_EVIDENCE"


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    WARNING = "warning"
    FAILED = "failed"


VALID_RUN_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.RUNNING, RunState.STOPPED, RunState.FAILED}),
    RunState.RUNNING: frozenset(
        {RunState.PAUSED, RunState.COMPLETED, RunState.STOPPED, RunState.FAILED}
    ),
    RunState.PAUSED: frozenset({RunState.RUNNING, RunState.STOPPED, RunState.FAILED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.STOPPED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class CameraSource:
    source_id: str
    name: str
    adapter_type: str
    enabled: bool
    attribution: str
    refresh_seconds: float
    terms_url: str | None = None
    location_label: str | None = None
    location_precision: str = "none"
    options: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.name or not self.adapter_type:
            raise ValueError("camera source identity fields cannot be empty")
        if self.refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")

    def public_dict(self) -> dict[str, JSONValue]:
        """Return browser-safe metadata; adapter options can contain credentials or URLs."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "adapter_type": self.adapter_type,
            "enabled": self.enabled,
            "attribution": self.attribution,
            "terms_url": self.terms_url,
            "refresh_seconds": self.refresh_seconds,
            "location_label": self.location_label,
            "location_precision": self.location_precision,
        }


@dataclass(frozen=True, slots=True)
class FramePacket:
    run_id: str
    source_id: str
    frame_id: int
    captured_at: datetime
    received_at: datetime
    image_bgr: np.ndarray
    source_metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape (height, width, 3)")
        if self.image_bgr.dtype != np.uint8:
            raise ValueError("image_bgr must use uint8 pixels")


@dataclass(frozen=True, slots=True)
class Detection:
    frame_id: int
    bbox_xyxy: BBox
    class_name: str
    confidence: float

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox_xyxy)
        _validate_unit_interval("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class CropQuality:
    apparent_height_px: int
    blur_score: float
    exposure_score: float
    occlusion_score: float
    usable: bool
    rejection_reasons: tuple[str, ...]
    fusion_weight: float

    def __post_init__(self) -> None:
        if self.apparent_height_px < 0:
            raise ValueError("apparent_height_px must be non-negative")
        for name in ("blur_score", "exposure_score", "occlusion_score"):
            _validate_unit_interval(name, getattr(self, name))
        if self.fusion_weight < 0:
            raise ValueError("fusion_weight must be non-negative")
        if self.usable and self.rejection_reasons:
            raise ValueError("usable crops cannot have rejection reasons")


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: str
    frame_id: int
    bbox_xyxy: BBox
    crop_bgr: np.ndarray
    quality: CropQuality

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox_xyxy)
        if self.crop_bgr.ndim != 3 or self.crop_bgr.shape[2] != 3:
            raise ValueError("crop_bgr must have shape (height, width, 3)")


@dataclass(frozen=True, slots=True)
class LabelPrediction:
    label: str | None
    confidence: float
    accepted: bool

    def __post_init__(self) -> None:
        _validate_unit_interval("confidence", self.confidence)
        if self.accepted and not self.label:
            raise ValueError("an accepted prediction requires a label")


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    method: str
    dataset_manifest_sha256: str
    validation_split_sha256: str
    test_split_sha256: str
    body_threshold: float
    make_threshold: float
    model_threshold: float

    def __post_init__(self) -> None:
        for name in ("body_threshold", "make_threshold", "model_threshold"):
            _validate_unit_interval(name, getattr(self, name))
        hashes = (
            self.dataset_manifest_sha256,
            self.validation_split_sha256,
            self.test_split_sha256,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("calibration identity fields must be SHA-256 hex digests")
        if self.validation_split_sha256 == self.test_split_sha256:
            raise ValueError("validation and test split identities must differ")


@dataclass(frozen=True, slots=True)
class VehiclePrediction:
    track_id: str
    body_type: LabelPrediction
    make: LabelPrediction
    model_family: LabelPrediction
    decision: Decision
    usable_frames: int
    disagreement: float
    model_version: str

    def __post_init__(self) -> None:
        if self.usable_frames < 0:
            raise ValueError("usable_frames must be non-negative")
        _validate_unit_interval("disagreement", self.disagreement)
        if self.make.accepted and not self.body_type.accepted:
            raise ValueError("make cannot be accepted when body type is abstained")
        if self.model_family.accepted and not self.make.accepted:
            raise ValueError("model cannot be accepted when make is abstained")
        expected = _decision_for(
            self.body_type.accepted, self.make.accepted, self.model_family.accepted
        )
        if self.decision is not expected:
            raise ValueError(f"decision {self.decision} does not match accepted hierarchy")

    def validate_hierarchy(self, model_to_make: Mapping[str, str]) -> None:
        if self.model_family.accepted:
            expected_make = model_to_make.get(self.model_family.label or "")
            if expected_make != self.make.label:
                raise ValueError("model family is incompatible with accepted make")


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    crop_id: str
    frame_id: int
    quality: CropQuality
    fusion_weight: float
    frame_prediction: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.fusion_weight < 0:
            raise ValueError("fusion_weight must be non-negative")
        if not self.frame_prediction or not all(np.isfinite(self.frame_prediction)):
            raise ValueError("frame_prediction must contain finite values")


@dataclass(frozen=True, slots=True)
class TrackEvidence:
    track_id: str
    items: tuple[EvidenceItem, ...]


@dataclass(frozen=True, slots=True)
class PrivacyRedactionResult:
    frame_id: int
    face_masks: tuple[BBox, ...]
    plate_masks: tuple[BBox, ...]
    redactor_version: str
    safe_for_display: bool

    def __post_init__(self) -> None:
        for bbox in (*self.face_masks, *self.plate_masks):
            _validate_bbox(bbox)


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    event_id: str
    sequence_id: int
    run_id: str
    stage: str
    status: StageStatus
    started_at: datetime
    frame_id: int | None = None
    track_id: str | None = None
    duration_ms: float | None = None
    input_summary: Mapping[str, JSONValue] = field(default_factory=dict)
    output_summary: Mapping[str, JSONValue] = field(default_factory=dict)
    warning: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        terminal = self.status not in {StageStatus.PENDING, StageStatus.RUNNING}
        if terminal and (self.duration_ms is None or self.duration_ms < 0):
            raise ValueError("terminal stage events require non-negative duration_ms")
        if self.status is StageStatus.FAILED and not self.error_code:
            raise ValueError("failed events require error_code")

    def to_dict(self) -> dict[str, JSONValue]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["started_at"] = self.started_at.astimezone(UTC).isoformat()
        return payload


def assert_run_transition(current: RunState, target: RunState) -> None:
    if target not in VALID_RUN_TRANSITIONS[current]:
        raise ValueError(f"invalid run transition: {current.value} -> {target.value}")


def _decision_for(body: bool, make: bool, model: bool) -> Decision:
    if model:
        return Decision.ACCEPT_BODY_MAKE_MODEL
    if make:
        return Decision.ACCEPT_BODY_MAKE
    if body:
        return Decision.ACCEPT_BODY_ONLY
    return Decision.INSUFFICIENT_VISUAL_EVIDENCE


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if min(bbox) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bounding box: {bbox}")
