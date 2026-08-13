"""Synchronous TrackLens inference orchestration with external checkpoint callbacks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import numpy as np

from roadid.contracts import (
    Detection,
    FramePacket,
    JSONValue,
    PrivacyRedactionResult,
    StageStatus,
    TrackObservation,
    VehiclePrediction,
)
from roadid.inference.calibration import HierarchicalDecisionEngine, TemperatureCalibrator
from roadid.inference.classifier import VehicleClassifier
from roadid.inference.detector import VehicleDetector
from roadid.inference.fusion import FusionResult, TrackFuser
from roadid.inference.privacy import PrivacyGuard
from roadid.inference.quality import CropQualityScorer
from roadid.inference.tracker import TrackedDetection

INFERENCE_STAGES = (
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


class InferenceCancelled(RuntimeError):
    """Raised at a stage boundary after cancellation is requested."""


@dataclass(frozen=True, slots=True)
class InferenceCheckpoint:
    run_id: str
    frame_id: int
    stage: str
    status: StageStatus
    track_id: str | None = None
    duration_ms: float | None = None
    summary: Mapping[str, JSONValue] = field(default_factory=dict)
    error_code: str | None = None


class CheckpointCallback(Protocol):
    def __call__(self, checkpoint: InferenceCheckpoint) -> None: ...


class Tracker(Protocol):
    def update(
        self, detections: tuple[Detection, ...], frame_id: int | None = None
    ) -> tuple[TrackedDetection, ...]: ...


@dataclass(frozen=True, slots=True)
class FrameInferenceResult:
    frame_id: int
    display_frame_bgr: np.ndarray
    detections: tuple[Detection, ...]
    observations: tuple[TrackObservation, ...]
    predictions: tuple[VehiclePrediction, ...]
    privacy: PrivacyRedactionResult


T = TypeVar("T")


class SynchronousInferencePipeline:
    """Run inference in the caller thread and publish bounded stage checkpoints."""

    def __init__(
        self,
        *,
        detector: VehicleDetector,
        tracker: Tracker,
        quality_scorer: CropQualityScorer,
        classifier: VehicleClassifier,
        fuser: TrackFuser,
        decision_engine: HierarchicalDecisionEngine,
        privacy_guard: PrivacyGuard,
        calibrator: TemperatureCalibrator | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        public_source_ids: Iterable[str] = (),
    ) -> None:
        if classifier.label_space != fuser.label_space:
            raise ValueError("classifier and fuser label spaces differ")
        if classifier.label_space != decision_engine.label_space:
            raise ValueError("classifier and decision label spaces differ")
        self.detector = detector
        self.tracker = tracker
        self.quality_scorer = quality_scorer
        self.classifier = classifier
        self.fuser = fuser
        self.decision_engine = decision_engine
        self.privacy_guard = privacy_guard
        self.calibrator = calibrator
        self.checkpoint_callback = checkpoint_callback
        self.cancellation_requested = cancellation_requested or (lambda: False)
        self.public_source_ids = frozenset(public_source_ids)

    def run(
        self, frames: Iterable[FramePacket], *, public_source: bool | None = None
    ) -> tuple[FrameInferenceResult, ...]:
        results = []
        for frame in frames:
            results.append(self.process_frame(frame, public_source=public_source))
        return tuple(results)

    def process_frame(
        self, frame: FramePacket, *, public_source: bool | None = None
    ) -> FrameInferenceResult:
        self._stage(
            frame,
            "frame_validation",
            lambda: _validate_frame(frame),
            summary=lambda _: {
                "height": frame.image_bgr.shape[0],
                "width": frame.image_bgr.shape[1],
            },
        )
        detections = self._stage(
            frame,
            "vehicle_detection",
            lambda: self.detector.detect(frame),
            summary=lambda values: {"detections": len(values)},
        )
        tracked = self._stage(
            frame,
            "track_association",
            lambda: self.tracker.update(detections, frame.frame_id),
            summary=lambda values: {"tracked": len(values)},
        )

        observations = []
        predictions = []
        for tracked_detection in tracked:
            self._check_cancelled()
            track_id = tracked_detection.track_id
            crop, quality = self._stage(
                frame,
                "crop_quality",
                lambda item=tracked_detection: self.quality_scorer.score_detection(
                    frame.image_bgr, item.detection.bbox_xyxy
                ),
                track_id=track_id,
                summary=lambda value: {
                    "usable": value[1].usable,
                    "apparent_height_px": value[1].apparent_height_px,
                },
            )
            observation = TrackObservation(
                track_id=track_id,
                frame_id=frame.frame_id,
                bbox_xyxy=tracked_detection.detection.bbox_xyxy,
                crop_bgr=crop,
                quality=quality,
            )
            observations.append(observation)
            if not quality.usable:
                self._emit(
                    frame,
                    "frame_classification",
                    StageStatus.SKIPPED,
                    track_id=track_id,
                    duration_ms=0.0,
                    summary={"reason_count": len(quality.rejection_reasons)},
                )
                continue

            scores = self._stage(
                frame,
                "frame_classification",
                lambda current=crop: self.classifier.classify(current),
                track_id=track_id,
                summary=lambda _: {"classified": True},
            )
            fusion = self._stage(
                frame,
                "track_fusion",
                lambda current_track=track_id, current_quality=quality, current_scores=scores: (
                    self.fuser.add(
                        track_id=current_track,
                        crop_id=f"{frame.source_id}:{frame.frame_id}:{current_track}",
                        frame_id=frame.frame_id,
                        quality=current_quality,
                        scores=current_scores,
                    )
                ),
                track_id=track_id,
                summary=lambda value, current_track=track_id: {
                    "evidence_items": len(self.fuser.evidence(current_track).items),
                    "disagreement": value.disagreement,
                },
            )
            calibrated = self._calibrate(frame, track_id, fusion)
            prediction = self._stage(
                frame,
                "hierarchy_decision",
                lambda current_track=track_id, current=calibrated: self.decision_engine.decide(
                    track_id=current_track,
                    scores=current.scores,
                    usable_frames=len(self.fuser.evidence(current_track).items),
                    disagreement=current.disagreement,
                ),
                track_id=track_id,
                summary=lambda value: {"decision": value.decision.value},
            )
            predictions.append(prediction)

        is_public = (
            frame.source_id in self.public_source_ids if public_source is None else public_source
        )
        protected = self._stage(
            frame,
            "privacy_render",
            lambda: self.privacy_guard.protect(
                frame.image_bgr, frame_id=frame.frame_id, public_source=is_public
            ),
            summary=lambda value: {"safe_for_display": value.result.safe_for_display},
        )
        return FrameInferenceResult(
            frame_id=frame.frame_id,
            display_frame_bgr=protected.image_bgr,
            detections=detections,
            observations=tuple(observations),
            predictions=tuple(predictions),
            privacy=protected.result,
        )

    def _calibrate(self, frame: FramePacket, track_id: str, fusion: FusionResult) -> FusionResult:
        if self.calibrator is None:
            self._emit(
                frame,
                "calibration",
                StageStatus.COMPLETED,
                track_id=track_id,
                duration_ms=0.0,
                summary={"method": "identity"},
            )
            return fusion
        scores = self._stage(
            frame,
            "calibration",
            lambda: self.calibrator.calibrate(fusion.scores),
            track_id=track_id,
            summary=lambda _: {"method": self.calibrator.contract.method},
        )
        return FusionResult(scores, fusion.disagreement, fusion.total_weight)

    def _stage(
        self,
        frame: FramePacket,
        stage: str,
        operation: Callable[[], T],
        *,
        track_id: str | None = None,
        summary: Callable[[T], Mapping[str, JSONValue]] = lambda _: {},
    ) -> T:
        self._check_cancelled()
        self._emit(frame, stage, StageStatus.RUNNING, track_id=track_id)
        started = time.perf_counter()
        try:
            value = operation()
        except Exception as error:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._emit(
                frame,
                stage,
                StageStatus.FAILED,
                track_id=track_id,
                duration_ms=duration_ms,
                error_code=type(error).__name__,
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._emit(
            frame,
            stage,
            StageStatus.COMPLETED,
            track_id=track_id,
            duration_ms=duration_ms,
            summary=summary(value),
        )
        return value

    def _check_cancelled(self) -> None:
        if self.cancellation_requested():
            raise InferenceCancelled("inference cancelled at stage boundary")

    def _emit(
        self,
        frame: FramePacket,
        stage: str,
        status: StageStatus,
        *,
        track_id: str | None = None,
        duration_ms: float | None = None,
        summary: Mapping[str, JSONValue] | None = None,
        error_code: str | None = None,
    ) -> None:
        if stage not in INFERENCE_STAGES:
            raise ValueError(f"unknown inference stage: {stage}")
        if self.checkpoint_callback is None:
            return
        self.checkpoint_callback(
            InferenceCheckpoint(
                run_id=frame.run_id,
                frame_id=frame.frame_id,
                track_id=track_id,
                stage=stage,
                status=status,
                duration_ms=duration_ms,
                summary=summary or {},
                error_code=error_code,
            )
        )


InferencePipeline = SynchronousInferencePipeline


def _validate_frame(frame: FramePacket) -> None:
    if frame.received_at < frame.captured_at:
        raise ValueError("frame received_at precedes captured_at")
