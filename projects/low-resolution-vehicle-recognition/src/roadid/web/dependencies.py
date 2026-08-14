"""Lazy integration wiring for standalone CarFace web startup."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from roadid.settings import Settings


class DependencyUnavailable(RuntimeError):
    """A runtime integration has not been installed or configured."""


class UnavailableSourceRegistry:
    def list_sources(self):
        raise DependencyUnavailable("Source integrations are unavailable.")

    def open(self, source_id):
        raise DependencyUnavailable("Source integrations are unavailable.")


class UnavailablePipelineFactory:
    def create(self, run_id, source, options):
        raise DependencyUnavailable("Inference integrations are unavailable.")


class DefaultPipelineFactory:
    """Build the offline deterministic replay profile only when a run starts."""

    def __init__(self, settings: Settings, recorder: Any) -> None:
        self._settings = settings
        self._recorder = recorder

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "device": "cpu",
            "model_version": "deterministic-replay-v1",
            "profile": "demo",
        }

    def create(self, run_id, source, options, cancellation_requested=None):
        if source.adapter_type != "replay":
            raise DependencyUnavailable("The default profile supports deterministic replay only.")

        from roadid.contracts import CalibrationContract, Detection
        from roadid.inference.calibration import HierarchicalDecisionEngine
        from roadid.inference.classifier import DeterministicDemoClassifier
        from roadid.inference.fusion import TrackFuser
        from roadid.inference.pipeline import SynchronousInferencePipeline
        from roadid.inference.privacy import DeterministicNoPIIRedactor, PrivacyGuard
        from roadid.inference.quality import CropQualityScorer, QualityConfig
        from roadid.inference.tracker import DeterministicTracker

        classifier = DeterministicDemoClassifier()
        runtime = self._settings.inference.get("runtime", {})
        quality_values = self._settings.inference.get("quality", {})
        tracker_values = self._settings.inference.get("tracker", {})
        quality = QualityConfig(
            minimum_height_px=int(quality_values.get("minimum_height_px", 12)),
            maximum_blur=float(quality_values.get("maximum_blur", 0.8)),
            maximum_exposure=float(quality_values.get("maximum_exposure", 0.85)),
            maximum_occlusion=float(quality_values.get("maximum_occlusion", 0.75)),
        )
        calibration = CalibrationContract(
            method="deterministic-demo",
            dataset_manifest_sha256="0" * 64,
            validation_split_sha256="1" * 64,
            test_split_sha256="2" * 64,
            body_threshold=0.5,
            make_threshold=0.5,
            model_threshold=0.5,
        )
        model_to_make = {"blue-line": "blue-motors", "red-line": "red-motors"}
        recorder = self._recorder.for_run(run_id)

        class ReplayMetadataDetector:
            def detect(self, frame):
                bbox = frame.source_metadata.get("vehicle_bbox_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return ()
                return (
                    Detection(
                        frame_id=frame.frame_id,
                        bbox_xyxy=tuple(int(value) for value in bbox),
                        class_name="car",
                        confidence=1.0,
                    ),
                )

        return SynchronousInferencePipeline(
            detector=ReplayMetadataDetector(),
            tracker=DeterministicTracker(
                lost_track_buffer=int(tracker_values.get("lost_track_buffer", 20))
            ),
            quality_scorer=CropQualityScorer(quality),
            classifier=classifier,
            fuser=TrackFuser(
                classifier.label_space,
                max_evidence_per_track=int(runtime.get("max_evidence_per_track", 16)),
            ),
            decision_engine=HierarchicalDecisionEngine(
                label_space=classifier.label_space,
                calibration=calibration,
                model_version="deterministic-replay-v1",
                model_to_make=model_to_make,
            ),
            privacy_guard=PrivacyGuard(DeterministicNoPIIRedactor()),
            checkpoint_callback=_checkpoint_recorder(recorder),
            cancellation_requested=cancellation_requested,
        )


class DetrDetectionPipelineFactory:
    """Build a pretrained generic-vehicle profile without make/model claims."""

    def __init__(
        self,
        settings: Settings,
        recorder: Any,
        *,
        detector: Any | None = None,
        tracker: Any | None = None,
    ) -> None:
        self._settings = settings
        self._recorder = recorder
        self._detector = detector
        self._tracker = tracker

    def status(self) -> dict[str, Any]:
        detector = self._settings.inference.get("detector", {})
        return {
            "ready": _module_available("torch") and _module_available("transformers"),
            "device": detector.get("device", "cpu"),
            "model_version": detector.get("model_id", "facebook/detr-resnet-50"),
            "profile": "detection-only",
        }

    def create(self, run_id, source, options, cancellation_requested=None):
        if source.adapter_type not in {"local_camera", "snapshot_http", "tfl_jamcam"}:
            raise DependencyUnavailable(
                "The DETR profile supports local cameras and reviewed snapshot sources only."
            )

        from roadid.contracts import CalibrationContract
        from roadid.inference.calibration import HierarchicalDecisionEngine
        from roadid.inference.classifier import DetectionOnlyClassifier
        from roadid.inference.detector import DetrConfig, HuggingFaceDetrDetector
        from roadid.inference.fusion import TrackFuser
        from roadid.inference.pipeline import SynchronousInferencePipeline
        from roadid.inference.privacy import FullFramePixelationRedactor, PrivacyGuard
        from roadid.inference.quality import CropQualityScorer, QualityConfig
        from roadid.inference.tracker import ByteTrackTracker, DeterministicTracker

        runtime = self._settings.inference.get("runtime", {})
        detector_values = self._settings.inference.get("detector", {})
        tracker_values = self._settings.inference.get("tracker", {})
        quality_values = self._settings.inference.get("quality", {})
        classifier = DetectionOnlyClassifier()
        detector = self._detector or HuggingFaceDetrDetector(
            DetrConfig(
                model_id=str(detector_values.get("model_id", "facebook/detr-resnet-50")),
                revision=detector_values.get("revision", "no_timm"),
                score_threshold=float(detector_values.get("score_threshold", 0.7)),
                vehicle_classes=tuple(
                    str(value)
                    for value in detector_values.get("vehicle_classes", ["car", "truck", "bus"])
                ),
                offline_only=bool(detector_values.get("offline_only", True)),
                device=str(detector_values.get("device", "cpu")),
            )
        )
        if self._tracker is not None:
            tracker = self._tracker
        elif tracker_values.get("implementation") == "bytetrack":
            tracker = ByteTrackTracker(
                activation_threshold=float(tracker_values.get("activation_threshold", 0.25)),
                matching_threshold=float(tracker_values.get("matching_threshold", 0.8)),
                lost_track_buffer=int(tracker_values.get("lost_track_buffer", 20)),
                frame_rate=max(1, round(float(runtime.get("processing_fps", 5.0)))),
            )
        else:
            tracker = DeterministicTracker(
                lost_track_buffer=int(tracker_values.get("lost_track_buffer", 20)),
                profile_label="deterministic-tracker-real-source-fallback",
            )
        calibration = CalibrationContract(
            method="detection-only-abstention",
            dataset_manifest_sha256="0" * 64,
            validation_split_sha256="1" * 64,
            test_split_sha256="2" * 64,
            body_threshold=0.49,
            make_threshold=1.0,
            model_threshold=1.0,
        )
        public_source = source.adapter_type in {"snapshot_http", "tfl_jamcam"}
        privacy_guard = (
            PrivacyGuard(FullFramePixelationRedactor(block_size=12))
            if public_source
            else PrivacyGuard(None, allow_raw_local_display=True)
        )
        recorder = self._recorder.for_run(run_id)
        return SynchronousInferencePipeline(
            detector=detector,
            tracker=tracker,
            quality_scorer=CropQualityScorer(
                QualityConfig(
                    minimum_height_px=int(quality_values.get("minimum_height_px", 12)),
                    maximum_blur=float(quality_values.get("maximum_blur", 0.8)),
                    maximum_exposure=float(quality_values.get("maximum_exposure", 0.85)),
                    maximum_occlusion=float(quality_values.get("maximum_occlusion", 0.75)),
                )
            ),
            classifier=classifier,
            fuser=TrackFuser(
                classifier.label_space,
                max_evidence_per_track=int(runtime.get("max_evidence_per_track", 16)),
            ),
            decision_engine=HierarchicalDecisionEngine(
                label_space=classifier.label_space,
                calibration=calibration,
                model_version=str(
                    detector_values.get("model_id", "facebook/detr-resnet-50")
                ),
                model_to_make={"model-not-trained": "make-not-trained"},
            ),
            privacy_guard=privacy_guard,
            checkpoint_callback=_checkpoint_recorder(recorder),
            cancellation_requested=cancellation_requested,
            public_source_ids=(source.source_id,) if public_source else (),
        )


class UnavailableRecorder:
    def ensure_run(self, run_id, source=None):
        raise DependencyUnavailable("Telemetry integrations are unavailable.")

    def events_after(self, run_id, sequence_id):
        raise DependencyUnavailable("Telemetry integrations are unavailable.")


class RecorderHub:
    """Own a bounded telemetry recorder for each independently ordered run."""

    def __init__(self, recorder_type: Any, capacity: int) -> None:
        self._recorder_type = recorder_type
        self._capacity = capacity
        self._recorders: dict[str, Any] = {}

    def ensure_run(self, run_id: str, source: Any = None) -> Any:
        if run_id not in self._recorders:
            metadata = {"source_id": source.source_id} if source is not None else None
            self._recorders[run_id] = self._recorder_type(
                run_id,
                capacity=self._capacity,
                metadata=metadata,
            )
        return self._recorders[run_id]

    def for_run(self, run_id: str) -> Any:
        try:
            return self._recorders[run_id]
        except KeyError as exc:
            raise LookupError("unknown telemetry run") from exc

    def events_after(self, run_id: str, sequence_id: int):
        return self.for_run(run_id).events_after(sequence_id)

    def wait_for_events(self, run_id: str, sequence_id: int, timeout: float):
        return self.for_run(run_id).wait_for_events(sequence_id, timeout=timeout)


@dataclass(frozen=True, slots=True)
class DefaultDependencies:
    source_registry: Any
    pipeline_factory: Any
    recorder: Any
    unavailable: tuple[str, ...]


def build_default_dependencies(settings: Settings) -> DefaultDependencies:
    """Import optional runtime packages only when an uninjected app needs them."""
    unavailable: list[str] = []

    try:
        source_module = importlib.import_module("roadid.sources.registry")
        source_builder = source_module.load_source_registry
        source_registry = source_builder(settings.paths.source_config)
    except (ImportError, AttributeError, OSError, ValueError):
        source_registry = UnavailableSourceRegistry()
        unavailable.append("sources")

    try:
        recorder_module = importlib.import_module("roadid.telemetry.recorder")
        recorder_type = recorder_module.EventRecorder
        capacity = int(settings.inference.get("runtime", {}).get("event_capacity", 5000))
        recorder = RecorderHub(recorder_type, capacity)
    except (ImportError, AttributeError, TypeError, ValueError):
        recorder = UnavailableRecorder()
        unavailable.append("telemetry")

    try:
        importlib.import_module("roadid.inference.pipeline")
        if isinstance(recorder, UnavailableRecorder):
            raise ImportError("telemetry is unavailable")
        profile = str(settings.inference.get("runtime", {}).get("profile", "demo"))
        pipeline_factory = (
            DetrDetectionPipelineFactory(settings, recorder)
            if profile == "detection-only"
            else DefaultPipelineFactory(settings, recorder)
        )
    except (ImportError, OSError, TypeError, ValueError):
        pipeline_factory = UnavailablePipelineFactory()
        unavailable.append("inference")

    return DefaultDependencies(
        source_registry=source_registry,
        pipeline_factory=pipeline_factory,
        recorder=recorder,
        unavailable=tuple(unavailable),
    )


def component_ready(component: Any) -> bool:
    return not isinstance(
        component,
        UnavailableSourceRegistry | UnavailablePipelineFactory | UnavailableRecorder,
    )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _checkpoint_recorder(recorder: Any):
    active: set[tuple[str, int | None, str | None]] = set()

    def record(checkpoint: Any) -> None:
        key = (checkpoint.stage, checkpoint.frame_id, checkpoint.track_id)
        common = {
            "frame_id": checkpoint.frame_id,
            "track_id": checkpoint.track_id,
        }
        if checkpoint.status.value == "running":
            recorder.start(checkpoint.stage, **common)
            active.add(key)
            return
        if key not in active:
            recorder.start(checkpoint.stage, **common)
        active.discard(key)
        if checkpoint.status.value == "failed":
            recorder.fail_stage_and_skip_remaining(
                checkpoint.stage,
                error_code=checkpoint.error_code or "INFERENCE_FAILED",
                duration_ms=checkpoint.duration_ms or 0.0,
                **common,
            )
        elif checkpoint.status.value == "skipped":
            recorder.skip(
                checkpoint.stage,
                reason="stage skipped",
                duration_ms=checkpoint.duration_ms or 0.0,
                **common,
            )
        elif checkpoint.status.value == "warning":
            recorder.warn(
                checkpoint.stage,
                warning="stage warning",
                output_summary=dict(checkpoint.summary),
                duration_ms=checkpoint.duration_ms or 0.0,
                **common,
            )
        else:
            recorder.complete(
                checkpoint.stage,
                output_summary=dict(checkpoint.summary),
                duration_ms=checkpoint.duration_ms or 0.0,
                **common,
            )

    return record
