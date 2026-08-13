"""Lazy integration wiring for standalone RoadID web startup."""

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
        pipeline_factory = DefaultPipelineFactory(settings, recorder)
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
