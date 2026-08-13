"""Thread-safe inference run orchestration for the Flask boundary."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from roadid.contracts import CameraSource, FramePacket, JSONValue, RunState, assert_run_transition


class SourceAdapter(Protocol):
    def capture(self, run_id: str) -> FramePacket: ...

    def close(self) -> None: ...


class SourceRegistry(Protocol):
    def list_sources(self) -> Iterable[CameraSource]: ...

    def create_adapter(self, source_id: str) -> SourceAdapter: ...


class InferencePipeline(Protocol):
    def process_frame(self, frame: FramePacket) -> Mapping[str, Any] | None: ...

    def close(self) -> None: ...


class PipelineFactory(Protocol):
    def create(
        self,
        run_id: str,
        source: CameraSource,
        options: Mapping[str, JSONValue],
        cancellation_requested: Any = None,
    ) -> InferencePipeline: ...


@dataclass(slots=True)
class RunRecord:
    run_id: str
    source: CameraSource
    options: dict[str, JSONValue] = field(default_factory=dict)
    state: RunState = RunState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    processed_frames: int = 0
    latest_frame_id: int | None = None
    tracks: dict[str, JSONValue] = field(default_factory=dict)
    current_jpeg: bytes | None = None
    report: dict[str, JSONValue] = field(default_factory=dict)
    pending_frame: FramePacket | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    worker: threading.Thread | None = None
    adapter: SourceAdapter | None = None

    def public_dict(self) -> dict[str, JSONValue]:
        with self.condition:
            return {
                "run_id": self.run_id,
                "source_id": self.source.source_id,
                "state": self.state.value,
                "created_at": _timestamp(self.created_at),
                "started_at": _timestamp(self.started_at),
                "finished_at": _timestamp(self.finished_at),
                "processed_frames": self.processed_frames,
                "latest_frame_id": self.latest_frame_id,
                "track_count": len(self.tracks),
                "error": (
                    {"code": self.error_code, "message": self.error_message}
                    if self.error_code
                    else None
                ),
            }


class RunManager:
    """Own one source/pipeline worker for each run and expose snapshot-only reads."""

    def __init__(
        self,
        source_registry: SourceRegistry,
        pipeline_factory: PipelineFactory,
        recorder: Any | None = None,
    ) -> None:
        self._source_registry = source_registry
        self._pipeline_factory = pipeline_factory
        self._recorder = recorder
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.RLock()
        self._shutdown = False

    def list_sources(self) -> tuple[CameraSource, ...]:
        return tuple(self._source_registry.list_sources())

    def create_run(
        self, source_id: str, options: Mapping[str, JSONValue] | None = None
    ) -> RunRecord:
        source = next(
            (item for item in self.list_sources() if item.source_id == source_id),
            None,
        )
        if source is None:
            raise LookupError("unknown source")
        if not source.enabled:
            raise PermissionError("source is disabled")
        with self._lock:
            if self._shutdown:
                raise RuntimeError("run manager is shutting down")
            if any(
                run.source.source_id == source_id
                and run.state in {RunState.PENDING, RunState.RUNNING, RunState.PAUSED}
                for run in self._runs.values()
            ):
                raise RuntimeError("source already has an active run")
            run = RunRecord(
                run_id=f"run-{uuid.uuid4().hex}",
                source=source,
                options=dict(options or {}),
            )
            ensure_run = getattr(self._recorder, "ensure_run", None)
            if callable(ensure_run):
                ensure_run(run.run_id, source)
            self._runs[run.run_id] = run
            worker = threading.Thread(
                target=self._worker_main,
                args=(run,),
                name=f"roadid-{run.run_id}",
                daemon=False,
            )
            run.worker = worker
            worker.start()
            return run

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise LookupError("unknown run") from exc

    def pause(self, run_id: str) -> RunRecord:
        run = self.get_run(run_id)
        self._transition(run, RunState.PAUSED)
        return run

    def resume(self, run_id: str) -> RunRecord:
        run = self.get_run(run_id)
        self._transition(run, RunState.RUNNING)
        return run

    def stop(self, run_id: str, join_timeout: float = 2.0) -> RunRecord:
        run = self.get_run(run_id)
        adapter: SourceAdapter | None
        with run.condition:
            if run.state not in {RunState.COMPLETED, RunState.FAILED, RunState.STOPPED}:
                assert_run_transition(run.state, RunState.STOPPED)
                run.state = RunState.STOPPED
                run.finished_at = datetime.now(UTC)
            run.stop_event.set()
            run.pending_frame = None
            adapter = run.adapter
            run.condition.notify_all()
        if adapter is not None:
            _close_quietly(adapter)
        worker = run.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(join_timeout)
        return run

    def tracks(self, run_id: str) -> list[JSONValue]:
        run = self.get_run(run_id)
        with run.condition:
            return list(run.tracks.values())

    def track(self, track_id: str) -> JSONValue:
        with self._lock:
            runs = tuple(self._runs.values())
        for run in runs:
            with run.condition:
                if track_id in run.tracks:
                    return run.tracks[track_id]
        raise LookupError("unknown track")

    def wait_for_change(self, run_id: str, timeout: float) -> RunState:
        run = self.get_run(run_id)
        with run.condition:
            run.condition.wait(timeout=max(timeout, 0.0))
            return run.state

    def shutdown(self, join_timeout: float = 2.0) -> None:
        with self._lock:
            self._shutdown = True
            run_ids = tuple(self._runs)
        for run_id in run_ids:
            self.stop(run_id, join_timeout=join_timeout)
        _close_quietly(self._source_registry)
        _close_quietly(self._pipeline_factory)

    def status(self) -> dict[str, JSONValue]:
        with self._lock:
            runs = tuple(self._runs.values())
            accepting_runs = not self._shutdown
        active = 0
        for run in runs:
            with run.condition:
                active += int(run.state in {RunState.PENDING, RunState.RUNNING, RunState.PAUSED})
        return {
            "ready": accepting_runs,
            "active_runs": active,
            "total_runs": len(runs),
        }

    def _transition(self, run: RunRecord, target: RunState) -> None:
        with run.condition:
            assert_run_transition(run.state, target)
            run.state = target
            run.condition.notify_all()

    def _worker_main(self, run: RunRecord) -> None:
        pipeline: InferencePipeline | None = None
        try:
            with run.condition:
                if run.state is RunState.STOPPED:
                    return
                assert_run_transition(run.state, RunState.RUNNING)
                run.state = RunState.RUNNING
                run.started_at = datetime.now(UTC)
                run.condition.notify_all()

            adapter = _open_adapter(self._source_registry, run.source.source_id)
            with run.condition:
                run.adapter = adapter
            pipeline = _create_pipeline(self._pipeline_factory, run)
            iterator = iter(_source_frames(adapter, run.run_id, run.stop_event))

            while not run.stop_event.is_set():
                frame = self._take_frame(run, iterator)
                if frame is None:
                    break
                _record_source_acquisition(self._recorder, run.run_id, frame)
                raw_result = pipeline.process_frame(frame)
                result = _web_result(pipeline, raw_result)
                self._apply_result(run, frame, result)

            with run.condition:
                if run.state is RunState.RUNNING:
                    assert_run_transition(run.state, RunState.COMPLETED)
                    run.state = RunState.COMPLETED
                    run.finished_at = datetime.now(UTC)
                    run.condition.notify_all()
        except Exception:
            with run.condition:
                if run.state not in {RunState.STOPPED, RunState.COMPLETED, RunState.FAILED}:
                    assert_run_transition(run.state, RunState.FAILED)
                    run.state = RunState.FAILED
                    run.error_code = "WORKER_FAILED"
                    run.error_message = "Inference worker failed."
                    run.finished_at = datetime.now(UTC)
                    run.condition.notify_all()
        finally:
            if pipeline is not None:
                _close_quietly(pipeline)
            if run.adapter is not None:
                _close_quietly(run.adapter)

    def _take_frame(self, run: RunRecord, iterator: Any) -> FramePacket | None:
        with run.condition:
            while run.state is RunState.PAUSED and not run.stop_event.is_set():
                run.condition.wait()
            if run.stop_event.is_set():
                return None
            if run.pending_frame is not None:
                frame = run.pending_frame
                run.pending_frame = None
                return frame

        try:
            frame = next(iterator)
        except StopIteration:
            return None

        with run.condition:
            if run.state is RunState.PAUSED:
                run.pending_frame = frame
                while run.state is RunState.PAUSED and not run.stop_event.is_set():
                    run.condition.wait()
                if run.stop_event.is_set():
                    run.pending_frame = None
                    return None
                frame = run.pending_frame
                run.pending_frame = None
            return frame

    def _apply_result(self, run: RunRecord, frame: FramePacket, result: Mapping[str, Any]) -> None:
        tracks = result.get("tracks", ())
        with run.condition:
            run.processed_frames += 1
            run.latest_frame_id = frame.frame_id
            jpeg = result.get("frame_jpeg")
            if isinstance(jpeg, bytes):
                run.current_jpeg = jpeg
            for track in tracks if isinstance(tracks, Iterable) else ():
                if isinstance(track, Mapping) and isinstance(track.get("track_id"), str):
                    run.tracks[track["track_id"]] = dict(track)
            report = result.get("report")
            if isinstance(report, Mapping):
                run.report = dict(report)
            run.condition.notify_all()


def _close_quietly(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _timestamp(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _open_adapter(registry: Any, source_id: str) -> Any:
    for name in ("open", "create_adapter", "get_adapter"):
        method = getattr(registry, name, None)
        if callable(method):
            adapter = method(source_id)
            reset = getattr(adapter, "reset", None)
            if callable(reset):
                reset()
            return adapter
    raise RuntimeError("source registry cannot create adapters")


def _source_frames(adapter: Any, run_id: str, stop_event: threading.Event) -> Iterable[FramePacket]:
    frames = getattr(adapter, "frames", None)
    if callable(frames):
        try:
            iterator = frames(run_id, stop_event)
        except TypeError:
            iterator = frames(run_id)
        for frame in iterator:
            if stop_event.is_set():
                return
            yield frame
        return

    capture = getattr(adapter, "capture", None)
    if not callable(capture):
        raise RuntimeError("source adapter cannot produce frames")
    while not stop_event.is_set():
        try:
            yield capture(run_id)
        except Exception as exc:
            if getattr(exc, "code", None) == "source_exhausted":
                return
            raise


def _create_pipeline(factory: Any, run: RunRecord) -> Any:
    create = factory.create
    try:
        return create(
            run.run_id,
            run.source,
            run.options,
            cancellation_requested=run.stop_event.is_set,
        )
    except TypeError as exc:
        if "cancellation_requested" not in str(exc):
            raise
        return create(run.run_id, run.source, run.options)


def _record_source_acquisition(recorder: Any, run_id: str, frame: FramePacket) -> None:
    for_run = getattr(recorder, "for_run", None)
    if not callable(for_run):
        return
    target = for_run(run_id)
    target.start(
        "source_acquisition",
        frame_id=frame.frame_id,
        input_summary={"source_id": frame.source_id},
    )
    target.complete(
        "source_acquisition",
        frame_id=frame.frame_id,
        output_summary={"captured": True},
        duration_ms=0.0,
    )


def _web_result(pipeline: Any, result: Any) -> Mapping[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return result
    image = getattr(result, "display_frame_bgr", None)
    predictions = getattr(result, "predictions", None)
    if image is None or predictions is None:
        raise TypeError("inference pipeline returned an unsupported result")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("JPEG rendering requires opencv-python-headless") from exc
    encoded, buffer = cv2.imencode(".jpg", image)
    if not encoded:
        raise RuntimeError("privacy-safe frame could not be encoded")

    observations = {
        observation.track_id: observation for observation in getattr(result, "observations", ())
    }
    tracks = []
    for prediction in predictions:
        payload = asdict(prediction)
        payload["decision"] = prediction.decision.value
        observation = observations.get(prediction.track_id)
        if observation is not None:
            payload["bbox_xyxy"] = list(observation.bbox_xyxy)
        evidence = getattr(getattr(pipeline, "fuser", None), "evidence", None)
        if callable(evidence):
            payload["evidence"] = asdict(evidence(prediction.track_id))
        tracks.append(payload)
    privacy = getattr(result, "privacy", None)
    return {
        "frame_jpeg": buffer.tobytes(),
        "tracks": tracks,
        "report": {
            "model_version": tracks[0]["model_version"] if tracks else None,
            "privacy": asdict(privacy) if privacy is not None else None,
        },
    }
