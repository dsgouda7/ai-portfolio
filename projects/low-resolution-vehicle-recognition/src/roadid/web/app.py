"""RoadID Flask application factory and HTTP API."""

from __future__ import annotations

import json
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from jinja2 import TemplateNotFound

from roadid.contracts import JSONValue, RunState
from roadid.settings import Settings, load_settings
from roadid.web.dependencies import (
    DependencyUnavailable,
    build_default_dependencies,
    component_ready,
)
from roadid.web.runs import RunManager
from roadid.web.serialization import safe_json

TERMINAL_STATES = {RunState.COMPLETED, RunState.FAILED, RunState.STOPPED}
ALLOWED_RUN_FIELDS = {"source_id", "processing_fps", "confidence_profile"}


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    code: str
    message: str
    status: int


def create_app(
    *,
    source_registry: Any | None = None,
    pipeline_factory: Any | None = None,
    recorder: Any | None = None,
    settings: Settings | None = None,
) -> Flask:
    settings = settings or load_settings()
    defaults = None
    if source_registry is None or pipeline_factory is None or recorder is None:
        defaults = build_default_dependencies(settings)
        source_registry = source_registry or defaults.source_registry
        pipeline_factory = pipeline_factory or defaults.pipeline_factory
        recorder = recorder or defaults.recorder

    app = Flask(
        __name__,
        template_folder=str(settings.paths.project_root / "templates"),
        static_folder=str(settings.paths.project_root / "static"),
    )
    app.json.sort_keys = False
    manager = RunManager(source_registry, pipeline_factory, recorder)
    app.extensions["roadid"] = {
        "manager": manager,
        "recorder": recorder,
        "settings": settings,
        "shutdown": manager.shutdown,
    }
    weakref.finalize(app, manager.shutdown)

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        payload = {"error": {"code": error.code, "message": error.message[:300]}}
        return jsonify(payload), error.status

    @app.get("/")
    def index():
        try:
            return render_template("index.html")
        except TemplateNotFound:
            return Response(
                "<!doctype html><title>RoadID</title><main><h1>RoadID</h1>"
                "<p>The web interface is not installed. The API is available.</p></main>",
                mimetype="text/html",
            )

    @app.get("/api/status")
    def status():
        components = {
            "sources": component_ready(source_registry),
            "inference": component_ready(pipeline_factory),
            "telemetry": component_ready(recorder),
        }
        factory_status = _component_status(pipeline_factory)
        detector = settings.inference.get("detector", {})
        device = factory_status.get("device") or detector.get("device") or "unavailable"
        model_version = factory_status.get("model_version")
        model = {
            "ready": components["inference"],
            "model_id": detector.get("model_id"),
            "version": model_version,
            "bundle_configured": settings.paths.model_bundle is not None,
        }
        return jsonify(
            {
                "service": "RoadID",
                "ready": all(components.values()) and manager.status()["ready"],
                "components": components,
                "model": safe_json(model),
                "model_version": safe_json(model_version),
                "device": safe_json(device),
                "worker": manager.status(),
                "run_states": [state.value for state in RunState],
            }
        )

    @app.get("/api/sources")
    def sources():
        try:
            payload = [source.public_dict() for source in manager.list_sources()]
        except DependencyUnavailable as exc:
            raise ApiError("SOURCES_UNAVAILABLE", str(exc), 503) from exc
        return jsonify({"sources": safe_json(payload)})

    @app.post("/api/runs")
    def create_run():
        if not component_ready(pipeline_factory):
            raise ApiError(
                "INFERENCE_UNAVAILABLE",
                "Inference integrations are unavailable.",
                503,
            )
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError("INVALID_REQUEST", "A JSON object is required.", 400)
        unknown_fields = set(payload) - ALLOWED_RUN_FIELDS
        if unknown_fields:
            raise ApiError("INVALID_REQUEST", "Unsupported run request fields.", 400)
        source_id = payload.get("source_id")
        if not isinstance(source_id, str) or not source_id or len(source_id) > 128:
            raise ApiError("INVALID_SOURCE_ID", "source_id must be a non-empty string.", 400)
        options = _run_options(payload)
        try:
            run = manager.create_run(source_id, options)
        except LookupError as exc:
            raise ApiError("SOURCE_NOT_FOUND", "Source ID was not found.", 404) from exc
        except PermissionError as exc:
            raise ApiError("SOURCE_DISABLED", "Source is disabled.", 409) from exc
        except DependencyUnavailable as exc:
            raise ApiError("DEPENDENCY_UNAVAILABLE", str(exc), 503) from exc
        except RuntimeError as exc:
            raise ApiError("SOURCE_BUSY", "Source already has an active run.", 409) from exc
        return jsonify({"run": run.public_dict()}), 201

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str):
        return jsonify({"run": _get_run(manager, run_id).public_dict()})

    @app.post("/api/runs/<run_id>/pause")
    def pause_run(run_id: str):
        return _transition_response(manager, run_id, manager.pause)

    @app.post("/api/runs/<run_id>/resume")
    def resume_run(run_id: str):
        return _transition_response(manager, run_id, manager.resume)

    @app.post("/api/runs/<run_id>/stop")
    def stop_run(run_id: str):
        run = _get_run(manager, run_id)
        return jsonify({"run": manager.stop(run.run_id).public_dict()})

    @app.get("/api/runs/<run_id>/events")
    def events(run_id: str):
        run = _get_run(manager, run_id)
        last_sequence = _last_event_id()
        initial_events = _events_after(recorder, run.run_id, last_sequence)
        heartbeat = _heartbeat_seconds(settings)

        @stream_with_context
        def generate():
            nonlocal last_sequence, initial_events
            missed = 0
            pending = initial_events
            while True:
                events_now = pending or _events_after(recorder, run.run_id, last_sequence)
                pending = []
                if events_now:
                    missed = 0
                    for event in events_now:
                        sequence = _event_sequence(event)
                        if sequence <= last_sequence:
                            continue
                        last_sequence = sequence
                        data = json.dumps(safe_json(event), separators=(",", ":"))
                        yield f"id: {sequence}\nevent: pipeline\ndata: {data}\n\n"
                    continue

                if run.state in TERMINAL_STATES:
                    break
                manager.wait_for_change(run.run_id, heartbeat)
                if _events_after(recorder, run.run_id, last_sequence):
                    continue
                missed += 1
                yield ": heartbeat\n\n"
                if missed >= 3:
                    break

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/<run_id>/tracks")
    def tracks(run_id: str):
        run = _get_run(manager, run_id)
        return jsonify({"run_id": run.run_id, "tracks": safe_json(manager.tracks(run.run_id))})

    @app.get("/api/tracks/<track_id>")
    def track(track_id: str):
        try:
            payload = manager.track(track_id)
        except LookupError as exc:
            raise ApiError("TRACK_NOT_FOUND", "Track ID was not found.", 404) from exc
        return jsonify({"track": safe_json(payload)})

    @app.get("/api/runs/<run_id>/report")
    def report(run_id: str):
        run = _get_run(manager, run_id)
        payload = {
            "schema_version": 1,
            "run": run.public_dict(),
            "tracks": manager.tracks(run.run_id),
            "pipeline": run.report,
        }
        response = jsonify({"report": safe_json(payload)})
        response.headers["Content-Disposition"] = f'attachment; filename="{run.run_id}-report.json"'
        return response

    @app.get("/api/runs/<run_id>/frame")
    @app.get("/api/runs/<run_id>/frame.jpg")
    def current_frame(run_id: str):
        run = _get_run(manager, run_id)
        with run.condition:
            jpeg = run.current_jpeg
        if jpeg is None:
            raise ApiError("FRAME_NOT_AVAILABLE", "No display-safe frame is available.", 404)
        return Response(
            jpeg,
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-ID": str(run.latest_frame_id or ""),
            },
        )

    return app


def _get_run(manager: RunManager, run_id: str):
    try:
        return manager.get_run(run_id)
    except LookupError as exc:
        raise ApiError("RUN_NOT_FOUND", "Run ID was not found.", 404) from exc


def _transition_response(manager: RunManager, run_id: str, transition):
    run = _get_run(manager, run_id)
    try:
        run = transition(run.run_id)
    except ValueError as exc:
        raise ApiError(
            "INVALID_RUN_TRANSITION",
            "Run state transition is not allowed.",
            409,
        ) from exc
    return jsonify({"run": run.public_dict()})


def _run_options(payload: Mapping[str, Any]) -> dict[str, JSONValue]:
    options: dict[str, JSONValue] = {}
    if "processing_fps" in payload:
        fps = payload["processing_fps"]
        if isinstance(fps, bool) or not isinstance(fps, int | float) or not 0.1 <= fps <= 60:
            raise ApiError(
                "INVALID_PROCESSING_FPS",
                "processing_fps must be between 0.1 and 60.",
                400,
            )
        options["processing_fps"] = float(fps)
    if "confidence_profile" in payload:
        profile = payload["confidence_profile"]
        if not isinstance(profile, str) or profile not in {"strict", "balanced", "permissive"}:
            raise ApiError(
                "INVALID_CONFIDENCE_PROFILE",
                "confidence_profile must be strict, balanced, or permissive.",
                400,
            )
        options["confidence_profile"] = profile
    return options


def _last_event_id() -> int:
    raw = request.headers.get("Last-Event-ID") or request.args.get("last_event_id") or "0"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApiError("INVALID_EVENT_ID", "Last-Event-ID must be an integer.", 400) from exc
    if value < 0:
        raise ApiError("INVALID_EVENT_ID", "Last-Event-ID must be non-negative.", 400)
    return value


def _events_after(recorder: Any, run_id: str, sequence_id: int) -> list[Any]:
    target = _recorder_for_run(recorder, run_id)
    read = getattr(target, "events_after", None)
    if not callable(read):
        raise ApiError("TELEMETRY_UNAVAILABLE", "Telemetry recorder is unavailable.", 503)
    try:
        if target is recorder and not hasattr(target, "run_id"):
            result = read(run_id, sequence_id)
        else:
            result = read(sequence_id)
    except DependencyUnavailable as exc:
        raise ApiError("TELEMETRY_UNAVAILABLE", str(exc), 503) from exc
    events = getattr(result, "events", result)
    events = list(events)
    return sorted(events, key=_event_sequence)


def _event_sequence(event: Any) -> int:
    value = (
        event.get("sequence_id")
        if isinstance(event, Mapping)
        else getattr(event, "sequence_id", None)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ApiError("INVALID_TELEMETRY", "Recorder returned an invalid sequence ID.", 503)
    return value


def _heartbeat_seconds(settings: Settings) -> float:
    value = settings.inference.get("runtime", {}).get("heartbeat_seconds", 10)
    try:
        return max(float(value), 0.01)
    except (TypeError, ValueError):
        return 10.0


def _recorder_for_run(recorder: Any, run_id: str) -> Any:
    for_run = getattr(recorder, "for_run", None)
    if callable(for_run):
        try:
            return for_run(run_id)
        except LookupError as exc:
            raise ApiError("TELEMETRY_UNAVAILABLE", "Run telemetry is unavailable.", 503) from exc
    recorder_run_id = getattr(recorder, "run_id", None)
    if recorder_run_id is not None and recorder_run_id != run_id:
        raise ApiError("TELEMETRY_UNAVAILABLE", "Run telemetry is unavailable.", 503)
    return recorder


def _component_status(component: Any) -> dict[str, Any]:
    status = getattr(component, "status", None)
    if not callable(status):
        return {}
    try:
        payload = status()
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}
