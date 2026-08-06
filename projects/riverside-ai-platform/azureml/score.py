from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from flask import Response
from pydantic import ValidationError

from src.artifact_validation import (
    ArtifactResolver,
    ArtifactValidationError,
    ChatCompletionRequest,
    DeploymentMetadata,
    ModelServingService,
    OverloadedError,
    ReleaseVerifier,
    RequestValidationError,
    RuntimeCompatibility,
    ServiceNotReadyError,
    TransformersPeftBackend,
    completion_response,
    error_response,
    new_trace_id,
    sse_encode,
    stream_events,
)


LOGGER = logging.getLogger("riverside.azureml.scoring")
SERVICE: ModelServingService | None = None
DEPLOYMENT: DeploymentMetadata | None = None


def _required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"required deployment setting is missing: {name}")
    return value


def init() -> None:
    global DEPLOYMENT, SERVICE

    if _required_environment("RIVERSIDE_STREAMING_MODE") != "buffered-sse":
        raise RuntimeError("RIVERSIDE_STREAMING_MODE must be buffered-sse")
    model_root = Path(_required_environment("AZUREML_MODEL_DIR")).resolve(strict=True)
    manifest_path = model_root / _required_environment("RIVERSIDE_RELEASE_MANIFEST")
    base_model_subdirectory = os.getenv("RIVERSIDE_BASE_MODEL_SUBDIR")
    base_model_directory = (
        (model_root / base_model_subdirectory).resolve(strict=True)
        if base_model_subdirectory
        else None
    )
    runtime = RuntimeCompatibility(
        name=_required_environment("RIVERSIDE_RUNTIME_NAME"),
        version=_required_environment("RIVERSIDE_RUNTIME_VERSION"),
        interface_version=_required_environment("RIVERSIDE_INTERFACE_VERSION"),
        model_profile=_required_environment("RIVERSIDE_MODEL_PROFILE"),
        precision=_required_environment("RIVERSIDE_PRECISION"),
        base_model_id=_required_environment("RIVERSIDE_BASE_MODEL_ID"),
        base_model_revision=_required_environment("RIVERSIDE_BASE_MODEL_REVISION"),
        adapter_type="lora",
    )
    service = ModelServingService(
        verifier=ReleaseVerifier(ArtifactResolver(model_root), runtime),
        backend=TransformersPeftBackend(
            device=os.getenv("RIVERSIDE_DEVICE", "cpu"),
            base_model_local_only=base_model_directory is not None,
            base_model_directory=base_model_directory,
        ),
        model_alias=_required_environment("RIVERSIDE_MODEL_ALIAS"),
        max_input_tokens=int(os.getenv("RIVERSIDE_MAX_INPUT_TOKENS", "4096")),
        max_output_tokens=int(os.getenv("RIVERSIDE_MAX_OUTPUT_TOKENS", "512")),
        max_concurrency=int(os.getenv("RIVERSIDE_MAX_CONCURRENCY", "1")),
    )
    try:
        service.initialize(manifest_path)
    except Exception as exc:
        LOGGER.error("release initialization failed", extra={"failure_type": type(exc).__name__})
        raise RuntimeError("release initialization failed") from exc

    release = service.release.manifest
    deployment = DeploymentMetadata(
        contract_version="1.0.0",
        kind="deployment_metadata",
        environment=_required_environment("RIVERSIDE_ENVIRONMENT"),
        release_id=release.release_id,
        model_alias=_required_environment("RIVERSIDE_MODEL_ALIAS"),
        deployment_name=_required_environment("RIVERSIDE_DEPLOYMENT_NAME"),
        deployment_slot=_required_environment("RIVERSIDE_DEPLOYMENT_SLOT"),
        region=_required_environment("RIVERSIDE_REGION"),
        runtime=runtime.name,
        runtime_version=runtime.version,
        index_version=_required_environment("RIVERSIDE_INDEX_VERSION"),
        deployed_at=datetime.fromisoformat(
            _required_environment("RIVERSIDE_DEPLOYED_AT").replace("Z", "+00:00")
        ),
        source_commit=release.source_commit,
    )
    SERVICE = service
    DEPLOYMENT = deployment
    LOGGER.info(
        "release ready",
        extra={"release_id": release.release_id, "deployment_slot": deployment.deployment_slot},
    )


def _json_response(body: dict[str, Any], status_code: int) -> AMLResponse:
    response = AMLResponse(json.dumps(body, separators=(",", ":")), status_code)
    response.headers["Content-Type"] = "application/json"
    response.headers["Cache-Control"] = "no-store"
    return response


@rawhttp
def run(request: Any) -> AMLResponse | Response:
    trace_id = new_trace_id()
    if SERVICE is None or DEPLOYMENT is None:
        return _json_response(
            error_response(
                code="release_unavailable",
                message="The verified release is not ready.",
                error_type="service_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=True,
            ),
            503,
        )
    if request.method != "POST":
        return _json_response(
            error_response(
                code="invalid_request",
                message="Only POST scoring requests are accepted.",
                error_type="request_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=False,
            ),
            405,
        )

    try:
        request_body = request.get_json(force=False, silent=False)
        chat_request = ChatCompletionRequest.model_validate(request_body)
        result = SERVICE.generate(chat_request)
        if chat_request.stream:
            response = Response(
                sse_encode(
                    stream_events(
                        result=result,
                        model_alias=chat_request.model,
                        deployment=DEPLOYMENT,
                        trace_id=trace_id,
                    )
                ),
                status=200,
                mimetype="text/event-stream",
            )
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["X-Riverside-Streaming-Mode"] = "buffered-sse"
            return response
        return _json_response(
            completion_response(
                result=result,
                model_alias=chat_request.model,
                deployment=DEPLOYMENT,
                trace_id=trace_id,
            ),
            200,
        )
    except (ValidationError, RequestValidationError):
        return _json_response(
            error_response(
                code="invalid_request",
                message="The request is invalid or exceeds a configured bound.",
                error_type="request_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=False,
            ),
            400,
        )
    except OverloadedError:
        return _json_response(
            error_response(
                code="overloaded",
                message="The service is at its bounded concurrency limit.",
                error_type="capacity_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=True,
                retry_after_seconds=5,
            ),
            429,
        )
    except ServiceNotReadyError:
        return _json_response(
            error_response(
                code="release_unavailable",
                message="The verified release is not ready.",
                error_type="service_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=True,
            ),
            503,
        )
    except ArtifactValidationError:
        return _json_response(
            error_response(
                code="release_unavailable",
                message="The verified release is unavailable.",
                error_type="service_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=False,
            ),
            503,
        )
    except Exception as exc:
        LOGGER.error("scoring failed", extra={"trace_id": trace_id, "failure_type": type(exc).__name__})
        return _json_response(
            error_response(
                code="backend_failure",
                message="The model backend could not complete the request.",
                error_type="backend_error",
                trace_id=trace_id,
                deployment=DEPLOYMENT,
                retryable=True,
            ),
            502,
        )
