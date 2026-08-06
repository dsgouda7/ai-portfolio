from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from endpoint_client import AppError, ChatCompletionRequest, ChatCompletionResponse, EndpointClientError
from endpoint_client.models import ErrorDetail, TraceMetadata
from rag_orchestrator import TrustedAuthContext

from .auth import BackendAuthenticationError
from .runtime import ApplicationRuntime, build_runtime


_TRACE_ID = __import__("re").compile(r"^[a-f0-9]{32}$")
_MAX_REQUEST_BYTES = 1_048_576


class AuthorizationContextError(ValueError):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, application: ASGIApp, max_bytes: int) -> None:
        self._application = application
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self._application(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared_length = headers.get(b"content-length")
        if declared_length is not None:
            try:
                if int(declared_length) > self._max_bytes:
                    await self._send_rejection(scope, receive, send)
                    return
            except ValueError:
                await self._send_rejection(scope, receive, send)
                return

        messages: list[Message] = []
        received_bytes = 0
        more_body = True
        while more_body:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._max_bytes:
                    await self._send_rejection(scope, receive, send)
                    return
                more_body = message.get("more_body", False)

        message_index = 0

        async def receive_buffered() -> Message:
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return await receive()

        await self._application(scope, receive_buffered, send)

    @staticmethod
    async def _send_rejection(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive=receive)
        response = _json_error(
            _error(
                request,
                code="invalid_request",
                message="The request is invalid or exceeds a configured bound.",
                error_type="request_error",
                retryable=False,
            ),
            413,
        )
        await response(scope, receive, send)


def _default_config_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    environment = os.getenv("RIVERSIDE_ENVIRONMENT", "dev")
    return project_root / "config" / f"{environment}.yaml"


def _trace_id(request: Request) -> str:
    candidate = request.headers.get("x-correlation-id", "").lower()
    return candidate if _TRACE_ID.fullmatch(candidate) else secrets.token_hex(16)


def _error(
    request: Request,
    *,
    code: str,
    message: str,
    error_type: str,
    retryable: bool,
    retry_after_seconds: int | None = None,
) -> AppError:
    return AppError(
        error=ErrorDetail(
            code=code,
            message=message,
            type=error_type,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
        ),
        trace=TraceMetadata(trace_id=_trace_id(request), retry_count=0),
    )


def _json_error(envelope: AppError, status_code: int) -> JSONResponse:
    headers = {"X-Correlation-ID": envelope.trace.trace_id, "Cache-Control": "no-store"}
    if envelope.error.retry_after_seconds is not None:
        headers["Retry-After"] = str(envelope.error.retry_after_seconds)
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def _auth_context(
    tenant_id: str | None,
    actor_id: str | None,
    tenant_tier: str | None,
    group_ids: str | None,
) -> TrustedAuthContext:
    if not tenant_id or not actor_id or not tenant_tier:
        raise AuthorizationContextError("trusted APIM authorization context is unavailable")
    groups = frozenset(value.strip() for value in (group_ids or "").split(",") if value.strip())
    return TrustedAuthContext(
        tenant_id=tenant_id,
        principal_id=actor_id,
        group_ids=groups,
        tenant_tier=tenant_tier,
    )


async def _buffered_sse(response: ChatCompletionResponse) -> AsyncIterator[str]:
    common: dict[str, Any] = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "trace": response.trace.model_dump(mode="json"),
        "deployment": response.deployment.model_dump(mode="json"),
    }
    content_event = {
        **common,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                },
                "finish_reason": None,
            }
        ],
    }
    final_event = {
        **common,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": response.choices[0].finish_reason,
            }
        ],
        "usage": response.usage.model_dump(mode="json"),
        "citations": [citation.model_dump(mode="json") for citation in response.citations],
    }
    yield f"data: {json.dumps(content_event, separators=(',', ':'))}\n\n"
    yield f"data: {json.dumps(final_event, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def create_app(
    *,
    config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime: ApplicationRuntime | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        selected_runtime = runtime
        try:
            if selected_runtime is None:
                selected_runtime = build_runtime(
                    config_path or os.getenv("RIVERSIDE_CONFIG") or _default_config_path(),
                    environ=environ,
                )
            application.state.runtime = selected_runtime
            await selected_runtime.initialize()
        except Exception as error:
            application.state.runtime = None
            application.state.readiness_error = type(error).__name__
        try:
            yield
        finally:
            active_runtime = getattr(application.state, "runtime", None)
            if active_runtime is not None:
                await active_runtime.close()

    application = FastAPI(
        title="Riverside AI Platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    application.add_middleware(RequestBodyLimitMiddleware, max_bytes=_MAX_REQUEST_BYTES)

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, _: RequestValidationError) -> JSONResponse:
        return _json_error(
            _error(
                request,
                code="invalid_request",
                message="The request is invalid or exceeds a configured bound.",
                error_type="request_error",
                retryable=False,
            ),
            400,
        )

    @application.exception_handler(EndpointClientError)
    async def endpoint_error_handler(_: Request, error: EndpointClientError) -> JSONResponse:
        status = error.status_code or (504 if error.envelope.error.code == "timeout" else 502)
        return _json_error(error.envelope, status)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "riverside-rag-orchestrator"}

    @application.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        active_runtime = getattr(request.app.state, "runtime", None)
        if active_runtime is not None and active_runtime.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "not_ready"},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
        tenant_id: str | None = Header(default=None, alias="X-Riverside-Tenant-ID"),
        actor_id: str | None = Header(default=None, alias="X-Riverside-Actor-ID"),
        tenant_tier: str | None = Header(default=None, alias="X-Riverside-Tenant-Tier"),
        group_ids: str | None = Header(default=None, alias="X-Riverside-Group-IDs"),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> JSONResponse | StreamingResponse:
        active_runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if active_runtime is None or not active_runtime.ready:
            return _json_error(
                _error(
                    request,
                    code="release_unavailable",
                    message="The configured release or retrieval index is not ready.",
                    error_type="service_error",
                    retryable=True,
                ),
                503,
            )
        try:
            await active_runtime.backend_identity.authenticate(authorization)
            auth = _auth_context(tenant_id, actor_id, tenant_tier, group_ids)
            if body.model != active_runtime.config.model.alias:
                raise ValueError("requested model alias is unavailable")
            if body.max_input_tokens > active_runtime.config.model.max_input_tokens:
                raise ValueError("max_input_tokens exceeds the configured profile")
            if body.max_tokens > active_runtime.config.model.max_output_tokens:
                raise ValueError("max_tokens exceeds the configured profile")
            request_telemetry = active_runtime.telemetry.for_tenant_tier(auth.tenant_tier)
            with request_telemetry.request_span() as measurement:
                try:
                    with measurement.stage("generation"):
                        response = await active_runtime.orchestrator.complete(body, auth)
                except EndpointClientError as error:
                    error_code = error.envelope.error.code
                    measurement.complete(
                        outcome="timeout" if error_code == "timeout" else "error",
                        cache_result="bypass",
                        prompt_tokens=0,
                        output_tokens=0,
                        retry_count=error.envelope.trace.retry_count,
                        error_code=error_code,
                        retrieval_top_k=(body.retrieval.top_k if body.retrieval else None),
                    )
                    raise
                except Exception:
                    measurement.complete(
                        outcome="error",
                        cache_result="bypass",
                        prompt_tokens=0,
                        output_tokens=0,
                        error_code="backend_failure",
                        retrieval_top_k=(body.retrieval.top_k if body.retrieval else None),
                    )
                    raise
                if body.stream and response.choices[0].message.content:
                    measurement.record_first_token()
                measurement.complete(
                    outcome="success",
                    cache_result="bypass",
                    prompt_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    retry_count=response.trace.retry_count,
                    retrieval_top_k=(body.retrieval.top_k if body.retrieval else None),
                )
        except BackendAuthenticationError:
            return _json_error(
                _error(
                    request,
                    code="unauthorized",
                    message="Backend authentication failed.",
                    error_type="authentication_error",
                    retryable=False,
                ),
                401,
            )
        except AuthorizationContextError:
            return _json_error(
                _error(
                    request,
                    code="forbidden",
                    message="Required authorization context is unavailable.",
                    error_type="authorization_error",
                    retryable=False,
                ),
                403,
            )
        except ValueError:
            return _json_error(
                _error(
                    request,
                    code="invalid_request",
                    message="The request is invalid or exceeds a configured bound.",
                    error_type="request_error",
                    retryable=False,
                ),
                400,
            )
        except EndpointClientError as error:
            status = error.status_code or (504 if error.envelope.error.code == "timeout" else 502)
            return _json_error(error.envelope, status)
        except Exception:
            return _json_error(
                _error(
                    request,
                    code="backend_failure",
                    message="The orchestration backend could not complete the request.",
                    error_type="backend_error",
                    retryable=True,
                ),
                502,
            )

        headers = {
            "X-Correlation-ID": response.trace.trace_id,
            "Cache-Control": "no-store",
        }
        if body.stream:
            headers["X-Accel-Buffering"] = "no"
            return StreamingResponse(
                _buffered_sse(response),
                media_type="text/event-stream",
                headers=headers,
            )
        return JSONResponse(
            response.model_dump(mode="json", exclude_none=True),
            headers=headers,
        )

    return application


app = create_app()
