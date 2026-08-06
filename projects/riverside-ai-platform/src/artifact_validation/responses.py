from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Iterator
from typing import Any, Literal

from .loader import GenerationResult
from .models import DeploymentMetadata


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_completion_id() -> str:
    return f"chatcmpl-{secrets.token_urlsafe(12)}"


def completion_response(
    *,
    result: GenerationResult,
    model_alias: str,
    deployment: DeploymentMetadata,
    trace_id: str,
    completion_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": model_alias,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text, "refusal": None},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "citations": [],
        "trace": {"trace_id": trace_id, "retry_count": 0},
        "deployment": deployment.model_dump(mode="json"),
    }


def stream_events(
    *,
    result: GenerationResult,
    model_alias: str,
    deployment: DeploymentMetadata,
    trace_id: str,
    completion_id: str | None = None,
    created: int | None = None,
) -> Iterator[dict[str, Any]]:
    event_id = completion_id or new_completion_id()
    event_created = int(time.time()) if created is None else created
    common = {
        "id": event_id,
        "object": "chat.completion.chunk",
        "created": event_created,
        "model": model_alias,
        "trace": {"trace_id": trace_id, "retry_count": 0},
        "deployment": deployment.model_dump(mode="json"),
    }
    chunks = re.findall(r"\S+\s*", result.text)
    for index, chunk in enumerate(chunks):
        delta: dict[str, str] = {"content": chunk}
        if index == 0:
            delta["role"] = "assistant"
        yield {
            **common,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    yield {
        **common,
        "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "citations": [],
    }


def sse_encode(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def error_response(
    *,
    code: Literal[
        "invalid_request",
        "overloaded",
        "backend_failure",
        "release_unavailable",
        "internal_error",
    ],
    message: str,
    error_type: Literal[
        "request_error", "capacity_error", "backend_error", "service_error"
    ],
    trace_id: str,
    deployment: DeploymentMetadata | None,
    retryable: bool,
    retry_after_seconds: int | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "type": error_type,
        "param": None,
        "retryable": retryable,
    }
    if retry_after_seconds is not None:
        error["retry_after_seconds"] = retry_after_seconds
    response: dict[str, Any] = {
        "error": error,
        "trace": {"trace_id": trace_id, "retry_count": 0},
    }
    if deployment is not None:
        response["deployment"] = deployment.model_dump(mode="json")
    return response
