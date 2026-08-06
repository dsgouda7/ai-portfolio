from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .errors import EndpointClientError, transport_error
from .models import (
    AssistantMessage,
    ChatCompletionResponse,
    ChatCompletionStreamEvent,
    Citation,
    CompletionChoice,
    DeploymentMetadata,
    StreamChoice,
    StreamDelta,
    TraceMetadata,
    Usage,
)


_COMPLETION_ID = re.compile(r"^chatcmpl-[A-Za-z0-9_-]{8,64}$")
_TRACE_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class NormalizationContext:
    model_alias: str
    trace_id: str
    retry_count: int
    deployment: DeploymentMetadata | None
    clock: Callable[[], float] = time.time


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("response", "result"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def _trace(payload: Mapping[str, Any], context: NormalizationContext) -> TraceMetadata:
    raw_trace = _mapping(payload.get("trace"))
    trace_id = raw_trace.get("trace_id")
    return TraceMetadata(
        trace_id=trace_id if isinstance(trace_id, str) and _TRACE_ID.fullmatch(trace_id) else context.trace_id,
        retry_count=context.retry_count,
    )


def _deployment(payload: Mapping[str, Any], context: NormalizationContext) -> DeploymentMetadata:
    raw = payload.get("deployment")
    if isinstance(raw, Mapping):
        try:
            return DeploymentMetadata.model_validate(raw)
        except ValidationError:
            pass
    if context.deployment is not None:
        return context.deployment
    raise transport_error(
        "release_unavailable",
        trace_id=context.trace_id,
        retry_count=context.retry_count,
    )


def _usage(payload: Mapping[str, Any]) -> Usage:
    raw = _mapping(payload.get("usage"))
    prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0))
    completion = raw.get("completion_tokens", raw.get("output_tokens", 0))
    prompt = prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else 0
    completion = completion if isinstance(completion, int) and not isinstance(completion, bool) else 0
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


def _citations(payload: Mapping[str, Any]) -> list[Citation]:
    raw = payload.get("citations", [])
    if not isinstance(raw, list):
        return []
    return [Citation.model_validate(item) for item in raw[:20]]


def _completion_id(payload: Mapping[str, Any]) -> str:
    raw = payload.get("id")
    if isinstance(raw, str) and _COMPLETION_ID.fullmatch(raw):
        return raw
    return f"chatcmpl-{secrets.token_urlsafe(12).replace('=', '')}"


def normalize_response(payload: Mapping[str, Any], context: NormalizationContext) -> ChatCompletionResponse:
    body = _unwrap(payload)
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        first = choices[0]
        message = _mapping(first.get("message"))
        content = message.get("content", "")
        refusal = message.get("refusal")
        finish_reason = first.get("finish_reason", "stop")
    else:
        output = body.get("output", body.get("generated_text", ""))
        content = output if isinstance(output, str) else ""
        refusal = None
        finish_reason = body.get("finish_reason", "stop")

    if finish_reason not in {"stop", "length", "content_filter", "error"}:
        finish_reason = "error"
    created = body.get("created")
    return ChatCompletionResponse(
        id=_completion_id(body),
        created=created if isinstance(created, int) and not isinstance(created, bool) else int(context.clock()),
        model=context.model_alias,
        choices=[
            CompletionChoice(
                message=AssistantMessage(
                    content=content if isinstance(content, str) else "",
                    refusal=refusal if isinstance(refusal, str) else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_usage(body),
        citations=_citations(body),
        trace=_trace(body, context),
        deployment=_deployment(body, context),
    )


def normalize_stream_event(payload: Mapping[str, Any], context: NormalizationContext) -> ChatCompletionStreamEvent:
    body = _unwrap(payload)
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    delta = _mapping(first.get("delta"))
    if not delta and isinstance(body.get("output"), str):
        delta = {"content": body["output"]}
    finish_reason = first.get("finish_reason")
    if finish_reason not in {None, "stop", "length", "content_filter", "error"}:
        finish_reason = "error"
    created = body.get("created")
    return ChatCompletionStreamEvent(
        id=_completion_id(body),
        created=created if isinstance(created, int) and not isinstance(created, bool) else int(context.clock()),
        model=context.model_alias,
        choices=[
            StreamChoice(
                delta=StreamDelta(
                    role="assistant" if delta.get("role") == "assistant" else None,
                    content=delta.get("content") if isinstance(delta.get("content"), str) else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_usage(body) if "usage" in body else None,
        citations=_citations(body) if "citations" in body else None,
        trace=_trace(body, context),
        deployment=_deployment(body, context),
    )


async def iter_sse_json(lines: AsyncIterable[str]) -> AsyncIterator[Mapping[str, Any]]:
    data_lines: list[str] = []
    async for line in lines:
        if line == "":
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data == "[DONE]":
                    return
                value = json.loads(data)
                if isinstance(value, Mapping):
                    yield value
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            value = json.loads(data)
            if isinstance(value, Mapping):
                yield value
