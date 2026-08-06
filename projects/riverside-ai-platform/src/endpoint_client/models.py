from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]
ModelAlias = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
TraceId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=32768)]
    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")] | None = None


class RetrievalFilters(ContractModel):
    classification: list[Literal["public", "internal", "confidential", "restricted"]] | None = None
    source_version: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")] | None = None


class RetrievalOptions(ContractModel):
    enabled: bool
    top_k: Annotated[int, Field(ge=1, le=20)]
    search_type: Literal["similarity", "mmr", "hybrid"] | None = None
    filters: RetrievalFilters | None = None


class ChatCompletionRequest(ContractModel):
    model: ModelAlias
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=128)]
    max_input_tokens: Annotated[int, Field(ge=1, le=8192)]
    max_tokens: Annotated[int, Field(ge=1, le=2048)]
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.1
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    stream: bool
    stop: Annotated[str, Field(min_length=1, max_length=128)] | Annotated[
        list[Annotated[str, Field(min_length=1, max_length=128)]],
        Field(min_length=1, max_length=4),
    ] | None = None
    retrieval: RetrievalOptions | None = None

    @model_validator(mode="after")
    def validate_stop_is_unique(self) -> ChatCompletionRequest:
        if isinstance(self.stop, list) and len(self.stop) != len(set(self.stop)):
            raise ValueError("stop entries must be unique")
        return self


class Usage(ContractModel):
    prompt_tokens: Annotated[int, Field(ge=0, le=8192)]
    completion_tokens: Annotated[int, Field(ge=0, le=2048)]
    total_tokens: Annotated[int, Field(ge=0, le=10240)]

    @model_validator(mode="after")
    def validate_total(self) -> Usage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        return self


class TraceMetadata(ContractModel):
    trace_id: TraceId
    retry_count: Annotated[int, Field(ge=0, le=3)]


class DeploymentMetadata(ContractModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["deployment_metadata"] = "deployment_metadata"
    environment: Literal["dev", "staging", "production"]
    release_id: Identifier
    model_alias: ModelAlias
    deployment_name: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9-]*$")]
    deployment_slot: Literal["blue", "green"]
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")]
    runtime: Annotated[str, Field(min_length=1, max_length=128)]
    runtime_version: Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")]
    index_version: Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")]
    deployed_at: datetime
    source_commit: Annotated[str, Field(pattern=r"^[a-f0-9]{7,40}$")]


class AnswerSpan(ContractModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_order(self) -> AnswerSpan:
        if self.end <= self.start:
            raise ValueError("answer span end must be greater than start")
        return self


class Citation(ContractModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["citation"] = "citation"
    citation_id: Identifier
    chunk_id: Identifier
    document_id: Identifier
    source_uri: Annotated[str, Field(max_length=2048)]
    source_version: Annotated[str, Field(min_length=1, max_length=128)]
    content_hash: Sha256
    title: Annotated[str, Field(min_length=1, max_length=512)]
    answer_spans: Annotated[list[AnswerSpan], Field(min_length=1, max_length=32)]
    retrieved_at: datetime

    @model_validator(mode="after")
    def validate_spans(self) -> Citation:
        previous_end = -1
        for span in self.answer_spans:
            if span.start < previous_end:
                raise ValueError("answer spans must be ordered and non-overlapping")
            previous_end = span.end
        return self


class AssistantMessage(ContractModel):
    role: Literal["assistant"] = "assistant"
    content: Annotated[str, Field(max_length=131072)]
    refusal: Annotated[str, Field(max_length=2048)] | None = None


class CompletionChoice(ContractModel):
    index: Literal[0] = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "content_filter", "error"]


class ChatCompletionResponse(ContractModel):
    id: Annotated[str, Field(pattern=r"^chatcmpl-[A-Za-z0-9_-]{8,64}$")]
    object: Literal["chat.completion"] = "chat.completion"
    created: Annotated[int, Field(ge=0)]
    model: ModelAlias
    choices: Annotated[list[CompletionChoice], Field(min_length=1, max_length=1)]
    usage: Usage
    citations: Annotated[list[Citation], Field(max_length=20)]
    trace: TraceMetadata
    deployment: DeploymentMetadata

    @model_validator(mode="after")
    def validate_citation_bounds(self) -> ChatCompletionResponse:
        answer_length = len(self.choices[0].message.content)
        if any(span.end > answer_length for citation in self.citations for span in citation.answer_spans):
            raise ValueError("citation answer span exceeds answer content")
        return self


class StreamDelta(ContractModel):
    role: Literal["assistant"] | None = None
    content: Annotated[str, Field(max_length=32768)] | None = None


class StreamChoice(ContractModel):
    index: Literal[0] = 0
    delta: StreamDelta
    finish_reason: Literal["stop", "length", "content_filter", "error"] | None


class ChatCompletionStreamEvent(ContractModel):
    id: Annotated[str, Field(pattern=r"^chatcmpl-[A-Za-z0-9_-]{8,64}$")]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: Annotated[int, Field(ge=0)]
    model: ModelAlias
    choices: Annotated[list[StreamChoice], Field(min_length=1, max_length=1)]
    usage: Usage | None = None
    citations: Annotated[list[Citation], Field(max_length=20)] | None = None
    trace: TraceMetadata
    deployment: DeploymentMetadata


ErrorCode = Literal[
    "invalid_request",
    "unauthorized",
    "forbidden",
    "policy_violation",
    "overloaded",
    "timeout",
    "backend_failure",
    "release_unavailable",
    "internal_error",
]


class ErrorDetail(ContractModel):
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=1024)]
    type: Literal[
        "request_error",
        "authentication_error",
        "authorization_error",
        "policy_error",
        "capacity_error",
        "deadline_error",
        "backend_error",
        "service_error",
    ]
    param: Annotated[str, Field(max_length=128)] | None = None
    retryable: bool
    retry_after_seconds: Annotated[int, Field(ge=1, le=300)] | None = None

    @model_validator(mode="after")
    def validate_overload_retry(self) -> ErrorDetail:
        if self.code == "overloaded" and (not self.retryable or self.retry_after_seconds is None):
            raise ValueError("overloaded errors must be retryable and include retry_after_seconds")
        return self


class AppError(ContractModel):
    error: ErrorDetail
    trace: TraceMetadata
    deployment: DeploymentMetadata | None = None
