"""Pydantic models for the legacy and Riverside v1 serving contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(ContractModel):
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 6
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.1


class QueryResponse(ContractModel):
    answer: str
    question: str
    sources_count: int


class HealthResponse(ContractModel):
    status: str
    mode: str
    retrieval: str
    generation: str


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
    model: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=128)]
    max_input_tokens: Annotated[int, Field(ge=1, le=8192)]
    max_tokens: Annotated[int, Field(ge=1, le=2048)]
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.1
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    stream: bool
    stop: str | list[str] | None = None
    retrieval: RetrievalOptions | None = None

    @model_validator(mode="after")
    def validate_stop(self) -> "ChatCompletionRequest":
        if isinstance(self.stop, list):
            if not 1 <= len(self.stop) <= 4 or len(self.stop) != len(set(self.stop)):
                raise ValueError("stop must contain one to four unique entries")
            if any(not 1 <= len(value) <= 128 for value in self.stop):
                raise ValueError("stop entries must contain 1 to 128 characters")
        elif isinstance(self.stop, str) and not 1 <= len(self.stop) <= 128:
            raise ValueError("stop must contain 1 to 128 characters")
        return self


class AnswerSpan(ContractModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]


class Citation(ContractModel):
    contract_version: Literal["1.0.0"]
    kind: Literal["citation"]
    citation_id: str
    chunk_id: str
    document_id: str
    source_uri: str
    source_version: str
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    title: str
    answer_spans: Annotated[list[AnswerSpan], Field(min_length=1, max_length=32)]
    retrieved_at: datetime


class Usage(ContractModel):
    prompt_tokens: Annotated[int, Field(ge=0, le=8192)]
    completion_tokens: Annotated[int, Field(ge=0, le=2048)]
    total_tokens: Annotated[int, Field(ge=0, le=10240)]


class TraceMetadata(ContractModel):
    trace_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    retry_count: Annotated[int, Field(ge=0, le=3)]


class DeploymentMetadata(ContractModel):
    contract_version: Literal["1.0.0"]
    kind: Literal["deployment_metadata"]
    environment: Literal["dev", "staging", "production"]
    release_id: str
    model_alias: str
    deployment_name: str
    deployment_slot: Literal["blue", "green"]
    region: str
    runtime: str
    runtime_version: str
    index_version: str
    deployed_at: datetime
    source_commit: str


class AssistantMessage(ContractModel):
    role: Literal["assistant"]
    content: str
    refusal: str | None = None


class CompletionChoice(ContractModel):
    index: Literal[0]
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "content_filter", "error"]


class ChatCompletionResponse(ContractModel):
    id: Annotated[str, Field(pattern=r"^chatcmpl-[A-Za-z0-9_-]{8,64}$")]
    object: Literal["chat.completion"]
    created: Annotated[int, Field(ge=0)]
    model: str
    choices: Annotated[list[CompletionChoice], Field(min_length=1, max_length=1)]
    usage: Usage
    citations: Annotated[list[Citation], Field(max_length=20)]
    trace: TraceMetadata
    deployment: DeploymentMetadata
