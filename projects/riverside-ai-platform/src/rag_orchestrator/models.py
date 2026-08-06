from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from endpoint_client.models import RetrievalFilters


class OrchestratorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrustedAuthContext(OrchestratorModel):
    tenant_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]
    principal_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]
    group_ids: Annotated[frozenset[str], Field(max_length=64)] = frozenset()
    tenant_tier: Literal["standard", "premium", "internal"]


class ACL(OrchestratorModel):
    visibility: Literal["tenant", "restricted"]
    principals: Annotated[frozenset[str], Field(max_length=64)] = frozenset()
    groups: Annotated[frozenset[str], Field(max_length=64)] = frozenset()


class SearchQuery(OrchestratorModel):
    text: Annotated[str, Field(min_length=1, max_length=32768)]
    top_k: Annotated[int, Field(ge=1, le=20)]
    search_type: Literal["similarity", "mmr", "hybrid"]
    filters: RetrievalFilters | None = None
    auth: TrustedAuthContext


class RetrievedChunk(OrchestratorModel):
    tenant_id: str
    chunk_id: Annotated[str, Field(min_length=1, max_length=128)]
    document_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_uri: Annotated[str, Field(min_length=1, max_length=2048)]
    source_version: Annotated[str, Field(min_length=1, max_length=128)]
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    title: Annotated[str, Field(min_length=1, max_length=512)]
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    score: float
    acl: ACL
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")]
    classification: Literal["public", "internal", "confidential", "restricted"]
    index_version: str
    deletion_status: Literal["active", "pending", "deleted"]
    indexed_at: datetime

    @model_validator(mode="after")
    def validate_finite_score(self) -> RetrievedChunk:
        if not self.score == self.score or self.score in {float("inf"), float("-inf")}:
            raise ValueError("score must be finite")
        return self


class OrchestratorConfig(OrchestratorModel):
    default_top_k: Annotated[int, Field(default=6, ge=1, le=20)]
    max_top_k: Annotated[int, Field(default=20, ge=1, le=20)]
    default_search_type: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    max_context_characters: Annotated[int, Field(default=24000, ge=1000, le=100000)]
    max_chunk_characters: Annotated[int, Field(default=6000, ge=256, le=100000)]
    refusal_message: Annotated[str, Field(default="I cannot answer from the authorized sources available.", min_length=1, max_length=2048)]

    @model_validator(mode="after")
    def validate_top_k(self) -> OrchestratorConfig:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot exceed max_top_k")
        return self
