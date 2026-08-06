from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


SEMANTIC_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
MODEL_ALIAS = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PROFILE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SOURCE_COMMIT = re.compile(r"^[a-f0-9]{7,40}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")

SemanticVersion = Annotated[str, Field(pattern=SEMANTIC_VERSION.pattern)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=IDENTIFIER.pattern)]
ModelAlias = Annotated[str, Field(min_length=1, max_length=128, pattern=MODEL_ALIAS.pattern)]
ModelProfile = Annotated[str, Field(min_length=1, max_length=128, pattern=PROFILE.pattern)]
SourceCommit = Annotated[str, Field(pattern=SOURCE_COMMIT.pattern)]
Sha256 = Annotated[str, Field(pattern=SHA256.pattern)]
Precision = Literal["fp32", "fp16", "bf16", "int8", "int4"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Digest(ContractModel):
    algorithm: Literal["sha256"]
    value: Sha256


class BaseModelDescriptor(ContractModel):
    id: Annotated[str, Field(min_length=1, max_length=256)]
    revision: Annotated[str, Field(min_length=7, max_length=128)]


class AdapterDescriptor(ContractModel):
    type: Literal["lora", "qlora", "ia3", "prefix", "full-model"]
    stage: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9-]+$")]
    uri: AnyUrl
    digest: Digest


class TokenizerDescriptor(ContractModel):
    revision: Annotated[str, Field(min_length=1, max_length=128)]
    uri: AnyUrl
    digest: Digest


class TrainingProvenance(ContractModel):
    manifest_uri: AnyUrl
    manifest_digest: Digest


class EvaluationThreshold(ContractModel):
    domain: Literal[
        "data_quality",
        "retrieval_quality",
        "generation_citation_quality",
        "adaptation_evidence",
        "safety_authorization",
        "operational_slos",
        "cost",
        "rollout_comparison",
    ]
    metric: Identifier
    operator: Literal["lt", "lte", "gt", "gte", "eq"]
    value: float


class EvaluationReference(ContractModel):
    report_uri: AnyUrl
    report_digest: Digest
    decision: Literal["promote", "hold", "reject"]
    thresholds: Annotated[list[EvaluationThreshold], Field(min_length=1, max_length=128)]


class ServingRuntime(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=128)]
    version: SemanticVersion
    interface_version: SemanticVersion
    compatible_model_profiles: Annotated[list[ModelProfile], Field(min_length=1, max_length=32)]
    supported_precisions: Annotated[list[Precision], Field(min_length=1, max_length=5)]

    @model_validator(mode="after")
    def require_unique_capabilities(self) -> ServingRuntime:
        if len(set(self.compatible_model_profiles)) != len(self.compatible_model_profiles):
            raise ValueError("compatible model profiles must be unique")
        if len(set(self.supported_precisions)) != len(self.supported_precisions):
            raise ValueError("supported precisions must be unique")
        return self


class ModelReleaseManifest(ContractModel):
    contract_version: Literal["1.0.0"]
    kind: Literal["model_release"]
    release_id: Identifier
    version: SemanticVersion
    base_model: BaseModelDescriptor
    adapter: AdapterDescriptor
    tokenizer: TokenizerDescriptor
    model_profile: ModelProfile
    precision: Precision
    training_provenance: TrainingProvenance
    evaluation: EvaluationReference
    serving_runtime: ServingRuntime
    created_at: datetime
    source_commit: SourceCommit

    @field_validator("created_at")
    @classmethod
    def require_created_at_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone offset")
        return value


class ChatMessage(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=32768)]
    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")] | None = None


class RetrievalFilters(ContractModel):
    classification: Annotated[
        list[Literal["public", "internal", "confidential", "restricted"]],
        Field(min_length=1, max_length=4),
    ] | None = None
    source_version: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")] | None = None

    @model_validator(mode="after")
    def require_unique_classifications(self) -> RetrievalFilters:
        if self.classification is not None and len(set(self.classification)) != len(
            self.classification
        ):
            raise ValueError("classification filters must be unique")
        return self


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
    def reject_duplicate_stop_sequences(self) -> ChatCompletionRequest:
        if isinstance(self.stop, list) and len(set(self.stop)) != len(self.stop):
            raise ValueError("stop sequences must be unique")
        return self


class RuntimeCompatibility(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=128)]
    version: SemanticVersion
    interface_version: SemanticVersion
    model_profile: ModelProfile
    precision: Precision
    base_model_id: Annotated[str, Field(min_length=1, max_length=256)]
    base_model_revision: Annotated[str, Field(min_length=7, max_length=128)]
    adapter_type: Literal["lora"] = "lora"


class ArtifactPaths(ContractModel):
    adapter: str
    tokenizer: str
    training_manifest: str
    evaluation_report: str


class VerifiedRelease(ContractModel):
    manifest: ModelReleaseManifest
    paths: ArtifactPaths


class DeploymentMetadata(ContractModel):
    contract_version: Literal["1.0.0"]
    kind: Literal["deployment_metadata"]
    environment: Literal["dev", "staging", "production"]
    release_id: Identifier
    model_alias: ModelAlias
    deployment_name: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9-]*$")]
    deployment_slot: Literal["blue", "green"]
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")]
    runtime: Annotated[str, Field(min_length=1, max_length=128)]
    runtime_version: SemanticVersion
    index_version: SemanticVersion
    deployed_at: datetime
    source_commit: SourceCommit

    @field_validator("deployed_at")
    @classmethod
    def require_deployed_at_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deployed_at must include a timezone offset")
        return value
