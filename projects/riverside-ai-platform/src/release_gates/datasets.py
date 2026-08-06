from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, ClassVar, Generic, Mapping, Protocol, TypeVar

from .models import (
    Domain,
    EvidenceValidationError,
    require_identifier,
    require_semver,
)

DATASET_SCHEMA_VERSION = "evaluation-dataset.v1"


class AdaptationStage(StrEnum):
    CPT = "cpt"
    SFT = "sft"
    DPO = "dpo"


class RolloutStage(StrEnum):
    OFFLINE = "offline"
    CLOUD_SMOKE = "cloud_smoke"
    BOUNDED_LOAD = "bounded_load"
    SHADOW = "shadow"
    CANARY = "canary"
    BROAD = "broad"


@dataclass(frozen=True, slots=True)
class CaseSlice:
    name: str
    value: str

    def __post_init__(self) -> None:
        if not self.name or not self.value:
            raise EvidenceValidationError("dataset case slice name and value are required")


class EvaluationCase(Protocol):
    domain: ClassVar[Domain]
    case_id: str
    slice: CaseSlice

    def to_payload(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DataQualityCase:
    domain: ClassVar[Domain] = Domain.DATA_QUALITY
    case_id: str
    slice: CaseSlice
    record_id: str
    required_fields: tuple[str, ...]
    present_fields: tuple[str, ...]
    parse_succeeded: bool
    quarantined: bool

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.record_id, "record_id")
        if not self.required_fields:
            raise EvidenceValidationError("data-quality cases require at least one required field")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "record_id": self.record_id,
            "required_fields": list(self.required_fields),
            "present_fields": list(self.present_fields),
            "parse_succeeded": self.parse_succeeded,
            "quarantined": self.quarantined,
        }


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    domain: ClassVar[Domain] = Domain.RETRIEVAL_QUALITY
    case_id: str
    slice: CaseSlice
    query: str
    relevant_document_ids: tuple[str, ...]
    expected_refusal: bool
    tenant_id: str
    authorized_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.tenant_id, "tenant_id")
        if not self.query:
            raise EvidenceValidationError("retrieval query cannot be empty")
        if not self.expected_refusal and not self.relevant_document_ids:
            raise EvidenceValidationError(
                "answerable retrieval cases require at least one relevant document"
            )

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "query": self.query,
            "relevant_document_ids": list(self.relevant_document_ids),
            "expected_refusal": self.expected_refusal,
            "tenant_id": self.tenant_id,
            "authorized_document_ids": list(self.authorized_document_ids),
        }


@dataclass(frozen=True, slots=True)
class GenerationCitationCase:
    domain: ClassVar[Domain] = Domain.GENERATION_CITATION_QUALITY
    case_id: str
    slice: CaseSlice
    prompt: str
    context_document_ids: tuple[str, ...]
    expected_citation_document_ids: tuple[str, ...]
    one_sentence_required: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        if not self.prompt:
            raise EvidenceValidationError("generation prompt cannot be empty")
        if not self.context_document_ids:
            raise EvidenceValidationError("generation cases require context documents")
        if not set(self.expected_citation_document_ids).issubset(self.context_document_ids):
            raise EvidenceValidationError(
                "expected citations must refer to supplied context documents"
            )

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "prompt": self.prompt,
            "context_document_ids": list(self.context_document_ids),
            "expected_citation_document_ids": list(self.expected_citation_document_ids),
            "one_sentence_required": self.one_sentence_required,
        }


@dataclass(frozen=True, slots=True)
class AdaptationEvidenceCase:
    domain: ClassVar[Domain] = Domain.ADAPTATION_EVIDENCE
    case_id: str
    slice: CaseSlice
    stage: AdaptationStage
    baseline_artifact_id: str
    candidate_artifact_id: str
    heldout_dataset_id: str
    required_metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.baseline_artifact_id, "baseline_artifact_id")
        require_identifier(self.candidate_artifact_id, "candidate_artifact_id")
        require_identifier(self.heldout_dataset_id, "heldout_dataset_id")
        if self.baseline_artifact_id == self.candidate_artifact_id:
            raise EvidenceValidationError("adaptation baseline and candidate must differ")
        if not self.required_metric_ids:
            raise EvidenceValidationError("adaptation cases require objective-aligned metrics")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "stage": self.stage.value,
            "baseline_artifact_id": self.baseline_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "heldout_dataset_id": self.heldout_dataset_id,
            "required_metric_ids": list(self.required_metric_ids),
        }


@dataclass(frozen=True, slots=True)
class SafetyAuthorizationCase:
    domain: ClassVar[Domain] = Domain.SAFETY_AUTHORIZATION
    case_id: str
    slice: CaseSlice
    actor_tenant_id: str
    resource_tenant_id: str
    requested_document_id: str
    expected_authorized: bool
    safety_category: str

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.actor_tenant_id, "actor_tenant_id")
        require_identifier(self.resource_tenant_id, "resource_tenant_id")
        require_identifier(self.requested_document_id, "requested_document_id")
        if not self.safety_category:
            raise EvidenceValidationError("safety_category is required")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "actor_tenant_id": self.actor_tenant_id,
            "resource_tenant_id": self.resource_tenant_id,
            "requested_document_id": self.requested_document_id,
            "expected_authorized": self.expected_authorized,
            "safety_category": self.safety_category,
        }


@dataclass(frozen=True, slots=True)
class OperationalCase:
    domain: ClassVar[Domain] = Domain.OPERATIONAL_SLOS
    case_id: str
    slice: CaseSlice
    scenario: str
    load_stage: str
    observation_window_seconds: int
    expected_outcome: str

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        if not self.scenario or not self.load_stage or not self.expected_outcome:
            raise EvidenceValidationError("operational scenario fields cannot be empty")
        if self.observation_window_seconds < 1:
            raise EvidenceValidationError("operational observation window must be positive")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "scenario": self.scenario,
            "load_stage": self.load_stage,
            "observation_window_seconds": self.observation_window_seconds,
            "expected_outcome": self.expected_outcome,
        }


@dataclass(frozen=True, slots=True)
class CostCase:
    domain: ClassVar[Domain] = Domain.COST
    case_id: str
    slice: CaseSlice
    workload_id: str
    successful_requests: int
    successful_output_tokens: int
    currency: str = "USD"

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.workload_id, "workload_id")
        if self.successful_requests < 1 or self.successful_output_tokens < 1:
            raise EvidenceValidationError("cost denominators must be positive")
        if self.currency != "USD":
            raise EvidenceValidationError("v1 cost cases use USD")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "workload_id": self.workload_id,
            "successful_requests": self.successful_requests,
            "successful_output_tokens": self.successful_output_tokens,
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class RolloutComparisonCase:
    domain: ClassVar[Domain] = Domain.ROLLOUT_COMPARISON
    case_id: str
    slice: CaseSlice
    stage: RolloutStage
    owner: str
    observation_window_seconds: int
    baseline_release_id: str
    candidate_release_id: str
    rollback_target: str
    abort_metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.baseline_release_id, "baseline_release_id")
        require_identifier(self.candidate_release_id, "candidate_release_id")
        require_identifier(self.rollback_target, "rollback_target")
        if self.baseline_release_id == self.candidate_release_id:
            raise EvidenceValidationError("rollout baseline and candidate must differ")
        if not self.owner or self.observation_window_seconds < 1:
            raise EvidenceValidationError("rollout owner and observation window are required")
        if not self.abort_metric_ids:
            raise EvidenceValidationError("rollout cases require abort criteria")

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "stage": self.stage.value,
            "owner": self.owner,
            "observation_window_seconds": self.observation_window_seconds,
            "baseline_release_id": self.baseline_release_id,
            "candidate_release_id": self.candidate_release_id,
            "rollback_target": self.rollback_target,
            "abort_metric_ids": list(self.abort_metric_ids),
        }


CaseT = TypeVar("CaseT", bound=EvaluationCase)


@dataclass(frozen=True, slots=True)
class VersionedDataset(Generic[CaseT]):
    dataset_id: str
    version: str
    domain: Domain
    evaluator_contract_version: str
    created_at: datetime
    source_digest: str
    cases: tuple[CaseT, ...]
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_identifier(self.dataset_id, "dataset_id")
        require_semver(self.evaluator_contract_version, "evaluator_contract_version")
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise EvidenceValidationError(
                f"schema_version must be {DATASET_SCHEMA_VERSION!r}"
            )
        if not self.version or len(self.version) > 128:
            raise EvidenceValidationError("dataset version must contain 1-128 characters")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise EvidenceValidationError("dataset created_at must be timezone-aware")
        if len(self.source_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_digest
        ):
            raise EvidenceValidationError("dataset source_digest must be lowercase SHA-256")
        if not self.cases:
            raise EvidenceValidationError("evaluation datasets cannot be empty")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise EvidenceValidationError("evaluation case IDs must be unique")
        if any(case.domain is not self.domain for case in self.cases):
            raise EvidenceValidationError("every case must match the dataset domain")

    def to_contract(self) -> dict[str, Any]:
        cases: list[dict[str, Any]] = []
        for case in self.cases:
            cases.append(
                {
                    "case_id": case.case_id,
                    "slice": {"name": case.slice.name, "value": case.slice.value},
                    **case.to_payload(),
                }
            )
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "domain": self.domain.value,
            "evaluator_contract_version": self.evaluator_contract_version,
            "created_at": self.created_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source_digest": self.source_digest,
            "cases": cases,
        }
