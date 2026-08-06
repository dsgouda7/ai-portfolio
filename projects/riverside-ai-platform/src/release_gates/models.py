from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping
from urllib.parse import urlparse

CONTRACT_VERSION = "1.0.0"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SOURCE_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{7,40}$")
_SLICE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class EvidenceValidationError(ValueError):
    """Raised when release evidence is malformed or internally contradictory."""


class Domain(StrEnum):
    DATA_QUALITY = "data_quality"
    RETRIEVAL_QUALITY = "retrieval_quality"
    GENERATION_CITATION_QUALITY = "generation_citation_quality"
    ADAPTATION_EVIDENCE = "adaptation_evidence"
    SAFETY_AUTHORIZATION = "safety_authorization"
    OPERATIONAL_SLOS = "operational_slos"
    COST = "cost"
    ROLLOUT_COMPARISON = "rollout_comparison"


class ThresholdOperator(StrEnum):
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"


class MetricStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class Decision(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


class FailureDisposition(StrEnum):
    HOLD = "hold"
    REJECT = "reject"


def require_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise EvidenceValidationError(f"{field_name} must be a v1 contract identifier")


def require_semver(value: str, field_name: str) -> None:
    if not _SEMVER_PATTERN.fullmatch(value):
        raise EvidenceValidationError(f"{field_name} must be a semantic version")


def require_source_commit(value: str) -> None:
    if not _SOURCE_COMMIT_PATTERN.fullmatch(value):
        raise EvidenceValidationError("source_commit must be a 7-40 character lowercase hex commit")


def require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise EvidenceValidationError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class DatasetRef:
    id: str
    version: str

    def __post_init__(self) -> None:
        require_identifier(self.id, "dataset.id")
        if not self.version or len(self.version) > 128:
            raise EvidenceValidationError("dataset.version must contain 1-128 characters")

    def to_contract(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version}


@dataclass(frozen=True, slots=True)
class EvaluatorRef:
    id: str
    version: str

    def __post_init__(self) -> None:
        require_identifier(self.id, "evaluator.id")
        require_semver(self.version, "evaluator.version")

    def to_contract(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version}


@dataclass(frozen=True, slots=True)
class Slice:
    name: str
    value: str

    def __post_init__(self) -> None:
        if not _SLICE_NAME_PATTERN.fullmatch(self.name) or len(self.name) > 64:
            raise EvidenceValidationError("slice.name must be a lowercase v1 slice key")
        if not self.value or len(self.value) > 128:
            raise EvidenceValidationError("slice.value must contain 1-128 characters")

    def to_contract(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class Threshold:
    operator: ThresholdOperator
    value: float
    unit: str

    def __post_init__(self) -> None:
        require_finite(self.value, "threshold.value")
        if not self.unit or len(self.unit) > 64:
            raise EvidenceValidationError("threshold.unit must contain 1-64 characters")

    def accepts(self, value: float) -> bool:
        require_finite(value, "observed.value")
        operations = {
            ThresholdOperator.LT: lambda: value < self.value,
            ThresholdOperator.LTE: lambda: value <= self.value,
            ThresholdOperator.GT: lambda: value > self.value,
            ThresholdOperator.GTE: lambda: value >= self.value,
            ThresholdOperator.EQ: lambda: value == self.value,
        }
        return operations[self.operator]()

    def to_contract(self) -> dict[str, str | float]:
        return {"operator": self.operator.value, "value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class Observation:
    value: float
    unit: str

    def __post_init__(self) -> None:
        require_finite(self.value, "observed.value")
        if not self.unit or len(self.unit) > 64:
            raise EvidenceValidationError("observed.unit must contain 1-64 characters")

    def to_contract(self) -> dict[str, str | float]:
        return {"value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class Uncertainty:
    method: str
    lower: float
    upper: float
    confidence_level: float
    sample_size: int

    def __post_init__(self) -> None:
        require_finite(self.lower, "uncertainty.lower")
        require_finite(self.upper, "uncertainty.upper")
        require_finite(self.confidence_level, "uncertainty.confidence_level")
        if not self.method or len(self.method) > 128:
            raise EvidenceValidationError("uncertainty.method must contain 1-128 characters")
        if self.lower > self.upper:
            raise EvidenceValidationError("uncertainty.lower cannot exceed uncertainty.upper")
        if not 0 < self.confidence_level < 1:
            raise EvidenceValidationError("uncertainty.confidence_level must be between 0 and 1")
        if self.sample_size < 1:
            raise EvidenceValidationError("uncertainty.sample_size must be positive")

    def conservative_value(self, operator: ThresholdOperator) -> float:
        if operator in (ThresholdOperator.GT, ThresholdOperator.GTE):
            return self.lower
        if operator in (ThresholdOperator.LT, ThresholdOperator.LTE):
            return self.upper
        if self.lower != self.upper:
            raise EvidenceValidationError(
                "equality thresholds require a zero-width interval for conservative evaluation"
            )
        return self.lower

    def to_contract(self) -> dict[str, str | float | int]:
        return {
            "method": self.method,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True, slots=True)
class MetricRecord:
    domain: Domain
    metric_id: str
    dataset: DatasetRef
    evaluator: EvaluatorRef
    slice: Slice
    threshold: Threshold
    observed: Observation
    uncertainty: Uncertainty | None = None
    status: MetricStatus | None = None

    def __post_init__(self) -> None:
        require_identifier(self.metric_id, "metric_id")
        if self.threshold.unit != self.observed.unit:
            raise EvidenceValidationError("threshold and observed units must match")
        if self.uncertainty is not None and not (
            self.uncertainty.lower <= self.observed.value <= self.uncertainty.upper
        ):
            raise EvidenceValidationError("observed value must lie inside its uncertainty interval")
        computed_status = (
            MetricStatus.PASS
            if self.threshold.accepts(self.observed.value)
            else MetricStatus.FAIL
        )
        if self.status is not None and self.status is not computed_status:
            raise EvidenceValidationError(
                f"recorded status {self.status.value!r} disagrees with threshold result "
                f"{computed_status.value!r}"
            )
        object.__setattr__(self, "status", computed_status)

    @property
    def key(self) -> tuple[Domain, str, str, str]:
        return (self.domain, self.metric_id, self.slice.name, self.slice.value)

    def passes_conservatively(self) -> bool:
        if self.uncertainty is None:
            return self.status is MetricStatus.PASS
        return self.threshold.accepts(
            self.uncertainty.conservative_value(self.threshold.operator)
        )

    def to_contract(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metric_id": self.metric_id,
            "dataset": self.dataset.to_contract(),
            "evaluator": self.evaluator.to_contract(),
            "slice": self.slice.to_contract(),
            "threshold": self.threshold.to_contract(),
            "observed": self.observed.to_contract(),
            "status": self.status.value if self.status is not None else MetricStatus.FAIL.value,
        }
        if self.uncertainty is not None:
            result["uncertainty"] = self.uncertainty.to_contract()
        return result


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    name: str
    uri: str
    digest: str

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 128:
            raise EvidenceValidationError("evidence.name must contain 1-128 characters")
        parsed = urlparse(self.uri)
        if not parsed.scheme or len(self.uri) > 2048:
            raise EvidenceValidationError("evidence.uri must be an absolute URI")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EvidenceValidationError(
                "evidence.uri cannot contain credentials, query parameters, or fragments"
            )
        if not _SHA256_PATTERN.fullmatch(self.digest):
            raise EvidenceValidationError("evidence.digest must be a lowercase SHA-256 digest")

    def to_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "digest": {"algorithm": "sha256", "value": self.digest},
        }


@dataclass(frozen=True, slots=True)
class ReleaseContext:
    report_id: str
    release_id: str
    release_version: str
    generated_at: datetime
    source_commit: str
    evidence: tuple[EvidenceRef, ...]
    baseline_release_id: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.report_id, "report_id")
        require_identifier(self.release_id, "release_id")
        require_semver(self.release_version, "release_version")
        require_source_commit(self.source_commit)
        if self.baseline_release_id is not None:
            require_identifier(self.baseline_release_id, "baseline_release_id")
        if not self.evidence:
            raise EvidenceValidationError("at least one evidence reference is required")
        if len(self.evidence) > 128:
            raise EvidenceValidationError("at most 128 evidence references are allowed")
        names = [item.name for item in self.evidence]
        if len(names) != len(set(names)):
            raise EvidenceValidationError("evidence names must be unique")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise EvidenceValidationError("generated_at must be timezone-aware")

    def generated_at_contract(self) -> str:
        return self.generated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


JsonObject = Mapping[str, Any]
