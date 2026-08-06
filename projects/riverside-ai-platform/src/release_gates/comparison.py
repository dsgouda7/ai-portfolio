from __future__ import annotations

from dataclasses import dataclass

from .models import (
    DatasetRef,
    Domain,
    EvaluatorRef,
    EvidenceValidationError,
    MetricRecord,
    Observation,
    Slice,
    Threshold,
    Uncertainty,
    require_identifier,
)


@dataclass(frozen=True, slots=True)
class BaselineCandidateComparison:
    baseline_release_id: str
    candidate_release_id: str
    baseline: Observation
    candidate: Observation

    def __post_init__(self) -> None:
        require_identifier(self.baseline_release_id, "baseline_release_id")
        require_identifier(self.candidate_release_id, "candidate_release_id")
        if self.baseline_release_id == self.candidate_release_id:
            raise EvidenceValidationError("baseline and candidate releases must differ")
        if self.baseline.unit != self.candidate.unit:
            raise EvidenceValidationError("baseline and candidate units must match")

    @property
    def absolute_delta(self) -> float:
        return self.candidate.value - self.baseline.value

    @property
    def relative_delta(self) -> float:
        if self.baseline.value == 0:
            raise EvidenceValidationError("relative delta is undefined for a zero baseline")
        return (self.candidate.value - self.baseline.value) / abs(self.baseline.value)

    def as_delta_metric(
        self,
        *,
        metric_id: str,
        dataset: DatasetRef,
        evaluator: EvaluatorRef,
        slice: Slice,
        threshold: Threshold,
        scale: float = 1.0,
        uncertainty: Uncertainty | None = None,
        domain: Domain = Domain.ROLLOUT_COMPARISON,
    ) -> MetricRecord:
        if scale <= 0:
            raise EvidenceValidationError("comparison scale must be positive")
        observed = Observation(value=self.absolute_delta * scale, unit=threshold.unit)
        return MetricRecord(
            domain=domain,
            metric_id=metric_id,
            dataset=dataset,
            evaluator=evaluator,
            slice=slice,
            threshold=threshold,
            observed=observed,
            uncertainty=uncertainty,
        )
