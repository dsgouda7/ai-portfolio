from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

from .datasets import CaseT, VersionedDataset
from .models import EvaluatorRef, MetricRecord


class ExternalEvaluationRequired(RuntimeError):
    """Raised when a caller requests an evaluator that was not explicitly configured."""


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    evaluator: EvaluatorRef
    metrics: tuple[MetricRecord, ...]


@runtime_checkable
class ExternalEvaluator(Protocol[CaseT]):
    """Interface for an explicitly configured judge, service, or human-review adapter."""

    @property
    def identity(self) -> EvaluatorRef: ...

    def evaluate(self, dataset: VersionedDataset[CaseT]) -> Sequence[MetricRecord]: ...


class DisabledExternalEvaluator(Generic[CaseT]):
    """Default evaluator that guarantees release-gate code performs no external calls."""

    def __init__(self, identity: EvaluatorRef) -> None:
        self._identity = identity

    @property
    def identity(self) -> EvaluatorRef:
        return self._identity

    def evaluate(self, dataset: VersionedDataset[CaseT]) -> Sequence[MetricRecord]:
        raise ExternalEvaluationRequired(
            f"external evaluator {self.identity.id!r} is disabled for dataset "
            f"{dataset.dataset_id!r}; inject an explicit adapter to run it"
        )


def collect_evaluation(
    evaluator: ExternalEvaluator[CaseT],
    dataset: VersionedDataset[CaseT],
) -> EvaluationBatch:
    metrics = tuple(evaluator.evaluate(dataset))
    for metric in metrics:
        if metric.dataset.id != dataset.dataset_id or metric.dataset.version != dataset.version:
            raise ValueError("evaluator returned a metric for a different dataset version")
        if metric.evaluator != evaluator.identity:
            raise ValueError("evaluator result identity does not match the configured adapter")
        if metric.domain is not dataset.domain:
            raise ValueError("evaluator returned a metric for a different domain")
    return EvaluationBatch(evaluator=evaluator.identity, metrics=metrics)
