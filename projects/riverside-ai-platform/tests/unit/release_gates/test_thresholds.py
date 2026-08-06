from __future__ import annotations

import pytest

from release_gates import (
    DatasetRef,
    Domain,
    EvaluatorRef,
    EvidenceValidationError,
    MetricRecord,
    MetricStatus,
    Observation,
    Slice,
    Threshold,
    ThresholdOperator,
    Uncertainty,
)


@pytest.mark.parametrize(
    ("operator", "observed", "threshold", "expected"),
    [
        (ThresholdOperator.LT, 0.4, 0.5, True),
        (ThresholdOperator.LTE, 0.5, 0.5, True),
        (ThresholdOperator.GT, 0.6, 0.5, True),
        (ThresholdOperator.GTE, 0.5, 0.5, True),
        (ThresholdOperator.EQ, 0.5, 0.5, True),
        (ThresholdOperator.LT, 0.5, 0.5, False),
        (ThresholdOperator.GT, 0.5, 0.5, False),
        (ThresholdOperator.EQ, 0.6, 0.5, False),
    ],
)
def test_threshold_operators(
    operator: ThresholdOperator,
    observed: float,
    threshold: float,
    expected: bool,
) -> None:
    assert Threshold(operator=operator, value=threshold, unit="ratio").accepts(observed) is expected


def test_metric_rejects_mismatched_units() -> None:
    with pytest.raises(EvidenceValidationError, match="units must match"):
        MetricRecord(
            domain=Domain.COST,
            metric_id="cost_per_successful_request",
            dataset=DatasetRef(id="cost-fixture", version="1.0"),
            evaluator=EvaluatorRef(id="cost-evaluator", version="1.0.0"),
            slice=Slice(name="tenant_tier", value="standard"),
            threshold=Threshold(ThresholdOperator.LTE, 0.05, "usd_per_request"),
            observed=Observation(0.04, "usd_per_token"),
        )


def test_metric_rejects_claimed_status_that_disagrees_with_threshold() -> None:
    with pytest.raises(EvidenceValidationError, match="disagrees with threshold"):
        MetricRecord(
            domain=Domain.RETRIEVAL_QUALITY,
            metric_id="recall_at_6",
            dataset=DatasetRef(id="retrieval-fixture", version="1.0"),
            evaluator=EvaluatorRef(id="retrieval-evaluator", version="1.0.0"),
            slice=Slice(name="query_type", value="all"),
            threshold=Threshold(ThresholdOperator.GTE, 0.8, "ratio"),
            observed=Observation(0.7, "ratio"),
            status=MetricStatus.PASS,
        )


def test_conservative_uncertainty_uses_lower_bound_for_minimum_gate() -> None:
    metric = MetricRecord(
        domain=Domain.RETRIEVAL_QUALITY,
        metric_id="recall_at_6",
        dataset=DatasetRef(id="retrieval-fixture", version="1.0"),
        evaluator=EvaluatorRef(id="retrieval-evaluator", version="1.0.0"),
        slice=Slice(name="query_type", value="all"),
        threshold=Threshold(ThresholdOperator.GTE, 0.8, "ratio"),
        observed=Observation(0.85, "ratio"),
        uncertainty=Uncertainty("bootstrap", 0.76, 0.90, 0.95, 200),
    )

    assert metric.status is MetricStatus.PASS
    assert metric.passes_conservatively() is False
