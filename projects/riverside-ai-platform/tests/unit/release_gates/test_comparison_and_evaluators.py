from __future__ import annotations

from datetime import datetime, timezone

import pytest

from release_gates import (
    BaselineCandidateComparison,
    CaseSlice,
    DataQualityCase,
    DatasetRef,
    DisabledExternalEvaluator,
    Domain,
    EvaluatorRef,
    ExternalEvaluationRequired,
    Observation,
    Slice,
    Threshold,
    ThresholdOperator,
    VersionedDataset,
)


def test_baseline_candidate_comparison_produces_delta_metric() -> None:
    comparison = BaselineCandidateComparison(
        baseline_release_id="baseline-release",
        candidate_release_id="candidate-release",
        baseline=Observation(0.965, "ratio"),
        candidate=Observation(0.980, "ratio"),
    )

    metric = comparison.as_delta_metric(
        metric_id="candidate_success_rate_delta",
        dataset=DatasetRef("shadow-traffic", "1.0"),
        evaluator=EvaluatorRef("rollout-evaluator", "1.0.0"),
        slice=Slice("deployment_slot", "green"),
        threshold=Threshold(ThresholdOperator.GTE, 0, "percentage_points"),
        scale=100,
    )

    assert metric.domain is Domain.ROLLOUT_COMPARISON
    assert metric.observed.value == pytest.approx(1.5)


def test_external_evaluator_is_disabled_by_default() -> None:
    dataset = VersionedDataset(
        dataset_id="data-quality-fixture",
        version="1.0",
        domain=Domain.DATA_QUALITY,
        evaluator_contract_version="1.0.0",
        created_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        source_digest="a" * 64,
        cases=(
            DataQualityCase(
                case_id="case-001",
                slice=CaseSlice("format", "text"),
                record_id="doc-001",
                required_fields=("document_id",),
                present_fields=("document_id",),
                parse_succeeded=True,
                quarantined=False,
            ),
        ),
    )
    evaluator = DisabledExternalEvaluator[DataQualityCase](
        EvaluatorRef("external-judge", "1.0.0")
    )

    with pytest.raises(ExternalEvaluationRequired, match="disabled"):
        evaluator.evaluate(dataset)
