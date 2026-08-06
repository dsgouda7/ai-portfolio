from __future__ import annotations

from dataclasses import replace

import pytest

from release_gates import (
    Decision,
    Domain,
    EvidenceValidationError,
    GatePolicy,
    MetricRecord,
    Observation,
    Threshold,
    ThresholdOperator,
    Uncertainty,
    evaluate_release,
)


def test_complete_passing_evidence_promotes(minimal_policy, passing_metrics) -> None:
    evaluation = evaluate_release(passing_metrics, minimal_policy)

    assert evaluation.decision is Decision.PROMOTE
    assert "All required domain thresholds passed" in evaluation.reasons[0]


def test_missing_domain_evidence_holds(minimal_policy, passing_metrics) -> None:
    incomplete = tuple(
        metric for metric in passing_metrics if metric.domain is not Domain.COST
    )

    evaluation = evaluate_release(incomplete, minimal_policy)

    assert evaluation.decision is Decision.HOLD
    assert any("Missing all evidence for domain cost" in reason for reason in evaluation.reasons)


def test_blocking_failure_rejects(minimal_policy, passing_metrics) -> None:
    failed = tuple(
        replace(metric, observed=Observation(0.25, "ratio"), status=None)
        if metric.domain is Domain.SAFETY_AUTHORIZATION
        else metric
        for metric in passing_metrics
    )

    evaluation = evaluate_release(failed, minimal_policy)

    assert evaluation.decision is Decision.REJECT
    assert any("Threshold failed" in reason for reason in evaluation.reasons)


def test_candidate_cannot_relax_policy_threshold(minimal_policy, passing_metrics) -> None:
    tampered = tuple(
        replace(
            metric,
            threshold=Threshold(ThresholdOperator.GTE, 0.1, "ratio"),
        )
        if metric.domain is Domain.RETRIEVAL_QUALITY
        else metric
        for metric in passing_metrics
    )

    with pytest.raises(EvidenceValidationError, match="policy threshold"):
        evaluate_release(tampered, minimal_policy)


def test_crossing_confidence_interval_holds(minimal_policy, passing_metrics) -> None:
    retrieval_requirement = next(
        requirement
        for requirement in minimal_policy.requirements
        if requirement.domain is Domain.RETRIEVAL_QUALITY
    )
    requirements = tuple(
        replace(
            requirement,
            uncertainty_required=True,
            minimum_sample_size=100,
        )
        if requirement is retrieval_requirement
        else requirement
        for requirement in minimal_policy.requirements
    )
    policy = GatePolicy(
        policy_id=minimal_policy.policy_id,
        version=minimal_policy.version,
        requirements=requirements,
    )
    metrics: tuple[MetricRecord, ...] = tuple(
        replace(
            metric,
            uncertainty=Uncertainty("bootstrap", 0.45, 0.85, 0.95, 200),
        )
        if metric.domain is Domain.RETRIEVAL_QUALITY
        else metric
        for metric in passing_metrics
    )

    evaluation = evaluate_release(metrics, policy)

    assert evaluation.decision is Decision.HOLD
    assert any("Confidence interval" in reason for reason in evaluation.reasons)
