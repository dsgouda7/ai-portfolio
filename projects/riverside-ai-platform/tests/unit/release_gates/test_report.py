from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from release_gates import (
    Decision,
    EvidenceRef,
    EvidenceValidationError,
    GateEvaluation,
    Observation,
    ReleaseContext,
    build_release_report,
    evaluate_release,
    validate_decision_consistency,
)


def test_release_report_matches_v1_domain_shape(minimal_policy, passing_metrics) -> None:
    evaluation = evaluate_release(passing_metrics, minimal_policy)
    context = ReleaseContext(
        report_id="report-fixture-001",
        release_id="candidate-release",
        release_version="1.0.0",
        baseline_release_id="baseline-release",
        generated_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        source_commit="0123456789abcdef0123456789abcdef01234567",
        evidence=(
            EvidenceRef(
                name="offline-results",
                uri="repo://evaluations/evidence/offline-results.json",
                digest="e" * 64,
            ),
        ),
    )

    report = build_release_report(context, evaluation)

    assert report["contract_version"] == "1.0.0"
    assert report["decision"] == "promote"
    assert len(report["domains"]) == 8
    assert all(report["domains"].values())


def test_gate_evaluation_rejects_promote_with_failed_metric(passing_metrics) -> None:
    failed_metric = replace(
        passing_metrics[0],
        observed=Observation(0.25, "ratio"),
        status=None,
    )
    contradictory_metrics = (failed_metric, *passing_metrics[1:])

    with pytest.raises(EvidenceValidationError, match="promote cannot contain failed"):
        GateEvaluation(
            policy_id="fixture-policy",
            policy_version="1.0.0",
            decision=Decision.PROMOTE,
            reasons=("contradictory fixture",),
            metrics=contradictory_metrics,
        )


def test_report_consistency_rejects_promote_with_failure() -> None:
    report = {
        "decision": "promote",
        "decision_reasons": ["contradictory fixture"],
        "domains": {
            domain: [{"status": "fail" if index == 0 else "pass"}]
            for index, domain in enumerate(
                [
                    "data_quality",
                    "retrieval_quality",
                    "generation_citation_quality",
                    "adaptation_evidence",
                    "safety_authorization",
                    "operational_slos",
                    "cost",
                    "rollout_comparison",
                ]
            )
        },
    }

    with pytest.raises(EvidenceValidationError, match="promote is inconsistent"):
        validate_decision_consistency(report)
