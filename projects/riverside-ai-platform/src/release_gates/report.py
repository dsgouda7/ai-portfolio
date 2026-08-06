from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .gates import GateEvaluation
from .models import CONTRACT_VERSION, Decision, Domain, EvidenceValidationError, ReleaseContext


def build_release_report(
    context: ReleaseContext,
    evaluation: GateEvaluation,
) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = {domain.value: [] for domain in Domain}
    for metric in evaluation.metrics:
        domains[metric.domain.value].append(metric.to_contract())

    empty_domains = [name for name, metrics in domains.items() if not metrics]
    if empty_domains:
        raise EvidenceValidationError(
            "release report requires metrics for every domain: " + ", ".join(empty_domains)
        )
    oversized_domains = [name for name, metrics in domains.items() if len(metrics) > 128]
    if oversized_domains:
        raise EvidenceValidationError(
            "release report exceeds 128 metrics for domain: "
            + ", ".join(oversized_domains)
        )

    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "kind": "evaluation_release_report",
        "report_id": context.report_id,
        "release_id": context.release_id,
        "release_version": context.release_version,
        "generated_at": context.generated_at_contract(),
        "source_commit": context.source_commit,
        "decision": evaluation.decision.value,
        "decision_reasons": list(evaluation.reasons),
        "evidence": [item.to_contract() for item in context.evidence],
        "domains": domains,
    }
    if context.baseline_release_id is not None:
        report["baseline_release_id"] = context.baseline_release_id

    validate_decision_consistency(report)
    return report


def validate_decision_consistency(report: Mapping[str, Any]) -> None:
    try:
        decision = Decision(str(report["decision"]))
        raw_domains = report["domains"]
    except (KeyError, ValueError) as error:
        raise EvidenceValidationError("report decision or domains are invalid") from error

    if not isinstance(raw_domains, Mapping):
        raise EvidenceValidationError("report domains must be an object")

    expected_domains = {domain.value for domain in Domain}
    actual_domains = set(raw_domains)
    if actual_domains != expected_domains:
        raise EvidenceValidationError("report must contain exactly the eight v1 domains")

    statuses: list[str] = []
    for domain_name in sorted(expected_domains):
        metrics = raw_domains[domain_name]
        if not isinstance(metrics, list) or not metrics:
            raise EvidenceValidationError(f"domain {domain_name} must contain metrics")
        for metric in metrics:
            if not isinstance(metric, Mapping) or metric.get("status") not in {"pass", "fail"}:
                raise EvidenceValidationError(f"domain {domain_name} contains an invalid status")
            statuses.append(str(metric["status"]))

    has_failure = "fail" in statuses
    if decision is Decision.PROMOTE and has_failure:
        raise EvidenceValidationError("promote is inconsistent with failed metrics")
    if decision is Decision.REJECT and not has_failure:
        raise EvidenceValidationError("reject requires at least one failed metric")

    reasons = report.get("decision_reasons")
    if not isinstance(reasons, list) or not reasons or not all(
        isinstance(reason, str) and reason for reason in reasons
    ):
        raise EvidenceValidationError("decision_reasons must contain non-empty strings")


def write_release_report(path: Path, report: Mapping[str, Any]) -> None:
    validate_decision_consistency(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
