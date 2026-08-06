from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from release_gates import EvidenceValidationError, validate_decision_consistency

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_REPORT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "valid"
    / "evaluation-release-report.json"
)


def _report() -> dict:
    return json.loads(VALID_REPORT_PATH.read_text(encoding="utf-8"))


def test_frozen_v1_report_example_is_decision_consistent() -> None:
    validate_decision_consistency(_report())


def test_promote_with_failed_metric_is_invalid() -> None:
    report = deepcopy(_report())
    report["domains"]["retrieval_quality"][0]["status"] = "fail"

    with pytest.raises(EvidenceValidationError, match="promote is inconsistent"):
        validate_decision_consistency(report)


def test_reject_without_failed_metric_is_invalid() -> None:
    report = deepcopy(_report())
    report["decision"] = "reject"
    report["decision_reasons"] = ["Contradictory static fixture."]

    with pytest.raises(EvidenceValidationError, match="reject requires"):
        validate_decision_consistency(report)


def test_missing_release_domain_is_invalid() -> None:
    report = deepcopy(_report())
    del report["domains"]["cost"]

    with pytest.raises(EvidenceValidationError, match="exactly the eight"):
        validate_decision_consistency(report)
