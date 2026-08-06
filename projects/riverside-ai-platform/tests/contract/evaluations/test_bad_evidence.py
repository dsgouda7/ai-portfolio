from __future__ import annotations

import json
from pathlib import Path

import pytest

from release_gates import EvidenceRef, EvidenceValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "evaluations"
    / "fixtures"
    / "invalid"
    / "bad-evidence.json"
)


def test_evidence_rejects_credentials_and_mutable_query() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    with pytest.raises(EvidenceValidationError, match="credentials, query parameters"):
        EvidenceRef(
            name=fixture["name"],
            uri=fixture["uri"],
            digest=fixture["digest"],
        )


def test_evidence_rejects_bad_digest_independently() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    with pytest.raises(EvidenceValidationError, match="lowercase SHA-256"):
        EvidenceRef(
            name=fixture["name"],
            uri="repo://evaluations/evidence/offline-results.json",
            digest=fixture["digest"],
        )
