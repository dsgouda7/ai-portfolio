from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATIONS_ROOT = PROJECT_ROOT / "evaluations"
SCHEMA_PATH = EVALUATIONS_ROOT / "schemas" / "v1" / "evaluation-dataset.schema.json"
DATASET_ROOT = EVALUATIONS_ROOT / "datasets" / "v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    return Draft202012Validator(_load(SCHEMA_PATH), format_checker=FormatChecker())


@pytest.mark.parametrize("dataset_path", sorted(DATASET_ROOT.glob("*.json")))
def test_v1_dataset_examples_validate(dataset_path: Path) -> None:
    errors = sorted(_validator().iter_errors(_load(dataset_path)), key=lambda error: list(error.path))

    assert errors == [], "\n".join(error.message for error in errors)


def test_v1_examples_cover_exactly_all_release_domains() -> None:
    domains = {_load(path)["domain"] for path in DATASET_ROOT.glob("*.json")}

    assert domains == {
        "data_quality",
        "retrieval_quality",
        "generation_citation_quality",
        "adaptation_evidence",
        "safety_authorization",
        "operational_slos",
        "cost",
        "rollout_comparison",
    }


def test_domain_case_mismatch_is_invalid() -> None:
    fixture = deepcopy(_load(DATASET_ROOT / "data-quality.json"))
    fixture["domain"] = "retrieval_quality"

    assert list(_validator().iter_errors(fixture))


def test_unknown_case_field_is_invalid() -> None:
    fixture = deepcopy(_load(DATASET_ROOT / "cost.json"))
    fixture["cases"][0]["unbounded_metadata"] = "not allowed"

    assert list(_validator().iter_errors(fixture))
