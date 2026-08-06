from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from endpoint_client import (
    AppError,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamEvent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_ROOT = PROJECT_ROOT / "contracts" / "v1"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "valid"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _registry() -> Registry[Any]:
    resources = []
    for path in SCHEMA_ROOT.glob("*.schema.json"):
        schema = _load_json(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@pytest.mark.parametrize(
    ("schema_name", "fixture_name", "model_type"),
    [
        ("app-chat-completion-request", "app-chat-completion-request", ChatCompletionRequest),
        ("app-chat-completion-response", "app-chat-completion-response", ChatCompletionResponse),
        ("app-chat-completion-stream-event", "app-chat-completion-stream-event", ChatCompletionStreamEvent),
        ("app-error", "app-error", AppError),
    ],
)
def test_typed_models_round_trip_through_frozen_v1_schema(
    schema_name: str,
    fixture_name: str,
    model_type: type[ChatCompletionRequest]
    | type[ChatCompletionResponse]
    | type[ChatCompletionStreamEvent]
    | type[AppError],
) -> None:
    fixture = _load_json(FIXTURE_ROOT / f"{fixture_name}.json")
    instance = model_type.model_validate(fixture).model_dump(mode="json", exclude_unset=True)
    schema = _load_json(SCHEMA_ROOT / f"{schema_name}.schema.json")
    validator = Draft202012Validator(
        schema,
        registry=_registry(),
        format_checker=FormatChecker(),
    )

    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))

    assert errors == []


def test_rag_orchestrator_has_no_ingestion_implementation_imports() -> None:
    source_root = PROJECT_ROOT / "src" / "rag_orchestrator"
    forbidden_prefixes = (
        "rag_knowledge_pipeline",
        "phase1_ingest",
        "phase2_vectorize",
        "databricks.ingestion",
    )

    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imported_modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module.startswith(forbidden_prefixes)
            for module in imported_modules
        ), f"{path.name} imports an ingestion implementation"
