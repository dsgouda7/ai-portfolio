"""Contract tests for deterministic v1 raw and parsed records."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from remote.contracts import (  # noqa: E402
    GovernanceContext,
    ParsedContent,
    SourceObject,
    build_parsed_record,
    build_raw_record,
)


FIXTURES = PHASE_ROOT / "tests" / "fixtures"
CONTRACTS = PROJECTS_ROOT / "riverside-ai-platform" / "contracts" / "v1"
NOW = datetime(2026, 8, 5, 10, 5, tzinfo=timezone.utc)


def _validator(schema_name: str) -> Draft202012Validator:
    resources = []
    root = None
    for path in CONTRACTS.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
        if path.name == schema_name:
            root = schema
    assert root is not None
    return Draft202012Validator(
        root,
        registry=Registry().with_resources(resources),
        format_checker=FormatChecker(),
    )


def _records() -> tuple[dict, dict]:
    content = (FIXTURES / "source" / "chapter-001.md").read_bytes()
    governance_values = json.loads((FIXTURES / "governance.json").read_text(encoding="utf-8"))
    governance_values["acl_principals"] = tuple(governance_values["acl_principals"])
    governance_values["acl_groups"] = tuple(governance_values["acl_groups"])
    source = SourceObject(
        source_uri="abfss://raw@riverside.dfs.core.windows.net/manuscripts/chapter-001.md",
        storage_uri="abfss://raw@riverside.dfs.core.windows.net/manuscripts/chapter-001.md",
        source_name="chapter-001.md",
        media_type="text/markdown",
        content=content,
        metadata={"collection": "the_weight_of_distant_light", "chapter_number": 1},
    )
    raw = build_raw_record(source, GovernanceContext(**governance_values), NOW)
    parsed = build_parsed_record(
        raw,
        ParsedContent(
            title="Harbor Lights",
            text=content.decode("utf-8"),
            language="en",
            parser_name="markdown-parser",
            parser_version="1.0.0",
        ),
        NOW,
    )
    return raw, parsed


def test_records_validate_against_frozen_v1_schemas() -> None:
    raw, parsed = _records()
    _validator("raw-document.schema.json").validate(raw)
    _validator("parsed-document.schema.json").validate(parsed)


def test_records_have_only_contract_fields() -> None:
    raw, parsed = _records()
    required = json.loads((FIXTURES / "v1-required-fields.json").read_text(encoding="utf-8"))
    assert set(raw) == set(required["raw_document"]) | {"metadata"}
    assert set(parsed) == set(required["parsed_document"]) | {"metadata"}


def test_document_identity_and_source_version_are_deterministic() -> None:
    first_raw, first_parsed = _records()
    second_raw, second_parsed = _records()
    assert first_raw["document_id"] == second_raw["document_id"]
    assert first_raw["source_version"] == second_raw["source_version"]
    assert first_raw["content_hash"] == first_parsed["raw_content_hash"]
    assert first_parsed["content_hash"] == second_parsed["content_hash"]


def test_content_change_creates_new_version_without_changing_document_identity() -> None:
    first_raw, _ = _records()
    changed_source = SourceObject(
        source_uri=first_raw["source_uri"],
        storage_uri=first_raw["storage_uri"],
        source_name=first_raw["source_name"],
        media_type=first_raw["media_type"],
        content=b"A revised source document.",
    )
    changed_raw = build_raw_record(
        changed_source,
        GovernanceContext(
            tenant_id="tenant-editorial-standard",
            region="eastus2",
            classification="confidential",
            acl_visibility="restricted",
            acl_groups=("editors",),
        ),
        NOW,
    )
    assert changed_raw["document_id"] == first_raw["document_id"]
    assert changed_raw["source_version"] != first_raw["source_version"]
