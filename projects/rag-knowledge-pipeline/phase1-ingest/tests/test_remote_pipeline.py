"""Partition-transform tests with no Spark session or external service."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from remote.config import IngestionSettings  # noqa: E402
from remote.pipeline import _transform_partition  # noqa: E402


class FixtureRow:
    def __init__(self, values: dict) -> None:
        self.values = values

    def asDict(self, recursive: bool = True) -> dict:
        return self.values


def _settings() -> IngestionSettings:
    return IngestionSettings(
        source_uri="abfss://raw@riverside.dfs.core.windows.net/manuscripts",
        catalog="main",
        schema="rag_demo",
        tenant_id="tenant-editorial-standard",
        region="eastus2",
        classification="confidential",
        run_id="fixture-run-001",
    )


def test_parse_failure_preserves_raw_record_and_emits_bounded_quarantine() -> None:
    content = (PHASE_ROOT / "tests" / "fixtures" / "source" / "malformed-json.txt").read_bytes()
    row = FixtureRow(
        {
            "source_uri": "abfss://raw@riverside.dfs.core.windows.net/malformed.json",
            "storage_uri": "abfss://raw@riverside.dfs.core.windows.net/malformed.json",
            "source_name": "malformed.json",
            "media_type": "application/json",
            "byte_size": len(content),
            "modified_at": datetime(2026, 8, 5, tzinfo=timezone.utc),
            "content": content,
        }
    )
    result = list(
        _transform_partition([row], _settings(), datetime(2026, 8, 5, tzinfo=timezone.utc))
    )[0]
    raw = json.loads(result[1])
    quarantine = json.loads(result[3])
    assert result[0] == "quarantined"
    assert raw["content_hash"] == quarantine["raw_content_hash"]
    assert quarantine["error_code"] == "invalid_json"
    assert "Incomplete document" not in quarantine["error_message"]


def test_supported_document_emits_raw_and_parsed_contract_records() -> None:
    content = (PHASE_ROOT / "tests" / "fixtures" / "source" / "chapter-001.md").read_bytes()
    row = FixtureRow(
        {
            "source_uri": "abfss://raw@riverside.dfs.core.windows.net/chapter-001.md",
            "storage_uri": "abfss://raw@riverside.dfs.core.windows.net/chapter-001.md",
            "source_name": "chapter-001.md",
            "media_type": "text/markdown",
            "byte_size": len(content),
            "modified_at": datetime(2026, 8, 5, tzinfo=timezone.utc),
            "content": content,
        }
    )
    result = list(
        _transform_partition([row], _settings(), datetime(2026, 8, 5, tzinfo=timezone.utc))
    )[0]
    assert result[0] == "parsed"
    assert json.loads(result[1])["document_id"] == json.loads(result[2])["document_id"]
    assert result[3] is None
