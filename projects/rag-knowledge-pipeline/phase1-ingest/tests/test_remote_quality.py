"""Quality report tests for durable pass/fail evidence."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from remote.config import IngestionSettings  # noqa: E402
from remote.quality import (  # noqa: E402
    DataQualityError,
    QualityCounts,
    build_quality_report,
    enforce_quality_report,
)


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


def test_quality_report_passes_complete_batch() -> None:
    report = build_quality_report(
        _settings(),
        QualityCounts(
            source_count=100,
            parsed_count=99,
            quarantine_count=1,
            duplicate_input_count=3,
            duplicate_content_count=4,
            duplicate_existing_count=2,
            schema_drift_count=0,
            required_field_count=3579,
            expected_required_field_count=3579,
            acl_covered_count=100,
            classification_covered_count=100,
            region_covered_count=100,
            lineage_complete_count=100,
            deletion_lineage_complete_count=100,
            freshest_source_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        ),
        datetime(2026, 8, 5, tzinfo=timezone.utc),
    )
    assert report["status"] == "passed"
    enforce_quality_report(report)


def test_quality_report_fails_after_durable_result_is_built() -> None:
    report = build_quality_report(
        _settings(),
        QualityCounts(
            source_count=10,
            parsed_count=8,
            quarantine_count=2,
            duplicate_input_count=0,
            duplicate_content_count=0,
            duplicate_existing_count=0,
            schema_drift_count=0,
            required_field_count=322,
            expected_required_field_count=322,
            acl_covered_count=10,
            classification_covered_count=10,
            region_covered_count=10,
            lineage_complete_count=10,
            deletion_lineage_complete_count=10,
        ),
        datetime(2026, 8, 5, tzinfo=timezone.utc),
    )
    assert report["status"] == "failed"
    with pytest.raises(DataQualityError, match="parse_success_rate, quarantine_rate"):
        enforce_quality_report(report)
