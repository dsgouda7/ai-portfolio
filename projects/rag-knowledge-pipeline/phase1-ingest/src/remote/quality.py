"""Durable ingestion quality metrics and release-blocking data gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import IngestionSettings


@dataclass(frozen=True)
class QualityCounts:
    source_count: int
    parsed_count: int
    quarantine_count: int
    duplicate_input_count: int
    duplicate_content_count: int
    duplicate_existing_count: int
    schema_drift_count: int
    required_field_count: int
    expected_required_field_count: int
    acl_covered_count: int
    classification_covered_count: int
    region_covered_count: int
    lineage_complete_count: int
    deletion_lineage_complete_count: int
    freshest_source_at: datetime | None = None


class DataQualityError(RuntimeError):
    pass


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def build_quality_report(
    settings: IngestionSettings,
    counts: QualityCounts,
    started_at: datetime,
    completed_at: datetime | None = None,
) -> dict[str, Any]:
    finished = completed_at or datetime.now(timezone.utc)
    parse_success_rate = _ratio(counts.parsed_count, counts.source_count)
    quarantine_rate = _ratio(counts.quarantine_count, counts.source_count)
    completeness = _ratio(counts.required_field_count, counts.expected_required_field_count)
    freshness_lag_seconds = (
        max(0.0, (finished - counts.freshest_source_at).total_seconds())
        if counts.freshest_source_at is not None
        else None
    )
    gates = [
        {
            "name": "source_count",
            "operator": ">",
            "threshold": 0,
            "observed": counts.source_count,
            "passed": counts.source_count > 0,
        },
        {
            "name": "parse_success_rate",
            "operator": ">=",
            "threshold": settings.min_parse_success_rate,
            "observed": parse_success_rate,
            "passed": parse_success_rate >= settings.min_parse_success_rate,
        },
        {
            "name": "quarantine_rate",
            "operator": "<=",
            "threshold": settings.max_quarantine_rate,
            "observed": quarantine_rate,
            "passed": quarantine_rate <= settings.max_quarantine_rate,
        },
        {
            "name": "required_field_completeness",
            "operator": "==",
            "threshold": 1.0,
            "observed": completeness,
            "passed": completeness == 1.0,
        },
        {
            "name": "schema_drift_count",
            "operator": "==",
            "threshold": 0,
            "observed": counts.schema_drift_count,
            "passed": counts.schema_drift_count == 0,
        },
    ]
    return {
        "run_id": settings.run_id or f"manual-{started_at.strftime('%Y%m%dT%H%M%SZ')}",
        "pipeline_version": settings.pipeline_version,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": finished.isoformat().replace("+00:00", "Z"),
        "status": "passed" if all(gate["passed"] for gate in gates) else "failed",
        "source_count": counts.source_count,
        "parsed_count": counts.parsed_count,
        "quarantine_count": counts.quarantine_count,
        "duplicate_input_count": counts.duplicate_input_count,
        "duplicate_content_count": counts.duplicate_content_count,
        "duplicate_existing_count": counts.duplicate_existing_count,
        "schema_drift_count": counts.schema_drift_count,
        "parse_success_rate": parse_success_rate,
        "quarantine_rate": quarantine_rate,
        "required_field_completeness": completeness,
        "acl_coverage": _ratio(counts.acl_covered_count, counts.source_count),
        "classification_coverage": _ratio(counts.classification_covered_count, counts.source_count),
        "region_coverage": _ratio(counts.region_covered_count, counts.source_count),
        "lineage_completeness": _ratio(counts.lineage_complete_count, counts.source_count),
        "deletion_lineage_coverage": _ratio(
            counts.deletion_lineage_complete_count, counts.source_count
        ),
        "freshest_source_at": (
            counts.freshest_source_at.isoformat().replace("+00:00", "Z")
            if counts.freshest_source_at is not None
            else None
        ),
        "freshness_lag_seconds": freshness_lag_seconds,
        "gates": gates,
    }


def enforce_quality_report(report: dict[str, Any]) -> None:
    failed = [gate["name"] for gate in report["gates"] if not gate["passed"]]
    if failed:
        raise DataQualityError(f"ingestion quality gates failed: {', '.join(failed)}")
