"""Fixture-based contract tests for load-test evidence normalization."""

import importlib.util
import io
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).parents[3]
LOAD_TESTS = PROJECT_ROOT / "load-tests"
FIXTURES = Path(__file__).parent / "fixtures"

SPEC = importlib.util.spec_from_file_location("riverside_result_parser", LOAD_TESTS / "result_parser.py")
assert SPEC is not None and SPEC.loader is not None
RESULT_PARSER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESULT_PARSER
SPEC.loader.exec_module(RESULT_PARSER)


def test_normalize_run_passes_complete_healthy_evidence() -> None:
    result_streams = [
        (FIXTURES / "azure-results-engine-1.csv").open(encoding="utf-8", newline=""),
        (FIXTURES / "azure-results-engine-2.csv").open(encoding="utf-8", newline=""),
    ]
    try:
        with (FIXTURES / "engine-health.json").open(encoding="utf-8") as health, (
            LOAD_TESTS / "success-criteria.json"
        ).open(encoding="utf-8") as criteria:
            normalized = RESULT_PARSER.normalize_run(result_streams, health, criteria)
    finally:
        for stream in result_streams:
            stream.close()

    assert normalized["valid"] is True
    assert normalized["passed"] is True
    assert normalized["stages"]["target"]["ttft"]["p95_ms"] == 125.0
    assert normalized["stages"]["recovery"]["total"]["p95_ms"] == 650.0


def test_unhealthy_engine_fails_evidence() -> None:
    result_stream = (FIXTURES / "azure-results-engine-1.csv").open(encoding="utf-8", newline="")
    try:
        with (FIXTURES / "engine-health-unhealthy.json").open(encoding="utf-8") as health, (
            LOAD_TESTS / "success-criteria.json"
        ).open(encoding="utf-8") as criteria:
            normalized = RESULT_PARSER.normalize_run([result_stream], health, criteria)
    finally:
        result_stream.close()

    assert normalized["valid"] is True
    assert normalized["passed"] is False
    assert normalized["engine_health"]["healthy"] is False


def test_parser_rejects_missing_documented_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        RESULT_PARSER.parse_azure_result_csv(io.StringIO("timeStamp,elapsed,label\n1,2,total\n"))


def test_engine_health_requires_every_configured_engine() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        RESULT_PARSER.parse_engine_health(
            io.StringIO(
                '{"engines":[{"engine_id":"engine-1","average_cpu_percentage":40,'
                '"average_memory_percentage":50,"average_network_bytes_per_second":1000,'
                '"maximum_virtual_users":50}]}'
            ),
            maximum_cpu=75,
            maximum_memory=75,
            expected_engine_count=2,
        )
