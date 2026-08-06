"""Normalize raw Azure Load Testing CSV files into release-gate evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO

STAGES = ("warm", "steady", "target", "overload", "recovery")
METRICS = ("total", "ttft", "tpot")
REQUIRED_RESULT_COLUMNS = frozenset({"timeStamp", "elapsed", "label", "success", "Latency", "responseCode"})


@dataclass(frozen=True, slots=True)
class Sample:
    timestamp_ms: int
    elapsed_ms: float
    label: str
    success: bool
    latency_ms: float
    response_code: str


@dataclass(frozen=True, slots=True)
class RequestSummary:
    sample_count: int
    error_count: int
    error_percentage: float
    average_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float


def parse_azure_result_csv(stream: TextIO) -> list[Sample]:
    """Parse one raw per-engine Azure result file in documented JMeter CSV format."""

    reader = csv.DictReader(stream)
    columns = frozenset(reader.fieldnames or ())
    missing = REQUIRED_RESULT_COLUMNS - columns
    if missing:
        raise ValueError(f"Azure result CSV is missing columns: {sorted(missing)}")
    samples: list[Sample] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            success_text = row["success"].strip().lower()
            if success_text not in {"true", "false"}:
                raise ValueError("success must be true or false")
            sample = Sample(
                timestamp_ms=int(row["timeStamp"]),
                elapsed_ms=float(row["elapsed"]),
                label=row["label"],
                success=success_text == "true",
                latency_ms=float(row["Latency"]),
                response_code=row["responseCode"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Azure result row {row_number}: {exc}") from exc
        if sample.elapsed_ms < 0 or sample.latency_ms < 0:
            raise ValueError(f"invalid Azure result row {row_number}: durations must be non-negative")
        samples.append(sample)
    if not samples:
        raise ValueError("Azure result CSV contains no samples")
    return samples


def summarize(samples: Iterable[Sample]) -> dict[str, RequestSummary]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    summaries: dict[str, RequestSummary] = {}
    for label, group in grouped.items():
        elapsed = sorted(sample.elapsed_ms for sample in group)
        errors = sum(not sample.success for sample in group)
        started_ms = min(sample.timestamp_ms for sample in group)
        ended_ms = max(sample.timestamp_ms + sample.elapsed_ms for sample in group)
        duration_seconds = max((ended_ms - started_ms) / 1000, 0.001)
        summaries[label] = RequestSummary(
            sample_count=len(group),
            error_count=errors,
            error_percentage=errors / len(group) * 100,
            average_ms=sum(elapsed) / len(elapsed),
            p50_ms=_percentile(elapsed, 0.50),
            p95_ms=_percentile(elapsed, 0.95),
            p99_ms=_percentile(elapsed, 0.99),
            requests_per_second=len(group) / duration_seconds,
        )
    return summaries


def parse_engine_health(
    stream: TextIO,
    *,
    maximum_cpu: float,
    maximum_memory: float,
    expected_engine_count: int | None = None,
) -> dict[str, Any]:
    """Validate normalized engine evidence exported by the pipeline integration."""

    data = json.load(stream)
    engines = data.get("engines") if isinstance(data, dict) else None
    if not isinstance(engines, list) or not engines:
        raise ValueError("engine health evidence must contain at least one engine")
    if expected_engine_count is not None and len(engines) != expected_engine_count:
        raise ValueError(
            f"engine health evidence contains {len(engines)} engines; expected {expected_engine_count}"
        )
    normalized = []
    for index, engine in enumerate(engines):
        try:
            engine_id = str(engine["engine_id"])
            cpu = float(engine["average_cpu_percentage"])
            memory = float(engine["average_memory_percentage"])
            network = float(engine["average_network_bytes_per_second"])
            users = int(engine["maximum_virtual_users"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid engine health item {index}: {exc}") from exc
        if (
            not engine_id
            or not all(math.isfinite(value) and value >= 0 for value in (cpu, memory, network))
            or cpu > 100
            or memory > 100
            or users < 0
        ):
            raise ValueError(f"invalid engine health item {index}: values must be finite and non-negative")
        normalized.append(
            {
                "engine_id": engine_id,
                "average_cpu_percentage": cpu,
                "average_memory_percentage": memory,
                "average_network_bytes_per_second": network,
                "maximum_virtual_users": users,
                "healthy": cpu < maximum_cpu and memory < maximum_memory,
            }
        )
    return {"healthy": all(engine["healthy"] for engine in normalized), "engines": normalized}


def normalize_run(
    result_streams: Iterable[TextIO],
    engine_health_stream: TextIO,
    criteria_stream: TextIO,
) -> dict[str, Any]:
    """Build deterministic pass/fail evidence from all engine result files."""

    criteria = json.load(criteria_stream)
    all_samples = [sample for stream in result_streams for sample in parse_azure_result_csv(stream)]
    summaries = summarize(all_samples)
    engine_limits = criteria["engine_health"]
    engine_health = parse_engine_health(
        engine_health_stream,
        maximum_cpu=float(engine_limits["maximum_average_cpu_percentage"]),
        maximum_memory=float(engine_limits["maximum_average_memory_percentage"]),
        expected_engine_count=(
            int(engine_limits["expected_engine_count"])
            if engine_limits.get("require_all_engines", True)
            else None
        ),
    )

    checks: list[dict[str, Any]] = []
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stage_summaries: dict[str, Any] = {}
        stage_criteria = criteria["stages"][stage]
        for metric in METRICS:
            label = f"chat.completions.{metric}.{stage}"
            summary = summaries.get(label)
            if summary is None:
                checks.append({"name": f"{stage}.{metric}.present", "passed": False, "observed": None})
                continue
            stage_summaries[metric] = asdict(summary)
            threshold_name = f"maximum_{metric}_p95_ms"
            checks.append(
                {
                    "name": f"{stage}.{metric}.p95_ms",
                    "passed": summary.p95_ms <= float(stage_criteria[threshold_name]),
                    "observed": summary.p95_ms,
                    "threshold": float(stage_criteria[threshold_name]),
                }
            )
        total = summaries.get(f"chat.completions.total.{stage}")
        if total is not None:
            checks.append(
                {
                    "name": f"{stage}.error_percentage",
                    "passed": total.error_percentage <= float(stage_criteria["maximum_error_percentage"]),
                    "observed": total.error_percentage,
                    "threshold": float(stage_criteria["maximum_error_percentage"]),
                }
            )
        stages[stage] = stage_summaries

    steady = summaries.get("chat.completions.total.steady")
    recovery = summaries.get("chat.completions.total.recovery")
    if steady is not None and recovery is not None:
        recovery_criteria = criteria["recovery"]
        p95_limit = steady.p95_ms * float(recovery_criteria["maximum_total_p95_multiplier_of_steady"])
        error_limit = steady.error_percentage + float(
            recovery_criteria["maximum_error_percentage_delta_from_steady"]
        )
        checks.extend(
            [
                {
                    "name": "recovery.total_p95_vs_steady",
                    "passed": recovery.p95_ms <= p95_limit,
                    "observed": recovery.p95_ms,
                    "threshold": p95_limit,
                },
                {
                    "name": "recovery.error_percentage_vs_steady",
                    "passed": recovery.error_percentage <= error_limit,
                    "observed": recovery.error_percentage,
                    "threshold": error_limit,
                },
            ]
        )

    checks.append({"name": "engine_health", "passed": engine_health["healthy"], "observed": engine_health["healthy"]})
    return {
        "schema_version": "1.0.0",
        "valid": all(not check["name"].endswith(".present") or check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "engine_health": engine_health,
        "stages": stages,
        "checks": checks,
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    rank = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[rank]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, type=Path)
    parser.add_argument("--engine-health", required=True, type=Path)
    parser.add_argument("--criteria", default=Path("success-criteria.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result_streams = [path.open(encoding="utf-8", newline="") for path in args.result]
    try:
        with args.engine_health.open(encoding="utf-8") as health_stream, args.criteria.open(
            encoding="utf-8"
        ) as criteria_stream:
            normalized = normalize_run(result_streams, health_stream, criteria_stream)
    finally:
        for stream in result_streams:
            stream.close()
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if normalized["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
