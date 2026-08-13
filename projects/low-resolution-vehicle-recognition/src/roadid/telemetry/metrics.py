"""Whole-run stage latency aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Final

from roadid.contracts import PipelineEvent, StageStatus
from roadid.telemetry.events import PIPELINE_STAGES

_TERMINAL_STATUSES: Final = frozenset(
    {StageStatus.COMPLETED, StageStatus.SKIPPED, StageStatus.WARNING, StageStatus.FAILED}
)


@dataclass(frozen=True, slots=True)
class StageLatencyStats:
    count: int
    total_ms: float
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(slots=True)
class _MutableLatency:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None


class StageLatencyAggregator:
    def __init__(self) -> None:
        self._values = {stage: _MutableLatency() for stage in PIPELINE_STAGES}
        self._lock = RLock()

    def observe(self, event: PipelineEvent) -> None:
        if event.status not in _TERMINAL_STATUSES or event.duration_ms is None:
            return
        with self._lock:
            value = self._values[event.stage]
            value.count += 1
            value.total_ms += event.duration_ms
            value.min_ms = (
                event.duration_ms if value.min_ms is None else min(value.min_ms, event.duration_ms)
            )
            value.max_ms = (
                event.duration_ms if value.max_ms is None else max(value.max_ms, event.duration_ms)
            )

    def snapshot(self) -> dict[str, StageLatencyStats]:
        with self._lock:
            return {
                stage: StageLatencyStats(
                    count=value.count,
                    total_ms=value.total_ms,
                    mean_ms=value.total_ms / value.count if value.count else None,
                    min_ms=value.min_ms,
                    max_ms=value.max_ms,
                )
                for stage, value in self._values.items()
            }
