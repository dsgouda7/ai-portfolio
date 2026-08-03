"""Small measurement helpers shared across OrderFlow notebooks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    trace_id: str
    step: int
    actor: str
    action: str
    status: str
    latency_ms: int = 0
    token_count: int = 0
    cost_units: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecorder:
    trace_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def record(
        self,
        actor: str,
        action: str,
        status: str,
        *,
        latency_ms: int = 0,
        token_count: int = 0,
        cost_units: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id,
            step=len(self.events) + 1,
            actor=actor,
            action=action,
            status=status,
            latency_ms=latency_ms,
            token_count=token_count,
            cost_units=cost_units,
            details=details or {},
        )
        self.events.append(event)
        return event

    def totals(self) -> dict[str, float]:
        return {
            "steps": len(self.events),
            "latency_ms": sum(event.latency_ms for event in self.events),
            "tokens": sum(event.token_count for event in self.events),
            "cost_units": round(sum(event.cost_units for event in self.events), 4),
        }

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]


def estimate_tokens(text: str) -> int:
    """Use a deterministic local estimate suitable for relative context comparisons."""
    return max(1, math.ceil(len(text) / 4))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_rate(actual: list[Any], expected: list[Any]) -> float:
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have equal lengths")
    if not expected:
        raise ValueError("expected cannot be empty")
    return sum(left == right for left, right in zip(actual, expected)) / len(expected)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[max(0, index)])
