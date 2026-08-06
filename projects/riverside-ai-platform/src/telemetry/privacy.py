"""Privacy-safe structured logging with a closed field allowlist."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Final, Mapping

SAFE_LOG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "service.name",
        "deployment.environment",
        "model.release_id",
        "model.alias",
        "deployment.name",
        "cloud.region",
        "http.route",
        "outcome.status",
        "cache.result",
        "gen_ai.usage.prompt_tokens_bucket",
        "gen_ai.usage.output_tokens_bucket",
        "tenant.tier",
        "error.code",
        "retrieval.top_k_bucket",
        "stage",
        "duration_ms",
        "ttft_ms",
        "tpot_ms",
        "retry_count",
        "rejection_reason",
        "trace_id",
    }
)

_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_TRACE_ID = re.compile(r"^[a-f0-9]{32}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DEPLOYMENT = re.compile(r"^riverside-(dev|staging|production)-(blue|green)$")
_REGION = re.compile(r"^[a-z0-9-]{2,64}$")
_ENUM_FIELDS = {
    "service.name": frozenset(
        {"riverside-gateway", "riverside-rag-orchestrator", "riverside-model-endpoint", "riverside-indexer"}
    ),
    "deployment.environment": frozenset({"dev", "staging", "production"}),
    "model.alias": frozenset({"riverside-editor"}),
    "http.route": frozenset({"/v1/chat/completions", "/health", "/ready"}),
    "outcome.status": frozenset({"success", "error", "rejected", "timeout"}),
    "cache.result": frozenset({"hit", "miss", "bypass", "error"}),
    "gen_ai.usage.prompt_tokens_bucket": frozenset({"0", "1-128", "129-512", "513-2048", "2049-8192"}),
    "gen_ai.usage.output_tokens_bucket": frozenset({"0", "1-128", "129-512", "513-2048", "2049-8192"}),
    "tenant.tier": frozenset({"standard", "premium", "internal"}),
    "error.code": frozenset(
        {
            "none",
            "invalid_request",
            "unauthorized",
            "forbidden",
            "policy_violation",
            "overloaded",
            "timeout",
            "backend_failure",
            "release_unavailable",
            "internal_error",
        }
    ),
    "retrieval.top_k_bucket": frozenset({"none", "1-5", "6-10", "11-20"}),
    "stage": frozenset({"queue", "retrieval", "generation"}),
    "rejection_reason": frozenset(
        {"none", "capacity", "rate_limit", "policy", "authorization", "invalid_request", "other"}
    ),
}


def sanitize_log_fields(fields: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only allowlisted scalar fields and reject malformed correlation data."""

    safe: dict[str, str | int | float | bool] = {}
    for key, value in fields.items():
        if key not in SAFE_LOG_FIELDS or not isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if key in {"duration_ms", "ttft_ms", "tpot_ms"} and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            continue
        if key == "retry_count" and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3
        ):
            continue
        if key == "trace_id" and (not isinstance(value, str) or not _TRACE_ID.fullmatch(value)):
            continue
        if key == "model.release_id" and (not isinstance(value, str) or not _IDENTIFIER.fullmatch(value)):
            continue
        if key == "deployment.name" and (not isinstance(value, str) or not _DEPLOYMENT.fullmatch(value)):
            continue
        if key == "cloud.region" and (not isinstance(value, str) or not _REGION.fullmatch(value)):
            continue
        allowed = _ENUM_FIELDS.get(key)
        if allowed is not None and value not in allowed:
            continue
        safe[key] = value
    return safe


class PrivacySafeLogger:
    """Emit JSON logs without accepting free-form messages or sensitive fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, level: int, event: str, **fields: Any) -> None:
        if not _EVENT_NAME.fullmatch(event):
            raise ValueError("event must be a controlled lowercase identifier")
        payload: dict[str, str | int | float | bool] = {"event": event}
        payload.update(sanitize_log_fields(fields))
        self._logger.log(level, "%s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
