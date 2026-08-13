"""Pipeline stage definitions and privacy-safe event payload helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from roadid.contracts import JSONValue

PIPELINE_STAGES: Final[tuple[str, ...]] = (
    "source_acquisition",
    "frame_validation",
    "vehicle_detection",
    "track_association",
    "crop_quality",
    "frame_classification",
    "track_fusion",
    "calibration",
    "hierarchy_decision",
    "privacy_render",
)
STAGE_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {stage: position for position, stage in enumerate(PIPELINE_STAGES)}
)

MAX_IDENTIFIER_LENGTH: Final = 128
MAX_STRING_LENGTH: Final = 512
MAX_SUMMARY_ITEMS: Final = 32
MAX_SUMMARY_DEPTH: Final = 4
MAX_SUMMARY_BYTES: Final = 4096

_SENSITIVE_KEY = re.compile(
    r"(?:auth|bearer|cookie|credential|image|password|passwd|plate|secret|token|"
    r"api[_-]?key|crop[_-]?bgr|frame[_-]?bytes|pixel)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s\]\[(){}<>\"']+", re.IGNORECASE)
_CREDENTIAL_VALUE = re.compile(
    r"(?:bearer\s+[a-z0-9._~+/=-]+|"
    r"(?:token|secret|password|api[_-]?key|plate(?:\s+(?:text|number))?)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def validate_stage(stage: str) -> str:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown pipeline stage: {stage}")
    return stage


def bounded_identifier(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} exceeds {MAX_IDENTIFIER_LENGTH} characters")
    return value


def bounded_text(value: str | None, *, limit: int = MAX_STRING_LENGTH) -> str | None:
    if value is None:
        return None
    safe = _redact_string(value)
    if len(safe) <= limit:
        return safe
    return f"{safe[: limit - 14]}...[truncated]"


def sanitize_summary(summary: Mapping[str, Any] | None) -> dict[str, JSONValue]:
    """Remove unsafe values and enforce deterministic payload limits."""
    if not summary:
        return {}
    sanitized = _sanitize_mapping(summary, depth=0)
    if _json_size(sanitized) <= MAX_SUMMARY_BYTES:
        return sanitized

    bounded: dict[str, JSONValue] = {"_truncated": True}
    for key, value in sanitized.items():
        candidate = {**bounded, key: value}
        if _json_size(candidate) > MAX_SUMMARY_BYTES:
            continue
        bounded[key] = value
    return bounded


def remaining_stages(stage: str) -> tuple[str, ...]:
    position = STAGE_ORDER[validate_stage(stage)]
    return PIPELINE_STAGES[position + 1 :]


def _sanitize_mapping(summary: Mapping[str, Any], *, depth: int) -> dict[str, JSONValue]:
    if depth >= MAX_SUMMARY_DEPTH:
        return {"_truncated": True}

    result: dict[str, JSONValue] = {}
    redacted_fields = 0
    for index, (raw_key, raw_value) in enumerate(summary.items()):
        if index >= MAX_SUMMARY_ITEMS:
            result["_truncated"] = True
            break
        key = bounded_text(str(raw_key), limit=MAX_IDENTIFIER_LENGTH) or "field"
        if _SENSITIVE_KEY.search(key):
            redacted_fields += 1
            continue
        result[key] = _sanitize_value(raw_value, depth=depth + 1)
    if redacted_fields:
        result["_redacted_fields"] = redacted_fields
    return result


def _sanitize_value(value: Any, *, depth: int) -> JSONValue:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return bounded_text(value) or ""
    if isinstance(value, bytes | bytearray | memoryview):
        return "[REDACTED_BINARY]"
    if value.__class__.__module__.startswith("numpy"):
        return "[REDACTED_ARRAY]"
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, depth=depth)
    if isinstance(value, Sequence):
        if _looks_like_image_array(value):
            return "[REDACTED_ARRAY]"
        if depth >= MAX_SUMMARY_DEPTH:
            return ["[TRUNCATED]"]
        items = [_sanitize_value(item, depth=depth + 1) for item in value[:MAX_SUMMARY_ITEMS]]
        if len(value) > MAX_SUMMARY_ITEMS:
            items.append("[TRUNCATED]")
        return items
    return bounded_text(str(value)) or ""


def _redact_string(value: str) -> str:
    without_urls = _URL.sub("[REDACTED_URL]", value)
    return _CREDENTIAL_VALUE.sub("[REDACTED]", without_urls)


def _json_size(value: Mapping[str, JSONValue]) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def _looks_like_image_array(value: Sequence[Any]) -> bool:
    if len(value) < 8 or not value:
        return False
    first = value[0]
    return isinstance(first, Sequence) and not isinstance(first, str | bytes | bytearray)
