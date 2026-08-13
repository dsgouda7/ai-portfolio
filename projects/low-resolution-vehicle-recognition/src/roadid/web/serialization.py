"""Bounded, privacy-conscious JSON serialization for the web boundary."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

MAX_STRING_LENGTH = 512
MAX_COLLECTION_ITEMS = 100
MAX_DEPTH = 6
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|auth|cookie|credential|password|plate|raw[_-]?url|secret|"
    r"source[_-]?url|token)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s\]\[(){}<>\"']+", re.IGNORECASE)
_CREDENTIAL = re.compile(
    r"(?:bearer\s+[a-z0-9._~+/=-]+|"
    r"(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def safe_json(value: Any, *, depth: int = 0) -> Any:
    """Convert known values to bounded JSON and redact sensitive fields."""
    if depth >= MAX_DEPTH:
        return "[truncated]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, bytes):
        return "[binary omitted]"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Enum):
        return safe_json(value.value, depth=depth + 1)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return safe_json(value.to_dict(), depth=depth + 1)
    if is_dataclass(value):
        return safe_json(asdict(value), depth=depth + 1)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_COLLECTION_ITEMS:
                break
            text_key = str(key)[:64]
            if _SENSITIVE_KEY.search(text_key):
                continue
            if text_key.lower() == "terms_url" and isinstance(item, str):
                result[text_key] = item[:MAX_STRING_LENGTH]
            else:
                result[text_key] = safe_json(item, depth=depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [safe_json(item, depth=depth + 1) for item in value[:MAX_COLLECTION_ITEMS]]
    return _safe_string(str(value))


def _safe_string(value: str) -> str:
    redacted = _CREDENTIAL.sub("[redacted]", value)
    redacted = _URL.sub("[redacted URL]", redacted)
    return redacted[:MAX_STRING_LENGTH]
