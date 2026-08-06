"""Unit tests for privacy-safe structured logging."""

import json
import logging

from telemetry.privacy import PrivacySafeLogger, sanitize_log_fields


class CapturingLogger(logging.Logger):
    def __init__(self) -> None:
        super().__init__("capturing")
        self.captured: tuple[int, str, tuple[object, ...]] | None = None

    def log(self, level: int, msg: object, *args: object, **kwargs: object) -> None:
        del kwargs
        self.captured = (level, str(msg), args)


def test_sanitizer_drops_content_and_identifiers() -> None:
    safe = sanitize_log_fields(
        {
            "outcome.status": "success",
            "cache.result": "secret-customer-value",
            "trace_id": "a" * 32,
            "prompt": "customer manuscript",
            "completion": "generated prose",
            "document_text": "private text",
            "request_id": "request-123",
            "tenant_id": "tenant-123",
            "user_id": "user-123",
            "authorization": "Bearer secret",
        }
    )

    assert safe == {"outcome.status": "success", "trace_id": "a" * 32}


def test_logger_emits_only_controlled_json_fields() -> None:
    logger = CapturingLogger()
    safe_logger = PrivacySafeLogger(logger)

    safe_logger.emit(
        logging.INFO,
        "request.completed",
        stage="generation",
        duration_ms=123.4,
        request_id="request-123",
        prompt="do not log me",
    )

    assert logger.captured is not None
    _, message_format, args = logger.captured
    assert message_format == "%s"
    payload = json.loads(str(args[0]))
    assert payload == {"duration_ms": 123.4, "event": "request.completed", "stage": "generation"}
