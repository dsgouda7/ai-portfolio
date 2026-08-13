import json

import numpy as np

from roadid.telemetry import MAX_STRING_LENGTH, MAX_SUMMARY_BYTES, EventRecorder, sanitize_summary


def test_sensitive_keys_values_and_binary_payloads_are_redacted() -> None:
    summary = sanitize_summary(
        {
            "api_token": "top-secret-token",
            "plate_text": "ABC123",
            "safe_url_note": "fetched https://camera.example/live?id=42",
            "authorization": "Bearer abc.def",
            "pixels": np.zeros((10, 10, 3), dtype=np.uint8),
            "blob": b"image bytes",
            "message": "plate text: XYZ789",
        }
    )
    encoded = json.dumps(summary).lower()

    assert "top-secret-token" not in encoded
    assert "abc123" not in encoded
    assert "abc.def" not in encoded
    assert "camera.example" not in encoded
    assert "xyz789" not in encoded
    assert "image bytes" not in encoded
    assert "api_token" not in encoded
    assert "plate_text" not in encoded
    assert summary["_redacted_fields"] == 4


def test_summary_and_strings_are_bounded() -> None:
    recorder = EventRecorder("run-limits")
    recorder.start(
        "track_fusion", input_summary={f"field-{index}": "x" * 5000 for index in range(80)}
    )
    completed = recorder.complete(
        "track_fusion",
        output_summary={"detail": "y" * 5000},
        duration_ms=2.0,
    )

    assert len(json.dumps(completed.input_summary).encode()) <= MAX_SUMMARY_BYTES
    assert len(completed.output_summary["detail"]) <= MAX_STRING_LENGTH
