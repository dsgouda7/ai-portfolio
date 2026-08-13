"""Privacy-redaction interface and fail-closed display policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from roadid.contracts import PrivacyRedactionResult


class PrivacyRedactionError(RuntimeError):
    """Raised when a frame cannot cross the configured privacy boundary."""


class PrivacyRedactor(Protocol):
    def redact(self, frame_id: int, image_bgr: np.ndarray) -> PrivacyRedactionResult: ...


class DeterministicNoPIIRedactor:
    """Replay-only redactor for fixtures that are guaranteed to contain no PII."""

    version = "deterministic-no-pii-replay-v1"

    def redact(self, frame_id: int, image_bgr: np.ndarray) -> PrivacyRedactionResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("privacy input must be a BGR image")
        return PrivacyRedactionResult(
            frame_id=frame_id,
            face_masks=(),
            plate_masks=(),
            redactor_version=self.version,
            safe_for_display=True,
        )


@dataclass(frozen=True, slots=True)
class RedactedFrame:
    image_bgr: np.ndarray
    result: PrivacyRedactionResult


class PrivacyGuard:
    def __init__(
        self,
        redactor: PrivacyRedactor | None,
        *,
        require_for_public_sources: bool = True,
        allow_raw_local_display: bool = False,
    ) -> None:
        self.redactor = redactor
        self.require_for_public_sources = require_for_public_sources
        self.allow_raw_local_display = allow_raw_local_display

    def protect(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: int,
        public_source: bool,
    ) -> RedactedFrame:
        if self.redactor is None:
            if public_source and self.require_for_public_sources:
                raise PrivacyRedactionError("public-source display requires a privacy redactor")
            if not public_source and self.allow_raw_local_display:
                result = PrivacyRedactionResult(
                    frame_id=frame_id,
                    face_masks=(),
                    plate_masks=(),
                    redactor_version="explicit-raw-local-display",
                    safe_for_display=True,
                )
                return RedactedFrame(image_bgr.copy(), result)
            raise PrivacyRedactionError("raw display is disabled without a privacy redactor")

        result = self.redactor.redact(frame_id, image_bgr)
        if result.frame_id != frame_id:
            raise PrivacyRedactionError("privacy result frame identity mismatch")
        if not result.safe_for_display:
            raise PrivacyRedactionError("privacy redactor did not mark the frame safe")
        redacted = image_bgr.copy()
        for x1, y1, x2, y2 in (*result.face_masks, *result.plate_masks):
            redacted[y1:y2, x1:x2] = 0
        return RedactedFrame(redacted, result)
