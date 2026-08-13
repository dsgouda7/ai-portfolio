"""Opt-in local OpenCV camera source with lazy device acquisition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from roadid.contracts import CameraSource, FramePacket
from roadid.sources.base import BaseSourceAdapter, SourceConfigurationError, SourceUnavailableError


class LocalCameraSource(BaseSourceAdapter):
    def __init__(
        self,
        source: CameraSource,
        *,
        capture_factory: Callable[[int], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(source, clock=clock)
        try:
            self._device_index = int(source.options.get("device_index", 0))
        except (TypeError, ValueError) as error:
            raise SourceConfigurationError(
                "local camera device_index must be an integer"
            ) from error
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._next_frame_id = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def _capture_once(self, run_id: str) -> FramePacket:
        capture = self._ensure_open()
        ok, image = capture.read()
        if not ok or image is None:
            raise SourceUnavailableError(
                f"local camera device {self._device_index} did not return a frame",
                source_id=self.source.source_id,
            )
        image_bgr = np.asarray(image)
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise SourceUnavailableError(
                f"local camera device {self._device_index} returned an invalid BGR frame",
                source_id=self.source.source_id,
            )
        captured_at = self._clock().astimezone(UTC)
        packet = FramePacket(
            run_id=run_id,
            source_id=self.source.source_id,
            frame_id=self._next_frame_id,
            captured_at=captured_at,
            received_at=captured_at,
            image_bgr=image_bgr,
            source_metadata={"device_index": self._device_index},
        )
        self._next_frame_id += 1
        return packet

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _ensure_open(self) -> Any:
        if self._capture is None:
            factory = self._capture_factory
            if factory is None:
                import cv2

                factory = cv2.VideoCapture
            self._capture = factory(self._device_index)
        if not self._capture.isOpened():
            self.close()
            raise SourceUnavailableError(
                f"local camera device {self._device_index} is unavailable",
                source_id=self.source.source_id,
            )
        return self._capture
