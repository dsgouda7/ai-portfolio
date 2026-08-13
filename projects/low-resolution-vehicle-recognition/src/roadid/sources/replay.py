"""Deterministic offline replay source used by tests and the default demo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from roadid.contracts import CameraSource, FramePacket, JSONValue
from roadid.sources.base import BaseSourceAdapter, SourceConfigurationError, SourceExhaustedError

DEFAULT_REPLAY_START = datetime(2025, 1, 1, tzinfo=UTC)


class ReplaySource(BaseSourceAdapter):
    def __init__(self, source: CameraSource) -> None:
        super().__init__(source)
        options = source.options
        self._frame_count = _bounded_int(options, "frame_count", 30, minimum=1, maximum=10_000)
        self._width = _bounded_int(options, "width", 960, minimum=64, maximum=4_096)
        self._height = _bounded_int(options, "height", 540, minimum=64, maximum=2_160)
        self._interval_seconds = _positive_float(
            options.get("capture_interval_seconds", source.refresh_seconds),
            "capture_interval_seconds",
        )
        self._start_at = _parse_start(options.get("start_at"))
        self._next_frame_id = 0

    def _capture_once(self, run_id: str) -> FramePacket:
        frame_id = self._next_frame_id
        if frame_id >= self._frame_count:
            raise SourceExhaustedError(
                f"replay source '{self.source.source_id}' has no frames remaining",
                source_id=self.source.source_id,
            )
        self._next_frame_id += 1
        captured_at = self._start_at + timedelta(seconds=frame_id * self._interval_seconds)
        image, vehicle_bbox = self._render_frame(frame_id)
        metadata: dict[str, JSONValue] = {
            "synthetic": True,
            "capture_interval_seconds": self._interval_seconds,
            "vehicle_track_active": vehicle_bbox is not None,
        }
        if vehicle_bbox is not None:
            metadata["track_id"] = f"{self.source.source_id}-track-0001"
            metadata["vehicle_bbox_xyxy"] = list(vehicle_bbox)
        return FramePacket(
            run_id=run_id,
            source_id=self.source.source_id,
            frame_id=frame_id,
            captured_at=captured_at,
            received_at=captured_at,
            image_bgr=image,
            source_metadata=metadata,
        )

    def reset(self) -> None:
        self._next_frame_id = 0

    def frames(self, run_id: str):
        while self._next_frame_id < self._frame_count:
            yield self.capture(run_id)

    def _render_frame(self, frame_id: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[:] = (168, 146, 118)
        horizon = self._height // 3
        image[horizon:] = (62, 66, 70)
        lane_y = min(self._height - 3, int(self._height * 0.78))
        image[lane_y : lane_y + 3, ::24] = (210, 210, 210)
        if not 3 <= frame_id <= 24:
            return image, None

        progress = (frame_id - 3) / 21
        vehicle_width = max(28, self._width // 9)
        vehicle_height = max(18, self._height // 12)
        x1 = int(-vehicle_width // 2 + progress * (self._width + vehicle_width))
        x1 = max(0, min(self._width - vehicle_width, x1))
        y1 = min(self._height - vehicle_height - 1, int(self._height * 0.62))
        x2 = x1 + vehicle_width
        y2 = y1 + vehicle_height
        image[y1:y2, x1:x2] = (42, 92, 196)
        roof_x1 = x1 + vehicle_width // 4
        roof_x2 = x2 - vehicle_width // 5
        roof_y1 = max(horizon, y1 - vehicle_height // 3)
        image[roof_y1:y1, roof_x1:roof_x2] = (36, 75, 156)
        wheel_radius = max(2, vehicle_height // 7)
        for wheel_x in (x1 + vehicle_width // 4, x2 - vehicle_width // 4):
            image[
                y2 - wheel_radius : y2 + wheel_radius,
                wheel_x - wheel_radius : wheel_x + wheel_radius,
            ] = (18, 18, 18)
        return image, (x1, roof_y1, x2, min(self._height, y2 + wheel_radius))


def _bounded_int(options: Any, name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(options.get(name, default))
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(f"replay option '{name}' must be an integer") from error
    if not minimum <= value <= maximum:
        raise SourceConfigurationError(
            f"replay option '{name}' must be between {minimum} and {maximum}"
        )
    return value


def _positive_float(value: JSONValue, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(f"replay option '{name}' must be numeric") from error
    if result <= 0:
        raise SourceConfigurationError(f"replay option '{name}' must be positive")
    return result


def _parse_start(value: JSONValue) -> datetime:
    if value is None:
        return DEFAULT_REPLAY_START
    if not isinstance(value, str):
        raise SourceConfigurationError("replay option 'start_at' must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SourceConfigurationError("replay option 'start_at' is not valid ISO-8601") from error
    if parsed.tzinfo is None:
        raise SourceConfigurationError("replay option 'start_at' must include a timezone")
    return parsed.astimezone(UTC)
