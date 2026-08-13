"""Typed source adapter primitives shared by all RoadID frame providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from time import sleep
from typing import Protocol, runtime_checkable

from roadid.contracts import CameraSource, FramePacket


class SourceError(Exception):
    """Base class for errors safe to surface at the source boundary."""

    code = "source_error"
    retryable = False

    def __init__(self, message: str, *, source_id: str | None = None) -> None:
        super().__init__(message)
        self.source_id = source_id


class SourceConfigurationError(SourceError):
    code = "source_configuration_error"


class SourceDisabledError(SourceConfigurationError):
    code = "source_disabled"


class SourceAccessError(SourceConfigurationError):
    code = "source_access_error"


class SourceTermsError(SourceAccessError):
    code = "source_terms_error"


class SourceSecurityError(SourceError):
    code = "source_security_error"


class SourceUnavailableError(SourceError):
    code = "source_unavailable"
    retryable = True


class SourceTimeoutError(SourceUnavailableError):
    code = "source_timeout"


class SourceFetchError(SourceUnavailableError):
    code = "source_fetch_error"


class SourceValidationError(SourceError):
    code = "source_validation_error"


class SourceOversizedError(SourceValidationError):
    code = "source_oversized"


class SourceStaleError(SourceValidationError):
    code = "source_stale"


class SourceExhaustedError(SourceError):
    code = "source_exhausted"


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SourceHealth:
    status: HealthStatus = HealthStatus.UNKNOWN
    successful_captures: int = 0
    consecutive_failures: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error_code: str | None = None


@runtime_checkable
class SourceAdapter(Protocol):
    @property
    def source(self) -> CameraSource: ...

    @property
    def health(self) -> SourceHealth: ...

    def capture(self, run_id: str) -> FramePacket: ...

    def close(self) -> None: ...


class BaseSourceAdapter(ABC):
    def __init__(
        self,
        source: CameraSource,
        *,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        if retry_backoff_seconds < 0 or retry_backoff_seconds > 30:
            raise ValueError("retry_backoff_seconds must be between 0 and 30")
        self._source = source
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleeper = sleeper
        self._clock = clock or (lambda: datetime.now(UTC))
        self._health = SourceHealth()

    @property
    def source(self) -> CameraSource:
        return self._source

    @property
    def health(self) -> SourceHealth:
        return self._health

    def capture(self, run_id: str) -> FramePacket:
        if not self.source.enabled:
            error = SourceDisabledError(
                f"source '{self.source.source_id}' is disabled",
                source_id=self.source.source_id,
            )
            self._record_failure(error)
            raise error

        for attempt in range(self._max_retries + 1):
            try:
                packet = self._capture_once(run_id)
            except SourceError as error:
                if not error.retryable or attempt >= self._max_retries:
                    self._record_failure(error)
                    raise
                self._sleeper(self._retry_backoff_seconds * (2**attempt))
            else:
                self._record_success()
                return packet
        raise AssertionError("bounded source retry loop terminated unexpectedly")

    def next_frame(self, run_id: str) -> FramePacket:
        return self.capture(run_id)

    @abstractmethod
    def _capture_once(self, run_id: str) -> FramePacket:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def _record_success(self) -> None:
        self._health = replace(
            self._health,
            status=HealthStatus.HEALTHY,
            successful_captures=self._health.successful_captures + 1,
            consecutive_failures=0,
            last_success_at=self._clock(),
            last_error_code=None,
        )

    def _record_failure(self, error: SourceError) -> None:
        failures = self._health.consecutive_failures + 1
        self._health = replace(
            self._health,
            status=(
                HealthStatus.DEGRADED
                if self._health.successful_captures
                else HealthStatus.UNAVAILABLE
            ),
            consecutive_failures=failures,
            last_failure_at=self._clock(),
            last_error_code=error.code,
        )
