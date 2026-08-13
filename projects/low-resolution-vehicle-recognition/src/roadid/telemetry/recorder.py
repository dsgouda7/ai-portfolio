"""Thread-safe event recording for one TrackLens inference run."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from threading import Condition, RLock
from time import monotonic
from typing import Any

from roadid.contracts import JSONValue, PipelineEvent, StageStatus
from roadid.telemetry.events import (
    bounded_identifier,
    bounded_text,
    remaining_stages,
    sanitize_summary,
    validate_stage,
)
from roadid.telemetry.metrics import StageLatencyAggregator, StageLatencyStats


@dataclass(frozen=True, slots=True)
class EventBatch:
    events: tuple[PipelineEvent, ...]
    dropped_count: int
    first_available_sequence: int
    next_sequence: int
    gap: bool


@dataclass(frozen=True, slots=True)
class _ActiveStage:
    started_at: datetime
    started_monotonic: float
    input_summary: dict[str, JSONValue]


class EventRecorder:
    """Record bounded, ordered telemetry for exactly one run."""

    def __init__(
        self,
        run_id: str,
        *,
        capacity: int = 1000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.run_id = bounded_identifier("run_id", run_id) or ""
        self.capacity = capacity
        self.metadata = sanitize_summary(metadata)
        self._events: deque[PipelineEvent] = deque()
        self._latest_by_stage: dict[str, PipelineEvent] = {}
        self._active: dict[tuple[str, int | None, str | None], _ActiveStage] = {}
        self._next_sequence = 1
        self._dropped_count = 0
        self._latencies = StageLatencyAggregator()
        self._lock = RLock()
        # All event/state mutations and observer wakeups share this condition.
        self._condition = Condition(self._lock)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def first_available_sequence(self) -> int:
        with self._lock:
            return self._first_available_sequence_unlocked()

    @property
    def next_sequence(self) -> int:
        with self._lock:
            return self._next_sequence

    def start_stage(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        input_summary: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> PipelineEvent:
        key = self._stage_key(stage, frame_id, track_id)
        with self._condition:
            if key in self._active:
                raise ValueError(f"stage is already running: {stage}")
            timestamp = _utc_datetime(started_at)
            active = _ActiveStage(timestamp, monotonic(), sanitize_summary(input_summary))
            self._active[key] = active
            return self._append_unlocked(
                stage=stage,
                status=StageStatus.RUNNING,
                started_at=timestamp,
                frame_id=frame_id,
                track_id=key[2],
                input_summary=active.input_summary,
            )

    def complete_stage(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        output_summary: Mapping[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> PipelineEvent:
        return self._terminal_stage(
            stage,
            StageStatus.COMPLETED,
            frame_id=frame_id,
            track_id=track_id,
            output_summary=output_summary,
            duration_ms=duration_ms,
        )

    def skip_stage(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
    ) -> PipelineEvent:
        return self._terminal_stage(
            stage,
            StageStatus.SKIPPED,
            frame_id=frame_id,
            track_id=track_id,
            warning=reason,
            duration_ms=duration_ms,
        )

    def warn_stage(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        warning: str,
        output_summary: Mapping[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> PipelineEvent:
        if not warning:
            raise ValueError("warning cannot be empty")
        return self._terminal_stage(
            stage,
            StageStatus.WARNING,
            frame_id=frame_id,
            track_id=track_id,
            output_summary=output_summary,
            warning=warning,
            duration_ms=duration_ms,
        )

    def fail_stage(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        error_code: str,
        warning: str | None = None,
        output_summary: Mapping[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> PipelineEvent:
        if not error_code:
            raise ValueError("error_code cannot be empty")
        return self._terminal_stage(
            stage,
            StageStatus.FAILED,
            frame_id=frame_id,
            track_id=track_id,
            output_summary=output_summary,
            warning=warning,
            error_code=error_code,
            duration_ms=duration_ms,
        )

    start = start_stage
    complete = complete_stage
    skip = skip_stage
    warn = warn_stage
    fail = fail_stage

    def fail_stage_and_skip_remaining(
        self,
        stage: str,
        *,
        frame_id: int | None = None,
        track_id: str | None = None,
        error_code: str,
        warning: str | None = None,
        duration_ms: float | None = None,
    ) -> tuple[PipelineEvent, ...]:
        failed = self.fail_stage(
            stage,
            frame_id=frame_id,
            track_id=track_id,
            error_code=error_code,
            warning=warning,
            duration_ms=duration_ms,
        )
        skipped: list[PipelineEvent] = []
        reason = f"downstream of failed stage {stage}"
        for downstream_stage in remaining_stages(stage):
            self.start_stage(downstream_stage, frame_id=frame_id, track_id=track_id)
            skipped.append(
                self.skip_stage(
                    downstream_stage,
                    frame_id=frame_id,
                    track_id=track_id,
                    reason=reason,
                    duration_ms=0.0,
                )
            )
        return failed, *skipped

    def events_after(self, sequence_id: int = 0) -> EventBatch:
        if sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        with self._lock:
            return self._batch_unlocked(sequence_id)

    def wait_for_events(self, sequence_id: int, timeout: float | None = None) -> EventBatch:
        if sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._next_sequence > sequence_id + 1
                    or self._first_available_sequence_unlocked() > sequence_id + 1
                ),
                timeout=timeout,
            )
            return self._batch_unlocked(sequence_id)

    def all_events(self) -> tuple[PipelineEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def latency_snapshot(self) -> dict[str, StageLatencyStats]:
        return self._latencies.snapshot()

    def graph_snapshot(self) -> dict[str, JSONValue]:
        from roadid.telemetry.graph import serialize_graph_snapshot

        with self._lock:
            events = tuple(self._latest_by_stage.values())
            latencies = self._latencies.snapshot()
        return serialize_graph_snapshot(events, latencies=latencies)

    def export_jsonl(self, destination: Any = None) -> str:
        from roadid.telemetry.export import export_jsonl

        return export_jsonl(self, destination)

    def _terminal_stage(
        self,
        stage: str,
        status: StageStatus,
        *,
        frame_id: int | None,
        track_id: str | None,
        output_summary: Mapping[str, Any] | None = None,
        warning: str | None = None,
        error_code: str | None = None,
        duration_ms: float | None,
    ) -> PipelineEvent:
        key = self._stage_key(stage, frame_id, track_id)
        with self._condition:
            active = self._active.get(key)
            if active is None:
                raise ValueError(f"stage must be running before {status.value}: {stage}")
            measured_duration = (
                max(0.0, (monotonic() - active.started_monotonic) * 1000.0)
                if duration_ms is None
                else duration_ms
            )
            if isinstance(measured_duration, bool) or not isfinite(measured_duration):
                raise ValueError("duration_ms must be finite")
            if measured_duration < 0:
                raise ValueError("duration_ms must be non-negative")
            event = self._append_unlocked(
                stage=stage,
                status=status,
                started_at=active.started_at,
                frame_id=frame_id,
                track_id=key[2],
                duration_ms=measured_duration,
                input_summary=active.input_summary,
                output_summary=sanitize_summary(output_summary),
                warning=bounded_text(warning),
                error_code=bounded_text(error_code, limit=128),
            )
            del self._active[key]
            return event

    def _stage_key(
        self, stage: str, frame_id: int | None, track_id: str | None
    ) -> tuple[str, int | None, str | None]:
        validate_stage(stage)
        if frame_id is not None and frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        safe_track_id = bounded_identifier("track_id", track_id)
        return stage, frame_id, safe_track_id

    def _append_unlocked(
        self,
        *,
        stage: str,
        status: StageStatus,
        started_at: datetime,
        frame_id: int | None,
        track_id: str | None,
        duration_ms: float | None = None,
        input_summary: dict[str, JSONValue] | None = None,
        output_summary: dict[str, JSONValue] | None = None,
        warning: str | None = None,
        error_code: str | None = None,
    ) -> PipelineEvent:
        sequence_id = self._next_sequence
        self._next_sequence += 1
        event = PipelineEvent(
            event_id=f"{self.run_id}:{sequence_id}",
            sequence_id=sequence_id,
            run_id=self.run_id,
            stage=stage,
            status=status,
            started_at=started_at,
            frame_id=frame_id,
            track_id=track_id,
            duration_ms=duration_ms,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            warning=warning,
            error_code=error_code,
        )
        if len(self._events) >= self.capacity:
            self._events.popleft()
            self._dropped_count += 1
        self._events.append(event)
        self._latest_by_stage[event.stage] = event
        self._latencies.observe(event)
        self._condition.notify_all()
        return event

    def _batch_unlocked(self, sequence_id: int) -> EventBatch:
        first = self._first_available_sequence_unlocked()
        return EventBatch(
            events=tuple(event for event in self._events if event.sequence_id > sequence_id),
            dropped_count=self._dropped_count,
            first_available_sequence=first,
            next_sequence=self._next_sequence,
            gap=sequence_id + 1 < first,
        )

    def _first_available_sequence_unlocked(self) -> int:
        return self._events[0].sequence_id if self._events else self._next_sequence


def _utc_datetime(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    return timestamp.astimezone(UTC)
