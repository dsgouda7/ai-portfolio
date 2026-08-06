"""Request spans and bounded metrics for LLM serving paths."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from time import perf_counter_ns
from typing import Any, Callable, ContextManager, Iterator, Literal, Mapping, Protocol, cast

from .conventions import CacheResult, ErrorCode, MetricContext, Outcome, build_metric_attributes

StageName = Literal["queue", "retrieval", "generation"]
RejectionReason = Literal[
    "none",
    "capacity",
    "rate_limit",
    "policy",
    "authorization",
    "invalid_request",
    "other",
]

_STAGES: tuple[StageName, ...] = ("queue", "retrieval", "generation")
_REJECTION_REASONS = frozenset(
    {"none", "capacity", "rate_limit", "policy", "authorization", "invalid_request", "other"}
)


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: str | int | float | bool) -> None: ...

    def set_status(self, status: Any) -> None: ...


class TracerLike(Protocol):
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
    ) -> ContextManager[SpanLike]: ...


class HistogramLike(Protocol):
    def record(self, amount: int | float, attributes: Mapping[str, str] | None = None) -> None: ...


class CounterLike(Protocol):
    def add(self, amount: int | float, attributes: Mapping[str, str] | None = None) -> None: ...


class MeterLike(Protocol):
    def create_histogram(self, name: str, *, unit: str, description: str) -> HistogramLike: ...

    def create_counter(self, name: str, *, unit: str, description: str) -> CounterLike: ...


@dataclass(frozen=True, slots=True)
class LatencyMeasurements:
    """Request latency values in milliseconds; streaming-only fields may be absent."""

    total_ms: float
    ttft_ms: float | None
    tpot_ms: float | None


class RequestTelemetry:
    """Creates OTel instruments and request measurements with v1 metric labels."""

    def __init__(
        self,
        tracer: TracerLike,
        meter: MeterLike,
        context: MetricContext,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._tracer = tracer
        self._context = context
        self._clock_ns = clock_ns
        self._request_duration = meter.create_histogram(
            "riverside.ai.request.duration",
            unit="ms",
            description="End-to-end request latency.",
        )
        self._ttft = meter.create_histogram(
            "riverside.ai.request.ttft",
            unit="ms",
            description="Time from request start to first output token.",
        )
        self._tpot = meter.create_histogram(
            "riverside.ai.request.tpot",
            unit="ms/token",
            description="Mean time per output token after the first token.",
        )
        self._stage_duration = {
            stage: meter.create_histogram(
                f"riverside.ai.stage.{stage}.duration",
                unit="ms",
                description=f"Duration of the {stage} stage.",
            )
            for stage in _STAGES
        }
        self._prompt_tokens = meter.create_histogram(
            "riverside.ai.usage.prompt_tokens",
            unit="token",
            description="Exact prompt-token count recorded with bounded labels.",
        )
        self._output_tokens = meter.create_histogram(
            "riverside.ai.usage.output_tokens",
            unit="token",
            description="Exact output-token count recorded with bounded labels.",
        )
        self._cost = meter.create_histogram(
            "riverside.ai.request.cost",
            unit="USD",
            description="Estimated request cost in US dollars.",
        )
        self._retries = meter.create_counter(
            "riverside.ai.request.retries",
            unit="{retry}",
            description="Retry attempts performed by request handlers.",
        )
        self._rejections = meter.create_counter(
            "riverside.ai.request.rejections",
            unit="{request}",
            description="Requests rejected before successful completion.",
        )
        self._cache_requests = meter.create_counter(
            "riverside.ai.cache.requests",
            unit="{request}",
            description="Requests by bounded cache outcome.",
        )

    @contextmanager
    def request_span(self) -> Iterator[RequestMeasurement]:
        """Start one request span and require callers to complete its measurement."""

        attributes = {
            "service.name": self._context.service_name,
            "deployment.environment": self._context.environment,
            "model.release_id": self._context.release_id,
            "model.alias": self._context.model_alias,
            "deployment.name": self._context.deployment_name,
            "cloud.region": self._context.region,
            "http.route": self._context.route,
            "tenant.tier": self._context.tenant_tier,
            "gen_ai.operation.name": "chat",
        }
        with self._tracer.start_as_current_span("riverside.chat.completions", attributes=attributes) as span:
            measurement = RequestMeasurement(self, span, self._clock_ns())
            try:
                yield measurement
            except Exception:
                if not measurement.completed:
                    measurement.complete(
                        outcome="error",
                        cache_result="bypass",
                        prompt_tokens=0,
                        output_tokens=0,
                        error_code="internal_error",
                    )
                raise
            if not measurement.completed:
                raise RuntimeError("request telemetry must be completed before leaving request_span")


class RequestMeasurement:
    """Mutable timing state scoped to one request span."""

    def __init__(self, telemetry: RequestTelemetry, span: SpanLike, started_ns: int) -> None:
        self._telemetry = telemetry
        self._span = span
        self._started_ns = started_ns
        self._first_token_ns: int | None = None
        self._stage_durations_ms: dict[StageName, float] = {}
        self.completed = False

    @contextmanager
    def stage(self, name: StageName) -> Iterator[None]:
        """Time one bounded request stage and create its child span."""

        if name not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}")
        if self.completed:
            raise RuntimeError("cannot record a stage after request completion")
        started_ns = self._telemetry._clock_ns()
        with self._telemetry._tracer.start_as_current_span(
            f"riverside.{name}",
            attributes={"riverside.stage.name": name},
        ):
            yield
        duration_ms = _milliseconds(self._telemetry._clock_ns() - started_ns)
        self._stage_durations_ms[name] = self._stage_durations_ms.get(name, 0.0) + duration_ms

    def record_first_token(self) -> None:
        """Capture TTFT once, when the first non-empty output token is observed."""

        if self.completed:
            raise RuntimeError("cannot record tokens after request completion")
        if self._first_token_ns is None:
            self._first_token_ns = self._telemetry._clock_ns()

    def complete(
        self,
        *,
        outcome: Outcome,
        cache_result: CacheResult,
        prompt_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        retry_count: int = 0,
        error_code: ErrorCode = "none",
        rejection_reason: RejectionReason = "none",
        retrieval_top_k: int | None = None,
    ) -> LatencyMeasurements:
        """Finalize the span and record all request metrics exactly once."""

        if self.completed:
            raise RuntimeError("request telemetry is already complete")
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("cost_usd must be a finite non-negative number")
        if isinstance(retry_count, bool) or not isinstance(retry_count, int) or not 0 <= retry_count <= 3:
            raise ValueError("retry_count must be an integer between 0 and 3")
        if rejection_reason not in _REJECTION_REASONS:
            raise ValueError(f"rejection_reason must be one of {sorted(_REJECTION_REASONS)}")
        if outcome == "rejected" and rejection_reason == "none":
            raise ValueError("rejected requests require a bounded rejection_reason")

        ended_ns = self._telemetry._clock_ns()
        total_ms = _milliseconds(ended_ns - self._started_ns)
        ttft_ms = None
        tpot_ms = None
        if self._first_token_ns is not None:
            ttft_ms = _milliseconds(self._first_token_ns - self._started_ns)
            if output_tokens > 1:
                tpot_ms = _milliseconds(ended_ns - self._first_token_ns) / (output_tokens - 1)

        attributes = build_metric_attributes(
            self._telemetry._context,
            outcome=outcome,
            cache_result=cache_result,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            error_code=error_code,
            retrieval_top_k=retrieval_top_k,
        )
        self._telemetry._request_duration.record(total_ms, attributes)
        if ttft_ms is not None:
            self._telemetry._ttft.record(ttft_ms, attributes)
        if tpot_ms is not None:
            self._telemetry._tpot.record(tpot_ms, attributes)
        for stage, duration_ms in self._stage_durations_ms.items():
            self._telemetry._stage_duration[stage].record(duration_ms, attributes)
        self._telemetry._prompt_tokens.record(prompt_tokens, attributes)
        self._telemetry._output_tokens.record(output_tokens, attributes)
        self._telemetry._cost.record(cost_usd, attributes)
        if retry_count:
            self._telemetry._retries.add(retry_count, attributes)
        if outcome == "rejected":
            self._telemetry._rejections.add(1, attributes)
        self._telemetry._cache_requests.add(1, attributes)

        span_attributes: dict[str, str | int | float | bool] = {
            "outcome.status": outcome,
            "cache.result": cache_result,
            "error.code": error_code,
            "gen_ai.usage.input_tokens": prompt_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "riverside.cost.usd": cost_usd,
            "riverside.retry.count": retry_count,
            "riverside.rejection.reason": rejection_reason,
            "riverside.latency.total_ms": total_ms,
        }
        if ttft_ms is not None:
            span_attributes["riverside.latency.ttft_ms"] = ttft_ms
        if tpot_ms is not None:
            span_attributes["riverside.latency.tpot_ms"] = tpot_ms
        for stage, duration_ms in self._stage_durations_ms.items():
            span_attributes[f"riverside.stage.{stage}.duration_ms"] = duration_ms
        for key, value in span_attributes.items():
            self._span.set_attribute(key, value)

        self.completed = True
        return LatencyMeasurements(total_ms=total_ms, ttft_ms=ttft_ms, tpot_ms=tpot_ms)


def _milliseconds(nanoseconds: int) -> float:
    if nanoseconds < 0:
        raise RuntimeError("monotonic clock moved backwards")
    return nanoseconds / 1_000_000
