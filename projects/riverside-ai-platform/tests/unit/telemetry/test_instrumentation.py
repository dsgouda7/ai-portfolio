"""Unit tests for request timing and OTel instrument use."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import pytest

from telemetry import MetricContext, RequestTelemetry


@dataclass
class Instrument:
    measurements: list[tuple[float, Mapping[str, str] | None]] = field(default_factory=list)

    def record(self, amount: int | float, attributes: Mapping[str, str] | None = None) -> None:
        self.measurements.append((float(amount), attributes))

    def add(self, amount: int | float, attributes: Mapping[str, str] | None = None) -> None:
        self.record(amount, attributes)


class Meter:
    def __init__(self) -> None:
        self.instruments: dict[str, Instrument] = {}

    def create_histogram(self, name: str, **_: str) -> Instrument:
        return self.instruments.setdefault(name, Instrument())

    def create_counter(self, name: str, **_: str) -> Instrument:
        return self.instruments.setdefault(name, Instrument())


class Span:
    def __init__(self) -> None:
        self.attributes: dict[str, str | int | float | bool] = {}

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        del status


class Tracer:
    def __init__(self) -> None:
        self.spans: list[tuple[str, Span, Mapping[str, object] | None]] = []

    @contextmanager
    def start_as_current_span(self, name: str, *, attributes: Mapping[str, object] | None = None) -> Iterator[Span]:
        span = Span()
        self.spans.append((name, span, attributes))
        yield span


class Clock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def context() -> MetricContext:
    return MetricContext(
        service_name="riverside-rag-orchestrator",
        environment="staging",
        release_id="release-1",
        model_alias="riverside-editor",
        deployment_name="riverside-staging-blue",
        region="eastus2",
        route="/v1/chat/completions",
        tenant_tier="premium",
    )


def test_streaming_request_records_total_ttft_tpot_and_stages() -> None:
    meter = Meter()
    tracer = Tracer()
    clock = Clock(
        0,
        10_000_000,
        30_000_000,
        40_000_000,
        70_000_000,
        100_000_000,
        200_000_000,
    )
    telemetry = RequestTelemetry(tracer, meter, context(), clock_ns=clock)

    with telemetry.request_span() as request:
        with request.stage("queue"):
            pass
        with request.stage("retrieval"):
            pass
        request.record_first_token()
        measurements = request.complete(
            outcome="success",
            cache_result="miss",
            prompt_tokens=600,
            output_tokens=3,
            cost_usd=0.002,
            retry_count=1,
            retrieval_top_k=8,
        )

    assert measurements.total_ms == 200.0
    assert measurements.ttft_ms == 100.0
    assert measurements.tpot_ms == 50.0
    assert meter.instruments["riverside.ai.stage.queue.duration"].measurements[0][0] == 20.0
    assert meter.instruments["riverside.ai.stage.retrieval.duration"].measurements[0][0] == 30.0
    metric_attributes = meter.instruments["riverside.ai.request.duration"].measurements[0][1]
    assert metric_attributes is not None
    assert metric_attributes["gen_ai.usage.prompt_tokens_bucket"] == "513-2048"
    assert "riverside.retry.count" not in metric_attributes
    request_span = tracer.spans[0][1]
    assert request_span.attributes["riverside.retry.count"] == 1
    assert request_span.attributes["riverside.cost.usd"] == 0.002


def test_rejected_request_requires_bounded_reason() -> None:
    telemetry = RequestTelemetry(Tracer(), Meter(), context(), clock_ns=Clock(0, 1_000_000))

    with pytest.raises(ValueError, match="rejection_reason"):
        with telemetry.request_span() as request:
            request.complete(
                outcome="rejected",
                cache_result="bypass",
                prompt_tokens=0,
                output_tokens=0,
                error_code="overloaded",
            )


def test_request_span_requires_completion() -> None:
    telemetry = RequestTelemetry(Tracer(), Meter(), context(), clock_ns=Clock(0))

    with pytest.raises(RuntimeError, match="must be completed"):
        with telemetry.request_span():
            pass
