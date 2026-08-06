from __future__ import annotations

import os
from collections.abc import Mapping

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from telemetry import MetricContext, RequestTelemetry

from .config import PlatformConfig


class RequestTelemetryFactory:
    def __init__(
        self,
        profile: PlatformConfig,
        *,
        release_id: str,
        deployment_name: str,
        exporter_endpoint: str,
    ) -> None:
        if deployment_name not in {
            profile.serving.blue_deployment,
            profile.serving.green_deployment,
        }:
            raise ValueError("active deployment must be the configured blue or green slot")
        self._profile = profile
        self._release_id = release_id
        self._deployment_name = deployment_name
        resource = Resource.create(
            {
                "service.name": profile.telemetry.service_name,
                "deployment.environment.name": profile.environment,
                "cloud.region": profile.region,
            }
        )
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{exporter_endpoint.rstrip('/')}/v1/traces"))
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=f"{exporter_endpoint.rstrip('/')}/v1/metrics")
                )
            ],
        )
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self._tracer = tracer_provider.get_tracer("riverside.app", "1.0.0")
        self._meter = meter_provider.get_meter("riverside.app", "1.0.0")
        self._requests: dict[str, RequestTelemetry] = {}

    @classmethod
    def from_environment(
        cls,
        profile: PlatformConfig,
        environ: Mapping[str, str] | None = None,
    ) -> RequestTelemetryFactory:
        if not profile.telemetry.enabled:
            raise ValueError("telemetry must be enabled for the deployable application host")
        values = os.environ if environ is None else environ
        release_id = values.get("RIVERSIDE_MODEL_RELEASE_ID")
        deployment_name = values.get("RIVERSIDE_ACTIVE_DEPLOYMENT_NAME")
        if not release_id or not deployment_name:
            raise ValueError(
                "telemetry requires RIVERSIDE_MODEL_RELEASE_ID and "
                "RIVERSIDE_ACTIVE_DEPLOYMENT_NAME"
            )
        return cls(
            profile,
            release_id=release_id,
            deployment_name=deployment_name,
            exporter_endpoint=str(profile.telemetry.exporter_endpoint),
        )

    def for_tenant_tier(self, tenant_tier: str) -> RequestTelemetry:
        request_telemetry = self._requests.get(tenant_tier)
        if request_telemetry is None:
            request_telemetry = RequestTelemetry(
                self._tracer,
                self._meter,
                MetricContext(
                    service_name=self._profile.telemetry.service_name,
                    environment=self._profile.environment,
                    release_id=self._release_id,
                    model_alias=self._profile.model.alias,
                    deployment_name=self._deployment_name,
                    region=self._profile.region,
                    route=self._profile.gateway.route,
                    tenant_tier=tenant_tier,
                ),
            )
            self._requests[tenant_tier] = request_telemetry
        return request_telemetry

    def close(self) -> None:
        self._meter_provider.shutdown()
        self._tracer_provider.shutdown()
