from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from endpoint_client import EndpointClientConfig, EndpointProvider, create_endpoint_client
from rag_orchestrator import OrchestratorConfig, RAGOrchestrator

from .auth import BackendIdentityConfig, BackendIdentityValidator
from .config import PlatformConfig, load_platform_config
from .search import DatabricksSearchConfig, DatabricksSearchIndex
from .telemetry import RequestTelemetryFactory


@dataclass(slots=True)
class ApplicationRuntime:
    config: PlatformConfig
    orchestrator: RAGOrchestrator
    endpoint_client: object
    search_index: DatabricksSearchIndex
    backend_identity: BackendIdentityValidator
    telemetry: RequestTelemetryFactory
    ready: bool = False
    readiness_error: str | None = None

    async def initialize(self) -> None:
        await self.search_index.check_ready()
        self.ready = True
        self.readiness_error = None

    async def close(self) -> None:
        await self.search_index.close()
        close = getattr(self.endpoint_client, "close", None)
        if close is not None:
            await close()
        self.telemetry.close()


def build_runtime(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ApplicationRuntime:
    values = os.environ if environ is None else environ
    profile = load_platform_config(config_path, environ=values)
    deployed_environment = values.get("AZURE_ENV_NAME")
    if deployed_environment and deployed_environment != profile.environment:
        raise ValueError("AZURE_ENV_NAME must match the selected Riverside profile")
    deployed_region = values.get("AZURE_LOCATION")
    if deployed_region and deployed_region != profile.region:
        raise ValueError("AZURE_LOCATION must match the selected Riverside profile")
    endpoint_config = EndpointClientConfig.from_environment(values)
    if endpoint_config.provider is EndpointProvider.APIM:
        raise ValueError("the orchestrator backend cannot target its own APIM gateway")
    if endpoint_config.timeout_seconds >= profile.serving.request_timeout_seconds:
        raise ValueError("model endpoint timeout must be shorter than the application deadline")
    endpoint_client = create_endpoint_client(endpoint_config)
    search_index = DatabricksSearchIndex(
        DatabricksSearchConfig.from_environment(profile, values)
    )
    orchestrator = RAGOrchestrator(
        search_index,
        endpoint_client,
        config=OrchestratorConfig(
            default_top_k=profile.retrieval.top_k,
            max_top_k=20,
            default_search_type=profile.retrieval.search_type,
        ),
    )
    return ApplicationRuntime(
        config=profile,
        orchestrator=orchestrator,
        endpoint_client=endpoint_client,
        search_index=search_index,
        backend_identity=BackendIdentityValidator(
            BackendIdentityConfig.from_environment(values)
        ),
        telemetry=RequestTelemetryFactory.from_environment(profile, values),
    )
