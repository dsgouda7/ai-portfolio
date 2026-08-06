from __future__ import annotations

import os
from enum import StrEnum
from typing import Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DeploymentMetadata


class EndpointProvider(StrEnum):
    AZURE_ML = "azure_ml"
    APIM = "apim"
    FOUNDRY = "foundry"


class EndpointClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: EndpointProvider
    endpoint_url: str
    token_scope: str
    route: str = "/v1/chat/completions"
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    max_retries: int = Field(default=2, ge=0, le=3)
    managed_identity_client_id: str | None = None
    azureml_deployment: str | None = None
    foundry_deployment: str | None = None
    foundry_api_version: str = "2024-10-21"
    deployment_metadata: DeploymentMetadata | None = None

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> EndpointClientConfig:
        parsed = urlparse(self.endpoint_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("endpoint_url must be an absolute HTTPS URL")
        if not self.token_scope.endswith("/.default"):
            raise ValueError("token_scope must be an Entra resource scope ending in /.default")
        if not self.route.startswith("/"):
            raise ValueError("route must start with /")
        if self.provider is EndpointProvider.FOUNDRY and not self.foundry_deployment:
            raise ValueError("foundry_deployment is required for the foundry provider")
        return self

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> EndpointClientConfig:
        values = os.environ if environ is None else environ

        forbidden_keys = {
            "RIVERSIDE_ENDPOINT_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        }
        configured_keys = sorted(name for name in forbidden_keys if values.get(name))
        if configured_keys:
            raise ValueError("API-key authentication is not supported; use managed or workload identity")

        def required(name: str) -> str:
            value = values.get(name)
            if not value:
                raise ValueError(f"missing required environment variable: {name}")
            return value

        return cls(
            provider=required("RIVERSIDE_ENDPOINT_PROVIDER"),
            endpoint_url=required("RIVERSIDE_ENDPOINT_URL"),
            token_scope=required("RIVERSIDE_ENDPOINT_TOKEN_SCOPE"),
            route=values.get("RIVERSIDE_ENDPOINT_ROUTE", "/v1/chat/completions"),
            timeout_seconds=int(values.get("RIVERSIDE_ENDPOINT_TIMEOUT_SECONDS", "30")),
            max_retries=int(values.get("RIVERSIDE_ENDPOINT_MAX_RETRIES", "2")),
            managed_identity_client_id=values.get("AZURE_CLIENT_ID"),
            azureml_deployment=values.get("RIVERSIDE_AZUREML_DEPLOYMENT"),
            foundry_deployment=values.get("RIVERSIDE_FOUNDRY_DEPLOYMENT"),
            foundry_api_version=values.get("RIVERSIDE_FOUNDRY_API_VERSION", "2024-10-21"),
        )
