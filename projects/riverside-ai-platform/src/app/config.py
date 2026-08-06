from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import yaml
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, model_validator


_ENVIRONMENT_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_FORBIDDEN_SECRET_NAMES = frozenset(
    {
        "AZURE_OPENAI_API_KEY",
        "DATABRICKS_TOKEN",
        "OPENAI_API_KEY",
        "RIVERSIDE_ENDPOINT_API_KEY",
    }
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IdentityConfig(ConfigModel):
    authentication: Literal["managed_identity", "workload_identity"]


class ModelConfig(ConfigModel):
    alias: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=128)]
    release_manifest_uri: AnyUrl
    max_input_tokens: Annotated[int, Field(ge=1, le=8192)]
    max_output_tokens: Annotated[int, Field(ge=1, le=2048)]
    precision: Literal["fp32", "fp16", "bf16", "int8", "int4"]


class DataConfig(ConfigModel):
    contract_version: Literal["1.0.0"]
    index_name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=128)]
    index_version: Annotated[
        str,
        Field(
            pattern=(
                r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
                r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
            )
        ),
    ]


class GatewayConfig(ConfigModel):
    base_url: AnyUrl
    route: Literal["/v1/chat/completions"]
    timeout_seconds: Annotated[int, Field(ge=1, le=120)]
    max_retries: Annotated[int, Field(ge=0, le=3)]


class ServingConfig(ConfigModel):
    endpoint_name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=128)]
    blue_deployment: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=128)]
    green_deployment: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=128)]
    request_timeout_seconds: Annotated[int, Field(ge=1, le=120)]


class RetrievalConfig(ConfigModel):
    top_k: Annotated[int, Field(ge=1, le=20)]
    search_type: Literal["similarity", "mmr", "hybrid"]


class TelemetryConfig(ConfigModel):
    enabled: bool
    service_name: Literal[
        "riverside-gateway",
        "riverside-rag-orchestrator",
        "riverside-model-endpoint",
    ]
    exporter_endpoint: AnyUrl


class EvaluationConfig(ConfigModel):
    release_report_uri: AnyUrl
    required_domains: Annotated[
        list[
            Literal[
                "data_quality",
                "retrieval_quality",
                "generation_citation_quality",
                "adaptation_evidence",
                "safety_authorization",
                "operational_slos",
                "cost",
                "rollout_comparison",
            ]
        ],
        Field(min_length=8, max_length=8),
    ]

    @model_validator(mode="after")
    def require_all_domains(self) -> EvaluationConfig:
        if len(set(self.required_domains)) != 8:
            raise ValueError("required_domains must contain each v1 evaluation domain once")
        return self


class PlatformConfig(ConfigModel):
    config_version: Literal["1.0.0"]
    project_name: Literal["riverside-ai-platform"]
    environment: Literal["dev", "staging", "production"]
    region: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")]
    identity: IdentityConfig
    model: ModelConfig
    data: DataConfig
    gateway: GatewayConfig
    serving: ServingConfig
    retrieval: RetrievalConfig
    telemetry: TelemetryConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def validate_deadlines_and_names(self) -> PlatformConfig:
        if self.serving.request_timeout_seconds >= self.gateway.timeout_seconds:
            raise ValueError("application serving deadline must be shorter than the gateway timeout")
        expected_prefix = f"riverside-{self.environment}"
        if self.serving.endpoint_name != expected_prefix:
            raise ValueError("endpoint_name must match the selected environment")
        if self.serving.blue_deployment != f"{expected_prefix}-blue":
            raise ValueError("blue_deployment must match the selected environment")
        if self.serving.green_deployment != f"{expected_prefix}-green":
            raise ValueError("green_deployment must match the selected environment")
        if not self.telemetry.enabled:
            raise ValueError("telemetry must be enabled for every deployable profile")
        if self.telemetry.service_name != "riverside-rag-orchestrator":
            raise ValueError("telemetry service_name must identify the RAG orchestrator")
        return self


def _substitute_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        match = _ENVIRONMENT_REFERENCE.fullmatch(value)
        if match is None:
            return value
        name = match.group(1)
        resolved = environ.get(name)
        if not resolved:
            raise ValueError(f"missing required environment variable: {name}")
        return resolved
    if isinstance(value, list):
        return [_substitute_environment(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _substitute_environment(item, environ) for key, item in value.items()}
    return value


def load_platform_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PlatformConfig:
    values = os.environ if environ is None else environ
    configured_secrets = sorted(name for name in _FORBIDDEN_SECRET_NAMES if values.get(name))
    if configured_secrets:
        raise ValueError(
            "API keys and Databricks personal access tokens are not supported; "
            "use managed or workload identity"
        )
    profile_path = Path(path)
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"configuration profile is not readable: {profile_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("configuration profile must contain a YAML object")
    resolved = _substitute_environment(raw, values)
    return PlatformConfig.model_validate(resolved)
