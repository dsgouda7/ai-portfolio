"""Configuration loading and validation for the RAG pipeline."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlparse


_MODES = {"local", "remote"}
_RETRIEVAL_PROVIDERS = {"auto", "local", "databricks"}
_GENERATION_PROVIDERS = {"openai_compatible", "azure_endpoint", "watsonx_legacy"}
_AZURE_ENDPOINT_PROVIDERS = {"apim", "azure_ml", "foundry"}
_FORBIDDEN_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "client_secret",
    "connection_string",
    "password",
    "token",
}
_REMOTE_ENV_REFERENCES = {
    "workspace_url": "DATABRICKS_HOST",
    "vector_search_endpoint": "RIVERSIDE_VECTOR_SEARCH_ENDPOINT",
}
_ENVIRONMENT_REFERENCE = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}$")
_PLACEHOLDER_MARKERS = (
    "example.invalid",
    "placeholder",
    "replace-with",
    "replace_with",
    "changeme",
)


class ConfigError(ValueError):
    """Raised when pipeline configuration is incomplete or unsafe."""


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _resolve_env_vars(config)
    return validate_config(config)


def _resolve_env_vars(config: Any) -> Any:
    """Recursively resolve environment variable references in config."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    else:
        return config


def _require(mapping: Mapping[str, Any], names: tuple[str, ...], section: str) -> None:
    missing = [name for name in names if mapping.get(name) in (None, "")]
    if missing:
        raise ConfigError(f"{section} requires: {', '.join(missing)}")


def _reject_deployment_placeholder(value: Any, field_name: str) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if (
        not text
        or _ENVIRONMENT_REFERENCE.fullmatch(text)
        or text.startswith("<") and text.endswith(">")
        or any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
    ):
        raise ConfigError(f"{field_name} must be resolved and must not be a placeholder")
    return text


def _require_https_url(value: Any, field_name: str, expected_host_suffix: str = "") -> str:
    text = str(value).strip()
    if (
        not text
        or _ENVIRONMENT_REFERENCE.fullmatch(text)
        or text.startswith("<") and text.endswith(">")
    ):
        _reject_deployment_placeholder(value, field_name)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ConfigError(f"{field_name} must be an absolute HTTPS URL without credentials")
    _reject_deployment_placeholder(text, field_name)
    if expected_host_suffix and not parsed.hostname.endswith(expected_host_suffix):
        raise ConfigError(f"{field_name} must use an {expected_host_suffix} host")
    return text


def _require_remote_environment_references(remote: Mapping[str, Any]) -> None:
    invalid = []
    for field_name, environment_name in _REMOTE_ENV_REFERENCES.items():
        expected_reference = f"${{{environment_name}}}"
        value = remote.get(field_name)
        environment_value = os.getenv(environment_name)
        if value == expected_reference or not environment_value:
            invalid.append(environment_name)
            continue
        try:
            resolved_environment_value = _reject_deployment_placeholder(
                environment_value, environment_name
            )
            resolved_value = _reject_deployment_placeholder(
                value, f"remote.{field_name}"
            )
        except ConfigError:
            invalid.append(environment_name)
            continue
        if resolved_value != resolved_environment_value:
            invalid.append(environment_name)
    if invalid:
        raise ConfigError(
            "Remote Databricks deployment requires environment variables: "
            + ", ".join(_REMOTE_ENV_REFERENCES.values())
        )


def _reject_inline_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in _FORBIDDEN_CONFIG_KEYS and child not in (None, ""):
                raise ConfigError(f"Credentials must not be stored in YAML: {child_path}")
            _reject_inline_secrets(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{index}]")


def validate_config(config: Any) -> Dict[str, Any]:
    """Validate the active execution mode and serving provider configuration."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping")
    _reject_inline_secrets(config)

    mode = config.get("mode", "local")
    if mode not in _MODES:
        raise ConfigError(f"Unsupported mode {mode!r}; expected local or remote")

    local = config.get("local")
    remote = config.get("remote")
    if not isinstance(local, dict) or not isinstance(remote, dict):
        raise ConfigError("Both local and remote configuration sections are required")
    if mode == "local":
        _require(
            local,
            ("delta_path", "vector_store_path", "embedding_model"),
            "local mode",
        )
    else:
        _require(
            remote,
            (
                "workspace_url",
                "vector_search_endpoint",
                "catalog",
                "schema",
                "embedding_model",
                "embedding_model_revision",
                "embedding_dimensions",
                "index_name",
                "index_version",
            ),
            "remote mode",
        )
        _require_remote_environment_references(remote)
        _require_https_url(
            remote["workspace_url"],
            "remote.workspace_url",
            ".azuredatabricks.net",
        )
        _reject_deployment_placeholder(
            remote["vector_search_endpoint"], "remote.vector_search_endpoint"
        )
        for field_name in (
            "catalog",
            "schema",
            "embedding_model",
            "embedding_model_revision",
            "index_name",
            "index_version",
        ):
            _reject_deployment_placeholder(remote[field_name], f"remote.{field_name}")

    serving = config.get("serving")
    if not isinstance(serving, dict):
        raise ConfigError("serving configuration is required")
    retrieval = serving.get("retrieval", {})
    generation = serving.get("generation", {})
    if not isinstance(retrieval, dict) or not isinstance(generation, dict):
        raise ConfigError("serving.retrieval and serving.generation must be mappings")

    retrieval_provider = retrieval.get("provider", "auto")
    if retrieval_provider not in _RETRIEVAL_PROVIDERS:
        raise ConfigError(f"Unsupported retrieval provider: {retrieval_provider!r}")
    if mode == "local" and retrieval_provider == "databricks":
        raise ConfigError("Databricks retrieval requires remote mode")
    if mode == "remote" and retrieval_provider == "local":
        raise ConfigError("Local retrieval requires local mode")

    generation_provider = generation.get("provider")
    if generation_provider not in _GENERATION_PROVIDERS:
        raise ConfigError(f"Unsupported generation provider: {generation_provider!r}")
    _require(generation, ("model_alias", "max_input_tokens", "max_tokens"), "generation")

    endpoint = generation.get("endpoint", {})
    if generation_provider in {"openai_compatible", "azure_endpoint"}:
        if not isinstance(endpoint, dict):
            raise ConfigError("generation.endpoint must be a mapping")
        _require(endpoint, ("url", "route"), "generation endpoint")
    if generation_provider == "azure_endpoint":
        endpoint_provider = endpoint.get("provider")
        if endpoint_provider not in _AZURE_ENDPOINT_PROVIDERS:
            raise ConfigError(f"Unsupported Azure endpoint provider: {endpoint_provider!r}")
        _require(endpoint, ("token_scope",), "Azure generation endpoint")
        _require_https_url(endpoint["url"], "Azure generation endpoint URL")
        token_scope = _reject_deployment_placeholder(
            endpoint["token_scope"], "Azure token scope"
        )
        if not token_scope.endswith("/.default"):
            raise ConfigError("Azure token scope must end with /.default")
    if generation_provider == "watsonx_legacy":
        legacy = generation.get("watsonx_legacy", {})
        if not isinstance(legacy, dict):
            raise ConfigError("generation.watsonx_legacy must be a mapping")
        _require(legacy, ("url", "project_id", "model_id"), "legacy Watsonx provider")

    return config


def get_mode(config: Dict[str, Any]) -> str:
    """Extract execution mode from config (local or remote)."""
    return config.get("mode", "local")


def get_local_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract local mode configuration."""
    return config.get("local", {})


def get_remote_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract remote mode configuration."""
    return config.get("remote", {})


def get_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract retrieval configuration."""
    return config.get("retrieval", {})


def get_serving_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validated serving configuration."""
    return config.get("serving", {})
