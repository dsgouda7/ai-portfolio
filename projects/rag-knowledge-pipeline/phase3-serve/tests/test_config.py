from __future__ import annotations

import os

import pytest

from shared.config_loader import ConfigError, validate_config


def config(mode: str = "local") -> dict:
    return {
        "mode": mode,
        "local": {
            "delta_path": "./data/delta_lake",
            "vector_store_path": "./data/chroma_db",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "remote": {
            "workspace_url": "https://adb-1234567890123456.7.azuredatabricks.net",
            "vector_search_endpoint": "riverside-search",
            "catalog": "main",
            "schema": "rag_demo",
            "embedding_model": "databricks-gte-large-en",
            "embedding_model_revision": "1",
            "embedding_dimensions": 1024,
            "index_name": "main.rag_demo.riverside_v1",
            "index_version": "1.0.0",
        },
        "serving": {
            "retrieval": {"provider": "auto"},
            "generation": {
                "provider": "openai_compatible",
                "model_alias": "riverside-editor",
                "max_input_tokens": 1024,
                "max_tokens": 128,
                "endpoint": {
                    "url": "http://localhost:8001",
                    "route": "/v1/chat/completions",
                },
            },
        },
    }


def test_local_and_remote_modes_share_one_validated_config_shape(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABRICKS_HOST",
        "https://adb-1234567890123456.7.azuredatabricks.net",
    )
    monkeypatch.setenv("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "riverside-search")
    assert validate_config(config("local"))["mode"] == "local"
    assert validate_config(config("remote"))["mode"] == "remote"


def test_remote_mode_rejects_incomplete_databricks_settings(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABRICKS_HOST",
        "https://adb-1234567890123456.7.azuredatabricks.net",
    )
    monkeypatch.setenv("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "riverside-search")
    values = config("remote")
    values["remote"]["index_name"] = ""
    with pytest.raises(ConfigError, match="index_name"):
        validate_config(values)


def test_remote_mode_requires_resolved_databricks_environment_references() -> None:
    values = config("remote")
    values["remote"]["workspace_url"] = "${DATABRICKS_HOST}"
    values["remote"]["vector_search_endpoint"] = "${RIVERSIDE_VECTOR_SEARCH_ENDPOINT}"

    with pytest.raises(ConfigError, match="DATABRICKS_HOST.*RIVERSIDE_VECTOR_SEARCH_ENDPOINT"):
        validate_config(values)


def test_remote_mode_rejects_literal_databricks_deployment_values(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABRICKS_HOST",
        "https://adb-1234567890123456.7.azuredatabricks.net",
    )
    monkeypatch.setenv("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "environment-endpoint")

    with pytest.raises(ConfigError, match="DATABRICKS_HOST.*RIVERSIDE_VECTOR_SEARCH_ENDPOINT"):
        validate_config(config("remote"))


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("DATABRICKS_HOST", "https://replace-with-workspace.azuredatabricks.net"),
        ("DATABRICKS_HOST", "https://adb-example.invalid"),
        ("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "${RIVERSIDE_VECTOR_SEARCH_ENDPOINT}"),
        ("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "replace-with-vector-search"),
    ],
)
def test_remote_mode_rejects_placeholder_environment_values(
    monkeypatch, environment_name: str, value: str
) -> None:
    monkeypatch.setenv(
        "DATABRICKS_HOST",
        "https://adb-1234567890123456.7.azuredatabricks.net",
    )
    monkeypatch.setenv("RIVERSIDE_VECTOR_SEARCH_ENDPOINT", "riverside-search")
    monkeypatch.setenv(environment_name, value)
    values = config("remote")
    values["remote"]["workspace_url"] = os.getenv("DATABRICKS_HOST")
    values["remote"]["vector_search_endpoint"] = os.getenv(
        "RIVERSIDE_VECTOR_SEARCH_ENDPOINT"
    )

    with pytest.raises(ConfigError, match=environment_name):
        validate_config(values)


def test_azure_endpoint_requires_https_and_entra_scope() -> None:
    values = config()
    generation = values["serving"]["generation"]
    generation["provider"] = "azure_endpoint"
    generation["endpoint"] = {
        "provider": "apim",
        "url": "http://gateway.example.invalid",
        "route": "/v1/chat/completions",
        "token_scope": "https://gateway.example.invalid/.default",
    }
    with pytest.raises(ConfigError, match="HTTPS"):
        validate_config(values)


def test_azure_endpoint_rejects_unresolved_and_documentation_only_values() -> None:
    values = config()
    generation = values["serving"]["generation"]
    generation["provider"] = "azure_endpoint"
    generation["endpoint"] = {
        "provider": "apim",
        "url": "https://gateway.example.invalid",
        "route": "/v1/chat/completions",
        "token_scope": "${RIVERSIDE_TOKEN_SCOPE}",
    }

    with pytest.raises(ConfigError, match="placeholder"):
        validate_config(values)


def test_yaml_credentials_are_rejected() -> None:
    values = config()
    values["serving"]["generation"]["endpoint"]["api_key"] = "not-allowed"
    with pytest.raises(ConfigError, match="Credentials"):
        validate_config(values)
