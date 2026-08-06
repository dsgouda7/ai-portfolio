from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.config import load_platform_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"


def _environment(environment: str) -> dict[str, str]:
    return {
        "RIVERSIDE_RELEASE_MANIFEST_URI": f"https://artifacts.example.invalid/{environment}/release.json",
        "RIVERSIDE_GATEWAY_BASE_URL": f"https://{environment}.gateway.example.invalid",
        "RIVERSIDE_SERVING_ENDPOINT_NAME": f"riverside-{environment}",
        "RIVERSIDE_BLUE_DEPLOYMENT_NAME": f"riverside-{environment}-blue",
        "RIVERSIDE_GREEN_DEPLOYMENT_NAME": f"riverside-{environment}-green",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"https://{environment}.otel.example.invalid",
        "RIVERSIDE_EVALUATION_REPORT_URI": f"https://artifacts.example.invalid/{environment}/report.json",
    }


def _schema_registry() -> Registry[Any]:
    resources = []
    for path in (PROJECT_ROOT / "contracts" / "v1").glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    config_schema = json.loads((CONFIG_ROOT / "schema.json").read_text(encoding="utf-8"))
    resources.append((config_schema["$id"], Resource.from_contents(config_schema)))
    return Registry().with_resources(resources)


@pytest.mark.integration
@pytest.mark.parametrize("environment", ["dev", "staging", "production"])
def test_profiles_are_schema_valid_and_resolve_bicep_names(environment: str) -> None:
    path = CONFIG_ROOT / f"{environment}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = json.loads((CONFIG_ROOT / "schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )

    assert list(validator.iter_errors(raw)) == []
    profile = load_platform_config(path, environ=_environment(environment))
    assert profile.environment == environment
    assert profile.serving.endpoint_name == f"riverside-{environment}"
    assert profile.serving.blue_deployment == f"riverside-{environment}-blue"
    assert profile.serving.green_deployment == f"riverside-{environment}-green"
    assert profile.serving.request_timeout_seconds < profile.gateway.timeout_seconds <= 120


@pytest.mark.integration
def test_no_local_environment_profile_exists() -> None:
    assert {path.name for path in CONFIG_ROOT.glob("*.yaml")} == {
        "dev.yaml",
        "staging.yaml",
        "production.yaml",
    }


@pytest.mark.integration
def test_production_region_matches_production_infrastructure_parameters() -> None:
    profile = load_platform_config(
        CONFIG_ROOT / "production.yaml",
        environ=_environment("production"),
    )
    parameters = json.loads(
        (CONFIG_ROOT / "production.parameters.example.json").read_text(encoding="utf-8")
    )

    assert profile.region == parameters["parameters"]["location"]["value"] == "uksouth"
