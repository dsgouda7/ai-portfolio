from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[3]
AZUREML_ROOT = PROJECT_ROOT / "azureml"


def _text(relative_path: str) -> str:
    return (AZUREML_ROOT / relative_path).read_text(encoding="utf-8")


def _project_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(path: Path) -> str:
    entries = [
        f"{candidate.relative_to(path).as_posix()}\t{_file_sha256(candidate)}"
        for candidate in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    return hashlib.sha256(("\n".join(entries) + "\n").encode()).hexdigest()


def _write_model_package(path: Path, release_id: str) -> None:
    artifacts = {
        "adapter_model.safetensors": b"fixture-adapter",
        "tokenizer.json": b"fixture-tokenizer",
        "training-manifest.json": b"fixture-training",
        "evaluation-report.json": b"fixture-evaluation",
    }
    path.mkdir(parents=True)
    for name, content in artifacts.items():
        (path / name).write_bytes(content)
    (path / "base-model").mkdir()
    (path / "base-model" / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "release_id": release_id,
        "adapter": {
            "uri": "repo://adapter_model.safetensors",
            "digest": {"algorithm": "sha256", "value": _file_sha256(path / "adapter_model.safetensors")},
        },
        "tokenizer": {
            "uri": "repo://tokenizer.json",
            "digest": {"algorithm": "sha256", "value": _file_sha256(path / "tokenizer.json")},
        },
        "training_provenance": {
            "manifest_uri": "repo://training-manifest.json",
            "manifest_digest": {"algorithm": "sha256", "value": _file_sha256(path / "training-manifest.json")},
        },
        "evaluation": {
            "report_uri": "repo://evaluation-report.json",
            "report_digest": {"algorithm": "sha256", "value": _file_sha256(path / "evaluation-report.json")},
        },
    }
    (path / "model-release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _pwsh() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is required for the offline materialization contract")
    return executable


def test_endpoint_uses_entra_identity_and_defers_traffic_assignment() -> None:
    endpoint = _text("endpoint.yml")

    assert "name: RIVERSIDE-ENDPOINT-NAME-REQUIRED" in endpoint
    assert "auth_mode: aad_token" in endpoint
    assert "type: system_assigned" in endpoint
    assert "public_network_access: disabled" in endpoint
    assert "environment: __RIVERSIDE_ENVIRONMENT__" in endpoint
    assert "region: __RIVERSIDE_REGION__" in endpoint
    assert "traffic:" not in endpoint


def test_blue_and_green_bind_materialized_assets_and_release_digests() -> None:
    required_digest_tags = {
        "release_manifest_sha256",
        "model_package_sha256",
        "code_package_sha256",
        "environment_image_sha256",
        "environment_conda_sha256",
        "environment_template_sha256",
    }
    for slot in ("blue", "green"):
        deployment = _text(f"deployments/{slot}.yml")
        upper_slot = slot.upper()
        assert f"name: __RIVERSIDE_{upper_slot}_SLOT_NAME__" in deployment
        assert (
            f"model: azureml:__RIVERSIDE_{upper_slot}_MODEL_NAME__:"
            f"__RIVERSIDE_{upper_slot}_MODEL_VERSION__"
        ) in deployment
        assert (
            "environment: azureml:__RIVERSIDE_ENVIRONMENT_ASSET_NAME__:"
            "__RIVERSIDE_ENVIRONMENT_ASSET_VERSION__"
        ) in deployment
        assert "endpoint_name: __RIVERSIDE_ENDPOINT_NAME__" in deployment
        assert "RIVERSIDE_REGION: __RIVERSIDE_REGION__" in deployment
        assert "app_insights_enabled: true" in deployment
        assert "max_concurrent_requests_per_instance: 1" in deployment
        assert 'WORKER_COUNT: "1"' in deployment
        assert "readiness_probe:" in deployment
        assert "RIVERSIDE_RELEASE_MANIFEST: model-release-manifest.json" in deployment
        assert "RIVERSIDE_STREAMING_MODE: buffered-sse" in deployment
        for tag in required_digest_tags:
            assert f"  {tag}:" in deployment
        assert not re.search(r"(?im)^\s*(client_secret|api_key|password|connection_string)\s*:", deployment)


def test_environment_uses_an_immutable_base_image() -> None:
    environment = _text("environment/environment.yml")

    assert "image: __RIVERSIDE_BASE_IMAGE_BY_DIGEST__" in environment
    assert "image_sha256: __RIVERSIDE_ENVIRONMENT_IMAGE_SHA256__" in environment
    assert not re.search(r"(?m)^image:\s+[^_].*:[^@\s]+$", environment)


def test_materializer_fails_closed_and_binds_all_deployment_inputs() -> None:
    materializer = _project_text("scripts/Materialize-AzureML.ps1")

    assert "base_image_by_digest must be pinned by sha256 digest" in materializer
    assert "Assert-AzureMlAssetName" in materializer
    assert "Assert-AzureMlAssetVersion" in materializer
    assert "release_manifest_sha256" in materializer
    assert "environment_image_sha256" in materializer
    assert "RIVERSIDE_BLUE_RELEASE_MANIFEST_SHA256" in materializer
    assert "RIVERSIDE_GREEN_RELEASE_MANIFEST_SHA256" in materializer
    assert "RIVERSIDE_ENVIRONMENT_IMAGE_SHA256" in materializer
    assert "RIVERSIDE_ENVIRONMENT_CONDA_SHA256" in materializer


def test_registration_validates_returned_asset_ids_and_all_digest_tags() -> None:
    registration = _project_text("scripts/Register-AzureMLAssets.ps1")

    assert "Assert-RegisteredAssetId" in registration
    assert "environment_image_sha256" in registration
    assert "environment_template_sha256" in registration
    assert "conda_sha256" in registration
    assert "release_manifest_sha256" in registration
    assert "package_sha256" in registration
    assert "registration-manifest.json" in registration
    assert "models = $registeredModels" in registration
    assert "$registrationAssets[$model.name]" not in registration


def test_orchestrator_images_are_required_by_digest() -> None:
    dockerfile = _project_text("infra/orchestrator.Dockerfile")
    container_host = _project_text("infra/modules/container-app-host.bicep")

    assert "ARG PYTHON_BASE_IMAGE" in dockerfile
    assert "FROM ${PYTHON_BASE_IMAGE}" in dockerfile
    assert not re.search(r"(?m)^FROM\s+[^$].*:[^@\s]+$", dockerfile)
    assert "param orchestratorImage string" in container_host
    assert "@sha256:" in container_host
    assert "orchestratorImageIsImmutable" in container_host
    assert "containerapps-helloworld:latest" not in container_host


def test_streaming_contract_discloses_buffered_sse_behavior() -> None:
    scoring = _text("score.py")
    readme = _text("README.md")

    assert 'response.headers["X-Riverside-Streaming-Mode"] = "buffered-sse"' in scoring
    assert "generation completes before SSE events are emitted" in readme
    assert "X-Accel-Buffering" in readme


def test_materialization_and_registration_dry_run_against_fixture(tmp_path: Path) -> None:
    code_package = tmp_path / "code"
    blue_package = tmp_path / "blue-model"
    green_package = tmp_path / "green-model"
    code_package.mkdir()
    (code_package / "score.py").write_text("def run(value):\n    return value\n", encoding="utf-8")
    _write_model_package(blue_package, "fixture-blue-release")
    _write_model_package(green_package, "fixture-green-release")

    config = {
        "subscription_id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "00000000-0000-0000-0000-000000000002",
        "resource_group": "fixture-rg",
        "workspace_name": "fixture-workspace",
        "environment": "staging",
        "region": "uksouth",
        "endpoint_name": "riverside-fixture",
        "blue_slot_name": "blue",
        "green_slot_name": "green",
        "blue_deployment_name": "riverside-fixture-blue",
        "green_deployment_name": "riverside-fixture-green",
        "blue_model_name": "riverside-model",
        "blue_model_version": "2026.08.05-blue",
        "green_model_name": "riverside-model",
        "green_model_version": "2026.08.05-green",
        "environment_asset_name": "riverside-runtime",
        "environment_asset_version": "2026.08.05",
        "base_image_by_digest": f"mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference@sha256:{'a' * 64}",
        "index_version": "index-2026.08.05",
        "deployed_at": "2026-08-05T12:00:00Z",
        "application_deadline_seconds": 100,
        "paths": {
            "code_package": str(code_package),
            "blue_model_package": str(blue_package),
            "green_model_package": str(green_package),
        },
        "expected_sha256": {
            "code_package": _directory_sha256(code_package),
            "blue_model_package": _directory_sha256(blue_package),
            "green_model_package": _directory_sha256(green_package),
            "conda_file": _file_sha256(AZUREML_ROOT / "environment/conda.yml"),
            "environment_template": _file_sha256(AZUREML_ROOT / "environment/environment.yml"),
        },
    }
    config_path = tmp_path / "azureml.json"
    output_path = tmp_path / "materialized"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    materialized = subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(PROJECT_ROOT / "scripts/Materialize-AzureML.ps1"),
            "-ConfigPath",
            str(config_path),
            "-OutputDirectory",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert materialized.returncode == 0, materialized.stdout + materialized.stderr

    blue = (output_path / "deployments/blue.yml").read_text(encoding="utf-8")
    endpoint = (output_path / "endpoint.yml").read_text(encoding="utf-8")
    manifest = json.loads((output_path / "materialization-manifest.json").read_text(encoding="utf-8"))
    assert "name: riverside-fixture" in endpoint
    assert "region: uksouth" in endpoint
    assert "RIVERSIDE_REGION: uksouth" in blue
    assert "release_manifest_sha256:" in blue
    assert "environment_image_sha256: " + "a" * 64 in blue
    assert manifest["region"] == "uksouth"
    assert manifest["endpoint_name"] == "riverside-fixture"
    assert manifest["packages"]["blue_model"]["release_manifest_sha256"] == _file_sha256(
        blue_package / "model-release-manifest.json"
    )

    registration = subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(PROJECT_ROOT / "scripts/Register-AzureMLAssets.ps1"),
            "-ConfigPath",
            str(config_path),
            "-MaterializedDirectory",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert registration.returncode == 0, registration.stdout + registration.stderr
    assert "DRY RUN: local digests are valid" in registration.stdout
    assert "az ml environment create" in registration.stdout
    assert "az ml model create" in registration.stdout

    config["base_image_by_digest"] = "python:3.12-slim"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    rejected = subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(PROJECT_ROOT / "scripts/Materialize-AzureML.ps1"),
            "-ConfigPath",
            str(config_path),
            "-OutputDirectory",
            str(tmp_path / "rejected"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "base_image_by_digest must be pinned by sha256 digest" in rejected.stderr


def test_rollout_profiles_are_bounded_and_sum_to_100() -> None:
    expected = {
        "rollout/blue-100.yml": (100, 0),
        "rollout/green-canary-10.yml": (90, 10),
        "rollout/green-100.yml": (0, 100),
    }
    for path, traffic in expected.items():
        profile = _text(path)
        blue = int(re.search(r"(?m)^\s*__RIVERSIDE_BLUE_SLOT_NAME__:\s*(\d+)$", profile).group(1))
        green = int(re.search(r"(?m)^\s*__RIVERSIDE_GREEN_SLOT_NAME__:\s*(\d+)$", profile).group(1))
        assert (blue, green) == traffic
        assert blue + green == 100


def test_sample_requests_match_the_v1_typed_contract() -> None:
    for path, stream in (
        ("samples/chat-request.json", False),
        ("samples/chat-request-stream.json", True),
    ):
        raw = _text(path)
        request = json.loads(raw)
        assert request["stream"] is stream
        assert request["model"] == "riverside-editor"
        assert isinstance(request["messages"], list) and request["messages"]
        assert all(message["role"] in {"system", "user", "assistant"} for message in request["messages"])
        assert 1 <= request["max_tokens"] <= 512
