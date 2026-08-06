from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.artifact_validation import (
    ArtifactResolutionError,
    ArtifactResolver,
    CompatibilityError,
    DigestMismatchError,
    ReleaseVerifier,
    RuntimeCompatibility,
)


BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
BASE_MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> str:
    return _write(path, json.dumps(value, sort_keys=True).encode("utf-8"))


def _release_fixture(root: Path) -> Path:
    adapter_directory = root / "checkpoints" / "instruction-lora"
    adapter_digest = _write(adapter_directory / "adapter_model.safetensors", b"adapter")
    tokenizer_digest = _write(adapter_directory / "tokenizer.json", b'{"version":"1"}')
    _write_json(
        adapter_directory / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": BASE_MODEL_ID,
            "revision": None,
        },
    )
    training_digest = _write_json(
        adapter_directory / "experiment-manifest.json",
        {
            "model": {"id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
            "stage": "supervised-fine-tuning-lora",
        },
    )
    evaluation_digest = _write_json(
        root / "evaluations" / "release.json",
        {
            "contract_version": "1.0.0",
            "kind": "evaluation_release_report",
            "release_id": "riverside-editor-2026-08-05",
            "release_version": "1.0.0",
            "source_commit": SOURCE_COMMIT,
            "decision": "promote",
            "domains": {
                "generation_citation_quality": [
                    {
                        "metric_id": "citation_correctness",
                        "threshold": {"operator": "gte", "value": 0.9, "unit": "ratio"},
                        "status": "pass",
                    }
                ]
            },
        },
    )
    manifest = {
        "contract_version": "1.0.0",
        "kind": "model_release",
        "release_id": "riverside-editor-2026-08-05",
        "version": "1.0.0",
        "base_model": {"id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
        "adapter": {
            "type": "lora",
            "stage": "supervised-fine-tuning-lora",
            "uri": "repo://checkpoints/instruction-lora/adapter_model.safetensors",
            "digest": {"algorithm": "sha256", "value": adapter_digest},
        },
        "tokenizer": {
            "revision": BASE_MODEL_REVISION,
            "uri": "repo://checkpoints/instruction-lora/tokenizer.json",
            "digest": {"algorithm": "sha256", "value": tokenizer_digest},
        },
        "model_profile": "cpu-small-360m",
        "precision": "fp32",
        "training_provenance": {
            "manifest_uri": "repo://checkpoints/instruction-lora/experiment-manifest.json",
            "manifest_digest": {"algorithm": "sha256", "value": training_digest},
        },
        "evaluation": {
            "report_uri": "repo://evaluations/release.json",
            "report_digest": {"algorithm": "sha256", "value": evaluation_digest},
            "decision": "promote",
            "thresholds": [
                {
                    "domain": "generation_citation_quality",
                    "metric": "citation_correctness",
                    "operator": "gte",
                    "value": 0.9,
                }
            ],
        },
        "serving_runtime": {
            "name": "azureml-riverside-runtime",
            "version": "1.0.0",
            "interface_version": "1.0.0",
            "compatible_model_profiles": ["cpu-small-360m"],
            "supported_precisions": ["fp32"],
        },
        "created_at": "2026-08-05T12:00:00Z",
        "source_commit": SOURCE_COMMIT,
    }
    path = root / "model-release-manifest.json"
    _write_json(path, manifest)
    return path


def _runtime(**updates: str) -> RuntimeCompatibility:
    values = {
        "name": "azureml-riverside-runtime",
        "version": "1.0.0",
        "interface_version": "1.0.0",
        "model_profile": "cpu-small-360m",
        "precision": "fp32",
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "adapter_type": "lora",
    }
    values.update(updates)
    return RuntimeCompatibility.model_validate(values)


def test_release_verifies_all_artifacts_and_compatibility(tmp_path: Path) -> None:
    manifest_path = _release_fixture(tmp_path)

    release = ReleaseVerifier(ArtifactResolver(tmp_path), _runtime()).verify_file(manifest_path)

    assert release.manifest.release_id == "riverside-editor-2026-08-05"
    assert Path(release.paths.adapter).name == "adapter_model.safetensors"


def test_digest_mismatch_keeps_release_unavailable(tmp_path: Path) -> None:
    manifest_path = _release_fixture(tmp_path)
    (tmp_path / "checkpoints" / "instruction-lora" / "adapter_model.safetensors").write_bytes(
        b"tampered"
    )

    with pytest.raises(DigestMismatchError):
        ReleaseVerifier(ArtifactResolver(tmp_path), _runtime()).verify_file(manifest_path)


def test_runtime_profile_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest_path = _release_fixture(tmp_path)

    with pytest.raises(CompatibilityError):
        ReleaseVerifier(
            ArtifactResolver(tmp_path), _runtime(model_profile="gpu-small-360m")
        ).verify_file(manifest_path)


def test_repo_uri_cannot_escape_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.bin").write_bytes(b"outside")

    with pytest.raises(ArtifactResolutionError):
        ArtifactResolver(root).resolve("repo://../outside.bin")
