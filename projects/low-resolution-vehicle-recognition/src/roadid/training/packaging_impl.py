"""Atomic RoadID bundle implementation used by the public packaging module."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roadid.contracts import CalibrationContract
from roadid.training.datasets import sha256_file
from roadid.training.hierarchy import LabelHierarchy
from roadid.training.model import build_tiny_classifier

REQUIRED_FILES = (
    "classifier/config.json",
    "classifier/model.safetensors",
    "classifier/preprocessor_config.json",
    "labels.json",
    "calibration.json",
    "thresholds.json",
    "training-manifest.json",
    "evaluation-report.json",
)


def package_bundle(
    destination: Path,
    *,
    model: Any,
    model_version: str,
    model_config: Mapping[str, object],
    hierarchy: LabelHierarchy,
    calibration: CalibrationContract,
    temperatures: Mapping[str, float],
    evaluation_report: Mapping[str, object],
    base_model: str,
    base_model_revision: str,
    dataset_manifest_sha256: str,
    git_revision: str = "unknown",
) -> Path:
    if destination.exists():
        raise FileExistsError(f"bundle version already exists: {destination}")
    if dataset_manifest_sha256 != calibration.dataset_manifest_sha256:
        raise ValueError("calibration and bundle dataset identities differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        classifier = staging / "classifier"
        classifier.mkdir()
        serialization = _save_weights(model, classifier / "model.safetensors")
        _write_json(
            classifier / "config.json",
            {**model_config, "serialization": serialization, "model_version": model_version},
        )
        _write_json(
            classifier / "preprocessor_config.json",
            {
                "do_resize": True,
                "size": int(model_config.get("image_size", 224)),
                "do_normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
        )
        _write_json(staging / "labels.json", hierarchy.to_dict())
        _write_json(staging / "calibration.json", asdict(calibration))
        _write_json(
            staging / "thresholds.json",
            {
                "body_type": calibration.body_threshold,
                "make": calibration.make_threshold,
                "model_family": calibration.model_threshold,
                "temperatures": dict(temperatures),
                "dataset_manifest_sha256": calibration.dataset_manifest_sha256,
                "validation_split_sha256": calibration.validation_split_sha256,
            },
        )
        _write_json(staging / "evaluation-report.json", evaluation_report)
        manifest = {
            "schema_version": 1,
            "model_version": model_version,
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "label_hierarchy_sha256": sha256_file(staging / "labels.json"),
            "classifier_sha256": sha256_file(classifier / "model.safetensors"),
            "image_processor_sha256": sha256_file(classifier / "preprocessor_config.json"),
            "calibration_sha256": sha256_file(staging / "calibration.json"),
            "thresholds_sha256": sha256_file(staging / "thresholds.json"),
            "evaluation_report_sha256": sha256_file(staging / "evaluation-report.json"),
            "metrics": evaluation_report.get("metrics", {}),
            "created_at": datetime.now(UTC).isoformat(),
            "git_revision": git_revision,
        }
        _write_json(staging / "training-manifest.json", manifest)
        verify_bundle(staging)
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle(path: Path) -> dict[str, object]:
    missing = [relative for relative in REQUIRED_FILES if not (path / relative).is_file()]
    if missing:
        raise ValueError(f"bundle is missing required files: {missing}")
    manifest = _read_json(path / "training-manifest.json")
    expected = {
        "label_hierarchy_sha256": "labels.json",
        "classifier_sha256": "classifier/model.safetensors",
        "image_processor_sha256": "classifier/preprocessor_config.json",
        "calibration_sha256": "calibration.json",
        "thresholds_sha256": "thresholds.json",
        "evaluation_report_sha256": "evaluation-report.json",
    }
    for field, relative in expected.items():
        if manifest.get(field) != sha256_file(path / relative):
            raise ValueError(f"bundle hash mismatch: {relative}")
    calibration = _read_json(path / "calibration.json")
    thresholds = _read_json(path / "thresholds.json")
    if calibration["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]:
        raise ValueError("calibration dataset identity does not match bundle manifest")
    if thresholds["validation_split_sha256"] != calibration["validation_split_sha256"]:
        raise ValueError("threshold validation identity does not match calibration")
    return manifest


def load_tiny_bundle(path: Path) -> Any:
    verify_bundle(path)
    config = _read_json(path / "classifier/config.json")
    if config.get("architecture") != "tiny_cnn":
        raise ValueError("bundle does not contain a tiny local classifier")
    model = build_tiny_classifier(
        {key: int(value) for key, value in config["class_counts"].items()},
        feature_dim=int(config["feature_dim"]),
        seed=int(config["seed"]),
    )
    _load_weights(model, path / "classifier/model.safetensors", str(config["serialization"]))
    model.eval()
    return model


def _save_weights(model: Any, path: Path) -> str:
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    try:
        from safetensors.torch import save_file
    except ImportError:
        import torch

        torch.save(state, path)
        return "torch_state_dict"
    save_file(state, str(path))
    return "safetensors"


def _load_weights(model: Any, path: Path, serialization: str) -> None:
    if serialization == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("install `safetensors` to load this bundle") from error
        state = load_file(str(path), device="cpu")
    elif serialization == "torch_state_dict":
        import torch

        with path.open("rb") as handle:
            state = torch.load(handle, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unsupported classifier serialization: {serialization}")
    model.load_state_dict(state, strict=True)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
