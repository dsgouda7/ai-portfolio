"""Fail-closed loading for versioned TrackLens model bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roadid.contracts import CalibrationContract

MANIFEST_NAMES = ("bundle-manifest.json", "manifest.json")
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_version",
        "base_model",
        "base_model_revision",
        "dataset_manifest_sha256",
        "label_hierarchy_sha256",
        "classifier_sha256",
        "image_processor_sha256",
        "calibration_sha256",
        "thresholds_sha256",
        "metrics",
        "created_at",
        "git_revision",
    }
)
HASHED_ARTIFACTS = {
    "label_hierarchy_sha256": Path("labels.json"),
    "classifier_sha256": Path("classifier/model.safetensors"),
    "image_processor_sha256": Path("classifier/preprocessor_config.json"),
    "calibration_sha256": Path("calibration.json"),
    "thresholds_sha256": Path("thresholds.json"),
}
REQUIRED_ARTIFACTS = frozenset(
    {
        Path("classifier/config.json"),
        Path("training-manifest.json"),
        Path("evaluation-report.json"),
        *HASHED_ARTIFACTS.values(),
    }
)


class BundleError(RuntimeError):
    """Raised when a model bundle cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    root: Path
    manifest: Mapping[str, Any]
    labels: Mapping[str, Any]
    calibration: CalibrationContract
    thresholds: Mapping[str, float]
    temperatures: Mapping[str, float]

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def classifier_path(self) -> Path:
        return self.root / "classifier"

    @property
    def model_to_make(self) -> Mapping[str, str]:
        hierarchy = self.labels.get("model_to_make", {})
        return hierarchy if isinstance(hierarchy, dict) else {}


class ModelBundleLoader:
    """Verify all bundle identities before returning parsed metadata."""

    def __init__(self, expected_base_model: str = "microsoft/resnet-50") -> None:
        self.expected_base_model = expected_base_model

    def load(self, root: str | Path) -> ModelBundle:
        bundle_root = Path(root).resolve()
        if not bundle_root.is_dir():
            raise BundleError(f"model bundle directory not found: {bundle_root}")

        manifest_path = self._manifest_path(bundle_root)
        manifest = _read_json_object(manifest_path)
        missing_fields = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
        if missing_fields:
            raise BundleError(f"bundle manifest missing fields: {', '.join(missing_fields)}")
        if manifest["schema_version"] != 1:
            raise BundleError(f"unsupported bundle schema version: {manifest['schema_version']}")
        if manifest["base_model"] != self.expected_base_model:
            raise BundleError(
                f"unexpected base model: {manifest['base_model']}; "
                f"expected {self.expected_base_model}"
            )
        _require_sha256("dataset_manifest_sha256", manifest["dataset_manifest_sha256"])

        for relative_path in sorted(REQUIRED_ARTIFACTS, key=str):
            artifact = bundle_root / relative_path
            if not artifact.is_file():
                raise BundleError(f"required bundle artifact missing: {relative_path.as_posix()}")

        for field, relative_path in HASHED_ARTIFACTS.items():
            expected = _require_sha256(field, manifest[field])
            actual = sha256_file(bundle_root / relative_path)
            if actual != expected:
                raise BundleError(
                    f"hash mismatch for {relative_path.as_posix()}: "
                    f"expected {expected}, got {actual}"
                )

        for json_path in (
            "classifier/config.json",
            "classifier/preprocessor_config.json",
            "training-manifest.json",
            "evaluation-report.json",
        ):
            _read_json_object(bundle_root / json_path)
        if "evaluation_report_sha256" in manifest:
            expected_report = _require_sha256(
                "evaluation_report_sha256", manifest["evaluation_report_sha256"]
            )
            actual_report = sha256_file(bundle_root / "evaluation-report.json")
            if actual_report != expected_report:
                raise BundleError("hash mismatch for evaluation-report.json")

        labels = _read_json_object(bundle_root / "labels.json")
        _validate_labels(labels)
        calibration_payload = _read_json_object(bundle_root / "calibration.json")
        threshold_payload = _read_json_object(bundle_root / "thresholds.json")
        calibration = _load_calibration(calibration_payload, threshold_payload)
        if calibration.dataset_manifest_sha256 != manifest["dataset_manifest_sha256"]:
            raise BundleError("calibration dataset identity does not match bundle manifest")
        _validate_threshold_identities(calibration_payload, threshold_payload)

        return ModelBundle(
            root=bundle_root,
            manifest=manifest,
            labels=labels,
            calibration=calibration,
            thresholds={
                "body_type": calibration.body_threshold,
                "make": calibration.make_threshold,
                "model_family": calibration.model_threshold,
            },
            temperatures=_load_temperatures(threshold_payload),
        )

    @staticmethod
    def _manifest_path(root: Path) -> Path:
        candidates = [root / name for name in MANIFEST_NAMES if (root / name).is_file()]
        if len(candidates) > 1:
            names = ", ".join(MANIFEST_NAMES)
            raise BundleError(f"bundle contains multiple manifests named {names}")
        if candidates:
            return candidates[0]
        training_manifest = root / "training-manifest.json"
        if training_manifest.is_file():
            return training_manifest
        names = ", ".join((*MANIFEST_NAMES, "training-manifest.json"))
        raise BundleError(f"bundle manifest not found; expected one of {names}")


def load_model_bundle(root: str | Path) -> ModelBundle:
    return ModelBundleLoader().load(root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid JSON artifact {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise BundleError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BundleError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise BundleError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def _validate_labels(labels: Mapping[str, Any]) -> None:
    for key in ("body_types", "makes", "model_families", "model_to_make"):
        if key not in labels:
            raise BundleError(f"labels.json missing hierarchy field: {key}")
    hierarchy_levels = ("body_types", "makes", "model_families")
    if not all(isinstance(labels[key], list) and labels[key] for key in hierarchy_levels):
        raise BundleError("label hierarchy levels must be non-empty lists")
    model_to_make = labels["model_to_make"]
    if not isinstance(model_to_make, dict):
        raise BundleError("model_to_make must be an object")
    models = set(labels["model_families"])
    makes = set(labels["makes"])
    invalid_parent = any(parent not in makes for parent in model_to_make.values())
    if set(model_to_make) != models or invalid_parent:
        raise BundleError("every model family must map to one known make")
    make_to_body = labels.get("make_to_body")
    if make_to_body is not None:
        if not isinstance(make_to_body, dict):
            raise BundleError("make_to_body must be an object")
        if set(make_to_body) != makes or any(
            parent not in set(labels["body_types"]) for parent in make_to_body.values()
        ):
            raise BundleError("every make must map to one known body type")


def _load_calibration(
    calibration: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> CalibrationContract:
    try:
        return CalibrationContract(
            method=str(calibration["method"]),
            dataset_manifest_sha256=str(calibration["dataset_manifest_sha256"]),
            validation_split_sha256=str(calibration["validation_split_sha256"]),
            test_split_sha256=str(calibration["test_split_sha256"]),
            body_threshold=float(thresholds["body_type"]),
            make_threshold=float(thresholds["make"]),
            model_threshold=float(thresholds["model_family"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BundleError(f"invalid calibration contract: {error}") from error


def _validate_threshold_identities(
    calibration: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> None:
    for field in ("dataset_manifest_sha256", "validation_split_sha256"):
        if field in thresholds and thresholds[field] != calibration.get(field):
            raise BundleError(f"threshold {field} does not match calibration identity")


def _load_temperatures(thresholds: Mapping[str, Any]) -> dict[str, float]:
    payload = thresholds.get("temperatures", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise BundleError("threshold temperatures must be an object")
    result = {}
    for level in ("body_type", "make", "model_family"):
        value = float(payload.get(level, 1.0))
        if not np.isfinite(value) or value <= 0:
            raise BundleError(f"temperature for {level} must be positive and finite")
        result[level] = value
    return result
