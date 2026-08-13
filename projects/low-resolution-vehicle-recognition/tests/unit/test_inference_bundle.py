import hashlib
import json
from pathlib import Path

import pytest

from roadid.inference.bundle import BundleError, load_model_bundle


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path) -> Path:
    dataset_hash = "1" * 64
    validation_hash = "2" * 64
    test_hash = "3" * 64
    _write_json(root / "classifier/config.json", {"architectures": ["TrackLensResNet"]})
    (root / "classifier/model.safetensors").write_bytes(b"fixture-weights")
    _write_json(root / "classifier/preprocessor_config.json", {"size": 224})
    _write_json(
        root / "labels.json",
        {
            "body_types": ["suv"],
            "makes": ["toyota"],
            "model_families": ["rav4"],
            "model_to_make": {"rav4": "toyota"},
        },
    )
    _write_json(
        root / "calibration.json",
        {
            "method": "temperature_scaling",
            "dataset_manifest_sha256": dataset_hash,
            "validation_split_sha256": validation_hash,
            "test_split_sha256": test_hash,
        },
    )
    _write_json(
        root / "thresholds.json",
        {
            "body_type": 0.7,
            "make": 0.75,
            "model_family": 0.8,
            "dataset_manifest_sha256": dataset_hash,
            "validation_split_sha256": validation_hash,
        },
    )
    _write_json(root / "training-manifest.json", {"dataset_manifest_sha256": dataset_hash})
    _write_json(root / "evaluation-report.json", {"status": "fixture"})
    _write_json(
        root / "bundle-manifest.json",
        {
            "schema_version": 1,
            "model_version": "roadid-fixture-v1",
            "base_model": "microsoft/resnet-50",
            "base_model_revision": "fixture",
            "dataset_manifest_sha256": dataset_hash,
            "label_hierarchy_sha256": _sha(root / "labels.json"),
            "classifier_sha256": _sha(root / "classifier/model.safetensors"),
            "image_processor_sha256": _sha(root / "classifier/preprocessor_config.json"),
            "calibration_sha256": _sha(root / "calibration.json"),
            "thresholds_sha256": _sha(root / "thresholds.json"),
            "metrics": {},
            "created_at": "2026-08-13T12:00:00Z",
            "git_revision": "fixture",
        },
    )
    return root


def test_model_bundle_loads_only_after_hash_and_identity_verification(tmp_path: Path) -> None:
    bundle = load_model_bundle(_bundle(tmp_path / "bundle"))

    assert bundle.model_version == "roadid-fixture-v1"
    assert bundle.model_to_make == {"rav4": "toyota"}
    assert bundle.calibration.model_threshold == 0.8


def test_model_bundle_rejects_corrupt_artifact(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "classifier/model.safetensors").write_bytes(b"corrupt")

    with pytest.raises(BundleError, match="hash mismatch"):
        load_model_bundle(root)


def test_training_manifest_is_the_canonical_manifest_when_no_alias_exists(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    (root / "training-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "bundle-manifest.json").unlink()

    assert load_model_bundle(root).model_version == "roadid-fixture-v1"
