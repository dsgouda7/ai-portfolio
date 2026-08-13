import hashlib

import pytest
import torch

from roadid.contracts import CalibrationContract
from roadid.training.hierarchy import LabelHierarchy
from roadid.training.model import build_tiny_classifier
from roadid.training.packaging import load_tiny_bundle, package_bundle, verify_bundle


def test_bundle_is_atomic_hash_verified_and_reloads_logits(tmp_path) -> None:
    class_counts = {"body_type": 2, "make": 2, "model_family": 2}
    model = build_tiny_classifier(class_counts, feature_dim=12, seed=9).eval()
    probe = torch.linspace(0, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(probe)
    hierarchy = LabelHierarchy(
        body_types=("sedan", "suv"),
        makes=("honda", "toyota"),
        model_families=("civic", "rav4"),
        make_to_body={"honda": "sedan", "toyota": "suv"},
        model_to_make={"civic": "honda", "rav4": "toyota"},
    )
    dataset_hash = hashlib.sha256(b"dataset").hexdigest()
    calibration = CalibrationContract(
        "temperature_scaling",
        dataset_hash,
        hashlib.sha256(b"validation").hexdigest(),
        hashlib.sha256(b"test").hexdigest(),
        0.6,
        0.7,
        0.8,
    )
    destination = tmp_path / "models" / "roadid-test"
    package_bundle(
        destination,
        model=model,
        model_version="roadid-test",
        model_config={
            "architecture": "tiny_cnn",
            "class_counts": class_counts,
            "feature_dim": 12,
            "seed": 9,
            "image_size": 32,
        },
        hierarchy=hierarchy,
        calibration=calibration,
        temperatures={level: 1.0 for level in class_counts},
        evaluation_report={"metrics": {"fixture": 1.0}},
        base_model="tiny-local",
        base_model_revision="fixture",
        dataset_manifest_sha256=dataset_hash,
    )
    loaded = load_tiny_bundle(destination)
    with torch.no_grad():
        actual = loaded(probe)

    verify_bundle(destination)
    for level in class_counts:
        torch.testing.assert_close(actual[level], expected[level])
    assert not list(destination.parent.glob("*.staging-*"))
    with pytest.raises(FileExistsError):
        package_bundle(
            destination,
            model=model,
            model_version="roadid-test",
            model_config={"architecture": "tiny_cnn"},
            hierarchy=hierarchy,
            calibration=calibration,
            temperatures={},
            evaluation_report={},
            base_model="tiny-local",
            base_model_revision="fixture",
            dataset_manifest_sha256=dataset_hash,
        )


def test_bundle_hash_tampering_fails_closed(tmp_path) -> None:
    path = tmp_path / "broken"
    path.mkdir()
    with pytest.raises(ValueError, match="missing required"):
        verify_bundle(path)
