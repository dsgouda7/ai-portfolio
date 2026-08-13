import json
from pathlib import Path

from roadid.training.config import load_training_config
from roadid.training.packaging import verify_bundle
from roadid.training.trainer import evaluate_bundle, prepare_dataset, train_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_offline_smoke_pipeline_produces_bound_verified_bundle(tmp_path) -> None:
    config = load_training_config(PROJECT_ROOT / "configs" / "train_resnet50.yaml")
    prepared = prepare_dataset(config, smoke=True, output_root=tmp_path / "dataset")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["split_before_derivation"] is True
    assert manifest["synthetic_pseudo_tracks"] is True
    assert all(manifest["split_owners"][split] for split in ("train", "validation", "test"))
    assert not (
        set(manifest["split_owners"]["train"]) & set(manifest["split_owners"]["validation"])
    )
    bundle = train_pipeline(
        config,
        prepared_root=prepared,
        smoke=True,
        output_root=tmp_path / "models",
    )
    bundle_manifest = verify_bundle(bundle)
    report = evaluate_bundle(bundle)

    assert bundle_manifest["dataset_manifest_sha256"] == manifest["dataset_manifest_sha256"]
    assert report["sealed_test_opened_after_calibration"] is True
    assert set(report["baselines"]) == {
        "random_init",
        "frozen_backbone",
        "partial_unfreeze",
    }
    assert report["metrics"]["slices"]["synthetic"]["count"] > 0
    assert report["metrics"]["slices"]["real"]["count"] == 0
