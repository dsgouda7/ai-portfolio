import csv

import pytest
from PIL import Image

from roadid.training.corpus_intake import build_corpus_manifest
from roadid.training.datasets import LocalManifestAdapter, assign_splits


def test_mio_tcd_intake_is_terms_gated_and_camera_owned(tmp_path) -> None:
    root = tmp_path / "mio"
    root.mkdir()
    Image.new("RGB", (32, 24), "red").save(root / "car.jpg")
    annotations = tmp_path / "mio.csv"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "body_type", "camera_id"])
        writer.writeheader()
        writer.writerow({"image_path": "car.jpg", "body_type": "car", "camera_id": "cam-7"})

    with pytest.raises(PermissionError, match="non-commercial"):
        build_corpus_manifest("mio-tcd", root, annotations, tmp_path / "manifest.json", accept_noncommercial_terms=False)

    manifest = build_corpus_manifest(
        "mio-tcd",
        root,
        annotations,
        tmp_path / "manifest.json",
        accept_noncommercial_terms=True,
    )
    items = LocalManifestAdapter(root, manifest).inventory()
    assigned = assign_splits(items, seed=12)

    assert items[0].source_license == "CC-BY-NC-SA-4.0"
    assert items[0].ownership_key == "mio-tcd:cam-7"
    assert assigned[0].split in {"train", "validation", "test"}


def test_compcars_intake_requires_complete_hierarchy_and_identity(tmp_path) -> None:
    root = tmp_path / "compcars"
    root.mkdir()
    Image.new("RGB", (32, 24), "blue").save(root / "vehicle.jpg")
    annotations = tmp_path / "compcars.csv"
    annotations.write_text(
        "image_path,body_type,make,model_family,identity_id\n"
        "vehicle.jpg,suv,toyota,,vehicle-1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_family"):
        build_corpus_manifest(
            "compcars",
            root,
            annotations,
            tmp_path / "manifest.json",
            accept_noncommercial_terms=True,
        )
