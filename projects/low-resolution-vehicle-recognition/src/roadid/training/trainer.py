"""Offline prepare, train, calibrate, evaluate, and package orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from roadid.training.calibration import (
    create_calibration_contract,
    fit_temperature,
    select_threshold,
    softmax,
)
from roadid.training.config import resolve_config_path
from roadid.training.datasets import (
    DatasetItem,
    assert_split_invariants,
    assign_splits,
    create_pseudo_tracks,
    sha256_file,
)
from roadid.training.degradations import degrade_file
from roadid.training.evaluation import FrameResult, evaluate_frame_and_tracks
from roadid.training.hierarchy import (
    LEVELS,
    LabelHierarchy,
    build_hierarchy,
    masked_hierarchical_loss,
)
from roadid.training.model import (
    build_hf_resnet50_classifier,
    build_tiny_classifier,
    configure_frozen_head,
    configure_partial_unfreeze,
)
from roadid.training.packaging import package_bundle, verify_bundle


def prepare_dataset(
    config: dict[str, Any], *, smoke: bool = False, output_root: Path | None = None
) -> Path:
    seed = int(config.get("seed", 2608))
    root = output_root or resolve_config_path(
        config, config["data"].get("prepared_root", "artifacts/datasets/roadid-v1")
    )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"prepared dataset directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    source_items = _smoke_source_items(root, seed) if smoke else _load_source_items(config)
    hierarchy = build_hierarchy(source_items)
    split_items = assign_splits(
        source_items,
        seed=seed,
        validation_fraction=float(config["data"].get("validation_fraction", 0.2)),
        test_fraction=float(config["data"].get("test_fraction", 0.2)),
    )
    _require_nonempty_splits(split_items)
    pseudo_length = int(config["data"].get("pseudo_track_length", 6))
    if smoke:
        pseudo_length = min(pseudo_length, 3)
    pseudo_items = create_pseudo_tracks(split_items, length=pseudo_length, seed=seed)

    image_size = int(config["data"].get("image_size", 224))
    if smoke:
        image_size = int(config["data"].get("smoke_image_size", 32))
    height_range = tuple(int(value) for value in config["data"]["apparent_height_range"])
    records: list[dict[str, object]] = []
    source_by_hash = {item.source_sha256: item for item in split_items}
    for pseudo in pseudo_items:
        source = source_by_hash[pseudo.source_sha256]
        destination = root / "images" / pseudo.split / f"{pseudo.item_id}.png"
        recipe, degraded_sha256 = degrade_file(
            Path(source.image_path),
            destination,
            seed=int(pseudo.transform_seed or seed),
            apparent_height_range=(height_range[0], height_range[1]),
            output_size=image_size,
        )
        record = pseudo.to_dict()
        record["image_path"] = destination.relative_to(root).as_posix()
        record["degraded_sha256"] = degraded_sha256
        record["degradation"] = recipe.to_dict()
        records.append(record)

    assert_split_invariants(pseudo_items)
    _write_json(root / "labels.json", hierarchy.to_dict())
    for split in ("train", "validation", "test"):
        _write_jsonl(root / f"{split}.jsonl", [row for row in records if row["split"] == split])
    split_owners = {
        split: sorted({item.ownership_key for item in split_items if item.split == split})
        for split in ("train", "validation", "test")
    }
    manifest_core = {
        "schema_version": 1,
        "dataset_version": "roadid-smoke-v1"
        if smoke
        else config["data"].get("dataset_version", "roadid-v1"),
        "seed": seed,
        "prepared_at": datetime.now(UTC).isoformat(),
        "synthetic_pseudo_tracks": True,
        "split_before_derivation": True,
        "source_count": len(source_items),
        "derived_count": len(records),
        "sources": sorted(
            {
                (
                    item.source_id,
                    item.source_version,
                    item.source_license,
                    item.source_terms_url,
                )
                for item in source_items
            }
        ),
        "split_owners": split_owners,
        "split_sha256": {
            split: sha256_file(root / f"{split}.jsonl") for split in ("train", "validation", "test")
        },
        "labels_sha256": sha256_file(root / "labels.json"),
        "class_counts": _class_counts(records),
        "excluded_items": [],
    }
    manifest_hash = _canonical_sha256(manifest_core)
    _write_json(
        root / "manifest.json",
        {**manifest_core, "dataset_manifest_sha256": manifest_hash},
    )
    return root


def train_pipeline(
    config: dict[str, Any],
    *,
    prepared_root: Path,
    smoke: bool = False,
    output_root: Path | None = None,
) -> Path:

    manifest = _read_json(prepared_root / "manifest.json")
    hierarchy = _read_hierarchy(prepared_root / "labels.json")
    rows = {
        split: _read_jsonl(prepared_root / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    class_counts = {
        "body_type": len(hierarchy.body_types),
        "make": len(hierarchy.makes),
        "model_family": len(hierarchy.model_families),
    }
    image_size = int(
        config["data"].get("smoke_image_size", 32) if smoke else config["data"]["image_size"]
    )
    feature_dim = int(config["training"].get("smoke_feature_dim", 16))
    if smoke:
        model = build_tiny_classifier(
            class_counts, feature_dim=feature_dim, seed=int(config["seed"])
        )
        base_model = "tiny-local"
        base_revision = "smoke-v1"
    else:
        base = config["base_model"]
        model = build_hf_resnet50_classifier(
            class_counts,
            model_id=str(base["model_id"]),
            revision=str(base["revision"]),
            offline_only=bool(base.get("offline_only", True)),
        )
        base_model = str(base["model_id"])
        base_revision = str(base["revision"])

    train_batch = _tensor_batch(rows["train"], prepared_root, hierarchy, image_size)
    validation_batch = _tensor_batch(rows["validation"], prepared_root, hierarchy, image_size)
    test_batch = _tensor_batch(rows["test"], prepared_root, hierarchy, image_size)
    baselines: dict[str, object] = {
        "random_init": _batch_accuracy(model, validation_batch),
    }

    configure_frozen_head(model)
    _train_epochs(
        model,
        train_batch,
        epochs=1 if smoke else int(config["training"]["head_epochs"]),
        learning_rate=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    baselines["frozen_backbone"] = _batch_accuracy(model, validation_batch)

    prefixes = (
        ("backbone.stage3",)
        if smoke
        else tuple(
            config["training"].get("partial_unfreeze_prefixes", ["backbone.encoder.stages.3"])
        )
    )
    configure_partial_unfreeze(model, backbone_prefixes=prefixes)
    _train_epochs(
        model,
        train_batch,
        epochs=1 if smoke else int(config["training"]["fine_tune_epochs"]),
        learning_rate=float(config["training"]["fine_tune_learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    baselines["partial_unfreeze"] = _batch_accuracy(model, validation_batch)

    validation_logits = _model_logits(model, validation_batch[0])
    temperatures: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    target_precision = config["calibration"]["target_precision"]
    for level in LEVELS:
        targets = validation_batch[1][level].numpy()
        logits = validation_logits[level]
        temperature = fit_temperature(logits, targets)
        probabilities = softmax(logits, temperature=temperature)
        valid = targets >= 0
        confidence = probabilities[valid].max(axis=1)
        prediction = probabilities[valid].argmax(axis=1)
        temperatures[level] = temperature
        thresholds[level] = select_threshold(
            confidence,
            prediction == targets[valid],
            target_precision=float(target_precision[level]),
        )

    calibration = create_calibration_contract(
        method=str(config["calibration"]["method"]),
        dataset_manifest_sha256=str(manifest["dataset_manifest_sha256"]),
        validation_identities=manifest["split_owners"]["validation"],
        test_identities=manifest["split_owners"]["test"],
        thresholds=thresholds,
    )
    evaluation = _evaluation_report(
        model,
        rows["test"],
        test_batch,
        baselines=baselines,
        temperatures=temperatures,
        thresholds=thresholds,
    )
    model_root = output_root or resolve_config_path(
        config, config["packaging"].get("model_root", "artifacts/models")
    )
    build_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_kind = "tiny" if smoke else "resnet50"
    dataset_id = str(manifest["dataset_manifest_sha256"])[:8]
    model_version = f"roadid-{model_kind}-{dataset_id}-{build_id}"
    destination = model_root / model_version
    package_bundle(
        destination,
        model=model,
        model_version=model_version,
        model_config={
            "architecture": "tiny_cnn" if smoke else "hf_resnet50_hierarchical",
            "class_counts": class_counts,
            "feature_dim": feature_dim,
            "seed": int(config["seed"]),
            "image_size": image_size,
        },
        hierarchy=hierarchy,
        calibration=calibration,
        temperatures=temperatures,
        evaluation_report=evaluation,
        base_model=base_model,
        base_model_revision=base_revision,
        dataset_manifest_sha256=str(manifest["dataset_manifest_sha256"]),
    )
    verify_bundle(destination)
    return destination


def evaluate_bundle(bundle: Path) -> dict[str, Any]:
    verify_bundle(bundle)
    return _read_json(bundle / "evaluation-report.json")


def _smoke_source_items(root: Path, seed: int) -> list[DatasetItem]:
    source_root = root / "source"
    source_root.mkdir()
    classes = (
        ("sedan", "honda", "civic", (190, 60, 50)),
        ("suv", "toyota", "rav4", (45, 105, 195)),
    )
    items: list[DatasetItem] = []
    for index in range(30):
        body, make, model, color = classes[index % 2]
        path = source_root / f"vehicle-{index:03d}.png"
        image = Image.new("RGB", (96, 64), (220, 220, 215))
        draw = ImageDraw.Draw(image)
        random = np.random.default_rng(seed + index)
        offset = int(random.integers(-3, 4))
        draw.rectangle((15 + offset, 24, 80 + offset, 49), fill=color)
        draw.rectangle(
            (31 + offset, 14, 68 + offset, 26), fill=tuple(max(0, value - 30) for value in color)
        )
        draw.ellipse((23 + offset, 43, 38 + offset, 58), fill=(25, 25, 25))
        draw.ellipse((60 + offset, 43, 75 + offset, 58), fill=(25, 25, 25))
        image.save(path, format="PNG")
        items.append(
            DatasetItem(
                item_id=f"vehicle-{index:03d}",
                image_path=str(path),
                source_id="roadid-smoke-fixture",
                source_version="1",
                source_terms_url="https://example.test/roadid-smoke-fixture-terms",
                source_license="generated-test-fixture-only",
                source_sha256=sha256_file(path),
                identity_id=f"vehicle-{index:03d}",
                body_type=body,
                make=make,
                model_family=model,
            )
        )
    return items


def _load_source_items(config: dict[str, Any]) -> list[DatasetItem]:
    manifest_path = resolve_config_path(config, config["data"]["manifest_path"])
    payload = _read_json(manifest_path)
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("source manifest requires an items list")
    dataset_root = resolve_config_path(config, config["data"]["dataset_root"])
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("source manifest item must be an object")
        path = Path(str(raw["image_path"]))
        path = path if path.is_absolute() else dataset_root / path
        digest = sha256_file(path)
        if raw.get("source_sha256") and raw["source_sha256"] != digest:
            raise ValueError(f"source hash mismatch: {path}")
        items.append(DatasetItem(**{**raw, "image_path": str(path), "source_sha256": digest}))
    return items


def _train_epochs(
    model: Any,
    batch: tuple[Any, Mapping[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total, _ = masked_hierarchical_loss(model(batch[0]), batch[1])
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()


def _tensor_batch(
    rows: Sequence[dict[str, Any]], root: Path, hierarchy: LabelHierarchy, image_size: int
) -> tuple[Any, dict[str, Any]]:
    import torch

    indices = hierarchy.label_to_index()
    pixels = []
    targets = {level: [] for level in LEVELS}
    for row in rows:
        with Image.open(root / row["image_path"]) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            pixels.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        for level in LEVELS:
            label = row.get(level)
            targets[level].append(indices[level].get(label, -1))
    return (
        torch.tensor(np.stack(pixels), dtype=torch.float32),
        {level: torch.tensor(values, dtype=torch.long) for level, values in targets.items()},
    )


def _model_logits(model: Any, pixels: Any) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    with torch.no_grad():
        return {level: value.cpu().numpy() for level, value in model(pixels).items()}


def _batch_accuracy(model: Any, batch: tuple[Any, Mapping[str, Any]]) -> dict[str, float | None]:
    logits = _model_logits(model, batch[0])
    result: dict[str, float | None] = {}
    for level in LEVELS:
        targets = batch[1][level].numpy()
        valid = targets >= 0
        result[level] = (
            float((logits[level][valid].argmax(axis=1) == targets[valid]).mean())
            if valid.any()
            else None
        )
    return result


def _evaluation_report(
    model: Any,
    rows: Sequence[dict[str, Any]],
    batch: tuple[Any, Mapping[str, Any]],
    *,
    baselines: Mapping[str, object],
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    logits = _model_logits(model, batch[0])
    frame_results = []
    for index, row in enumerate(rows):
        degradation = row["degradation"]
        frame_results.append(
            FrameResult(
                frame_id=str(row["item_id"]),
                track_id=str(row["pseudo_track_id"]),
                logits={level: tuple(logits[level][index].tolist()) for level in LEVELS},
                target_indices={level: int(batch[1][level][index]) for level in LEVELS},
                quality_weight=max(0.01, 1.0 - float(degradation["blur_radius"]) / 3.0),
                apparent_height_px=int(degradation["apparent_height_px"]),
                blur_score=min(1.0, float(degradation["blur_radius"]) / 3.0),
                synthetic=bool(row["synthetic"]),
                source_id=str(row["source_id"]),
            )
        )
    metrics = evaluate_frame_and_tracks(frame_results)
    return {
        "schema_version": 1,
        "sealed_test_opened_after_calibration": True,
        "metrics": metrics,
        "baselines": dict(baselines),
        "temperatures": dict(temperatures),
        "thresholds": dict(thresholds),
        "limitations": [
            "Smoke mode uses generated pseudo-tracks and does not claim real-camera accuracy.",
            "Real and synthetic track metrics are reported separately.",
        ],
    }


def _read_hierarchy(path: Path) -> LabelHierarchy:
    payload = _read_json(path)
    return LabelHierarchy(
        body_types=tuple(payload["body_types"]),
        makes=tuple(payload["makes"]),
        model_families=tuple(payload["model_families"]),
        make_to_body=payload["make_to_body"],
        model_to_make=payload["model_to_make"],
    )


def _require_nonempty_splits(items: Iterable[DatasetItem]) -> None:
    counts = {
        split: sum(item.split == split for item in items)
        for split in ("train", "validation", "test")
    }
    if not all(counts.values()):
        raise ValueError(f"deterministic split produced an empty partition: {counts}")


def _class_counts(records: Sequence[dict[str, object]]) -> dict[str, dict[str, int]]:
    return {
        level: {
            label: sum(row.get(level) == label for row in records)
            for label in sorted({str(row[level]) for row in records if row.get(level)})
        }
        for level in LEVELS
    }


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
