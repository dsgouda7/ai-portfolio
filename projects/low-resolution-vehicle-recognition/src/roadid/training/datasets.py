"""Dataset provenance and leakage-safe split construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image, ImageDraw

from roadid.training.degradations import sample_recipe

Split = Literal["train", "validation", "test"]


class DatasetAdapter(Protocol):
    def inventory(self) -> tuple[DatasetItem, ...]: ...


@dataclass(frozen=True, slots=True)
class LocalManifestAdapter:
    dataset_root: Path
    manifest_path: Path

    def inventory(self) -> tuple[DatasetItem, ...]:
        payload = _read_json_object(self.manifest_path)
        source = payload.get("source")
        rows = payload.get("items")
        if not isinstance(source, dict) or not isinstance(rows, list):
            raise ValueError("source manifest requires source metadata and an items list")
        required = ("id", "version", "terms_url", "license")
        if any(not source.get(field) for field in required):
            raise ValueError(f"source metadata requires fields: {required}")
        items: list[DatasetItem] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"manifest item {index} must be an object")
            relative = Path(str(row.get("image_path", "")))
            image = (self.dataset_root / relative).resolve()
            if not image.is_file() or self.dataset_root.resolve() not in image.parents:
                raise ValueError(f"manifest image is missing or outside dataset root: {relative}")
            items.append(
                DatasetItem(
                    item_id=str(row.get("item_id", f"item-{index:06d}")),
                    image_path=image.as_posix(),
                    source_id=str(source["id"]),
                    source_version=str(source["version"]),
                    source_terms_url=str(source["terms_url"]),
                    source_license=str(source["license"]),
                    source_sha256=sha256_file(image),
                    identity_id=_optional_text(row.get("identity_id")),
                    track_id=_optional_text(row.get("track_id")),
                    camera_id=_optional_text(row.get("camera_id")),
                    geography=_optional_text(row.get("geography")),
                    body_type=_optional_text(row.get("body_type")),
                    make=_optional_text(row.get("make")),
                    model_family=_optional_text(row.get("model_family")),
                )
            )
        if not items:
            raise ValueError("source manifest contains no items")
        return tuple(items)


@dataclass(frozen=True, slots=True)
class DatasetItem:
    item_id: str
    image_path: str
    source_id: str
    source_version: str
    source_terms_url: str
    source_license: str
    source_sha256: str
    identity_id: str | None = None
    track_id: str | None = None
    camera_id: str | None = None
    geography: str | None = None
    body_type: str | None = None
    make: str | None = None
    model_family: str | None = None
    split: Split | None = None
    synthetic: bool = False
    pseudo_track_id: str | None = None
    pseudo_frame_index: int | None = None
    transform_seed: int | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.item_id,
                self.image_path,
                self.source_id,
                self.source_version,
                self.source_terms_url,
                self.source_license,
            )
        ):
            raise ValueError("dataset provenance fields cannot be empty")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest")
        int(self.source_sha256, 16)
        if self.synthetic and (self.split is None or self.pseudo_track_id is None):
            raise ValueError("synthetic items require an assigned split and pseudo-track ID")
        if not self.synthetic and self.pseudo_track_id is not None:
            raise ValueError("real items cannot name a pseudo-track")

    @property
    def ownership_key(self) -> str:
        boundary = self.identity_id or self.track_id or self.source_sha256
        return f"{self.source_id}:{boundary}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(
    items: Iterable[DatasetItem],
    *,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[DatasetItem, ...]:
    """Assign one split to each source identity/track ownership group."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must leave training data")

    materialized = tuple(items)
    if any(item.synthetic or item.split is not None for item in materialized):
        raise ValueError("split assignment accepts only unsplit real source items")

    owner_split: dict[str, Split] = {}
    for owner in sorted({item.ownership_key for item in materialized}):
        value = (
            int.from_bytes(hashlib.sha256(f"{seed}:{owner}".encode()).digest()[:8], "big") / 2**64
        )
        if value < test_fraction:
            owner_split[owner] = "test"
        elif value < test_fraction + validation_fraction:
            owner_split[owner] = "validation"
        else:
            owner_split[owner] = "train"

    assigned = tuple(replace(item, split=owner_split[item.ownership_key]) for item in materialized)
    assert_split_invariants(assigned)
    return assigned


def create_pseudo_tracks(
    items: Iterable[DatasetItem], *, length: int, seed: int
) -> tuple[DatasetItem, ...]:
    """Expand split-owned still images into synthetic frame records."""
    if length < 2:
        raise ValueError("pseudo-track length must be at least two")
    expanded: list[DatasetItem] = []
    for item in items:
        if item.split is None or item.synthetic:
            raise ValueError("pseudo-tracks can only be created from split-owned real items")
        pseudo_track_id = f"pseudo-{item.source_sha256[:16]}-{item.split}"
        for frame_index in range(length):
            frame_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{item.source_sha256}:{frame_index}".encode()).digest()[:8],
                "big",
            )
            expanded.append(
                replace(
                    item,
                    item_id=f"{item.item_id}-pseudo-{frame_index:03d}",
                    synthetic=True,
                    pseudo_track_id=pseudo_track_id,
                    pseudo_frame_index=frame_index,
                    transform_seed=frame_seed,
                )
            )
    assert_split_invariants(expanded)
    return tuple(expanded)


def assert_split_invariants(items: Iterable[DatasetItem]) -> None:
    owners: dict[str, Split] = {}
    pseudo_tracks: dict[str, Split] = {}
    for item in items:
        if item.split is None:
            raise ValueError(f"item has no split: {item.item_id}")
        previous = owners.setdefault(item.ownership_key, item.split)
        if previous != item.split:
            raise ValueError(f"source ownership crosses splits: {item.ownership_key}")
        if item.pseudo_track_id:
            pseudo_previous = pseudo_tracks.setdefault(item.pseudo_track_id, item.split)
            if pseudo_previous != item.split:
                raise ValueError(f"pseudo-track crosses splits: {item.pseudo_track_id}")


def prepare_dataset_artifacts(
    adapter: DatasetAdapter,
    output_root: Path,
    *,
    seed: int,
    pseudo_track_length: int,
    apparent_height_range: tuple[int, int],
) -> Path:
    source_items = adapter.inventory()
    assigned = assign_splits(source_items, seed=seed)
    pseudo = create_pseudo_tracks(assigned, length=pseudo_track_length, seed=seed)
    rows = []
    for item in pseudo:
        if item.transform_seed is None:
            raise AssertionError("pseudo-track item has no transform seed")
        rows.append(
            {
                **item.to_dict(),
                "degradation_recipe": sample_recipe(
                    item.transform_seed,
                    apparent_height_range=apparent_height_range,
                ).to_dict(),
            }
        )
    output_root.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise ValueError(f"prepared dataset has no {split} ownership groups")
        _write_jsonl(output_root / f"{split}.jsonl", split_rows)
    from roadid.training.hierarchy import build_hierarchy

    hierarchy = build_hierarchy(source_items)
    _write_json(output_root / "labels.json", hierarchy.to_dict())
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "item_count": len(rows),
        "source_item_count": len(source_items),
        "synthetic_item_count": len(rows),
        "real_track_item_count": 0,
        "source_ids": sorted({item.source_id for item in source_items}),
        "source_versions": sorted({item.source_version for item in source_items}),
        "source_terms_urls": sorted({item.source_terms_url for item in source_items}),
        "source_licenses": sorted({item.source_license for item in source_items}),
        "source_sha256": sorted({item.source_sha256 for item in source_items}),
        "split_sha256": {
            split: sha256_file(output_root / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
        "labels_sha256": sha256_file(output_root / "labels.json"),
        "pseudo_tracks_created_after_split": True,
    }
    _write_json(output_root / "manifest.json", manifest)
    return output_root


def create_smoke_source(root: Path, *, item_count: int = 48) -> LocalManifestAdapter:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(item_count):
        body, make, model = (
            ("sedan", "honda", "civic") if index % 2 == 0 else ("suv", "toyota", "rav4")
        )
        path = images / f"vehicle-{index:03d}.png"
        image = Image.new("RGB", (96, 64), (35, 38, 42))
        draw = ImageDraw.Draw(image)
        color = (200, 65, 45) if index % 2 == 0 else (35, 100, 210)
        top = 27 if body == "sedan" else 20
        draw.rectangle((12, top, 84, 53), fill=color)
        draw.rectangle((28, top - 9, 69, top), fill=tuple(max(0, value - 30) for value in color))
        draw.ellipse((20, 48, 34, 62), fill=(12, 12, 12))
        draw.ellipse((64, 48, 78, 62), fill=(12, 12, 12))
        image.save(path, format="PNG", compress_level=9)
        rows.append(
            {
                "item_id": f"smoke-{index:03d}",
                "image_path": path.relative_to(root).as_posix(),
                "identity_id": f"vehicle-{index:03d}",
                "camera_id": f"camera-{index % 4}",
                "geography": f"zone-{index % 3}",
                "body_type": body,
                "make": make,
                "model_family": model,
            }
        )
    manifest_path = root / "source-manifest.json"
    _write_json(
        manifest_path,
        {
            "source": {
                "id": "roadid-smoke",
                "version": "1",
                "terms_url": "https://example.test/roadid-smoke-terms",
                "license": "generated-fixture-only",
            },
            "items": rows,
        },
    )
    return LocalManifestAdapter(root, manifest_path)


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None and str(value) else None
