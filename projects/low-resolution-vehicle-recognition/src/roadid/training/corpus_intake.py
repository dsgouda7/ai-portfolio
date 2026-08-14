"""License-gated conversion of external vehicle corpora into CarFace manifests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from roadid.training.datasets import sha256_file


@dataclass(frozen=True, slots=True)
class CorpusContract:
    source_id: str
    version: str
    terms_url: str
    license_name: str
    required_labels: tuple[str, ...]
    ownership_field: str


CORPORA = {
    "mio-tcd": CorpusContract(
        source_id="mio-tcd",
        version="2018",
        terms_url="https://tcd.miovision.com/challenge/dataset.html",
        license_name="CC-BY-NC-SA-4.0",
        required_labels=("body_type",),
        ownership_field="camera_id",
    ),
    "compcars": CorpusContract(
        source_id="compcars",
        version="cvpr-2015",
        terms_url="https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html",
        license_name="non-commercial-research-only",
        required_labels=("body_type", "make", "model_family"),
        ownership_field="identity_id",
    ),
}


def build_corpus_manifest(
    dataset: str,
    dataset_root: Path,
    annotations_csv: Path,
    output_path: Path,
    *,
    accept_noncommercial_terms: bool,
) -> Path:
    try:
        contract = CORPORA[dataset]
    except KeyError as error:
        raise ValueError(f"unsupported corpus: {dataset}") from error
    if not accept_noncommercial_terms:
        raise PermissionError(
            f"{dataset} is non-commercial; pass --accept-noncommercial-terms only after review"
        )
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    rows = list(csv.DictReader(annotations_csv.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise ValueError("annotation CSV contains no rows")
    items = []
    identifiers = set()
    for index, row in enumerate(rows):
        relative = Path((row.get("image_path") or "").strip())
        image = (root / relative).resolve()
        if root not in image.parents or not image.is_file():
            raise ValueError(f"row {index + 2} image is missing or outside dataset root")
        missing = [name for name in contract.required_labels if not (row.get(name) or "").strip()]
        if missing:
            raise ValueError(f"row {index + 2} is missing labels: {', '.join(missing)}")
        ownership = (row.get(contract.ownership_field) or "").strip()
        if not ownership:
            raise ValueError(
                f"row {index + 2} requires {contract.ownership_field} for leakage-safe splitting"
            )
        item_id = (row.get("item_id") or f"{dataset}-{index:07d}").strip()
        if item_id in identifiers:
            raise ValueError(f"duplicate item_id: {item_id}")
        identifiers.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "image_path": relative.as_posix(),
                "source_id": contract.source_id,
                "source_version": contract.version,
                "source_terms_url": contract.terms_url,
                "source_license": contract.license_name,
                "source_sha256": sha256_file(image),
                "identity_id": _optional(row.get("identity_id")),
                "track_id": _optional(row.get("track_id")),
                "camera_id": _optional(row.get("camera_id")),
                "geography": _optional(row.get("geography")),
                "body_type": _optional(row.get("body_type")),
                "make": _optional(row.get("make")),
                "model_family": _optional(row.get("model_family")),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": {
            "id": contract.source_id,
            "version": contract.version,
            "terms_url": contract.terms_url,
            "license": contract.license_name,
            "noncommercial_terms_accepted": True,
        },
        "items": items,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a license-gated CarFace corpus manifest.")
    parser.add_argument("--dataset", choices=sorted(CORPORA), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-noncommercial-terms", action="store_true")
    arguments = parser.parse_args(argv)
    output = build_corpus_manifest(
        arguments.dataset,
        arguments.dataset_root,
        arguments.annotations,
        arguments.output,
        accept_noncommercial_terms=arguments.accept_noncommercial_terms,
    )
    print(output)
    return 0


def _optional(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None
