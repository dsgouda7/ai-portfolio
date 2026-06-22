"""
Download public image corpus data for benchmarking the Context Optimizer.

Data sources (all public domain / CC):
  - COCO 2017 val captions JSON (25 MB zip, no images required for caption benchmarks)
  - Optionally, COCO 2017 val images (JPEG, ~1 GB for the full 5 000-image set)

Usage
-----
    # Download captions only (fast, required for image_corpus_benchmarks.py)
    python download_images.py --mode captions

    # Download captions + a small subset of images
    python download_images.py --mode small

    # Download captions + medium image subset (~1 000 images)
    python download_images.py --mode medium

Output layout
-------------
    benchmarks/image_data/
        captions_val2017.json          # All 202 520 captions
        val2017/                       # Downloaded JPEG images (optional)
            000000000139.jpg
            ...
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

# ── Constants ─────────────────────────────────────────────────────────────────
BENCH_DIR  = Path(__file__).parent
IMAGE_DIR  = BENCH_DIR / "image_data"

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGE_URL   = "http://images.cocodataset.org/val2017/{image_id:012d}.jpg"

# Number of images to download per mode (small/medium)
IMAGE_COUNTS: dict[str, int] = {
    "small":  200,
    "medium": 1_000,
}


# ── Caption download ──────────────────────────────────────────────────────────

def download_coco_captions(data_dir: Path = IMAGE_DIR) -> Path:
    """
    Download the COCO 2017 validation captions JSON (via annotations zip).

    Returns the path to captions_val2017.json.
    Skips the download if the file already exists.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "captions_val2017.json"

    if out.exists():
        print(f"  [captions] Already exists → {out}")
        return out

    if requests is None:
        raise ImportError("'requests' is required: pip install requests")

    print(f"  [captions] Downloading annotations zip from COCO …")
    response = requests.get(COCO_ANNOTATIONS_URL, stream=True, timeout=120)
    response.raise_for_status()

    buf = io.BytesIO()
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
        total += len(chunk)
        print(f"\r  [captions] Downloaded {total / 1e6:.1f} MB …", end="", flush=True)
    print()

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        target = "annotations/captions_val2017.json"
        if target not in zf.namelist():
            raise RuntimeError(f"Expected '{target}' inside zip, not found")
        with zf.open(target) as src, out.open("wb") as dst:
            dst.write(src.read())

    print(f"  [captions] Saved → {out}")
    return out


def load_coco_captions(json_path: Path, max_images: int | None = None) -> list[str]:
    """
    Load COCO captions as a flat list of strings.

    Each returned string is a single caption sentence.
    """
    with json_path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    annotations = data.get("annotations", [])

    # Optionally limit to the first `max_images` unique image IDs
    if max_images is not None:
        seen: set[int] = set()
        filtered = []
        for ann in annotations:
            seen.add(ann["image_id"])
            filtered.append(ann)
            if len(seen) >= max_images:
                break
        annotations = filtered

    return [ann["caption"].strip() for ann in annotations if ann.get("caption")]


# ── Image download (optional) ─────────────────────────────────────────────────

def download_coco_images(
    data_dir: Path,
    image_ids: list[int],
    *,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Download COCO 2017 validation JPEG images by image_id.

    Returns list of local paths for images that were successfully saved.
    """
    if requests is None:
        raise ImportError("'requests' is required: pip install requests")

    out_dir = data_dir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx, img_id in enumerate(image_ids):
        filename = f"{img_id:012d}.jpg"
        dest = out_dir / filename
        if skip_existing and dest.exists():
            saved.append(dest)
            continue

        url = COCO_VAL_IMAGE_URL.format(image_id=img_id)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            saved.append(dest)
            print(f"\r  [images] {idx + 1}/{len(image_ids)} downloaded ({filename})", end="", flush=True)
        except Exception as exc:
            print(f"\n  [warn] Failed to download {filename}: {exc}")

    if image_ids:
        print()
    return saved


def _get_image_ids(json_path: Path, max_images: int) -> list[int]:
    """Return the first `max_images` unique image IDs from the captions JSON."""
    with json_path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    seen: list[int] = []
    for ann in data.get("annotations", []):
        iid = ann["image_id"]
        if iid not in seen:
            seen.append(iid)
        if len(seen) >= max_images:
            break
    return seen


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO image/caption data for the Context Optimizer image benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["captions", "small", "medium"],
        default="captions",
        help=(
            "captions = JSON only (fast, ~25 MB). "
            "small = captions + 200 images. "
            "medium = captions + 1 000 images."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(IMAGE_DIR),
        help=f"Output directory (default: {IMAGE_DIR})",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"\n{'='*60}")
    print(f"  Image corpus download — mode: {args.mode}")
    print(f"  Output: {data_dir}")
    print(f"{'='*60}\n")

    captions_path = download_coco_captions(data_dir)
    captions = load_coco_captions(captions_path, max_images=100)
    print(f"  [info] Loaded {len(captions)} sample captions (first 100 images)")

    if args.mode in IMAGE_COUNTS:
        n_images = IMAGE_COUNTS[args.mode]
        print(f"\n  [images] Downloading {n_images} JPEG images …")
        image_ids = _get_image_ids(captions_path, n_images)
        saved = download_coco_images(data_dir, image_ids)
        print(f"  [images] Saved {len(saved)}/{len(image_ids)} images → {data_dir / 'val2017'}")

    print(f"\n  Done.  Caption JSON → {captions_path}\n")


if __name__ == "__main__":
    main()
