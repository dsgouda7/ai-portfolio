"""
Context Optimizer Image Corpus Benchmark — raw vs optimised on COCO image captions.

Usage
-----
    # Download captions first
    python download_images.py --mode captions

    # Then run the benchmark
    python image_corpus_benchmarks.py [--corpus small|medium|large]

Steps
-----
1. Load COCO captions (or generate synthetic image descriptions as fallback).
2. Run ground-truth queries against both strategies:
      Raw baseline  — full caption scan (monolithic approach)
      Optimised     — compress → ChromaDB index → ToT-retrieve
3. Gather metrics and write image_results.md in the benchmarks directory.

Corpus sizes (approximate unique-image counts)
----------------------------------------------
    small   ~  1 000 images  (~  5 000 captions)
    medium  ~  5 000 images  (~ 25 000 captions)
    large   ~ 25 000 images  (~125 000 captions)

Environment variables (optional)
---------------------------------
    OLLAMA_BASE_URL    default: http://localhost:11434
    GROQ_API_KEY       required when provider=groq
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Project paths ─────────────────────────────────────────────────────────────
BENCH_DIR = Path(__file__).parent
PROJECT_ROOT = BENCH_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Corpus sizes (unique image counts) ─────────────────────────────────────────
CORPUS_IMAGES: dict[str, int] = {
    "small": 1_000,
    "medium": 5_000,
    "large": 25_000,
}

# ── Ground-truth queries (image / caption domain) ─────────────────────────────
IMAGE_GROUND_TRUTH_QUERIES: list[dict[str, Any]] = [
    {
        "query": "dog playing outdoor",
        "must_contain": ["dog", "outdoor"],
    },
    {
        "query": "person riding bicycle street",
        "must_contain": ["person", "bicycle"],
    },
    {
        "query": "cat sitting indoors",
        "must_contain": ["cat", "sit"],
    },
    {
        "query": "food on table meal",
        "must_contain": ["food", "table"],
    },
    {
        "query": "car driving road city",
        "must_contain": ["car", "road"],
    },
]

# ── Fallback synthetic caption generator ─────────────────────────────────────


def _generate_synthetic_captions(n_images: int) -> list[str]:
    """
    Generate deterministic synthetic image captions when COCO data is unavailable.
    Each image gets approximately 5 captions (COCO average).
    """
    caption_templates = [
        "A dog is playing in an outdoor park near a grassy area.",
        "Two people are riding bicycles down a busy city street.",
        "A cat is sitting on a sofa indoors next to a window.",
        "A plate of food is placed on a wooden table at a meal.",
        "A car is driving along a road through the city.",
        "A person is standing near a bicycle on the street.",
        "An outdoor scene with a dog running across the grass.",
        "A close-up of food arranged on a table for a meal.",
        "A cat and a dog are sitting together indoors.",
        "Several cars are parked along a road in the city.",
    ]
    captions: list[str] = []
    for i in range(n_images):
        # Each image gets ~5 captions (cycling through templates)
        for j in range(5):
            idx = (i * 5 + j) % len(caption_templates)
            captions.append(f"[img-{i:06d}] {caption_templates[idx]}")
    return captions[: n_images * 5]


# ── Step 1 — Load image captions ─────────────────────────────────────────────


def load_image_corpus(size: str) -> list[str]:
    """
    Return caption strings for the requested corpus size.

    Priority:
    1. Real COCO captions from benchmarks/image_data/captions_val2017.json
    2. Inline synthetic captions (no download required)
    """
    n_images = CORPUS_IMAGES[size]
    captions_path = BENCH_DIR / "image_data" / "captions_val2017.json"

    if captions_path.exists():
        try:
            sys.path.insert(0, str(BENCH_DIR))
            from download_images import load_coco_captions

            captions = load_coco_captions(captions_path, max_images=n_images)
            print(f"  [corpus] {len(captions):,} COCO captions ({n_images} images)")
            return captions
        except Exception as exc:
            print(
                f"  [warn] COCO load failed ({exc}), falling back to synthetic captions"
            )

    captions = _generate_synthetic_captions(n_images)
    print(
        f"  [corpus] Generated {len(captions):,} synthetic captions ({n_images} images)"
    )
    return captions


# ── Step 2a — Raw baseline ────────────────────────────────────────────────────


def run_raw_baseline(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Baseline: scan all captions for each query. No compression, no index."""
    results = []
    total_tokens = sum(len(c) for c in corpus) // 4

    for q in queries:
        start = time.perf_counter()
        must = q["must_contain"]
        hits = [c for c in corpus if any(kw.lower() in c.lower() for kw in must)]
        latency_ms = (time.perf_counter() - start) * 1000

        results.append(
            {
                "query": q["query"],
                "strategy": "raw",
                "tokens_processed": total_tokens,
                "lines_scanned": len(corpus),
                "lines_retrieved": len(hits),
                "latency_ms": latency_ms,
                "recall": _recall(hits, must),
            }
        )
    return results


def _recall(lines: list[str], must_contain: list[str]) -> float:
    if not must_contain:
        return 1.0
    blob = "\n".join(lines).lower()
    hits = sum(1 for kw in must_contain if kw.lower() in blob)
    return hits / len(must_contain)


def _recall_from_answer(answer: str, must_contain: list[str]) -> float:
    """Check how many must_contain keywords appear in the LLM's answer text."""
    blob = answer.lower()
    hits = sum(1 for kw in must_contain if kw.lower() in blob)
    return hits / max(len(must_contain), 1)


# ── Step 2b — Optimised pipeline ─────────────────────────────────────────────


def run_optimized(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int, int]:
    """
    Optimised pipeline for image captions:
      1. Compress captions with rolling-window LLM.
      2. Store in ephemeral ChromaDB.
      3. ToTReasoner retrieves evidence per query.
    """
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.tot_reasoner import ToTReasoner

    t0 = time.perf_counter()
    chunks = compress_corpus_rolling(corpus)
    compress_time = time.perf_counter() - t0

    original_tokens = sum(c.original_tokens for c in chunks)
    compressed_tokens = sum(c.compressed_tokens for c in chunks)
    ratio = compressed_tokens / max(original_tokens, 1)
    print(
        f"  [compress] {len(chunks)} chunks | "
        f"{original_tokens:,} → {compressed_tokens:,} tokens "
        f"({ratio:.1%} ratio) | {compress_time:.1f}s"
    )

    retriever = _build_retriever(chunks)
    reasoner = ToTReasoner(retriever=retriever)

    results = []
    for q in queries:
        branch_specs = [
            {"id": "main", "title": q["query"], "search_terms": q["must_contain"]}
        ]
        start = time.perf_counter()
        tot = reasoner.reason(
            type("_Ctx", (), {"entities": q["must_contain"]})(),
            branch_specs=branch_specs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        results.append(
            {
                "query": q["query"],
                "strategy": "optimized",
                "tokens_processed": compressed_tokens,
                "lines_scanned": tot.total_retrieved_lines,
                "lines_retrieved": tot.total_retrieved_lines,
                "latency_ms": latency_ms,
                "selected_branch": tot.selected_branch_id,
                "recall": _recall_from_snippets(
                    tot.winner.evidence_snippets, q["must_contain"]
                ),
            }
        )

    _cleanup_retriever(retriever)
    return results, compress_time, original_tokens, compressed_tokens


def _build_retriever(chunks: list[Any]) -> Any:
    try:
        from context_optimizer.cached_retriever import CachedChromaRetriever

        tmp_dir = tempfile.mkdtemp(prefix="co_img_bench_")
        retriever = CachedChromaRetriever(
            collection_name="image_benchmark", persist_directory=tmp_dir
        )
        retriever.add_chunks(chunks)
        retriever._tmp_dir = tmp_dir
        return retriever
    except Exception as exc:
        print(f"  [info] ChromaDB unavailable ({exc}), using DualStorageRetriever")
        from context_optimizer.retriever import DualStorageRetriever

        retriever = DualStorageRetriever(chunks)
        retriever._tmp_dir = None
        return retriever


def _cleanup_retriever(retriever: Any) -> None:
    tmp = getattr(retriever, "_tmp_dir", None)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


def _recall_from_snippets(snippets: list[str], must_contain: list[str]) -> float:
    if not snippets:
        return 0.0
    blob = "\n".join(snippets).lower()
    hits = sum(1 for kw in must_contain if kw.lower() in blob)
    return hits / max(len(must_contain), 1)


# ── Step 3 — Write image_results.md ──────────────────────────────────────────


def write_results(
    corpus_size: str,
    n_captions: int,
    raw_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
    compress_time: float,
    original_tokens: int,
    compressed_tokens: int,
) -> Path:
    out = BENCH_DIR / "image_results.md"

    raw_avg_latency = sum(r["latency_ms"] for r in raw_results) / max(
        len(raw_results), 1
    )
    opt_avg_latency = sum(r["latency_ms"] for r in opt_results) / max(
        len(opt_results), 1
    )
    raw_tokens = raw_results[0]["tokens_processed"] if raw_results else 0
    compression_pct = (1 - compressed_tokens / max(original_tokens, 1)) * 100
    speedup = raw_avg_latency / max(opt_avg_latency, 0.001)
    raw_recall_avg = sum(r["recall"] for r in raw_results) / max(len(raw_results), 1)
    opt_recall_avg = sum(r["recall"] for r in opt_results) / max(len(opt_results), 1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_images = CORPUS_IMAGES[corpus_size]

    token_pass = (
        "✅ PASS"
        if compression_pct >= 90
        else f"⚠️  {compression_pct:.1f}% (target ≥ 90%)"
    )
    speedup_note = (
        f"✅ {speedup:.1f}×" if speedup >= 10 else f"⚠️  {speedup:.1f}× (target ≥ 10×)"
    )

    md: list[str] = [
        "# Context Optimizer — Image Corpus Benchmark Results",
        "",
        f"**Run**: {now}  |  **Corpus**: `{corpus_size}` ({n_images:,} images / {n_captions:,} captions)  |  **Modality**: Image Captions",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Raw Baseline | Optimised (ToT) | Δ |",
        "|--------|:------------|:----------------|:--|",
        f"| Tokens processed (avg/query) | {raw_tokens:,} | {compressed_tokens:,} | **−{compression_pct:.1f}%** |",
        f"| Avg query latency | {raw_avg_latency:.1f} ms | {opt_avg_latency:.1f} ms | **{speedup:.1f}× faster** |",
        f"| Avg recall (ground-truth keywords) | {raw_recall_avg:.0%} | {opt_recall_avg:.0%} | — |",
        f"| One-time compression cost | — | {compress_time:.1f} s | — |",
        "",
        "### Threshold Gates",
        "",
        "| Gate | Result |",
        "|------|--------|",
        f"| Token reduction ≥ 90% | {token_pass} |",
        f"| Query speedup ≥ 10× | {speedup_note} |",
        "",
        "---",
        "",
        "## Per-Query Results",
        "",
        "| Query | Raw latency | Opt latency | Token reduction | Opt recall |",
        "|-------|------------|-------------|-----------------|------------|",
    ]

    for rr, oo in zip(raw_results, opt_results):
        tok_red = (1 - oo["tokens_processed"] / max(rr["tokens_processed"], 1)) * 100
        md.append(
            f"| {rr['query'][:55]} | {rr['latency_ms']:.1f} ms | "
            f"{oo['latency_ms']:.1f} ms | {tok_red:.1f}% | {oo['recall']:.0%} |"
        )

    md += [
        "",
        "---",
        "",
        "## Data Source",
        "",
        "Captions sourced from **COCO 2017 Validation** (`captions_val2017.json`).",
        "Download with: `python benchmarks/download_images.py --mode captions`",
        "",
        "## Integrating These Results",
        "",
        "> **Update the whitepaper and design doc with the numbers above.**",
        ">",
        "> ### 1 · Whitepaper (`docs/whitepaper/proposed-whitepaper.md`)",
        ">",
        "> Add an *Image Corpus Benchmark* subsection under *Performance Evidence*:",
        f">",
        f">     Corpus: {corpus_size} ({n_images:,} images, image-caption modality)",
        f">     Token reduction:       {compression_pct:.1f}%",
        f">     Avg query latency:     {opt_avg_latency:.1f} ms",
        f">     One-time compression:  {compress_time:.1f} s",
        ">",
        "> ### 2 · Architecture doc (`docs/design/ARCHITECTURE.md` §12)",
        ">",
        "> Add a new image-modality row to the benchmark table:",
        f">",
        f">     | {corpus_size.capitalize()} (image) | {n_images:,} | image-captions | {compression_pct:.1f}% | {opt_avg_latency:.1f} ms | ... |",
        ">",
        "> Threshold checks:",
        f">   - Token reduction ≥ 90%: {token_pass}",
        f">   - Query speedup ≥ 10×:  {speedup_note}",
        "",
        "---",
        f"*Generated by `benchmarks/image_corpus_benchmarks.py` — do not edit manually.*",
    ]

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"  [results] Written → {out.relative_to(PROJECT_ROOT)}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context Optimizer image corpus benchmark (raw vs optimised on COCO captions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size in terms of unique images (default: small = 1 000 images)",
    )
    args = parser.parse_args()
    corpus_size = args.corpus

    print(f"\n{'='*60}")
    print(
        f"  Image Corpus Benchmark — {corpus_size} ({CORPUS_IMAGES[corpus_size]:,} images)"
    )
    print(f"{'='*60}\n")

    print("[1/3] Loading image corpus …")
    corpus = load_image_corpus(corpus_size)
    n_captions = len(corpus)

    print(f"\n[2/3] Running {len(IMAGE_GROUND_TRUTH_QUERIES)} ground-truth queries …")
    print("  → Raw baseline …")
    raw_results = run_raw_baseline(corpus, IMAGE_GROUND_TRUTH_QUERIES)
    raw_avg = sum(r["latency_ms"] for r in raw_results) / len(raw_results)
    print(
        f"     avg latency: {raw_avg:.1f} ms | tokens: {raw_results[0]['tokens_processed']:,}"
    )

    print("  → Optimised (compress → ToT local) …")
    opt_results, compress_time, original_tokens, compressed_tokens = run_optimized(
        corpus, IMAGE_GROUND_TRUTH_QUERIES
    )
    opt_avg = sum(r["latency_ms"] for r in opt_results) / len(opt_results)
    reduction = (1 - compressed_tokens / max(original_tokens, 1)) * 100
    print(f"     avg latency: {opt_avg:.1f} ms | token reduction: {reduction:.1f}%")

    print("\n[3/3] Writing results …")
    out = write_results(
        corpus_size,
        n_captions,
        raw_results,
        opt_results,
        compress_time,
        original_tokens,
        compressed_tokens,
    )

    print(f"\n{'='*60}")
    print(f"  Done.  Results → {out.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
