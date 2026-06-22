"""
Context Optimizer Benchmark — three-step ground-truth comparison.

Usage
-----
    python run_benchmark.py [--corpus small|medium|large] [--provider mock|ollama|groq]

Steps
-----
1. Download / generate a corpus of the requested size.
2. Run ground-truth queries against both strategies:
      Raw baseline  — full corpus scan (monolithic LLM approach)
      Optimized     — compress → ChromaDB index → ToT-retrieve
3. Gather metrics from both runs and write results.md in this directory.
   results.md also contains instructions for integrating the latest
   numbers into the whitepaper and design doc.

Corpus sizes (approximate line counts)
---------------------------------------
    small   ~  5 000 lines  (fast, good for CI / quick checks)
    medium  ~ 25 000 lines  (representative, ~15-30 min with real LLM)
    large   ~100 000 lines  (stress test, production-scale)

Environment variables (optional)
---------------------------------
    CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER   ollama|groq|mock  (default: mock)
    CONTEXT_OPTIMIZER_COMPRESSOR_MODEL      model name for compression LLM
    OLLAMA_BASE_URL                         default: http://localhost:11434
    GROQ_API_KEY                            required when provider=groq
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Project paths ────────────────────────────────────────────────────────────
BENCH_DIR    = Path(__file__).parent
PROJECT_ROOT = BENCH_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Corpus sizes ─────────────────────────────────────────────────────────────
CORPUS_LINES: dict[str, int] = {
    "small":  5_000,
    "medium": 25_000,
    "large":  100_000,
}

# ── Ground-truth queries ──────────────────────────────────────────────────────
# For each query we know which keywords MUST appear in relevant results.
# Recall = fraction of must_contain keywords found in retrieved content.
GROUND_TRUTH_QUERIES: list[dict[str, Any]] = [
    {
        "query":        "CosmosDB timeout error code 21012",
        "must_contain": ["CosmosDB", "timeout", "21012"],
    },
    {
        "query":        "ingress upstream timed out",
        "must_contain": ["upstream", "ingress", "timeout"],
    },
    {
        "query":        "payment-service cancellation error",
        "must_contain": ["payment-service", "cancellation"],
    },
    {
        "query":        "HTTP 504 latency spike checkout",
        "must_contain": ["504", "checkout"],
    },
    {
        "query":        "CosmosDB RU charge partition hot-key",
        "must_contain": ["ru_charge", "partition"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download / generate corpus
# ─────────────────────────────────────────────────────────────────────────────

def download_corpus(size: str) -> list[str]:
    """
    Return corpus lines for the requested size.

    Priority order:
    1. Real text files already in benchmarks/tot/test_data/books_*.txt
    2. Generate a synthetic AKS/CosmosDB incident log corpus (always works,
       no downloads required — ideal for CI and mock-mode benchmarking).
    """
    n = CORPUS_LINES[size]
    test_data_dir = BENCH_DIR / "tot" / "test_data"

    # Try real book data first (already downloaded by download_test_data.py)
    txt_files = sorted(test_data_dir.glob("books_*.txt"))
    if txt_files:
        raw: list[str] = []
        for f in txt_files:
            try:
                raw.extend(f.read_text(encoding="utf-8").splitlines())
            except Exception:
                pass
        if raw:
            # tile to requested size if needed
            while len(raw) < n:
                raw = raw + raw
            lines = raw[:n]
            print(f"  [corpus] {len(lines):,} lines from {len(txt_files)} book file(s) in test_data/")
            return lines

    # Fallback: generate synthetic AKS incident logs (no I/O required)
    _co_bench = _load_benchmark_module()
    lines = _co_bench.build_mock_log_cache(total_lines=n)
    print(f"  [corpus] Generated {len(lines):,} synthetic AKS log lines (mock mode)")
    return lines


def _load_benchmark_module() -> Any:
    """Dynamically load context_optimizer_benchmark so we can reuse helpers."""
    bench_file = PROJECT_ROOT / "context_optimizer_benchmark.py"
    spec = importlib.util.spec_from_file_location("_co_bench", bench_file)
    mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                    # type: ignore[union-attr]
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Step 2a — Raw baseline (monolithic scan)
# ─────────────────────────────────────────────────────────────────────────────

def run_raw_baseline(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Simulate the monolithic LLM approach: scan the full corpus for each query.
    No compression, no index — every query sees the entire corpus.
    """
    results = []
    total_tokens = sum(len(ln) for ln in corpus) // 4  # rough 4-chars-per-token estimate

    for q in queries:
        start = time.perf_counter()
        must = q["must_contain"]
        hits = [ln for ln in corpus if any(kw.lower() in ln.lower() for kw in must)]
        latency_ms = (time.perf_counter() - start) * 1000

        results.append({
            "query":            q["query"],
            "strategy":         "raw",
            "tokens_processed": total_tokens,
            "lines_scanned":    len(corpus),
            "lines_retrieved":  len(hits),
            "latency_ms":       latency_ms,
            "recall":           _recall(hits, must),
        })

    return results


def _recall(lines: list[str], must_contain: list[str]) -> float:
    if not must_contain:
        return 1.0
    blob = "\n".join(lines).lower()
    hits = sum(1 for kw in must_contain if kw.lower() in blob)
    return hits / len(must_contain)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Optimized pipeline (compress → ToT-retrieve)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimized(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int, int]:
    """
    Optimized pipeline:
      1. Compress corpus with rolling-window LLM (one-time write cost).
      2. Store compressed chunks in an ephemeral ChromaDB.
      3. For each query, use ToTReasoner to retrieve evidence and report metrics.

    Returns (results, compress_time_s, original_tokens, compressed_tokens).
    Falls back to DualStorageRetriever if chromadb is not installed.
    """
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.tot_reasoner import ToTReasoner

    # --- Compression (one-time) ---
    t0     = time.perf_counter()
    chunks = compress_corpus_rolling(corpus)
    compress_time = time.perf_counter() - t0

    original_tokens   = sum(c.original_tokens   for c in chunks)
    compressed_tokens = sum(c.compressed_tokens for c in chunks)
    ratio             = compressed_tokens / max(original_tokens, 1)
    print(
        f"  [compress] {len(chunks)} chunks | "
        f"{original_tokens:,} → {compressed_tokens:,} tokens "
        f"({ratio:.1%} compression ratio) | {compress_time:.1f}s"
    )

    # --- Build retriever ---
    retriever = _build_retriever(chunks)
    reasoner  = ToTReasoner(retriever=retriever)

    # --- Query ---
    results = []
    for q in queries:
        branch_specs = [
            {
                "id":           "main",
                "title":        q["query"],
                "search_terms": q["must_contain"],
            }
        ]
        start     = time.perf_counter()
        tot       = reasoner.reason(
            type("_Ctx", (), {"entities": q["must_contain"]})(),
            branch_specs=branch_specs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        results.append({
            "query":            q["query"],
            "strategy":         "optimized",
            "tokens_processed": compressed_tokens,
            "lines_scanned":    tot.total_retrieved_lines,
            "lines_retrieved":  tot.total_retrieved_lines,
            "latency_ms":       latency_ms,
            "selected_branch":  tot.selected_branch_id,
            "recall":           _recall_from_snippets(tot.winner.evidence_snippets, q["must_contain"]),
        })

    # Cleanup ephemeral DB
    _cleanup_retriever(retriever)

    return results, compress_time, original_tokens, compressed_tokens


def _build_retriever(chunks: list[Any]) -> Any:
    """
    Build CachedChromaRetriever if chromadb is available, else fall back to
    DualStorageRetriever (zero dependencies, keyword/entity scoring).
    """
    try:
        from context_optimizer.cached_retriever import CachedChromaRetriever
        tmp_dir  = tempfile.mkdtemp(prefix="co_bench_")
        retriever = CachedChromaRetriever(
            collection_name="benchmark",
            persist_directory=tmp_dir,
        )
        retriever.add_chunks(chunks)
        retriever._tmp_dir = tmp_dir  # stash for cleanup
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Write results.md
# ─────────────────────────────────────────────────────────────────────────────

def write_results(
    corpus_size: str,
    n_lines: int,
    raw_results:   list[dict[str, Any]],
    opt_results:   list[dict[str, Any]],
    compress_time: float,
    original_tokens:   int,
    compressed_tokens: int,
) -> Path:
    out = BENCH_DIR / "results.md"

    raw_avg_latency = sum(r["latency_ms"] for r in raw_results) / max(len(raw_results), 1)
    opt_avg_latency = sum(r["latency_ms"] for r in opt_results) / max(len(opt_results), 1)
    raw_tokens      = raw_results[0]["tokens_processed"] if raw_results else 0
    compression_pct = (1 - compressed_tokens / max(original_tokens, 1)) * 100
    speedup         = raw_avg_latency / max(opt_avg_latency, 0.001)
    raw_recall_avg  = sum(r["recall"] for r in raw_results) / max(len(raw_results), 1)
    opt_recall_avg  = sum(r["recall"] for r in opt_results) / max(len(opt_results), 1)
    now             = datetime.now().strftime("%Y-%m-%d %H:%M")

    token_pass    = "✅ PASS" if compression_pct >= 90 else f"⚠️  {compression_pct:.1f}% (target ≥ 90%)"
    speedup_note  = f"✅ {speedup:.1f}×" if speedup >= 10 else f"⚠️  {speedup:.1f}× (target ≥ 10×)"

    md: list[str] = [
        "# Context Optimizer — Benchmark Results",
        "",
        f"**Run**: {now}  |  **Corpus**: `{corpus_size}` ({n_lines:,} lines)",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Raw Baseline | Optimized (ToT) | Δ |",
        "|--------|:------------|:----------------|:--|",
        f"| Tokens processed (avg/query) | {raw_tokens:,} | {compressed_tokens:,} | **−{compression_pct:.1f}%** |",
        f"| Avg query latency | {raw_avg_latency:.1f} ms | {opt_avg_latency:.1f} ms | **{speedup:.1f}× faster** |",
        f"| Avg recall (ground-truth keywords) | {raw_recall_avg:.0%} | {opt_recall_avg:.0%} | — |",
        f"| One-time compression cost | — | {compress_time:.1f} s | — |",
        "",
        "### Threshold Gates",
        "",
        f"| Gate | Result |",
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
        "## Integrating These Results",
        "",
        "> **Instructions — update the whitepaper and design doc with the numbers above.**",
        ">",
        "> ### 1 · Whitepaper (`docs/whitepaper/proposed-whitepaper.md`)",
        ">",
        "> Copy the **Summary** table into the *Performance Evidence* section.",
        "> Replace any previous corpus-size row for `{corpus_size}` with:",
        f">",
        f">     Corpus: {corpus_size} ({n_lines:,} lines)",
        f">     Token reduction:       {compression_pct:.1f}%",
        f">     Avg query latency:     {opt_avg_latency:.1f} ms",
        f">     One-time compression:  {compress_time:.1f} s",
        ">",
        "> ### 2 · Architecture doc (`docs/design/ARCHITECTURE.md` §12)",
        ">",
        "> Update the benchmark table row for this corpus size:",
        f">",
        f">     | {corpus_size.capitalize()} | {n_lines:,} | ... | {compression_pct:.1f}% | {opt_avg_latency:.1f} ms | ... |",
        ">",
        "> Verify the threshold assertions still hold:",
        f">   - Token reduction ≥ 90%:  {token_pass}",
        f">   - Query speedup ≥ 10×:    {speedup_note}",
        ">",
        "> ### 3 · If using a real LLM (not mock)",
        ">",
        "> Also update the *E2E Experiment Results* table in §12 with:",
        ">   - Run date",
        ">   - Model names used for compression / embedding / reasoning",
        ">   - Judge score delta (if you added LLM-as-judge evaluation)",
        "",
        "---",
        f"*Generated by `benchmarks/run_benchmark.py` — do not edit manually.*",
    ]

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"  [results] Written → {out.relative_to(PROJECT_ROOT)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context Optimizer ground-truth benchmark (raw vs optimized).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size to benchmark (default: small)",
    )
    args = parser.parse_args()
    corpus_size = args.corpus

    print(f"\n{'='*60}")
    print(f"  Context Optimizer Benchmark — corpus: {corpus_size}")
    print(f"{'='*60}\n")

    # ── Step 1: corpus ────────────────────────────────────────────────────────
    print("[1/3] Preparing corpus …")
    corpus = download_corpus(corpus_size)
    n_lines = len(corpus)

    # ── Step 2: run both strategies ───────────────────────────────────────────
    print(f"\n[2/3] Running ground-truth queries ({len(GROUND_TRUTH_QUERIES)} queries) …")

    print("  → Raw baseline …")
    raw_results = run_raw_baseline(corpus, GROUND_TRUTH_QUERIES)
    raw_avg = sum(r["latency_ms"] for r in raw_results) / len(raw_results)
    print(f"     avg latency: {raw_avg:.1f} ms | tokens: {raw_results[0]['tokens_processed']:,}")

    print("  → Optimized (compress → ToT) …")
    opt_results, compress_time, original_tokens, compressed_tokens = run_optimized(
        corpus, GROUND_TRUTH_QUERIES
    )
    opt_avg = sum(r["latency_ms"] for r in opt_results) / len(opt_results)
    reduction = (1 - compressed_tokens / max(original_tokens, 1)) * 100
    print(f"     avg latency: {opt_avg:.1f} ms | token reduction: {reduction:.1f}%")

    # ── Step 3: write results.md ──────────────────────────────────────────────
    print("\n[3/3] Writing results …")
    out = write_results(
        corpus_size, n_lines,
        raw_results, opt_results,
        compress_time, original_tokens, compressed_tokens,
    )

    print(f"\n{'='*60}")
    print(f"  Done.  Results → {out.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
