"""
Context Optimizer Text Corpus Benchmark — raw baseline vs optimised (compress + ToT) on text data.

Usage
-----
    python text_corpus_benchmarks.py [--corpus small|medium|large]

Steps
-----
1. Load / generate a text corpus of the requested size.
2. Run ground-truth queries against both strategies:
      Raw baseline  — full corpus scan (monolithic approach)
      Optimised     — compress → ChromaDB index → ToT-retrieve
3. Gather metrics and write results.md in the benchmarks directory.

Corpus sizes
------------
    small   ~  5 000 lines  (fast, good for CI / quick checks)
    medium  ~ 25 000 lines  (representative)
    large   ~100 000 lines  (stress test, production-scale)

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── Project paths ─────────────────────────────────────────────────────────────
BENCH_DIR    = Path(__file__).parent
PROJECT_ROOT = BENCH_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Corpus sizes ──────────────────────────────────────────────────────────────
CORPUS_LINES: dict[str, int] = {
    "small":  5_000,
    "medium": 25_000,
    "large":  100_000,
}

# ── Ground-truth queries ──────────────────────────────────────────────────────
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


# ── Synthetic log generator ───────────────────────────────────────────────────

def _generate_synthetic_aks_logs(n: int) -> list[str]:
    """Generate deterministic AKS incident-style log lines for benchmarking (no randomness)."""
    templates = [
        "ERROR order-service [pod-k2m8q] System.TimeoutException CosmosDB substatus=21012 region=eastus2",
        "WARN ingress-nginx upstream timed out client=10.42.7.19 route=/v1/checkout",
        "ERROR api-gateway HTTP 504 upstream timeout route=/v1/checkout p95=8.7s",
        "WARN payment-service CosmosDB timeout retries=3 ru_charge=128.44 partition=tenant-445",
        "ERROR order-service PaymentConnector cancellation waterfall at SubmitAsync",
        "INFO recommendation-service request completed status=200 latency_ms=85",
        "WARN order-service CosmosDB retry ru_charge=64.0 partition=tenant-103",
        "INFO api-gateway request completed status=200 latency_ms=220",
    ]
    logs: list[str] = []
    base = datetime(2026, 6, 16, 1, 45, 0)
    for i in range(n):
        ts = (base + timedelta(seconds=i * 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logs.append(f"{ts} {templates[i % len(templates)]}")
    return logs


# ── Step 1 — Load / generate corpus ──────────────────────────────────────────

def load_corpus(size: str) -> list[str]:
    """
    Return corpus lines for the requested size.

    Always uses the inline deterministic AKS synthetic log generator so that
    the ground-truth queries (which reference AKS/CosmosDB terms) have a
    corpus that actually contains those keywords.  The books_*.txt files in
    data/ are general-purpose text and do NOT contain the AKS-domain
    terms the ground-truth queries require, which causes 0% recall when they
    are used here.
    """
    n = CORPUS_LINES[size]
    lines = _generate_synthetic_aks_logs(n)
    print(f"  [corpus] Generated {len(lines):,} synthetic AKS log lines")
    return lines


# ── Step 2a — Raw baseline (full corpus scan) ─────────────────────────────────

def run_raw_baseline(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Baseline: scan the full corpus for each query. No compression, no index."""
    results = []
    total_tokens = sum(len(ln) for ln in corpus) // 4

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


def _recall_from_answer(answer: str, must_contain: list[str]) -> float:
    """Check how many must_contain keywords appear in the LLM's answer text."""
    blob = answer.lower()
    hits = sum(1 for kw in must_contain if kw.lower() in blob)
    return hits / max(len(must_contain), 1)


# ── Step 2b — Optimised pipeline (compress → ToT-retrieve) ───────────────────

def run_optimized(
    corpus: list[str],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int, int]:
    """
    Optimised pipeline:
      1. Compress corpus with rolling-window LLM (one-time write cost).
      2. Store compressed chunks in an ephemeral ChromaDB.
      3. For each query use ToTReasoner to retrieve evidence and report metrics.
    """
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.tot_reasoner import ToTReasoner

    t0     = time.perf_counter()
    chunks = compress_corpus_rolling(corpus)
    compress_time = time.perf_counter() - t0

    original_tokens   = sum(c.original_tokens   for c in chunks)
    compressed_tokens = sum(c.compressed_tokens for c in chunks)
    ratio             = compressed_tokens / max(original_tokens, 1)
    print(
        f"  [compress] {len(chunks)} chunks | "
        f"{original_tokens:,} → {compressed_tokens:,} tokens "
        f"({ratio:.1%} ratio) | {compress_time:.1f}s"
    )

    retriever = _build_retriever(chunks)
    reasoner  = ToTReasoner(retriever=retriever)

    results = []
    for q in queries:
        branch_specs = [
            {"id": "main", "title": q["query"], "search_terms": q["must_contain"]}
        ]
        start      = time.perf_counter()
        tot        = reasoner.reason(
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

    _cleanup_retriever(retriever)
    return results, compress_time, original_tokens, compressed_tokens


def _build_retriever(chunks: list[Any]) -> Any:
    try:
        from context_optimizer.cached_retriever import CachedChromaRetriever
        tmp_dir   = tempfile.mkdtemp(prefix="co_text_bench_")
        retriever = CachedChromaRetriever(collection_name="text_benchmark", persist_directory=tmp_dir)
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


# ── Step 3 — Write results.md ─────────────────────────────────────────────────

def write_results(
    corpus_size: str,
    n_lines: int,
    raw_results:       list[dict[str, Any]],
    opt_results:       list[dict[str, Any]],
    compress_time:     float,
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

    token_pass   = "✅ PASS" if compression_pct >= 90 else f"⚠️  {compression_pct:.1f}% (target ≥ 90%)"
    speedup_note = f"✅ {speedup:.1f}×" if speedup >= 10 else f"⚠️  {speedup:.1f}× (target ≥ 10×)"

    md: list[str] = [
        "# Context Optimizer — Text Corpus Benchmark Results",
        "",
        f"**Run**: {now}  |  **Corpus**: `{corpus_size}` ({n_lines:,} lines)  |  **Modality**: Text",
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
        "## Integrating These Results",
        "",
        "> **Update the whitepaper and design doc with the numbers above.**",
        ">",
        "> ### 1 · Whitepaper (`docs/whitepaper/proposed-whitepaper.md`)",
        ">",
        "> Copy the **Summary** table into the *Performance Evidence* section.",
        f">",
        f">     Corpus: {corpus_size} ({n_lines:,} lines, text modality)",
        f">     Token reduction:       {compression_pct:.1f}%",
        f">     Avg query latency:     {opt_avg_latency:.1f} ms",
        f">     One-time compression:  {compress_time:.1f} s",
        ">",
        "> ### 2 · Architecture doc (`docs/design/ARCHITECTURE.md` §12)",
        ">",
        "> Update the benchmark table row for this corpus size:",
        f">",
        f">     | {corpus_size.capitalize()} | {n_lines:,} | text | {compression_pct:.1f}% | {opt_avg_latency:.1f} ms | ... |",
        ">",
        "> Threshold checks:",
        f">   - Token reduction ≥ 90%: {token_pass}",
        f">   - Query speedup ≥ 10×:  {speedup_note}",
        "",
        "---",
        f"*Generated by `benchmarks/text_corpus_benchmarks.py` — do not edit manually.*",
    ]

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"  [results] Written → {out.relative_to(PROJECT_ROOT)}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context Optimizer text corpus benchmark (raw vs optimised).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size (default: small)",
    )
    args = parser.parse_args()
    corpus_size = args.corpus

    print(f"\n{'='*60}")
    print(f"  Text Corpus Benchmark — {corpus_size}")
    print(f"{'='*60}\n")

    print("[1/3] Preparing corpus …")
    corpus = load_corpus(corpus_size)
    n_lines = len(corpus)

    print(f"\n[2/3] Running {len(GROUND_TRUTH_QUERIES)} ground-truth queries …")
    print("  → Raw baseline …")
    raw_results = run_raw_baseline(corpus, GROUND_TRUTH_QUERIES)
    raw_avg = sum(r["latency_ms"] for r in raw_results) / len(raw_results)
    print(f"     avg latency: {raw_avg:.1f} ms | tokens: {raw_results[0]['tokens_processed']:,}")

    print("  → Optimised (compress → ToT local) …")
    opt_results, compress_time, original_tokens, compressed_tokens = run_optimized(
        corpus, GROUND_TRUTH_QUERIES
    )
    opt_avg = sum(r["latency_ms"] for r in opt_results) / len(opt_results)
    reduction = (1 - compressed_tokens / max(original_tokens, 1)) * 100
    print(f"     avg latency: {opt_avg:.1f} ms | token reduction: {reduction:.1f}%")

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
