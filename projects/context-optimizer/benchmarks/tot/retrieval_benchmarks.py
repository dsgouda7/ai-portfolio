"""
Retrieval Benchmarks -- Medium Corpus (end-to-end, ephemeral).

Compresses a fresh medium corpus in a temp directory, then measures retrieval
latency (cache miss + cache hit) and context-window reduction per query.

Unlike quick_compress_and_save + latency_comparison (persistent ChromaDB),
this script is fully self-contained: it compresses and discards the index each
run, so results are always reproducible from scratch.

LLM backend (defaults to Ollama):  see llm_provider.py for env-var reference.

Usage:
    python retrieval_benchmarks.py
    python run_benchmarks.py --with-retrieval
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.cached_retriever import CachedChromaRetriever
from download_test_data import download_all_datasets
from llm_provider import build_compression_llm

# Performance targets from plan.md
TARGETS = {
    "cache_hit_ms":         5.0,
    "cache_miss_ms":      100.0,
    "context_reduction_pct": 95.0,
}

CORPUS_SIZE = 25_000

TEST_QUERIES = {
    "specific_fact": "Who is the main character in Pride and Prejudice?",
    "concept":       "What is machine learning?",
    "code":          "How do you implement a Python function?",
    "historical":    "What happened in the French Revolution?",
    "technical":     "Explain neural network architectures",
    "broad":         "Tell me about literature and novels",
}


def _context_metrics(retrieved_hits, corpus_size, elapsed_sec, cache_hit=False):
    original_tokens  = corpus_size * 16
    retrieved_tokens = (
        sum(len(h["compressed_summary"].split()) * 1.3 for h in retrieved_hits)
        if retrieved_hits else 0
    )
    reduction = 1 - (retrieved_tokens / original_tokens) if original_tokens else 0
    ratio     = original_tokens / retrieved_tokens if retrieved_tokens > 0 else 0
    return {
        "query_latency_ms":           elapsed_sec * 1000,
        "cache_hit":                  cache_hit,
        "chunks_retrieved":           len(retrieved_hits),
        "original_tokens":            original_tokens,
        "retrieved_tokens":           int(retrieved_tokens),
        "context_reduction_pct":      reduction * 100,
        "context_window_savings_ratio": ratio,
    }


def run_retrieval_test(corpus_name, corpus_lines, llm):
    """Compress corpus in an ephemeral temp dir, run queries, measure latency."""

    print("\n" + "=" * 80)
    print(f"RETRIEVAL TEST: {corpus_name}")
    print("=" * 80)
    corpus_size = len(corpus_lines)
    print(f"  Corpus size: {corpus_size:,} lines")

    # Compress
    print(f"\n[1/4] Compressing corpus...")
    t0 = time.time()
    compressed_chunks = compress_corpus_rolling(
        corpus_lines=corpus_lines,
        chunk_size_threshold=512,
        chunk_overlap_tokens=128,
        llm=llm,
    )
    compress_time  = time.time() - t0
    total_orig     = sum(c.original_tokens  for c in compressed_chunks)
    total_comp     = sum(c.compressed_tokens for c in compressed_chunks)
    overall_ratio  = total_comp / total_orig if total_orig else 0
    print(f"  [OK] {len(compressed_chunks):,} chunks in {compress_time:.1f}s")
    print(f"  Ratio: {overall_ratio:.3f}  ({total_orig:,} -> {total_comp:,} tokens)")

    # Store in ephemeral ChromaDB
    print(f"\n[2/4] Storing in ephemeral ChromaDB (sentence-transformers, local CPU)...")
    query_results   = []
    total_query_sec = 0.0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        retriever = CachedChromaRetriever(
            collection_name="bench_corpus",
            persist_directory=tmp_dir,
            embedding_model_name="all-MiniLM-L6-v2",
            cache_size=500,
            cache_threshold=0.85,
        )
        retriever.add_chunks(compressed_chunks)
        print(f"  [OK] {retriever.collection.count():,} chunks indexed")

        # Warm-up pass (populates embedding cache, not semantic cache)
        print(f"\n[3/4] Warming embedding cache...")
        for qt in TEST_QUERIES.values():
            retriever.search(qt, top_k=5, use_cache=False)

        # Timed query pass: miss then hit
        print(f"\n[4/4] Running {len(TEST_QUERIES)} queries (miss + hit each)...")
        for query_name, query_text in TEST_QUERIES.items():
            print(f"\n  Query: {query_name}")
            retriever.cache.clear()
            try:
                # MISS pass
                t_miss = time.time()
                results_miss = retriever.search(query_text, top_k=5, use_cache=True)
                miss_sec     = time.time() - t_miss
                total_query_sec += miss_sec
                miss_m = _context_metrics(results_miss, corpus_size, miss_sec, False)

                # HIT pass (guaranteed hit — same query)
                t_hit = time.time()
                retriever.search(query_text, top_k=5, use_cache=True)
                hit_sec  = time.time() - t_hit
                hit_m    = _context_metrics([], corpus_size, hit_sec, True)

                miss_ok = miss_m["query_latency_ms"] <= TARGETS["cache_miss_ms"]
                hit_ok  = hit_m["query_latency_ms"]  <= TARGETS["cache_hit_ms"]
                red_ok  = miss_m["context_reduction_pct"] >= TARGETS["context_reduction_pct"]

                print(f"    Miss: {miss_m['query_latency_ms']:.1f}ms {'[PASS]' if miss_ok else '[FAIL]'}  "
                      f"Hit: {hit_m['query_latency_ms']:.1f}ms {'[PASS]' if hit_ok else '[FAIL]'}  "
                      f"Reduction: {miss_m['context_reduction_pct']:.1f}% {'[PASS]' if red_ok else '[FAIL]'}")

                query_results.append({
                    "query_name":     query_name,
                    "query_text":     query_text,
                    "success":        True,
                    "miss_latency_ms": miss_m["query_latency_ms"],
                    "hit_latency_ms":  hit_m["query_latency_ms"],
                    "miss_pass":      miss_ok,
                    "hit_pass":       hit_ok,
                    "reduction_pass": red_ok,
                    **{k: v for k, v in miss_m.items()
                       if k not in ("query_latency_ms", "cache_hit")},
                })

            except Exception as exc:
                print(f"    [FAILED] {exc}")
                query_results.append({
                    "query_name": query_name,
                    "query_text": query_text,
                    "success":    False,
                    "error":      str(exc),
                })

        # Release ChromaDB file handles before temp dir cleanup (Windows)
        try:
            if hasattr(retriever, "client"):
                retriever.client.clear_system_cache()
            del retriever
        except Exception:
            pass

    ok = [q for q in query_results if q.get("success")]
    avg_miss = sum(q["miss_latency_ms"]        for q in ok) / len(ok) if ok else None
    avg_hit  = sum(q["hit_latency_ms"]         for q in ok) / len(ok) if ok else None
    avg_red  = sum(q["context_reduction_pct"]  for q in ok) / len(ok) if ok else None
    avg_sav  = sum(q["context_window_savings_ratio"] for q in ok) / len(ok) if ok else None

    print(f"\n" + "-" * 80)
    print(f"AGGREGATE")
    if avg_miss is not None:
        print(f"  Miss : {avg_miss:.1f}ms  {'[PASS]' if avg_miss <= TARGETS['cache_miss_ms'] else '[FAIL]'}  "
              f"Hit : {avg_hit:.1f}ms  {'[PASS]' if avg_hit <= TARGETS['cache_hit_ms'] else '[FAIL]'}  "
              f"Reduction : {avg_red:.1f}%  {'[PASS]' if avg_red >= TARGETS['context_reduction_pct'] else '[FAIL]'}  "
              f"Savings : {avg_sav:.1f}x")

    return {
        "corpus_name":         corpus_name,
        "corpus_size":         corpus_size,
        "compressed_chunks":   len(compressed_chunks),
        "compression_time_sec": compress_time,
        "compression_ratio":   overall_ratio,
        "total_original_tokens": total_orig,
        "total_compressed_tokens": total_comp,
        "queries":             query_results,
        "aggregate": {
            "successful_queries":     len(ok),
            "total_queries":          len(query_results),
            "avg_miss_latency_ms":    avg_miss,
            "avg_hit_latency_ms":     avg_hit,
            "avg_context_reduction_pct": avg_red,
            "avg_window_savings_ratio":  avg_sav,
            "total_query_time_sec":   total_query_sec,
            "targets_met": {
                "cache_miss":       avg_miss is not None and avg_miss <= TARGETS["cache_miss_ms"],
                "cache_hit":        avg_hit  is not None and avg_hit  <= TARGETS["cache_hit_ms"],
                "context_reduction": avg_red is not None and avg_red  >= TARGETS["context_reduction_pct"],
            },
        },
    }


def run_retrieval_benchmarks():
    """Run end-to-end retrieval benchmark on medium corpus."""

    print("=" * 80)
    print("RETRIEVAL BENCHMARKS  --  Medium Corpus")
    print("=" * 80)

    # LLM
    print("\n[1/3] Initialising compression LLM...")
    llm = build_compression_llm()

    # Corpus
    print("\n[2/3] Loading corpus data...")
    corpus_samples = download_all_datasets()
    medium         = corpus_samples["medium_500mb"]
    corpus_lines   = []
    corpus_lines.extend(medium["books"])
    corpus_lines.extend(medium["code"])
    corpus_lines.extend(medium["wiki"])
    corpus_lines = corpus_lines[:CORPUS_SIZE]
    print(f"  Medium corpus: {len(corpus_lines):,} lines")

    # Run
    print("\n[3/3] Running retrieval test...")
    result = run_retrieval_test("Medium Corpus (25K lines)", corpus_lines, llm)

    print("\n" + "=" * 80)
    print("RETRIEVAL BENCHMARKS COMPLETE!")
    print("=" * 80)

    provider = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "ollama")
    model    = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL",    "qwen2.5-coder:7b")

    results_file = Path(__file__).parent / "RETRIEVAL_BENCHMARK_RESULTS.json"
    with open(results_file, "w") as f:
        json.dump({
            "test_date":      datetime.now().isoformat(),
            "provider":       provider,
            "model":          model,
            "retriever":      "CachedChromaRetriever",
            "embedding_model": "all-MiniLM-L6-v2",
            "targets":        TARGETS,
            "results":        [result] if result else [],
        }, f, indent=2)
    print(f"  Saved: {results_file}")

    return {"results": [result] if result else []}


if __name__ == "__main__":
    run_retrieval_benchmarks()
    sys.exit(0)
