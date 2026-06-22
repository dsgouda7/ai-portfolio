"""
Accuracy Benchmarks -- Medium Corpus (precision / recall / F1).

Loads compressed chunks from the persistent ChromaDB populated by
quick_compress_and_save.py and measures retrieval quality (F1) plus
latency (cache miss and hit) on a set of ground-truth queries.

Prerequisites:
    python quick_compress_and_save.py   # populates chroma_db/

Usage:
    python accuracy_benchmarks.py
    python run_benchmarks.py            # runs this after correctness + latency
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from context_optimizer.cached_retriever import CachedChromaRetriever

# Performance targets
F1_TARGETS      = {"easy": 0.85, "medium": 0.70, "hard": 0.60}
LATENCY_TARGETS = {"cache_miss_ms": 100.0, "cache_hit_ms": 5.0}

CHROMA_DIR      = Path(__file__).parent / "chroma_db"
COLLECTION      = "medium_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class GroundTruthQuery:
    query_text:          str
    expected_chunk_ids:  list
    domain:              str
    difficulty:          str   # easy | medium | hard


# Queries with auto-derived ground truth (populated at runtime by keyword overlap)
GROUND_TRUTH_QUERIES = [
    GroundTruthQuery("Who is Elizabeth Bennet?",                                       [], "books",     "easy"),
    GroundTruthQuery("What is a Python list comprehension?",                           [], "code",      "easy"),
    GroundTruthQuery("How does machine learning differ from traditional programming?", [], "technical", "medium"),
    GroundTruthQuery("Explain the French Revolution timeline",                         [], "history",   "medium"),
    GroundTruthQuery("Compare neural network architectures for computer vision",       [], "technical", "hard"),
    GroundTruthQuery("Literary themes of social class in 19th century novels",         [], "books",     "hard"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _precision_recall_f1(retrieved, expected_ids):
    ret_ids = {r["chunk_id"] for r in retrieved}
    exp_ids = set(expected_ids)
    if not ret_ids or not exp_ids:
        return 0.0, 0.0, 0.0
    tp        = len(ret_ids & exp_ids)
    precision = tp / len(ret_ids)
    recall    = tp / len(exp_ids)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _derive_ground_truth(retriever, query_text, total_chunks):
    """Auto-label relevant chunks by keyword overlap (> 40% match)."""
    candidates  = retriever.search(query_text, top_k=min(20, total_chunks), use_cache=False)
    query_toks  = set(query_text.lower().split())
    relevant    = []
    for hit in candidates:
        chunk_text = f"{hit['compressed_summary']} {' '.join(hit['entities'])} {' '.join(hit['keywords'])}".lower()
        overlap = len(query_toks & set(chunk_text.split()))
        if overlap / len(query_toks) > 0.4:
            relevant.append(hit["chunk_id"])
    return relevant


# ── Core test ─────────────────────────────────────────────────────────────────

def run_accuracy_test(corpus_label, retriever, ground_truth_queries):
    total_chunks = retriever.collection.count()
    print("\n" + "=" * 80)
    print(f"ACCURACY TEST: {corpus_label}")
    print("=" * 80)
    print(f"  Chunks: {total_chunks:,}   Queries: {len(ground_truth_queries)}")

    results = []
    for i, gt in enumerate(ground_truth_queries, 1):
        print(f"\n  Query {i}/{len(ground_truth_queries)}: {gt.difficulty.upper()}")
        print(f"    Text: '{gt.query_text[:70]}'")

        if not gt.expected_chunk_ids:
            gt.expected_chunk_ids = _derive_ground_truth(retriever, gt.query_text, total_chunks)

        if not gt.expected_chunk_ids:
            print(f"    [SKIP] No ground-truth chunks found for this query")
            continue

        try:
            # MISS pass
            retriever.clear_cache()
            t_miss   = time.time()
            retrieved = retriever.search(gt.query_text, top_k=5, use_cache=True)
            miss_ms  = (time.time() - t_miss) * 1000

            # HIT pass (same query)
            t_hit  = time.time()
            retriever.search(gt.query_text, top_k=5, use_cache=True)
            hit_ms = (time.time() - t_hit) * 1000

            precision, recall, f1 = _precision_recall_f1(retrieved, gt.expected_chunk_ids)

            f1_target = F1_TARGETS.get(gt.difficulty, 0.60)
            f1_ok     = f1      >= f1_target
            miss_ok   = miss_ms <= LATENCY_TARGETS["cache_miss_ms"]
            hit_ok    = hit_ms  <= LATENCY_TARGETS["cache_hit_ms"]

            print(f"    P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
                  f"{'[PASS]' if f1_ok else '[FAIL]'} (target >={f1_target})")
            print(f"    Miss: {miss_ms:.1f}ms {'[PASS]' if miss_ok else '[FAIL]'}  "
                  f"Hit: {hit_ms:.1f}ms {'[PASS]' if hit_ok else '[FAIL]'}")

            results.append({
                "query_text":        gt.query_text,
                "domain":            gt.domain,
                "difficulty":        gt.difficulty,
                "expected_chunks":   len(gt.expected_chunk_ids),
                "retrieved_chunks":  len(retrieved),
                "precision":         precision,
                "recall":            recall,
                "f1":                f1,
                "f1_target":         f1_target,
                "f1_pass":           f1_ok,
                "miss_latency_ms":   miss_ms,
                "hit_latency_ms":    hit_ms,
                "miss_latency_pass": miss_ok,
                "hit_latency_pass":  hit_ok,
                "success":           True,
            })

        except Exception as exc:
            print(f"    [FAILED] {exc}")
            results.append({
                "query_text": gt.query_text,
                "domain":     gt.domain,
                "difficulty": gt.difficulty,
                "error":      str(exc),
                "success":    False,
            })

    ok = [r for r in results if r.get("success")]
    if ok:
        avg_p    = sum(r["precision"] for r in ok) / len(ok)
        avg_r    = sum(r["recall"]    for r in ok) / len(ok)
        avg_f1   = sum(r["f1"]        for r in ok) / len(ok)
        avg_miss = sum(r["miss_latency_ms"] for r in ok) / len(ok)
        avg_hit  = sum(r["hit_latency_ms"]  for r in ok) / len(ok)

        print(f"\n" + "-" * 80)
        print(f"AGGREGATE  ({len(ok)}/{len(results)} queries)")
        print(f"  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}  "
              f"Miss={avg_miss:.1f}ms  Hit={avg_hit:.1f}ms")

        for diff in ("easy", "medium", "hard"):
            sub = [r for r in ok if r["difficulty"] == diff]
            if sub:
                df1    = sum(r["f1"] for r in sub) / len(sub)
                passes = sum(1 for r in sub if r["f1_pass"])
                print(f"  {diff.capitalize():6s}: F1={df1:.3f}  {passes}/{len(sub)} PASS")
    else:
        avg_p = avg_r = avg_f1 = avg_miss = avg_hit = None

    return {
        "corpus_name":  corpus_label,
        "total_chunks": total_chunks,
        "queries":      results,
        "aggregate": {
            "successful_queries": len(ok),
            "total_queries":      len(results),
            "avg_precision":      avg_p,
            "avg_recall":         avg_r,
            "avg_f1":             avg_f1,
            "avg_miss_latency_ms": avg_miss,
            "avg_hit_latency_ms":  avg_hit,
            "f1_targets":         F1_TARGETS,
            "latency_targets":    LATENCY_TARGETS,
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run_accuracy_benchmarks():
    print("=" * 80)
    print("ACCURACY BENCHMARKS  --  ChromaDB Semantic Search")
    print("=" * 80)

    if not CHROMA_DIR.exists():
        print(f"\n[ERROR] ChromaDB not found: {CHROMA_DIR}")
        print("[ERROR] Run  python quick_compress_and_save.py  first.")
        return None

    print(f"\n[1/2] Loading ChromaDB ({EMBEDDING_MODEL})...")
    retriever = CachedChromaRetriever(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_model_name=EMBEDDING_MODEL,
        cache_size=1000,
        cache_threshold=0.85,
    )
    print(f"  medium_corpus: {retriever.collection.count():,} chunks")

    print(f"\n[2/2] Running accuracy tests...")
    result = run_accuracy_test("Medium Corpus (25K lines)", retriever, GROUND_TRUTH_QUERIES)
    all_results = [result] if result else []

    print("\n" + "=" * 80)
    print("ACCURACY BENCHMARKS COMPLETE!")
    print("=" * 80)

    out_file = Path(__file__).parent / "ACCURACY_BENCHMARK_RESULTS.json"
    with open(out_file, "w") as f:
        json.dump({
            "test_date":  datetime.now().isoformat(),
            "storage":    f"CachedChromaRetriever ({EMBEDDING_MODEL})",
            "chroma_dir": str(CHROMA_DIR),
            "f1_targets":      F1_TARGETS,
            "latency_targets": LATENCY_TARGETS,
            "results":    all_results,
        }, f, indent=2)
    print(f"  Saved: {out_file}")

    return all_results


if __name__ == "__main__":
    results = run_accuracy_benchmarks()
    sys.exit(0 if results else 1)
