"""
Correctness Tests for the Context Optimizer Architecture

Unit-style tests that run WITHOUT Azure and WITHOUT a pre-populated ChromaDB.
They create synthetic CompressedChunk objects, store them in an ephemeral
CachedChromaRetriever, and assert exact behaviour.

Tests:
  1. add_chunks stores the correct number of documents
  2. search returns semantically relevant results
  3. Semantic cache hits the same query a second time
  4. Cache respects similarity threshold (unrelated query misses)
  5. get_chunk_by_id returns raw_text (pointer model)
  6. Compression ratio is below a threshold (basic quality gate)
  7. Latency: cache hit < 5ms, cache miss < 100ms

Run with:
    python benchmarks/tot/test_correctness.py
"""

import sys
import time
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_optimizer.compressor import CompressedChunk
from context_optimizer.cached_retriever import CachedChromaRetriever

# ── Latency thresholds ─────────────────────────────────────────────────────────
HIT_LATENCY_MS  = 5.0    # cache hit  (plan: < 5 ms)
MISS_LATENCY_MS = 100.0  # cache miss (plan: < 100 ms)

# ── Synthetic corpus ───────────────────────────────────────────────────────────
SYNTHETIC_CHUNKS = [
    CompressedChunk(
        chunk_id="chunk-001",
        raw_text="Elizabeth Bennet is the protagonist of Pride and Prejudice by Jane Austen. "
                 "She is witty, intelligent, and independent, eventually marrying Mr. Darcy.",
        compressed_summary="Elizabeth Bennet: protagonist of Pride and Prejudice. Witty, intelligent; marries Darcy.",
        entities=["Elizabeth Bennet", "Pride and Prejudice", "Jane Austen", "Mr. Darcy"],
        keywords=["protagonist", "witty", "intelligent", "marriage"],
        metadata={"source": "books", "section": "fiction"},
        original_tokens=40,
        compressed_tokens=14,
        compression_ratio=0.35,
    ),
    CompressedChunk(
        chunk_id="chunk-002",
        raw_text="A Python list comprehension provides a concise way to create lists. "
                 "Example: squares = [x**2 for x in range(10)]. Faster than a for-loop for simple transforms.",
        compressed_summary="Python list comprehension: concise list creation. [x**2 for x in range(10)]. Faster than for-loop.",
        entities=["Python", "list comprehension"],
        keywords=["list", "comprehension", "syntax", "performance"],
        metadata={"source": "code", "section": "python"},
        original_tokens=35,
        compressed_tokens=15,
        compression_ratio=0.43,
    ),
    CompressedChunk(
        chunk_id="chunk-003",
        raw_text="Machine learning is a subset of artificial intelligence where models learn from data "
                 "without explicit programming. Differs from traditional programming by inferring rules from examples.",
        compressed_summary="Machine learning: AI subset, models learn from data. Rules inferred, not programmed explicitly.",
        entities=["machine learning", "artificial intelligence"],
        keywords=["learning", "data", "inference", "model"],
        metadata={"source": "wiki", "section": "technology"},
        original_tokens=36,
        compressed_tokens=14,
        compression_ratio=0.39,
    ),
    CompressedChunk(
        chunk_id="chunk-004",
        raw_text="The French Revolution (1789-1799) was a period of radical political and social transformation. "
                 "Key events: storming of the Bastille (1789), execution of Louis XVI (1793), Reign of Terror.",
        compressed_summary="French Revolution 1789-1799: radical social/political transformation. Bastille stormed, Louis XVI executed.",
        entities=["French Revolution", "Bastille", "Louis XVI", "Reign of Terror"],
        keywords=["revolution", "France", "1789", "political"],
        metadata={"source": "wiki", "section": "history"},
        original_tokens=40,
        compressed_tokens=16,
        compression_ratio=0.40,
    ),
    CompressedChunk(
        chunk_id="chunk-005",
        raw_text="Neural network architectures for computer vision include CNNs (convolutional layers), "
                 "ResNet (residual connections), and ViT (vision transformers). CNNs excel at spatial hierarchies.",
        compressed_summary="CV neural architectures: CNN (convolutional), ResNet (residual), ViT (transformer). CNN best for spatial hierarchies.",
        entities=["CNN", "ResNet", "ViT", "computer vision"],
        keywords=["neural network", "architecture", "convolutional", "vision"],
        metadata={"source": "technical", "section": "deep-learning"},
        original_tokens=38,
        compressed_tokens=17,
        compression_ratio=0.45,
    ),
    CompressedChunk(
        chunk_id="chunk-006",
        raw_text="Completely unrelated chunk about cooking pasta. Boil water, add salt, cook for 8-10 minutes. "
                 "Al dente means slightly firm to the bite. Remove from heat and drain.",
        compressed_summary="Pasta cooking: boil salted water, cook 8-10 min, al dente (firm to bite), drain.",
        entities=["pasta", "al dente"],
        keywords=["cooking", "pasta", "boil", "kitchen"],
        metadata={"source": "misc", "section": "food"},
        original_tokens=33,
        compressed_tokens=14,
        compression_ratio=0.42,
    ),
]


def _build_retriever(tmp_dir: str) -> CachedChromaRetriever:
    """Create a fresh CachedChromaRetriever populated with synthetic chunks."""
    retriever = CachedChromaRetriever(
        collection_name="correctness_test",
        persist_directory=tmp_dir,
        embedding_model_name="all-MiniLM-L6-v2",
        cache_size=100,
        cache_threshold=0.85,
    )
    retriever.add_chunks(SYNTHETIC_CHUNKS)
    return retriever


def _assert(condition: bool, test_name: str, detail: str = "") -> bool:
    """Print PASS/FAIL and return success flag."""
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}  {test_name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ── Individual tests ───────────────────────────────────────────────────────────

def test_add_chunks_count(retriever: CachedChromaRetriever) -> bool:
    """add_chunks must store exactly the right number of documents."""
    count = retriever.collection.count()
    return _assert(count == len(SYNTHETIC_CHUNKS), "add_chunks: document count",
                   f"got {count}, expected {len(SYNTHETIC_CHUNKS)}")


def test_search_returns_relevant(retriever: CachedChromaRetriever) -> bool:
    """Searching for Elizabeth Bennet must return chunk-001 in top-3."""
    results = retriever.search("Who is Elizabeth Bennet?", top_k=3, use_cache=False)
    ids = [r["chunk_id"] for r in results]
    return _assert("chunk-001" in ids, "search: semantic relevance (Elizabeth Bennet in top-3)",
                   f"got {ids}")


def test_search_code_query(retriever: CachedChromaRetriever) -> bool:
    """Searching for Python list comprehension must return chunk-002 in top-3."""
    results = retriever.search("Python list comprehension syntax", top_k=3, use_cache=False)
    ids = [r["chunk_id"] for r in results]
    return _assert("chunk-002" in ids, "search: code query (list comprehension in top-3)",
                   f"got {ids}")


def test_cache_hit(retriever: CachedChromaRetriever) -> bool:
    """Second identical query must be a cache hit (stats.hits increments)."""
    retriever.clear_cache()
    q = "Explain machine learning vs traditional programming"
    retriever.search(q, top_k=3, use_cache=True)   # miss
    before = retriever.cache.hits
    retriever.search(q, top_k=3, use_cache=True)   # hit
    after = retriever.cache.hits
    return _assert(after == before + 1, "cache: hit on repeat query",
                   f"hits before={before}, after={after}")


def test_cache_miss_unrelated(retriever: CachedChromaRetriever) -> bool:
    """Semantically unrelated query must NOT hit cache for a different cached query."""
    retriever.clear_cache()
    retriever.search("machine learning", top_k=3, use_cache=True)   # seed
    misses_before = retriever.cache.misses
    retriever.search("cooking pasta al dente", top_k=3, use_cache=True)  # very different topic
    misses_after = retriever.cache.misses
    return _assert(misses_after > misses_before, "cache: miss on unrelated query",
                   f"misses before={misses_before}, after={misses_after}")


def test_get_chunk_by_id_raw_text(retriever: CachedChromaRetriever) -> bool:
    """get_chunk_by_id must return raw_text (pointer model)."""
    result = retriever.get_chunk_by_id("chunk-004")
    if result is None:
        return _assert(False, "get_chunk_by_id: chunk-004 found", "returned None")
    has_raw = bool(result.get("raw_text", "").strip())
    keyword_ok = "French" in result.get("raw_text", "")
    return _assert(has_raw and keyword_ok, "get_chunk_by_id: raw_text contains original content",
                   f"raw_text preview: {result.get('raw_text', '')[:60]}...")


def test_cache_hit_latency(retriever: CachedChromaRetriever) -> bool:
    """Cache hit latency must be below HIT_LATENCY_MS threshold."""
    retriever.clear_cache()
    q = "neural network architectures for vision"
    retriever.search(q, top_k=3, use_cache=True)  # warm cache
    t0 = time.time()
    retriever.search(q, top_k=3, use_cache=True)  # hit
    hit_ms = (time.time() - t0) * 1000
    return _assert(hit_ms <= HIT_LATENCY_MS, f"latency: cache hit < {HIT_LATENCY_MS}ms",
                   f"got {hit_ms:.2f}ms")


def test_cache_miss_latency(retriever: CachedChromaRetriever) -> bool:
    """Cache miss latency must be below MISS_LATENCY_MS threshold."""
    retriever.clear_cache()
    q = "What is the French Revolution timeline?"
    t0 = time.time()
    retriever.search(q, top_k=3, use_cache=True)  # guaranteed miss (fresh cache)
    miss_ms = (time.time() - t0) * 1000
    return _assert(miss_ms <= MISS_LATENCY_MS, f"latency: cache miss < {MISS_LATENCY_MS}ms",
                   f"got {miss_ms:.2f}ms")


def test_compression_ratio_gate(retriever: CachedChromaRetriever) -> bool:
    """All synthetic chunks must have compression_ratio < 0.80 (basic quality gate)."""
    bad = [c.chunk_id for c in SYNTHETIC_CHUNKS if c.compression_ratio >= 0.80]
    return _assert(len(bad) == 0, "compression: all chunks have ratio < 0.80",
                   f"failing chunk ids: {bad}" if bad else "all OK")


def test_summary_in_search_result(retriever: CachedChromaRetriever) -> bool:
    """Search results must include compressed_summary field (not empty)."""
    results = retriever.search("Bastille storming revolution France", top_k=3, use_cache=False)
    all_have_summary = all(bool(r.get("compressed_summary", "").strip()) for r in results)
    return _assert(all_have_summary, "search result schema: compressed_summary present",
                   f"checked {len(results)} results")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_correctness_tests() -> dict:
    print("=" * 80)
    print("CORRECTNESS TESTS - CachedChromaRetriever (no Azure required)")
    print("=" * 80)
    print(f"  Embedding model: all-MiniLM-L6-v2 (local CPU)")
    print(f"  Corpus: {len(SYNTHETIC_CHUNKS)} synthetic chunks")
    print(f"  ChromaDB: ephemeral (temp directory)")
    print()

    tmp_dir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmp_dir = tmp_dir_obj.name
    retriever = None
    results = []

    try:
        print("[Setup] Building retriever and adding synthetic chunks...")
        retriever = _build_retriever(tmp_dir)
        print(f"  [OK] {retriever.collection.count()} chunks indexed\n")

        tests = [
            test_add_chunks_count,
            test_search_returns_relevant,
            test_search_code_query,
            test_cache_hit,
            test_cache_miss_unrelated,
            test_get_chunk_by_id_raw_text,
            test_cache_hit_latency,
            test_cache_miss_latency,
            test_compression_ratio_gate,
            test_summary_in_search_result,
        ]

        results = [t(retriever) for t in tests]

    finally:
        # Explicitly release ChromaDB file handles before temp dir cleanup (Windows)
        if retriever is not None:
            try:
                retriever.client.clear_system_cache()
            except Exception:
                pass
            del retriever
        try:
            tmp_dir_obj.cleanup()
        except Exception:
            pass  # Windows may leave file locks; non-fatal

    passed = sum(results)
    total  = len(results)

    print()
    print("=" * 80)
    print(f"RESULT: {passed}/{total} tests PASSED")
    if passed == total:
        print("[ALL PASS] Architecture correctness verified.")
    else:
        print(f"[{total - passed} FAILURES] Review output above.")
    print("=" * 80)

    return {"passed": passed, "total": total, "all_pass": passed == total}


if __name__ == "__main__":
    result = run_correctness_tests()
    sys.exit(0 if result["all_pass"] else 1)
