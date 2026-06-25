"""
Tests for SemanticCache and CachedChromaRetriever.

SemanticCache is tested with a FakeEmbedder (no sentence-transformers download).
CachedChromaRetriever is tested by patching chromadb.PersistentClient with
a real in-memory EphemeralClient so ChromaDB itself is exercised but no
disk writes occur.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import FakeChromaEmbeddingFn, FakeEmbedder
from context_optimizer.cached_retriever import SemanticCache
from context_optimizer.compressor import CompressedChunk
from context_optimizer.raw_index import RawIndex

# ── SemanticCache ─────────────────────────────────────────────────────────────


def test_semantic_cache_empty_returns_none(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(embedding_model=fake_embedder, max_size=10)
    assert cache.get("any query") is None


def test_semantic_cache_exact_hit(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(embedding_model=fake_embedder, max_size=10)
    results = [{"chunk_id": "c0", "compressed_summary": "foo"}]
    cache.put("hello world", results)
    hit = cache.get("hello world")
    assert hit is not None
    assert hit[0]["chunk_id"] == "c0"


def test_semantic_cache_miss_for_different_query(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(
        embedding_model=fake_embedder,
        max_size=10,
        similarity_threshold=0.99,  # very high threshold → only exact hits
    )
    cache.put("hello world", [{"a": 1}])
    # Completely unrelated query — should miss
    assert cache.get("database connection error") is None


def test_semantic_cache_clear_resets_state(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(embedding_model=fake_embedder, max_size=10)
    cache.put("q1", [{"x": 1}])
    cache.clear()
    # After clear, the cache is empty
    assert len(cache.cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0
    # Getting from an empty cache returns None (and increments misses)
    assert cache.get("q1") is None


def test_semantic_cache_lru_eviction(fake_embedder: FakeEmbedder) -> None:
    # Use very distinct strings so SHA-256 embeddings have near-zero cosine
    # similarity with each other, preventing false cache hits.
    cache = SemanticCache(
        embedding_model=fake_embedder, max_size=3, similarity_threshold=0.99
    )
    cache.put("alpha_corpus", [{"id": 1}])
    cache.put("beta_database", [{"id": 2}])
    cache.put("gamma_network", [{"id": 3}])
    # Adding a 4th entry should evict the oldest (alpha_corpus)
    cache.put("delta_timeout", [{"id": 4}])
    assert len(cache.cache) == 3
    # The first item was evicted; exact match would return it, but it's gone
    assert "alpha_corpus" not in cache.cache


def test_semantic_cache_get_stats_structure(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(embedding_model=fake_embedder, max_size=10)
    cache.put("q", [{"x": 1}])
    cache.get("q")  # hit
    cache.get("miss_query_xyz")  # miss

    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert 0.0 <= stats["hit_rate_pct"] <= 100.0
    assert stats["max_size"] == 10
    assert "similarity_threshold" in stats


def test_semantic_cache_hit_rate_100_percent(fake_embedder: FakeEmbedder) -> None:
    cache = SemanticCache(embedding_model=fake_embedder, max_size=10)
    cache.put("q", [{"x": 1}])
    cache.get("q")
    stats = cache.get_stats()
    assert stats["hit_rate_pct"] == pytest.approx(100.0)


# ── CachedChromaRetriever ─────────────────────────────────────────────────────

# Strategy: patch chromadb.PersistentClient → chromadb.EphemeralClient
# and SentenceTransformer → FakeEmbedder.  This lets us test the full
# add_chunks / search / get_chunk_by_id logic path.


def _make_chromadb_patch():
    """Return a context-manager that replaces PersistentClient with EphemeralClient."""
    import chromadb

    class _EphemeralFactory:
        def __init__(self, *args, **kwargs):
            self._client = chromadb.EphemeralClient()

        def get_or_create_collection(self, name, **kwargs):
            # Use a simple default embedding function for the test
            return self._client.get_or_create_collection(
                name=name,
                embedding_function=FakeChromaEmbeddingFn(),
                metadata=kwargs.get("metadata", {}),
            )

        def __getattr__(self, item):
            return getattr(self._client, item)

    return patch(
        "context_optimizer.cached_retriever.chromadb.PersistentClient",
        new=_EphemeralFactory,
    )


def _make_retriever(tmp_path=None, raw_index=None):
    """Build a CachedChromaRetriever backed by an in-memory ChromaDB."""
    from context_optimizer.cached_retriever import CachedChromaRetriever

    fake_emb = FakeEmbedder()

    with _make_chromadb_patch():
        with patch(
            "context_optimizer.cached_retriever.SENTENCE_TRANSFORMERS_AVAILABLE", True
        ):
            with patch(
                "context_optimizer.cached_retriever.SentenceTransformer",
                return_value=fake_emb,
            ):
                with patch(
                    "context_optimizer.cached_retriever.chromadb.utils.embedding_functions"
                    ".SentenceTransformerEmbeddingFunction",
                    return_value=FakeChromaEmbeddingFn(),
                ):
                    retriever = CachedChromaRetriever(
                        collection_name="test_col",
                        persist_directory=str(tmp_path) if tmp_path else "/tmp/test",
                        raw_index=raw_index,
                    )
    return retriever


def _make_sample_chunks(n: int = 3) -> list[CompressedChunk]:
    return [
        CompressedChunk(
            chunk_id=f"chunk_{i:06d}",
            raw_text=f"raw text for chunk {i}",
            compressed_summary=f"summary {i} about python databases",
            entities=["entity_a"],
            keywords=["python", "database"],
            metadata={"source": "test"},
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
        )
        for i in range(n)
    ]


def test_cached_retriever_add_chunks_and_count(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    chunks = _make_sample_chunks(3)
    retriever.add_chunks(chunks)
    assert retriever.collection.count() == 3


def test_cached_retriever_search_returns_results(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    retriever.add_chunks(_make_sample_chunks(3))
    results = retriever.search("python database", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    if results:
        assert "chunk_id" in results[0]
        assert "compressed_summary" in results[0]


def test_cached_retriever_search_cache_hit(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    retriever.add_chunks(_make_sample_chunks(2))
    # First call → cache miss
    r1 = retriever.search("python", top_k=2)
    # Second identical call → cache hit
    r2 = retriever.search("python", top_k=2)
    assert r1 == r2
    assert retriever.cache.hits >= 1


def test_cached_retriever_search_skip_cache(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    retriever.add_chunks(_make_sample_chunks(2))
    retriever.search("query", top_k=2, use_cache=False)
    # Cache should not have been populated
    assert retriever.cache.get("query") is None


def test_cached_retriever_get_stats_structure(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    retriever.add_chunks(_make_sample_chunks(1))
    stats = retriever.get_stats()
    assert "total_chunks" in stats
    assert "collection_name" in stats
    assert "cache" in stats


def test_cached_retriever_clear_cache(tmp_path) -> None:
    retriever = _make_retriever(tmp_path)
    retriever.add_chunks(_make_sample_chunks(2))
    retriever.search("python", top_k=2)
    assert retriever.cache.hits + retriever.cache.misses >= 1
    retriever.clear_cache()
    assert retriever.cache.hits == 0


def test_get_chunk_by_id_from_chromadb(tmp_path) -> None:
    retriever = _make_retriever(tmp_path, raw_index=None)
    retriever.add_chunks(_make_sample_chunks(2))
    result = retriever.get_chunk_by_id("chunk_000000")
    assert result is not None
    assert result["chunk_id"] == "chunk_000000"
    assert "raw_text" in result


def test_get_chunk_by_id_missing_returns_none(tmp_path) -> None:
    retriever = _make_retriever(tmp_path, raw_index=None)
    retriever.add_chunks(_make_sample_chunks(1))
    assert retriever.get_chunk_by_id("nonexistent_id") is None


def test_get_chunk_by_id_uses_raw_index_fast_path(tmp_path) -> None:
    raw_idx = RawIndex(":memory:")
    raw_idx.add("chunk_000000", "FULL RAW TEXT — not truncated")
    retriever = _make_retriever(tmp_path, raw_index=raw_idx)
    retriever.add_chunks(_make_sample_chunks(1))
    result = retriever.get_chunk_by_id("chunk_000000")
    assert result is not None
    assert result["raw_text"] == "FULL RAW TEXT — not truncated"
    assert result.get("source") == "raw_index"
    raw_idx.close()
