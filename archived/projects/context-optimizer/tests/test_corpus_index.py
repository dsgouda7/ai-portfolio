"""
Tests for context_optimizer.index — CorpusIndex, IngestStats, QueryResult.

All external dependencies are mocked:
- LLM (compressor) is replaced with MockLLM
- ChromaDB is replaced with an in-memory EphemeralClient
- sentence-transformers is replaced with FakeEmbedder
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from context_optimizer.index import CorpusIndex, IngestStats, QueryResult

from conftest import FakeChromaEmbeddingFn, FakeEmbedder, MockLLM


# ── Helpers ───────────────────────────────────────────────────────────────────


def _patch_stack():
    """
    Return a list of patches that replace all external services with
    fast, in-memory fakes.  Must be applied with contextlib.ExitStack
    or nested with-blocks.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        import chromadb

        class _EphemeralFactory:
            def __init__(self, *args, **kwargs):
                self._client = chromadb.EphemeralClient()

            def get_or_create_collection(self, name, **kwargs):
                return self._client.get_or_create_collection(
                    name=name,
                    embedding_function=FakeChromaEmbeddingFn(),
                )

            def __getattr__(self, item):
                return getattr(self._client, item)

        fake_emb = FakeEmbedder()
        patches = [
            patch("context_optimizer.cached_retriever.chromadb.PersistentClient", new=_EphemeralFactory),
            patch("context_optimizer.cached_retriever.SENTENCE_TRANSFORMERS_AVAILABLE", True),
            patch("context_optimizer.cached_retriever.SentenceTransformer", return_value=fake_emb),
            patch(
                "context_optimizer.cached_retriever.chromadb.utils.embedding_functions"
                ".SentenceTransformerEmbeddingFunction",
                return_value=FakeChromaEmbeddingFn(),
            ),
            patch("context_optimizer.compressor._build_local_llm", return_value=MockLLM()),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    return _ctx()


SAMPLE_LINES = [
    "This is line one about Python.",
    "This is line two about databases.",
    "This is line three about machine learning.",
] * 60  # 180 lines → easily exceeds the 512-token threshold


# ── IngestStats dataclass ─────────────────────────────────────────────────────


def test_ingest_stats_fields() -> None:
    stats = IngestStats(
        chunks=5,
        original_tokens=1000,
        compressed_tokens=200,
        compression_ratio=0.2,
        elapsed_s=1.23,
    )
    assert stats.chunks == 5
    assert stats.compression_ratio == pytest.approx(0.2)
    assert stats.elapsed_s == pytest.approx(1.23)


# ── QueryResult dataclass ─────────────────────────────────────────────────────


def test_query_result_fields() -> None:
    result = QueryResult(
        answer="The answer is 42.",
        evidence=["snippet 1", "snippet 2"],
        tokens_used=50,
        latency_ms=12.5,
        branch_id="primary",
    )
    assert result.answer == "The answer is 42."
    assert len(result.evidence) == 2
    assert result.branch_id == "primary"


def test_query_result_branch_id_optional() -> None:
    result = QueryResult(answer="ans", evidence=[], tokens_used=0, latency_ms=0.0)
    assert result.branch_id is None


# ── CorpusIndex construction ──────────────────────────────────────────────────


def test_corpus_index_repr_before_ingest() -> None:
    idx = CorpusIndex()
    r = repr(idx)
    assert "CorpusIndex" in r
    assert "none" in r.lower() or "<none>" in r


def test_corpus_index_repr_after_ingest() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES, collection="test_col")
        r = repr(idx)
        assert "test_col" in r


# ── CorpusIndex.ingest ────────────────────────────────────────────────────────


def test_ingest_returns_ingest_stats() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        stats = idx.ingest(SAMPLE_LINES)
        assert isinstance(stats, IngestStats)
        assert stats.chunks >= 1
        assert stats.original_tokens > 0
        assert stats.compressed_tokens > 0
        assert 0.0 <= stats.compression_ratio <= 2.0
        assert stats.elapsed_s >= 0.0


def test_ingest_multiple_collections_are_independent() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES, collection="col_a")
        idx.ingest(SAMPLE_LINES, collection="col_b")
        assert "col_a" in idx._ingested_collections
        assert "col_b" in idx._ingested_collections


def test_ingest_twice_same_collection_appends() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        s1 = idx.ingest(SAMPLE_LINES[:60], collection="col")
        s2 = idx.ingest(SAMPLE_LINES[60:], collection="col")
        assert s1.chunks >= 1
        assert s2.chunks >= 1


def test_ingest_creates_raw_index_in_memory_for_ephemeral() -> None:
    with _patch_stack():
        idx = CorpusIndex()  # no persist_dir → ephemeral
        idx.ingest(SAMPLE_LINES, collection="test")
        raw_idx = idx._raw_indexes.get("test")
        assert raw_idx is not None
        assert raw_idx.count() > 0


# ── CorpusIndex.query ─────────────────────────────────────────────────────────


def test_query_raises_if_not_ingested() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        with pytest.raises(RuntimeError, match="no data"):
            idx.query("what happened?", collection="missing_col")


def test_query_returns_query_result() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        result = idx.query("what is Python?")
        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.evidence, list)
        assert result.latency_ms >= 0.0


def test_query_tot_strategy() -> None:
    with _patch_stack():
        idx = CorpusIndex(retrieval_strategy="tot")
        idx.ingest(SAMPLE_LINES)
        result = idx.query("database timeout")
        assert result.branch_id is not None


def test_query_simple_strategy() -> None:
    with _patch_stack():
        idx = CorpusIndex(retrieval_strategy="simple")
        idx.ingest(SAMPLE_LINES)
        result = idx.query("Python language")
        assert isinstance(result, QueryResult)


# ── CorpusIndex.query_many ────────────────────────────────────────────────────


def test_query_many_returns_list_of_results() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        results = idx.query_many(["q1", "q2", "q3"])
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)


def test_query_many_empty_list_returns_empty() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        assert idx.query_many([]) == []


# ── CorpusIndex.raw_lookup ────────────────────────────────────────────────────


def test_raw_lookup_after_ingest() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        # chunk_000000 should exist after ingest
        raw = idx.raw_lookup("chunk_000000")
        assert raw is not None
        assert len(raw) > 0


def test_raw_lookup_missing_returns_none() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        assert idx.raw_lookup("chunk_999999") is None


def test_raw_lookup_missing_collection_returns_none() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        assert idx.raw_lookup("chunk_000000", collection="no_such_col") is None


# ── CorpusIndex.raw_search ────────────────────────────────────────────────────


def test_raw_search_returns_results() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        hits = idx.raw_search("Python", top_k=5)
        assert isinstance(hits, list)
        if hits:
            assert "chunk_id" in hits[0]
            assert "raw_text" in hits[0]
            assert "rank" in hits[0]


def test_raw_search_missing_collection_returns_empty() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        assert idx.raw_search("python", collection="no_such_col") == []


# ── CorpusIndex lifecycle ─────────────────────────────────────────────────────


def test_context_manager_closes_index() -> None:
    with _patch_stack():
        with CorpusIndex() as idx:
            idx.ingest(SAMPLE_LINES)
        # After __exit__, raw_indexes should be cleared
        assert idx._raw_indexes == {}


def test_close_is_idempotent() -> None:
    with _patch_stack():
        idx = CorpusIndex()
        idx.ingest(SAMPLE_LINES)
        idx.close()
        idx.close()  # Second close must not raise


def test_persistent_index_does_not_delete_on_close(tmp_path) -> None:
    with _patch_stack():
        idx = CorpusIndex(persist_dir=str(tmp_path))
        idx.ingest(SAMPLE_LINES)
        idx.close()
    # User-supplied persist_dir must survive close()
    assert tmp_path.exists()
