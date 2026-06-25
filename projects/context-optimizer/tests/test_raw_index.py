"""
Tests for context_optimizer.raw_index.RawIndex.

All tests use SQLite in-memory databases (:memory:) — no disk I/O,
no external dependencies.
"""

from __future__ import annotations

import threading

import pytest
from context_optimizer.raw_index import RawHit, RawIndex

# ── Helpers ───────────────────────────────────────────────────────────────────


def _populated_index(pairs: list[tuple[str, str]] | None = None) -> RawIndex:
    """Return an in-memory RawIndex pre-populated with sample chunks."""
    idx = RawIndex(":memory:")
    data = pairs or [
        ("chunk_000000", "The quick brown fox jumps over the lazy dog."),
        ("chunk_000001", "Python is a high-level programming language."),
        ("chunk_000002", "SQLite supports full-text search via FTS5."),
    ]
    for cid, text in data:
        idx.add(cid, text)
    return idx


# ── Construction & schema ─────────────────────────────────────────────────────


def test_raw_index_creates_without_error() -> None:
    idx = RawIndex(":memory:")
    assert idx.count() == 0


def test_raw_index_repr_shows_db_path_and_count() -> None:
    idx = RawIndex(":memory:")
    idx.add("c0", "hello world")
    r = repr(idx)
    assert ":memory:" in r
    assert "1" in r


def test_raw_index_context_manager() -> None:
    with RawIndex(":memory:") as idx:
        idx.add("c0", "text")
        assert idx.count() == 1
    # After __exit__, connection is closed; close() is idempotent
    idx.close()


# ── add / get ─────────────────────────────────────────────────────────────────


def test_add_and_get_roundtrip() -> None:
    idx = _populated_index()
    assert idx.get("chunk_000000") == "The quick brown fox jumps over the lazy dog."
    assert idx.get("chunk_000001") == "Python is a high-level programming language."


def test_get_returns_none_for_missing_chunk() -> None:
    idx = _populated_index()
    assert idx.get("chunk_999999") is None


def test_add_replaces_existing_chunk() -> None:
    idx = RawIndex(":memory:")
    idx.add("c0", "original text")
    idx.add("c0", "updated text")  # INSERT OR REPLACE
    assert idx.get("c0") == "updated text"
    assert idx.count() == 1


def test_add_many_batch_insert() -> None:
    idx = RawIndex(":memory:")
    pairs = [(f"chunk_{i:06d}", f"content {i}") for i in range(10)]
    idx.add_many(pairs)
    assert idx.count() == 10
    assert idx.get("chunk_000005") == "content 5"


def test_add_many_empty_list_is_noop() -> None:
    idx = RawIndex(":memory:")
    idx.add_many([])
    assert idx.count() == 0


# ── count ─────────────────────────────────────────────────────────────────────


def test_count_increments() -> None:
    idx = RawIndex(":memory:")
    assert idx.count() == 0
    idx.add("c0", "a")
    assert idx.count() == 1
    idx.add("c1", "b")
    assert idx.count() == 2


# ── FTS5 search ───────────────────────────────────────────────────────────────


def test_search_returns_hits_for_matching_keyword() -> None:
    idx = _populated_index()
    hits = idx.search("python", top_k=5)
    assert len(hits) >= 1
    assert any("Python" in h.raw_text for h in hits)


def test_search_returns_empty_for_no_match() -> None:
    idx = _populated_index()
    hits = idx.search("zzznomatchzzz", top_k=5)
    assert hits == []


def test_search_top_k_limits_results() -> None:
    pairs = [(f"chunk_{i:06d}", f"the fox {i}") for i in range(10)]
    idx = _populated_index(pairs)
    hits = idx.search("fox", top_k=3)
    assert len(hits) <= 3


def test_search_returns_raw_hit_namedtuples() -> None:
    idx = _populated_index()
    hits = idx.search("sqlite fts5", top_k=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert isinstance(hit, RawHit)
    assert isinstance(hit.chunk_id, str)
    assert isinstance(hit.raw_text, str)
    assert isinstance(hit.rank, float)


def test_search_best_match_ranked_first() -> None:
    """
    Chunk with more keyword matches should rank higher (more-negative BM25 score).
    """
    idx = _populated_index(
        [
            ("c_high", "fox fox fox fox jumping fox"),
            ("c_low", "fox"),
        ]
    )
    hits = idx.search("fox", top_k=5)
    assert len(hits) == 2
    # BM25 rank: more-negative = better match; hits are sorted ascending by rank
    assert hits[0].rank <= hits[1].rank


def test_search_malformed_fts5_query_returns_empty() -> None:
    """Malformed FTS5 syntax should not raise — returns empty list."""
    idx = _populated_index()
    hits = idx.search('"unclosed phrase', top_k=5)
    assert hits == []


# ── Thread-safety ─────────────────────────────────────────────────────────────


def test_concurrent_writes_are_safe(tmp_path) -> None:
    """Multiple threads writing to the same file-backed RawIndex must not corrupt data."""
    db = tmp_path / "concurrent.db"
    idx = RawIndex(db)
    errors: list[Exception] = []

    def _write(n: int) -> None:
        try:
            for i in range(n):
                # Use a unique prefix per thread to avoid key collisions
                idx.add(f"thread_{threading.get_ident()}_{i}", f"content {i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(20,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Each of 4 threads writes 20 unique chunks = 80 total
    assert idx.count() == 80
    idx.close()


# ── Persistence (file-backed) ─────────────────────────────────────────────────


def test_persistent_index_survives_reopen(tmp_path) -> None:
    db = tmp_path / "test.db"
    idx = RawIndex(db)
    idx.add("c0", "persisted content")
    idx.close()

    # Reopen
    idx2 = RawIndex(db)
    assert idx2.get("c0") == "persisted content"
    idx2.close()
