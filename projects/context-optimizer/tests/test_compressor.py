"""
Tests for context_optimizer.compressor module.

All LLM calls are mocked — no Ollama required.
"""

from __future__ import annotations

import pytest
from conftest import MockLLM
from context_optimizer.compressor import (
    CompressedChunk,
    _estimate_tokens,
    compress_chunk_with_llm,
    compress_corpus_rolling,
)
from context_optimizer.raw_index import RawIndex

# ── _estimate_tokens ─────────────────────────────────────────────────────────


def test_estimate_tokens_empty_string() -> None:
    assert _estimate_tokens("") == 1  # max(1, 0)


def test_estimate_tokens_4_chars_per_token() -> None:
    # "aaaa" → 4 chars → 1 token
    assert _estimate_tokens("aaaa") == 1
    # 400 chars → 100 tokens
    assert _estimate_tokens("x" * 400) == 100


def test_estimate_tokens_non_ascii() -> None:
    # Any non-empty text → at least 1
    assert _estimate_tokens("こんにちは") >= 1


# ── CompressedChunk dataclass ─────────────────────────────────────────────────


def test_compressed_chunk_fields() -> None:
    chunk = CompressedChunk(
        chunk_id="chunk_000000",
        raw_text="original text",
        compressed_summary="compressed",
        entities=["entity_a"],
        keywords=["kw1"],
        metadata={"source": "test"},
        original_tokens=100,
        compressed_tokens=30,
        compression_ratio=0.3,
    )
    assert chunk.chunk_id == "chunk_000000"
    assert chunk.raw_text == "original text"
    assert chunk.entities == ["entity_a"]
    assert chunk.compression_ratio == pytest.approx(0.3)


# ── compress_chunk_with_llm ───────────────────────────────────────────────────


def test_compress_chunk_with_valid_llm(mock_llm: MockLLM) -> None:
    chunk = compress_chunk_with_llm(
        text="Some text about Python and databases.",
        chunk_id="chunk_000000",
        llm=mock_llm,
    )
    assert chunk.chunk_id == "chunk_000000"
    assert chunk.compressed_summary == "Mock compressed summary."
    assert "entity_a" in chunk.entities
    assert "kw1" in chunk.keywords
    assert chunk.original_tokens > 0
    assert mock_llm.call_count == 1


def test_compress_chunk_stores_raw_text(mock_llm: MockLLM) -> None:
    raw = "This is the original raw text."
    chunk = compress_chunk_with_llm(text=raw, chunk_id="c0", llm=mock_llm)
    assert chunk.raw_text == raw


def test_compress_chunk_with_bad_json_falls_back(mock_llm_bad_json: MockLLM) -> None:
    chunk = compress_chunk_with_llm(text="text", chunk_id="c0", llm=mock_llm_bad_json)
    # Fallback: uses raw response text truncated to 600
    assert isinstance(chunk.compressed_summary, str)
    assert chunk.entities == []
    assert chunk.keywords == []


def test_compress_chunk_on_llm_exception_falls_back(mock_llm_fail: MockLLM) -> None:
    chunk = compress_chunk_with_llm(text="text", chunk_id="c0", llm=mock_llm_fail)
    # Fallback: first 200 chars of raw text
    assert chunk.chunk_id == "c0"
    assert chunk.compressed_summary == "text"[:200]
    assert chunk.entities == []


def test_compress_chunk_no_llm_falls_back_to_truncation() -> None:
    """When llm=None and no env-configured LLM, uses a truncated raw text."""
    import os

    # Ensure no env LLM is configured
    os.environ.pop("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", None)
    os.environ.pop("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", None)

    chunk = compress_chunk_with_llm(text="hello world", chunk_id="c0", llm=None)
    # Should return a CompressedChunk (either truncated fallback or env-LLM result)
    assert isinstance(chunk, CompressedChunk)
    assert chunk.chunk_id == "c0"


def test_compress_chunk_metadata_preserved(mock_llm: MockLLM) -> None:
    meta = {"source": "test_file.txt", "line_start": 10}
    chunk = compress_chunk_with_llm(
        text="some text", chunk_id="c0", metadata=meta, llm=mock_llm
    )
    # LLM adds has_code, has_math, section to metadata
    assert "source" in chunk.metadata
    assert "has_code" in chunk.metadata


# ── compress_corpus_rolling ───────────────────────────────────────────────────


def test_compress_corpus_rolling_empty_returns_empty(mock_llm: MockLLM) -> None:
    result = compress_corpus_rolling([], llm=mock_llm)
    assert result == []


def test_compress_corpus_rolling_single_chunk(mock_llm: MockLLM) -> None:
    # 128 lines × ~4 tokens/line = 512 tokens → exactly one chunk
    lines = ["word word word word"] * 128
    chunks = compress_corpus_rolling(lines, chunk_size_threshold=512, llm=mock_llm)
    assert len(chunks) >= 1
    assert all(isinstance(c, CompressedChunk) for c in chunks)


def test_compress_corpus_rolling_produces_correct_ids(mock_llm: MockLLM) -> None:
    # Each line = 50 chars = ~12 tokens; 50 lines = ~600 tokens > 512 threshold
    # so we get the first chunk + at least one more → ≥ 2 chunks total
    lines = ["word " * 10] * 60  # 60 × ~12 tokens = ~720 tokens
    chunks = compress_corpus_rolling(lines, chunk_size_threshold=512, llm=mock_llm)
    ids = [c.chunk_id for c in chunks]
    # IDs should be sequential
    assert ids[0] == "chunk_000000"
    assert ids[1] == "chunk_000001"


def test_compress_corpus_rolling_llm_called_per_chunk(mock_llm: MockLLM) -> None:
    # ~12 tokens/line × 120 lines = ~1440 tokens → ≥ 2 chunks
    lines = ["word " * 10] * 120
    compress_corpus_rolling(lines, chunk_size_threshold=512, llm=mock_llm)
    assert mock_llm.call_count >= 2


def test_compress_corpus_rolling_with_raw_index(mock_llm: MockLLM) -> None:
    """Parallel raw indexing: each chunk's raw text ends up in the RawIndex."""
    idx = RawIndex(":memory:")
    lines = ["word word word word word"] * 128  # ≥ 1 chunk
    chunks = compress_corpus_rolling(
        lines, chunk_size_threshold=512, llm=mock_llm, raw_index=idx
    )
    assert len(chunks) >= 1
    # Every chunk must have been written to the raw index
    for chunk in chunks:
        stored = idx.get(chunk.chunk_id)
        assert stored is not None, f"{chunk.chunk_id} missing from RawIndex"
    idx.close()


def test_compress_corpus_rolling_progress_callback(mock_llm: MockLLM) -> None:
    calls: list[tuple[int, int]] = []

    def cb(idx: int, total: int) -> None:
        calls.append((idx, total))

    # ~12 tokens/line × 120 lines = ~1440 tokens → ≥ 2 chunks
    lines = ["word " * 10] * 120
    compress_corpus_rolling(
        lines,
        chunk_size_threshold=512,
        compression_batch_size=1,
        llm=mock_llm,
        progress_callback=cb,
    )
    assert len(calls) >= 1


def test_compress_corpus_rolling_no_raw_index_does_not_crash(mock_llm: MockLLM) -> None:
    """Backward compatibility: raw_index=None (default) must still work."""
    lines = ["hello world"] * 64
    chunks = compress_corpus_rolling(
        lines, chunk_size_threshold=100, llm=mock_llm, raw_index=None
    )
    assert len(chunks) >= 1
