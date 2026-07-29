"""
Tests for context_optimizer.raw_index.BlockIndex.

Verifies the file-pointer store:
  - Metadata round-trip (add → get_meta)
  - Raw text read (get_text seeks correct byte range in source file)
  - Error handling (missing file, unknown block_id)
  - Batch insert (add_many)
  - Sorting, counting, upsert semantics
  - Thread safety
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest
from context_optimizer.raw_index import BlockIndex, BlockPointer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _tmp_corpus(blocks: list[str]) -> tuple[Path, list[tuple[int, int]]]:
    """
    Write *blocks* sequentially to a temp file.
    Returns (file_path, [(byte_start, byte_end), ...]).
    """
    f = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".txt", delete=False, prefix="co_blockindex_test_"
    )
    ranges: list[tuple[int, int]] = []
    pos = 0
    for block in blocks:
        raw = block.encode("utf-8")
        f.write(raw)
        ranges.append((pos, pos + len(raw)))
        pos += len(raw)
    f.close()
    return Path(f.name), ranges


# ── add_block / get_meta ──────────────────────────────────────────────────────


def test_add_and_get_meta_roundtrip():
    idx = BlockIndex(":memory:")
    idx.add_block("b0", "/data/corpus.txt", 0, 1000)
    meta = idx.get_meta("b0")
    assert meta is not None
    assert meta.block_id == "b0"
    assert meta.file_path == "/data/corpus.txt"
    assert meta.byte_start == 0
    assert meta.byte_end == 1000


def test_size_bytes_property():
    idx = BlockIndex(":memory:")
    idx.add_block("b0", "/f.txt", 100, 600)
    assert idx.get_meta("b0").size_bytes == 500


def test_get_meta_returns_none_for_unknown_id():
    idx = BlockIndex(":memory:")
    assert idx.get_meta("nonexistent") is None


def test_duplicate_id_upserts():
    """Last write wins — INSERT OR REPLACE semantics."""
    idx = BlockIndex(":memory:")
    idx.add_block("b0", "/f.txt", 0, 100)
    idx.add_block("b0", "/f.txt", 200, 300)
    meta = idx.get_meta("b0")
    assert meta.byte_start == 200
    assert meta.byte_end == 300
    assert idx.count() == 1  # still one entry


# ── get_text — core file-pointer verification ─────────────────────────────────


def test_get_text_reads_correct_byte_range():
    """
    CRITICAL: verifies that get_text() seeks to byte_start and reads exactly
    byte_end - byte_start bytes, returning the correct content.

    This is the primary test of the BlockIndex file-pointer mechanism.
    If this test fails the fallback path is broken.
    """
    # Write 5 distinct blocks to a single file
    blocks = [f"BLOCK_{i:03d} DATA {'x' * 200}\n" for i in range(5)]
    corpus_path, ranges = _tmp_corpus(blocks)
    try:
        idx = BlockIndex(":memory:")
        for i, (start, end) in enumerate(ranges):
            idx.add_block(f"block_{i}", str(corpus_path), start, end)

        for i, block in enumerate(blocks):
            text = idx.get_text(f"block_{i}")
            assert text is not None, f"block_{i} returned None"
            assert f"BLOCK_{i:03d}" in text, f"block_{i} content mismatch"
            # Confirm NO content from adjacent blocks leaked in
            for j in range(5):
                if j != i:
                    assert f"BLOCK_{j:03d}" not in text, (
                        f"block_{i} contains content from block_{j}"
                    )
    finally:
        corpus_path.unlink(missing_ok=True)


def test_get_text_needle_fact_recovery():
    """
    Simulates the production fallback scenario:
    A needle fact ('SECRET_CODE:ZEBRA-7731') is embedded mid-block and would
    be dropped by summarisation.  get_text() must recover it verbatim.
    """
    needle = "SECRET_CODE:ZEBRA-7731 approved=2026-03-17 by=authorityX"
    filler = "Lorem ipsum dolor sit amet. " * 500  # ~14 KB of filler
    block_text = filler[:7000] + "\n" + needle + "\n" + filler[7000:14000]

    corpus_path, ranges = _tmp_corpus([block_text])
    try:
        idx = BlockIndex(":memory:")
        idx.add_block("b0", str(corpus_path), ranges[0][0], ranges[0][1])

        recovered = idx.get_text("b0")
        assert recovered is not None
        assert needle in recovered, "Needle fact not found in recovered raw block"
    finally:
        corpus_path.unlink(missing_ok=True)


def test_get_text_returns_none_for_missing_file():
    idx = BlockIndex(":memory:")
    idx.add_block("b0", "/nonexistent/path/corpus.txt", 0, 1000)
    assert idx.get_text("b0") is None


def test_get_text_returns_none_for_unknown_block_id():
    idx = BlockIndex(":memory:")
    assert idx.get_text("unknown") is None


def test_get_text_partial_file_read():
    """Only the registered byte range is read, not the full file."""
    corpus_path, ranges = _tmp_corpus(["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"])
    try:
        idx = BlockIndex(":memory:")
        # Register only block 1 (the B block)
        idx.add_block("b1", str(corpus_path), ranges[1][0], ranges[1][1])
        text = idx.get_text("b1")
        assert "BBB" in text
        assert "AAA" not in text
        assert "CCC" not in text
    finally:
        corpus_path.unlink(missing_ok=True)


# ── add_many ──────────────────────────────────────────────────────────────────


def test_add_many_batch_insert():
    idx = BlockIndex(":memory:")
    idx.add_many([
        ("b0", "/f.txt", 0, 500),
        ("b1", "/f.txt", 500, 1000),
        ("b2", "/g.txt", 0, 750),
    ])
    assert idx.count() == 3
    assert idx.get_meta("b1").byte_start == 500
    assert idx.get_meta("b2").file_path == "/g.txt"


def test_add_many_empty_list():
    idx = BlockIndex(":memory:")
    idx.add_many([])  # should not raise
    assert idx.count() == 0


# ── all_ids / count ───────────────────────────────────────────────────────────


def test_all_ids_returns_sorted():
    idx = BlockIndex(":memory:")
    idx.add_many([("c", "/f", 0, 1), ("a", "/f", 1, 2), ("b", "/f", 2, 3)])
    assert idx.all_ids() == ["a", "b", "c"]


def test_count_increases_with_adds():
    idx = BlockIndex(":memory:")
    assert idx.count() == 0
    idx.add_block("b0", "/f", 0, 100)
    assert idx.count() == 1
    idx.add_block("b1", "/f", 100, 200)
    assert idx.count() == 2


# ── Thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_writes_are_safe(tmp_path):
    """Multiple writer threads must not corrupt the index."""
    db = tmp_path / "blocks.db"
    idx = BlockIndex(str(db))

    errors: list[Exception] = []

    def _write(start_id: int) -> None:
        try:
            for i in range(20):
                idx.add_block(f"block_{start_id + i}", "/f", i * 100, (i + 1) * 100)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(t * 100,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert idx.count() == 100  # 5 threads × 20 blocks each


# ── BlockPointer NamedTuple ───────────────────────────────────────────────────


def test_block_pointer_is_named_tuple():
    ptr = BlockPointer(block_id="x", file_path="/a", byte_start=0, byte_end=1000)
    assert ptr.size_bytes == 1000
    assert ptr.block_id == "x"
    # Should be unpacking-compatible
    bid, fp, bs, be = ptr
    assert bid == "x"
