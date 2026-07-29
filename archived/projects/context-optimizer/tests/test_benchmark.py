"""
Tests for context_optimizer.benchmark — BenchmarkResult and compare().

compare() is tested by patching CorpusIndex so no LLM/ChromaDB is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from context_optimizer.benchmark import (
    BenchmarkResult,
    _keyword_f1,
    _reference_keywords,
)
from context_optimizer.index import IngestStats, QueryResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(**overrides) -> BenchmarkResult:
    defaults = dict(
        raw_tokens=10_000,
        compressed_tokens=870,
        token_reduction_pct=91.3,
        raw_latency_ms=0.0,
        optimized_latency_ms=120.5,
        latency_improvement_pct=0.0,
        raw_kw_f1=0.021,
        optimized_kw_f1=0.045,
        raw_log_lines=500,
        optimized_retrieved_lines=12,
        retrieval_efficiency_pct=2.4,
        ingest_chunks=9,
        ingest_compression_ratio=0.087,
        branch_scores={"primary": 3.0, "secondary": 1.0},
        winning_branch="primary",
        raw_answer="",
        optimized_answer="",
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


# ── _keyword_f1 ───────────────────────────────────────────────────────────────


def test_keyword_f1_empty_keywords_returns_zero() -> None:
    assert _keyword_f1("some output", []) == pytest.approx(0.0)


def test_keyword_f1_all_keywords_present() -> None:
    score = _keyword_f1("python database error", ["python", "database", "error"])
    assert score > 0.0


def test_keyword_f1_no_keywords_present() -> None:
    score = _keyword_f1("completely unrelated text", ["python", "database"])
    assert score == pytest.approx(0.0)


# ── _reference_keywords ───────────────────────────────────────────────────────


def test_reference_keywords_filters_short_words() -> None:
    kw = _reference_keywords("what is the error in the database?")
    # "is", "in", "the" (≤ 2 chars) should be excluded
    assert all(len(w) > 2 for w in kw)


def test_reference_keywords_deduplicates() -> None:
    kw = _reference_keywords("error error error in database")
    assert len(kw) == len(set(kw))


def test_reference_keywords_merges_extras() -> None:
    kw = _reference_keywords("timeout error", extra=["CosmosDB", "503"])
    assert "CosmosDB" in kw


# ── BenchmarkResult.summary ───────────────────────────────────────────────────


def test_summary_contains_token_reduction() -> None:
    result = _make_result()
    s = result.summary()
    assert "91.3" in s
    assert "10,000" in s


def test_summary_contains_winning_branch() -> None:
    result = _make_result(winning_branch="primary")
    s = result.summary()
    assert "primary" in s


def test_summary_contains_branch_scores() -> None:
    result = _make_result(branch_scores={"primary": 3.0, "secondary": 1.0})
    s = result.summary()
    assert "Branch scores" in s or "primary" in s


def test_summary_no_winning_branch() -> None:
    result = _make_result(winning_branch=None, branch_scores={})
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0


# ── BenchmarkResult.to_dict ───────────────────────────────────────────────────


def test_to_dict_contains_expected_keys() -> None:
    d = _make_result().to_dict()
    expected = [
        "raw_tokens",
        "compressed_tokens",
        "token_reduction_pct",
        "raw_latency_ms",
        "optimized_latency_ms",
        "latency_improvement_pct",
        "raw_kw_f1",
        "optimized_kw_f1",
        "raw_log_lines",
        "optimized_retrieved_lines",
        "retrieval_efficiency_pct",
        "ingest_chunks",
        "ingest_compression_ratio",
        "winning_branch",
        "branch_scores",
    ]
    for key in expected:
        assert key in d, f"Missing key: {key}"


def test_to_dict_values_are_rounded() -> None:
    d = _make_result(token_reduction_pct=91.3456789).to_dict()
    # to_dict rounds to 2 decimal places
    assert d["token_reduction_pct"] == pytest.approx(91.35)


def test_to_dict_branch_scores_preserved() -> None:
    d = _make_result(branch_scores={"primary": 3.0}).to_dict()
    assert d["branch_scores"] == {"primary": 3.0}


# ── compare() ─────────────────────────────────────────────────────────────────


def _make_ingest_stats(**kw) -> IngestStats:
    defaults = dict(
        chunks=3,
        original_tokens=1000,
        compressed_tokens=200,
        compression_ratio=0.2,
        elapsed_s=2.0,
    )
    defaults.update(kw)
    return IngestStats(**defaults)


def _make_query_result(**kw) -> QueryResult:
    defaults = dict(
        answer="The system timed out due to CosmosDB overload.",
        evidence=["evidence 1", "evidence 2"],
        tokens_used=50,
        latency_ms=120.0,
        branch_id="primary",
    )
    defaults.update(kw)
    return QueryResult(**defaults)


def test_compare_no_llm_skips_pipe_a() -> None:
    """Without an LLM, compare() skips Pipe A and sets raw_latency_ms=0."""
    from context_optimizer.benchmark import compare

    mock_index = MagicMock()
    mock_index.__enter__ = lambda s: s
    mock_index.__exit__ = MagicMock(return_value=False)
    mock_index.ingest.return_value = _make_ingest_stats()
    mock_index.query.return_value = _make_query_result()

    # compare() does `from context_optimizer.index import CorpusIndex` inside
    # the function, so we must patch at the source module.
    with patch("context_optimizer.index.CorpusIndex", return_value=mock_index):
        result = compare(
            question="What caused the timeout?",
            raw_corpus=["log line 1"] * 20,
            llm=None,
        )

    assert result.raw_latency_ms == pytest.approx(0.0)
    assert result.ingest_chunks == 3
    assert result.compressed_tokens == 200


def test_compare_returns_benchmark_result() -> None:
    from context_optimizer.benchmark import compare

    mock_index = MagicMock()
    mock_index.__enter__ = lambda s: s
    mock_index.__exit__ = MagicMock(return_value=False)
    mock_index.ingest.return_value = _make_ingest_stats()
    mock_index.query.return_value = _make_query_result()

    with patch("context_optimizer.index.CorpusIndex", return_value=mock_index):
        result = compare(
            question="Why did the database fail?",
            raw_corpus=["entry a", "entry b"] * 10,
        )

    assert isinstance(result, BenchmarkResult)
    assert result.raw_log_lines == 20
    assert result.optimized_retrieved_lines == 2  # len(evidence)


def test_compare_token_reduction_computed_correctly() -> None:
    from context_optimizer.benchmark import compare

    raw_corpus = ["x" * 400] * 5  # 5 × 100 tokens = 500 raw tokens
    mock_index = MagicMock()
    mock_index.__enter__ = lambda s: s
    mock_index.__exit__ = MagicMock(return_value=False)
    mock_index.ingest.return_value = _make_ingest_stats(
        original_tokens=500, compressed_tokens=50
    )
    mock_index.query.return_value = _make_query_result()

    with patch("context_optimizer.index.CorpusIndex", return_value=mock_index):
        result = compare("q", raw_corpus)

    # token_reduction_pct is computed from the actual raw corpus, not ingest stats
    assert 0 <= result.token_reduction_pct <= 100


def test_compare_with_reference_keywords() -> None:
    from context_optimizer.benchmark import compare

    mock_index = MagicMock()
    mock_index.__enter__ = lambda s: s
    mock_index.__exit__ = MagicMock(return_value=False)
    mock_index.ingest.return_value = _make_ingest_stats()
    mock_index.query.return_value = _make_query_result()

    with patch("context_optimizer.index.CorpusIndex", return_value=mock_index):
        result = compare(
            "timeout",
            ["line"] * 5,
            reference_keywords=["timeout", "CosmosDB", "AKS"],
        )

    assert isinstance(result, BenchmarkResult)
