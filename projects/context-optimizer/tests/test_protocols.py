"""
Tests for context_optimizer.protocols — Retriever structural typing.

These tests verify that:
1. Any class with a compatible `.search()` method satisfies the Retriever Protocol.
2. Classes without the right method do not.
3. The Protocol is exported from the package __init__.
"""
from __future__ import annotations

from typing import runtime_checkable

import pytest

from context_optimizer.protocols import Retriever


# ── Protocol compliance ───────────────────────────────────────────────────────


class _ValidRetriever:
    """Duck-type implementation of Retriever."""

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return [{"compressed_summary": f"result for {query}"}]


class _MissingSearchMethod:
    """Does NOT implement search() — should not satisfy the protocol."""

    def fetch(self, query: str) -> list[dict]:
        return []


def test_valid_retriever_is_instance_of_protocol() -> None:
    # Protocol must be @runtime_checkable for isinstance() to work
    assert isinstance(_ValidRetriever(), Retriever)


def test_invalid_class_is_not_instance_of_protocol() -> None:
    assert not isinstance(_MissingSearchMethod(), Retriever)


def test_retriever_search_is_callable() -> None:
    r = _ValidRetriever()
    results = r.search("test query", top_k=3)
    assert isinstance(results, list)


def test_retriever_exported_from_package() -> None:
    from context_optimizer import Retriever as R
    assert R is Retriever


# ── RawIndex exported from package ───────────────────────────────────────────


def test_raw_index_exported_from_package() -> None:
    from context_optimizer import RawIndex, RawHit
    assert RawIndex is not None
    assert RawHit is not None


# ── All public symbols present in __all__ ────────────────────────────────────


def test_package_all_symbols_importable() -> None:
    import context_optimizer as co

    expected = [
        "CorpusIndex", "QueryResult", "IngestStats",
        "BenchmarkResult",
        "ToTReasoner", "ToTResult", "Branch",
        "Retriever",
        "RawIndex", "RawHit",
    ]
    for name in expected:
        assert hasattr(co, name), f"context_optimizer.{name} is missing"
