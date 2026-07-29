"""
Tests for context_optimizer.tot_reasoner — ToTReasoner, Branch, ToTResult.

No LLM or ChromaDB calls; a mock Retriever implementation is used.
"""
from __future__ import annotations

import pytest

from context_optimizer.compressor import CompressedChunk
from context_optimizer.tot_reasoner import Branch, ToTReasoner, ToTResult


# ── Mock Retriever ────────────────────────────────────────────────────────────


class MockRetriever:
    """
    Minimal Retriever that returns canned snippets for any query that
    contains one of the registered keywords.
    """

    def __init__(self, snippets: dict[str, list[dict]] | None = None) -> None:
        self._snippets = snippets or {}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        for keyword, results in self._snippets.items():
            if keyword.lower() in query.lower():
                return results[:top_k]
        return []


# ── Branch dataclass ──────────────────────────────────────────────────────────


def test_branch_defaults() -> None:
    b = Branch(id="primary", title="Primary branch", search_terms=["a", "b"])
    assert b.evidence_snippets == []
    assert b.evidence_hits == 0
    assert b.score == 0.0


# ── ToTResult.winner property ─────────────────────────────────────────────────


def test_tot_result_winner_property() -> None:
    b1 = Branch(id="primary", title="P", search_terms=[], score=10.0)
    b2 = Branch(id="secondary", title="S", search_terms=[], score=5.0)
    result = ToTResult(
        branches=[b1, b2],
        selected_branch_id="primary",
        selected_summary="summary",
        total_retrieved_lines=0,
        latency_s=0.0,
    )
    assert result.winner is b1


# ── ToTReasoner._derive_branch_specs ─────────────────────────────────────────


def test_derive_branch_specs_from_string() -> None:
    specs = ToTReasoner._derive_branch_specs("foo bar baz qux quux")
    assert len(specs) == 3
    assert specs[0]["id"] == "primary"
    assert len(specs[0]["search_terms"]) >= 1


def test_derive_branch_specs_from_compressed_chunk() -> None:
    chunk = CompressedChunk(
        chunk_id="c0",
        raw_text="raw",
        compressed_summary="summary",
        entities=["entity_a", "entity_b", "entity_c", "entity_d", "entity_e"],
        keywords=["kw1"],
        metadata={},
        original_tokens=100,
        compressed_tokens=30,
        compression_ratio=0.3,
    )
    specs = ToTReasoner._derive_branch_specs(chunk)
    assert len(specs) == 3
    # Entities should be distributed across branches
    all_terms = [t for s in specs for t in s["search_terms"]]
    assert "entity_a" in all_terms


def test_derive_branch_specs_from_dict_with_entities() -> None:
    ctx = {"entities": ["alpha", "beta", "gamma", "delta"]}
    specs = ToTReasoner._derive_branch_specs(ctx)
    assert len(specs) == 3


def test_derive_branch_specs_from_dict_with_technical_identifiers() -> None:
    ctx = {"technical_identifiers": ["CosmosDB", "504", "AKS"]}
    specs = ToTReasoner._derive_branch_specs(ctx)
    assert len(specs) == 3


def test_derive_branch_specs_empty_context_returns_defaults() -> None:
    specs = ToTReasoner._derive_branch_specs({})
    assert len(specs) == 3
    assert specs[0]["id"] == "primary"
    assert specs[1]["id"] == "secondary"
    assert specs[2]["id"] == "tertiary"


# ── ToTReasoner.reason — no retriever ─────────────────────────────────────────


def test_reason_without_retriever_produces_result() -> None:
    reasoner = ToTReasoner()
    result = reasoner.reason("database timeout CosmosDB AKS ingress")
    assert isinstance(result, ToTResult)
    assert result.selected_branch_id in {"primary", "secondary", "tertiary"}
    assert len(result.branches) == 3
    assert result.latency_s >= 0.0


def test_reason_without_retriever_evidence_is_empty() -> None:
    reasoner = ToTReasoner()
    result = reasoner.reason("some context string")
    for branch in result.branches:
        assert branch.evidence_snippets == []
        assert branch.evidence_hits == 0


# ── ToTReasoner.reason — with mock retriever ─────────────────────────────────


def test_reason_with_retriever_populates_evidence() -> None:
    retriever = MockRetriever(
        snippets={
            "CosmosDB": [{"compressed_summary": "Cosmos timeout at 21:00"}],
            "AKS":      [{"compressed_summary": "AKS pod restart"}, {"compressed_summary": "Node draining"}],
        }
    )
    reasoner = ToTReasoner(retriever=retriever, top_k_per_term=3)
    result = reasoner.reason("CosmosDB timeout AKS ingress 504 retry cancel")
    assert result.total_retrieved_lines > 0
    winner = result.winner
    assert len(winner.evidence_snippets) > 0


def test_reason_selects_highest_evidence_branch() -> None:
    """Branch with more evidence hits should win."""
    retriever = MockRetriever(
        snippets={
            "rich":  [{"compressed_summary": "hit1"}, {"compressed_summary": "hit2"}],
            "poor":  [{"compressed_summary": "hit1"}],
        }
    )
    reasoner = ToTReasoner(retriever=retriever)
    result = reasoner.reason(
        None,
        branch_specs=[
            {"id": "rich_branch", "title": "Rich", "search_terms": ["rich", "rich"]},
            {"id": "poor_branch", "title": "Poor", "search_terms": ["poor"]},
        ],
    )
    assert result.selected_branch_id == "rich_branch"


def test_reason_with_explicit_branch_specs() -> None:
    reasoner = ToTReasoner()
    specs = [
        {"id": "b1", "title": "Branch 1", "search_terms": ["a"]},
        {"id": "b2", "title": "Branch 2", "search_terms": ["b"]},
    ]
    result = reasoner.reason("some context", branch_specs=specs)
    assert len(result.branches) == 2
    assert result.selected_branch_id in {"b1", "b2"}


def test_reason_summary_contains_branch_title() -> None:
    reasoner = ToTReasoner()
    result = reasoner.reason("foo bar baz")
    assert "ToT-selected branch" in result.selected_summary


def test_reason_retriever_exception_is_handled_gracefully() -> None:
    """If the retriever raises, evidence should simply be empty."""

    class BrokenRetriever:
        def search(self, query: str, top_k: int = 5) -> list[dict]:
            raise RuntimeError("retriever down")

    reasoner = ToTReasoner(retriever=BrokenRetriever())
    result = reasoner.reason("context string")
    assert isinstance(result, ToTResult)
    assert result.total_retrieved_lines == 0
