"""
Tree-of-Thought (ToT) Reasoner — default reasoning strategy.

Baked into the core library so any pipeline that imports context_optimizer
gets ToT reasoning without extra configuration.

Design
------
- Generates N hypothesis branches derived from the compressed context.
- Each branch retrieves evidence from the corpus via the retriever.
- Branches are scored by evidence density; the winner is returned.
- Works with any Retriever (CachedChromaRetriever, DualStorageRetriever,
  or any object with a `.search()` method).

Usage::

    from context_optimizer.tot_reasoner import ToTReasoner

    # With a real corpus retriever
    reasoner = ToTReasoner(retriever=cached_chroma_retriever)
    result = reasoner.reason(compressed_chunk)
    print(result.selected_summary)

    # With explicit branch specs (e.g. from incident entities)
    result = reasoner.reason(
        compressed_incident,
        branch_specs=[
            {"id": "cosmos",  "title": "CosmosDB hypothesis",  "search_terms": ["CosmosDB", "21012"]},
            {"id": "ingress", "title": "Ingress hypothesis",   "search_terms": ["upstream", "504"]},
            {"id": "retry",   "title": "Retry cascade",        "search_terms": ["retry", "cancel"]},
        ],
    )

    # Without a retriever — branches still populated, evidence empty
    reasoner = ToTReasoner()
    result = reasoner.reason("CosmosDB timeout cascade AKS ingress 504")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .protocols import Retriever  # noqa: F401 — re-exported for callers

# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class Branch:
    """A single reasoning hypothesis explored during a ToT pass."""

    id: str
    title: str
    search_terms: list[str]
    evidence_snippets: list[str] = field(default_factory=list)
    evidence_hits: int = 0
    score: float = 0.0


@dataclass
class ToTResult:
    """Outcome of a Tree-of-Thought reasoning pass."""

    branches: list[Branch]
    selected_branch_id: str
    selected_summary: str
    total_retrieved_lines: int
    latency_s: float

    @property
    def winner(self) -> Branch:
        return next(b for b in self.branches if b.id == self.selected_branch_id)


# ── Reasoner ────────────────────────────────────────────────────────────────


class ToTReasoner:
    """
    Tree-of-Thought multi-branch reasoner — default reasoning strategy.

    Each call to ``reason()`` follows three steps:
    1. Derive (or accept) N branch specs from the compressed context.
    2. For every branch, retrieve evidence snippets from the corpus.
    3. Score branches by evidence density; return the winner + summary.

    Parameters
    ----------
    retriever:
        Any object satisfying the ``Retriever`` protocol.  If omitted,
        branches are generated but evidence retrieval is skipped.
    llm:
        Reserved for future LLM-guided branch generation.  Currently unused;
        branch specs are derived deterministically from entities.
    top_k_per_term:
        How many corpus chunks to retrieve per search term (default 3).
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: Any = None,
        *,
        top_k_per_term: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k_per_term

    # ── Public API ──────────────────────────────────────────────────────────

    def reason(
        self,
        compressed_context: Any,
        branch_specs: list[dict[str, Any]] | None = None,
    ) -> ToTResult:
        """
        Run Tree-of-Thought reasoning over *compressed_context*.

        Parameters
        ----------
        compressed_context:
            Accepts any of the following:
            - A ``CompressedChunk`` (has ``.entities``)
            - A ``CompressedIncident`` (has ``.technical_identifiers``)
            - A plain dict with ``"entities"`` or ``"technical_identifiers"``
            - A plain string (split into words and used as search terms)
        branch_specs:
            Optional explicit branch list.  Each entry is a dict with keys
            ``id`` (str), ``title`` (str), and ``search_terms`` (list[str]).
            When omitted, branches are derived from the context's entities.

        Returns
        -------
        ToTResult
            Contains all branches (scored), the winning branch id, a
            human-readable summary, and timing info.
        """
        specs = (
            branch_specs
            if branch_specs is not None
            else self._derive_branch_specs(compressed_context)
        )
        start = time.perf_counter()

        branches: list[Branch] = []
        total_lines = 0

        for spec in specs:
            branch = Branch(
                id=spec["id"],
                title=spec["title"],
                search_terms=spec.get("search_terms", []),
            )
            for term in branch.search_terms:
                scored = self._retrieve_snippets(term)
                branch.evidence_snippets.extend(s for s, _ in scored)
                total_lines += len(scored)
                if scored:
                    branch.evidence_hits += 1
                    # Accumulate mean similarity across all returned chunks for this term.
                    # ChromaDB cosine distance is in [0, 2]; similarity = 1 - distance
                    # giving [-1, 1] where 1 = identical.  Average over top_k results.
                    branch.score += sum(sim for _, sim in scored) / len(scored)

            branches.append(branch)

        branches.sort(key=lambda b: b.score, reverse=True)
        winner = branches[0]
        entity_str = self._entity_str(compressed_context)

        summary = (
            f"ToT-selected branch: {winner.title}.\n"
            f"Mean similarity: {winner.score:.3f} "
            f"({winner.evidence_hits}/{len(winner.search_terms)} search term(s) matched).\n"
            + (f"Key entities: {entity_str}.\n" if entity_str else "")
            + "Next: address the dominant failure mode identified by the winning branch."
        )

        return ToTResult(
            branches=branches,
            selected_branch_id=winner.id,
            selected_summary=summary,
            total_retrieved_lines=total_lines,
            latency_s=time.perf_counter() - start,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _retrieve_snippets(self, term: str) -> list[tuple[str, float]]:
        """Fetch top_k snippets with similarity scores for *term*.

        Returns
        -------
        list of (compressed_summary, similarity) pairs.
        similarity = 1 - cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
        Returns [] on any error or when no retriever is configured.
        """
        if self._retriever is None:
            return []
        try:
            results = self._retriever.search(term, top_k=self._top_k)
            out: list[tuple[str, float]] = []
            for r in results:
                snippet = r.get("compressed_summary", str(r))
                dist = r.get("distance")
                # ChromaDB cosine space: distance in [0, 2].  Convert to similarity.
                # When distance is absent (e.g. non-ChromaDB retriever), default to 0.5.
                similarity = 1.0 - float(dist) if dist is not None else 0.5
                out.append((snippet, similarity))
            return out
        except Exception:
            return []

    @staticmethod
    def _derive_branch_specs(ctx: Any) -> list[dict[str, Any]]:
        """
        Build branch specs from the entities embedded in *ctx*.

        Splits entities into thirds: primary (high-confidence), secondary
        (supporting), tertiary (residual / weak signals).
        """
        entities: list[str] = []

        if isinstance(ctx, str):
            entities = [w for w in ctx.split() if len(w) > 2][:9]
        elif hasattr(ctx, "technical_identifiers"):
            entities = list(ctx.technical_identifiers)[:9]
        elif hasattr(ctx, "entities"):
            entities = list(ctx.entities)[:9]
        elif isinstance(ctx, dict):
            entities = (ctx.get("technical_identifiers") or ctx.get("entities") or [])[
                :9
            ]

        if not entities:
            return [
                {
                    "id": "primary",
                    "title": "Primary hypothesis branch",
                    "search_terms": [],
                },
                {
                    "id": "secondary",
                    "title": "Secondary hypothesis branch",
                    "search_terms": [],
                },
                {
                    "id": "tertiary",
                    "title": "Tertiary hypothesis branch",
                    "search_terms": [],
                },
            ]

        n = len(entities)
        t = max(1, n // 3)

        # Group entities into thirds, then join each group into a single
        # composite hypothesis sentence.  A sentence embedding carries far
        # more semantic signal than individual keyword embeddings — ChromaDB
        # and the semantic cache both benefit from a richer query vector.
        # This also halves the number of retriever calls: 1 per branch instead
        # of up to t calls per branch.
        groups = [
            entities[:t],
            entities[t : 2 * t] or entities[:1],
            entities[2 * t :] or entities[:1],
        ]

        return [
            {
                "id": bid,
                "title": f"Branch {bid.title()} — {', '.join(group)}",
                # Single composite sentence per branch: richer embedding,
                # fewer retriever round-trips.
                "search_terms": [" ".join(group)],
            }
            for bid, group in zip(("primary", "secondary", "tertiary"), groups)
        ]

    @staticmethod
    def _entity_str(ctx: Any) -> str:
        for attr in ("technical_identifiers", "entities"):
            val = getattr(ctx, attr, None)
            if val:
                return ", ".join(list(val)[:8])
        if isinstance(ctx, dict):
            for key in ("technical_identifiers", "entities"):
                if ctx.get(key):
                    return ", ".join(ctx[key][:8])
        return ""
