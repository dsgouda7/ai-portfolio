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
    synthesized_answer: str = (
        ""  # Non-empty when a reasoning LLM synthesised the answer
    )

    @property
    def winner(self) -> Branch:
        return next(b for b in self.branches if b.id == self.selected_branch_id)


# ── Synthesis prompt (used by the optional reasoning LLM) ──────────────────

_SYNTHESIS_PROMPT = """\
You are a precise question-answering assistant.

Evidence snippets retrieved from a document corpus:
{evidence}

Question: {question}

Answer concisely in 1-3 sentences using only information from the evidence above.
If the evidence does not contain enough information to answer, say "Insufficient evidence."
"""


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
        raw_fallback_threshold: float = 0.40,
        raw_index: Any | None = None,
        block_index: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k_per_term
        # If the winning branch similarity score stays below this value after
        # the compressed pass, automatically re-retrieve raw data.
        self._raw_fallback_threshold = raw_fallback_threshold
        # RawIndex (SQLite+FTS5): BM25 keyword search over raw chunk text.
        # Used when raw text was copied into SQLite at ingestion time.
        self._raw_index = raw_index
        # BlockIndex (file-pointer store): for large-corpus block ingestion
        # where raw text is NOT copied but pointed to on disk.  When a summary
        # retrieval score is below the threshold, the winning block is fetched
        # directly from disk via its file pointer.
        self._block_index = block_index

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
                    branch.score += sum(sim for _, sim in scored) / len(scored)

            branches.append(branch)

        branches.sort(key=lambda b: b.score, reverse=True)
        winner = branches[0]

        # ── Raw-text fallback ────────────────────────────────────────────────
        # When compressed evidence confidence is low (winner.score below
        # threshold), re-retrieve using raw_text so the exact source vocabulary
        # is available.  The index ranking stays the same (embeddings are still
        # computed from compressed_summary); only the returned snippet text
        # switches to raw.  This short-circuits to verbatim data automatically
        # without any query classification.
        used_raw = False
        needs_fallback = (
            self._raw_fallback_threshold > 0
            and winner.score < self._raw_fallback_threshold
            and (self._raw_index is not None or self._block_index is not None)
        )
        if needs_fallback:
            raw_branches: list[Branch] = []
            for spec in specs:
                branch = Branch(
                    id=spec["id"],
                    title=spec["title"],
                    search_terms=spec.get("search_terms", []),
                )
                for term in branch.search_terms:
                    # Prefer FTS5/BM25 on RawIndex (true short-circuit — no
                    # embedding, no ChromaDB round-trip); fall back to
                    # ChromaDB raw_text metadata if neither index is wired in.
                    if self._raw_index is not None:
                        scored = self._retrieve_raw_snippets(term)
                    else:
                        scored = self._retrieve_snippets(term, use_raw=True)
                    branch.evidence_snippets.extend(s for s, _ in scored)
                    total_lines += len(scored)
                    if scored:
                        branch.evidence_hits += 1
                        branch.score += sum(sim for _, sim in scored) / len(scored)

                # ── BlockIndex fallback (large-corpus file pointer model) ──────
                # When block_index is wired in, the winning compressed chunk's
                # metadata contains the block_id.  Fetch the raw block text from
                # disk and add it as an evidence snippet so the reasoning model
                # can answer granular questions that the summary omitted.
                if self._block_index is not None and self._retriever is not None:
                    try:
                        top_hits = self._retriever.search(term, top_k=1)
                        for hit in top_hits[:1]:
                            block_id = hit.get("metadata", {}).get(
                                "block_id"
                            ) or hit.get("chunk_id", "")
                            if block_id:
                                raw_text = self._block_index.get_text(block_id)
                                if raw_text:
                                    branch.evidence_snippets.append(raw_text[:2000])
                    except Exception:
                        pass

                raw_branches.append(branch)
            raw_branches.sort(key=lambda b: b.score, reverse=True)
            # Replace the winning compressed branch with the raw version so that
            # ToTResult.winner returns evidence_snippets from raw_text.
            winner = raw_branches[0]
            branches = [b if b.id != winner.id else winner for b in branches]
            used_raw = True
        entity_str = self._entity_str(compressed_context)
        fallback_note = "  [raw-text fallback]\n" if used_raw else ""

        # ── Optional LLM synthesis ───────────────────────────────────────────
        # When a reasoning LLM is wired in, synthesise a concise natural-
        # language answer from the top evidence snippets.  The raw snippets are
        # still available on each Branch for debugging / keyword scoring.
        synthesized_answer = ""
        if self._llm is not None:
            all_snips: list[str] = []
            for b in sorted(branches, key=lambda b: b.score, reverse=True):
                all_snips.extend(b.evidence_snippets)
            synthesized_answer = self._synthesize(
                str(compressed_context), all_snips[:6]
            )

        summary = (
            f"ToT-selected branch: {winner.title}.\n"
            f"Mean similarity: {winner.score:.3f} "
            f"({winner.evidence_hits}/{len(winner.search_terms)} search term(s) matched).\n"
            + fallback_note
            + (f"Key entities: {entity_str}.\n" if entity_str else "")
            + "Next: address the dominant failure mode identified by the winning branch."
        )

        return ToTResult(
            branches=branches,
            selected_branch_id=winner.id,
            selected_summary=summary,
            total_retrieved_lines=total_lines,
            latency_s=time.perf_counter() - start,
            synthesized_answer=synthesized_answer,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _synthesize(self, question: str, snippets: list[str]) -> str:
        """
        Call the reasoning LLM to produce a concise natural-language answer
        from the top evidence snippets.  Falls back silently to an empty
        string on any error so callers can fall back to raw snippet aggregation.
        """
        if not snippets or self._llm is None:
            return ""
        evidence = "\n---\n".join(snippets)
        prompt = _SYNTHESIS_PROMPT.format(evidence=evidence, question=question)
        try:
            resp = self._llm.invoke(prompt)
            return (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception:
            return ""

    def _retrieve_raw_snippets(self, term: str) -> list[tuple[str, float]]:
        """
        BM25/FTS5 keyword search directly on the RawIndex — true short-circuit.

        Bypasses ChromaDB and the embedding layer entirely.  The SQLite FTS5
        engine ranks results by BM25; scores are normalised to a 0–1
        pseudo-similarity so downstream branch scoring stays consistent.

        Returns
        -------
        list of (raw_text, similarity) pairs, best match first.
        """
        if self._raw_index is None:
            return []
        try:
            hits = self._raw_index.search(term, top_k=self._top_k)
            if not hits:
                return []
            # BM25 rank is negative (more negative = better match in SQLite).
            scores = [abs(h.rank) for h in hits]
            max_score = max(scores) or 1.0
            return [(h.raw_text, min(s / max_score, 1.0)) for h, s in zip(hits, scores)]
        except Exception:
            return []

    def _retrieve_snippets(
        self, term: str, use_raw: bool = False
    ) -> list[tuple[str, float]]:
        """Fetch top_k snippets with similarity scores for *term*.

        The ChromaDB index is always queried via the compressed embedding
        (ranking is based on semantic similarity to the compressed summary).
        When *use_raw* is True the returned text is taken from the chunk's
        ``raw_text`` field instead of ``compressed_summary``, preserving the
        exact source vocabulary for keyword-sensitive queries.

        Returns
        -------
        list of (text, similarity) pairs.
        """
        if self._retriever is None:
            return []
        try:
            results = self._retriever.search(term, top_k=self._top_k)
            out: list[tuple[str, float]] = []
            for r in results:
                if use_raw:
                    # Prefer top-level raw_text (bubbled up by CachedChromaRetriever);
                    # fall back to metadata["raw_text"] for older retriever versions.
                    snippet = (
                        r.get("raw_text")
                        or r.get("metadata", {}).get("raw_text")
                        or r.get("compressed_summary", str(r))
                    )
                else:
                    snippet = r.get("compressed_summary", str(r))
                dist = r.get("distance")
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
