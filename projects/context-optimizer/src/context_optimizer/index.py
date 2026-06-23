"""
CorpusIndex — high-level facade for context-optimizer.

Wires together three internal components without changing any of them:
  1. ``compressor.compress_corpus_rolling``   — LLM rolling-window compression
  2. ``cached_retriever.CachedChromaRetriever`` — two-tier vector store + cache
  3. ``tot_reasoner.ToTReasoner``              — Tree-of-Thought multi-branch reasoning

Quick-start::

    from context_optimizer import CorpusIndex

    index = CorpusIndex(compression_model="llama3.2:3b", persist_dir="./my_index")
    stats  = index.ingest(log_lines)
    result = index.query("what caused the CosmosDB timeout?")
    print(result.answer)
    for chunk in result.evidence:
        print(" -", chunk[:80])

Use as a context manager for automatic cleanup of ephemeral storage::

    with CorpusIndex() as index:
        index.ingest(lines)
        result = index.query("...")
"""
from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Public data classes ──────────────────────────────────────────────────────

@dataclass
class IngestStats:
    """Statistics returned by :meth:`CorpusIndex.ingest`."""

    chunks: int
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float   # compressed / original  (< 1.0 = reduction)
    elapsed_s: float


@dataclass
class QueryResult:
    """Result returned by :meth:`CorpusIndex.query`."""

    answer: str
    evidence: list[str]
    tokens_used: int
    latency_ms: float
    branch_id: str | None = None


# ── CorpusIndex ──────────────────────────────────────────────────────────────

class CorpusIndex:
    """
    High-level interface for corpus ingestion and semantic querying.

    Parameters
    ----------
    compression_model:
        Name of the Ollama model used for rolling-window compression.
        Passed to ``compress_corpus_rolling`` via the
        ``CONTEXT_OPTIMIZER_COMPRESSOR_MODEL`` environment variable.
    embedding_model:
        Sentence-transformers model name for vector embeddings.
        Passed to ``CachedChromaRetriever`` as ``embedding_model_name``.
    persist_dir:
        Directory to persist the ChromaDB data.  When ``None`` (default)
        a temporary directory is created and cleaned up on :meth:`close`.
    chunk_tokens:
        Token budget for each compressed chunk (passed to
        ``compress_corpus_rolling`` as ``chunk_size_threshold``).
    retrieval_strategy:
        ``"tot"`` (default) — Tree-of-Thought multi-branch reasoning.
        ``"simple"`` — plain vector search, no ToT overhead.
    provider:
        LLM backend used for compression.  Currently only ``"ollama"``
        is supported for local compression; use the Docker services for
        remote compression.
    base_url:
        Base URL for the Ollama server (ignored for non-Ollama providers).
    """

    def __init__(
        self,
        compression_model: str = "llama3.2:3b",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str | None = None,
        chunk_tokens: int = 512,
        retrieval_strategy: str = "tot",
        provider: str = "ollama",
        base_url: str = "http://localhost:11434",
    ) -> None:
        import os

        self._compression_model   = compression_model
        self._embedding_model     = embedding_model
        self._chunk_tokens        = chunk_tokens
        self._strategy            = retrieval_strategy
        self._provider            = provider
        self._base_url            = base_url

        # Persist dir — user-supplied or managed temp
        self._user_persist = persist_dir is not None
        self._persist_dir  = persist_dir or tempfile.mkdtemp(prefix="co_index_")

        # Per-collection retriever cache: collection_name → CachedChromaRetriever
        self._retrievers: dict[str, Any] = {}
        # Per-collection reasoner cache: collection_name → ToTReasoner
        self._reasoners: dict[str, Any] = {}
        # Track which collections have at least one chunk ingested
        self._ingested_collections: set[str] = set()

        # Patch env vars so the existing compressor picks up the right settings
        os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_MODEL"]    = compression_model
        os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = provider
        if base_url != "http://localhost:11434":
            os.environ["OLLAMA_BASE_URL"] = base_url

    # ── Ingest ───────────────────────────────────────────────────────────────

    def ingest(
        self,
        lines: list[str],
        collection: str = "default",
    ) -> IngestStats:
        """
        Compress *lines* and add them to the index.

        Can be called multiple times; each call appends to the same
        collection without discarding previous data.

        Parameters
        ----------
        lines:
            Corpus lines — e.g. log lines, document paragraphs.
        collection:
            ChromaDB collection name.  Use different names to keep
            separate corpora in the same index directory.

        Returns
        -------
        IngestStats
            Chunk count, token counts, compression ratio, and wall time.
        """
        from context_optimizer.compressor      import compress_corpus_rolling
        from context_optimizer.cached_retriever import CachedChromaRetriever
        from context_optimizer.tot_reasoner     import ToTReasoner

        t0     = time.perf_counter()
        chunks = compress_corpus_rolling(lines, chunk_size_threshold=self._chunk_tokens)
        elapsed = time.perf_counter() - t0

        orig_tok = sum(c.original_tokens  for c in chunks)
        comp_tok = sum(c.compressed_tokens for c in chunks)

        # Lazy-create or reuse the retriever for this collection
        if collection not in self._retrievers:
            self._retrievers[collection] = CachedChromaRetriever(
                collection_name=collection,
                persist_directory=str(Path(self._persist_dir) / collection),
                embedding_model_name=self._embedding_model,
            )

        self._retrievers[collection].add_chunks(chunks)

        # Lazy-create or reuse the ToT reasoner for this collection
        if collection not in self._reasoners and self._strategy == "tot":
            self._reasoners[collection] = ToTReasoner(
                retriever=self._retrievers[collection]
            )

        self._ingested_collections.add(collection)

        return IngestStats(
            chunks=len(chunks),
            original_tokens=orig_tok,
            compressed_tokens=comp_tok,
            compression_ratio=comp_tok / max(orig_tok, 1),
            elapsed_s=round(elapsed, 2),
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        collection: str = "default",
        top_k: int = 6,
    ) -> QueryResult:
        """
        Retrieve evidence and answer *question* from the indexed corpus.

        Parameters
        ----------
        question:
            Natural-language query.
        collection:
            Which ChromaDB collection to search (must have been ingested first).
        top_k:
            How many evidence snippets to include in the result
            (used by the ``"simple"`` strategy; ToT uses its own k per branch).

        Returns
        -------
        QueryResult
            Structured result with ``answer``, ``evidence`` list, token count,
            and latency.

        Raises
        ------
        RuntimeError
            If the collection has never been ingested.
        """
        if collection not in self._ingested_collections:
            raise RuntimeError(
                f"Collection '{collection}' has no data. Call ingest() first."
            )

        t0 = time.perf_counter()

        # ── Tree-of-Thought path ──────────────────────────────────────────
        if self._strategy == "tot" and collection in self._reasoners:
            tot = self._reasoners[collection].reason(question)
            latency_ms = (time.perf_counter() - t0) * 1000
            return QueryResult(
                answer=tot.selected_summary,
                evidence=tot.winner.evidence_snippets,
                tokens_used=sum(len(s) // 4 for s in tot.winner.evidence_snippets),
                latency_ms=round(latency_ms, 2),
                branch_id=tot.selected_branch_id,
            )

        # ── Simple vector-search fallback ─────────────────────────────────
        results    = self._retrievers[collection].search(question, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        snippets   = [r.get("compressed_summary", str(r)) for r in results]

        return QueryResult(
            answer=snippets[0] if snippets else "",
            evidence=snippets,
            tokens_used=sum(len(s) // 4 for s in snippets),
            latency_ms=round(latency_ms, 2),
        )

    def query_many(
        self,
        questions: list[str],
        collection: str = "default",
    ) -> list[QueryResult]:
        """Run :meth:`query` for each item in *questions* and return all results."""
        return [self.query(q, collection=collection) for q in questions]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Delete the ephemeral index directory (no-op for user-supplied dirs)."""
        if not self._user_persist and Path(self._persist_dir).exists():
            shutil.rmtree(self._persist_dir, ignore_errors=True)

    def __enter__(self) -> "CorpusIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        collections = list(self._ingested_collections) or ["<none>"]
        return (
            f"CorpusIndex(model={self._compression_model!r}, "
            f"strategy={self._strategy!r}, "
            f"collections={collections})"
        )
