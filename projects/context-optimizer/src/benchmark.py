"""
Benchmark utilities — compare raw (Pipe A) vs CorpusIndex/ToT (Pipe C).

Exposes a typed ``BenchmarkResult`` dataclass and a standalone ``compare()``
function so benchmark metrics are part of the library, not just the scripts.

Usage::

    from context_optimizer import CorpusIndex
    from context_optimizer.benchmark import compare, BenchmarkResult

    with open("incident.log") as f:
        lines = f.read().splitlines()

    result: BenchmarkResult = compare(
        question="What caused the CosmosDB timeout?",
        raw_corpus=lines,
        compression_model="llama3.2:3b",
    )
    print(result.summary())

The ``compare()`` function does NOT require an LLM for the Pipe A baseline —
it uses a token-count and keyword-overlap proxy so the function stays fast
and offline-friendly (no extra LLM round-trip).  Pass ``llm=`` to get an
actual LLM answer for Pipe A.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


# ── BenchmarkResult ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Comparison metrics for raw (Pipe A) vs CorpusIndex/ToT (Pipe C).

    Attributes
    ----------
    raw_tokens:
        Estimated token count of the full raw corpus (Pipe A prompt size).
    compressed_tokens:
        Estimated token count after rolling-window compression (Pipe C prompt).
    token_reduction_pct:
        Percentage of tokens eliminated by compression.
    raw_latency_ms:
        Wall-clock ms for the Pipe A call (0.0 when no LLM is provided).
    optimized_latency_ms:
        Wall-clock ms for the Pipe C ToT query.
    latency_improvement_pct:
        Percentage speedup of Pipe C over Pipe A (positive = faster).
    raw_kw_f1:
        Keyword F1 of the raw-corpus answer vs reference keywords.
    optimized_kw_f1:
        Keyword F1 of the ToT answer vs reference keywords.
    raw_log_lines:
        Total lines passed to Pipe A (entire corpus).
    optimized_retrieved_lines:
        Lines actually retrieved by Pipe C (targeted retrieval).
    retrieval_efficiency_pct:
        Percentage of corpus retrieved by Pipe C (lower = more targeted).
    ingest_chunks:
        Number of compressed chunks created during ingestion.
    ingest_compression_ratio:
        compressed_tokens / raw_tokens for the ingested corpus.
    branch_scores:
        Per-branch evidence hit scores from ToT (branch_id → score).
    winning_branch:
        Branch id selected by ToT reasoning.
    raw_answer:
        Pipe A LLM answer (empty string when no LLM is provided).
    optimized_answer:
        Pipe C ToT summary.
    """

    raw_tokens: int
    compressed_tokens: int
    token_reduction_pct: float

    raw_latency_ms: float
    optimized_latency_ms: float
    latency_improvement_pct: float

    raw_kw_f1: float
    optimized_kw_f1: float

    raw_log_lines: int
    optimized_retrieved_lines: int
    retrieval_efficiency_pct: float

    ingest_chunks: int
    ingest_compression_ratio: float

    branch_scores: dict[str, float] = field(default_factory=dict)
    winning_branch: str | None = None

    raw_answer: str = ""
    optimized_answer: str = ""

    def summary(self) -> str:
        """Return a compact human-readable report."""
        lines = [
            "┌─ Context Optimizer Benchmark ─────────────────────────────────┐",
            f"│  Token reduction      {self.token_reduction_pct:>6.1f}%"
            f"  ({self.raw_tokens:,} → {self.compressed_tokens:,} tokens)           │",
            f"│  Retrieval efficiency {self.retrieval_efficiency_pct:>6.1f}%"
            f"  ({self.optimized_retrieved_lines:,} / {self.raw_log_lines:,} lines retrieved)  │",
            f"│  Latency improvement  {self.latency_improvement_pct:>6.1f}%"
            f"  ({self.raw_latency_ms:.0f}ms → {self.optimized_latency_ms:.0f}ms)          │",
            f"│  KW-F1  raw={self.raw_kw_f1:.3f}  optimized={self.optimized_kw_f1:.3f}"
            f"  Δ={self.optimized_kw_f1 - self.raw_kw_f1:+.3f}                  │",
            f"│  Chunks {self.ingest_chunks}  compression ratio {self.ingest_compression_ratio:.2f}",
        ]
        if self.winning_branch:
            lines.append(f"│  ToT winner: {self.winning_branch}")
        if self.branch_scores:
            scores = "  ".join(f"{k}={v:.0f}" for k, v in self.branch_scores.items())
            lines.append(f"│  Branch scores: {scores}")
        lines.append("└───────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict suitable for JSON serialisation or DataFrame insertion."""
        return {
            "raw_tokens":               self.raw_tokens,
            "compressed_tokens":        self.compressed_tokens,
            "token_reduction_pct":      round(self.token_reduction_pct, 2),
            "raw_latency_ms":           round(self.raw_latency_ms, 2),
            "optimized_latency_ms":     round(self.optimized_latency_ms, 2),
            "latency_improvement_pct":  round(self.latency_improvement_pct, 2),
            "raw_kw_f1":                round(self.raw_kw_f1, 4),
            "optimized_kw_f1":          round(self.optimized_kw_f1, 4),
            "raw_log_lines":            self.raw_log_lines,
            "optimized_retrieved_lines": self.optimized_retrieved_lines,
            "retrieval_efficiency_pct": round(self.retrieval_efficiency_pct, 2),
            "ingest_chunks":            self.ingest_chunks,
            "ingest_compression_ratio": round(self.ingest_compression_ratio, 4),
            "winning_branch":           self.winning_branch,
            "branch_scores":            self.branch_scores,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _keyword_f1(output: str, keywords: list[str]) -> float:
    """Keyword F1 proxy — measures how densely the output references expected terms."""
    normalized = output.lower()
    answer_words = set(re.findall(r"[a-z0-9]+", normalized))
    matched = sum(1 for kw in keywords if kw.lower() in normalized)
    precision = matched / len(answer_words) if answer_words else 0.0
    recall = matched / len(keywords) if keywords else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _reference_keywords(question: str, extra: list[str] | None = None) -> list[str]:
    """Build a reference keyword list from the question + optional extras."""
    tokens = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2]
    return list(dict.fromkeys(tokens + (extra or [])))


# ── compare() ─────────────────────────────────────────────────────────────────

def compare(
    question: str,
    raw_corpus: list[str],
    *,
    compression_model: str = "llama3.2:3b",
    reference_keywords: list[str] | None = None,
    collection: str = "benchmark",
    llm: Any = None,
    **index_kwargs: Any,
) -> "BenchmarkResult":
    """
    Run Pipe A (raw corpus → LLM) vs Pipe C (CorpusIndex / ToT) and return
    structured comparison metrics.

    Parameters
    ----------
    question:
        The query used for both pipelines.
    raw_corpus:
        List of raw text lines (e.g. log lines, document paragraphs).
    compression_model:
        Ollama model to use for rolling-window compression.
    reference_keywords:
        Keywords expected to appear in a good answer.  Derived from *question*
        automatically when omitted.
    collection:
        ChromaDB collection name for the temporary index.
    llm:
        Optional LLM object for the Pipe A raw-dump baseline.  When ``None``
        the raw answer is skipped and ``raw_latency_ms`` is set to 0.
    **index_kwargs:
        Extra keyword arguments forwarded to :class:`~context_optimizer.CorpusIndex`.

    Returns
    -------
    BenchmarkResult
        Fully populated comparison metrics.
    """
    from context_optimizer.index import CorpusIndex

    kw = reference_keywords or _reference_keywords(question)

    # ── Pipe A — raw baseline ────────────────────────────────────────────────
    raw_text = "\n".join(raw_corpus)
    raw_tokens = _estimate_tokens(raw_text)
    raw_answer = ""
    raw_latency_ms = 0.0

    if llm is not None:
        import textwrap
        prompt = textwrap.dedent(
            f"Analyse the following logs and answer: {question}\n\n"
            f"Logs ({len(raw_corpus)} lines):\n{raw_text}"
        )
        t0 = time.perf_counter()
        resp = llm.invoke(prompt)
        raw_latency_ms = (time.perf_counter() - t0) * 1000
        raw_answer = getattr(resp, "content", str(resp))

    raw_kw_f1 = _keyword_f1(raw_answer or raw_text, kw)

    # ── Pipe C — CorpusIndex / ToT ───────────────────────────────────────────
    with CorpusIndex(
        compression_model=compression_model,
        **index_kwargs,
    ) as index:
        stats = index.ingest(raw_corpus, collection=collection)
        result = index.query(question, collection=collection)

    optimized_latency_ms = result.latency_ms
    optimized_kw_f1 = _keyword_f1(result.answer + "\n".join(result.evidence), kw)

    # ── Derived metrics ───────────────────────────────────────────────────────
    compressed_tokens = stats.compressed_tokens
    token_reduction = (
        max(0.0, (raw_tokens - compressed_tokens) / raw_tokens * 100.0)
        if raw_tokens else 0.0
    )

    latency_improvement = (
        (raw_latency_ms - optimized_latency_ms) / raw_latency_ms * 100.0
        if raw_latency_ms > 0 else 0.0
    )

    optimized_lines = result.tokens_used * 4  # rough inverse of _estimate_tokens
    retrieval_efficiency = (
        optimized_lines / len(raw_corpus) * 100.0
        if raw_corpus else 0.0
    )

    branch_scores: dict[str, float] = {}
    winning_branch: str | None = result.branch_id

    return BenchmarkResult(
        raw_tokens=raw_tokens,
        compressed_tokens=compressed_tokens,
        token_reduction_pct=round(token_reduction, 2),
        raw_latency_ms=round(raw_latency_ms, 2),
        optimized_latency_ms=round(optimized_latency_ms, 2),
        latency_improvement_pct=round(latency_improvement, 2),
        raw_kw_f1=round(raw_kw_f1, 4),
        optimized_kw_f1=round(optimized_kw_f1, 4),
        raw_log_lines=len(raw_corpus),
        optimized_retrieved_lines=len(result.evidence),
        retrieval_efficiency_pct=round(retrieval_efficiency, 2),
        ingest_chunks=stats.chunks,
        ingest_compression_ratio=round(stats.compression_ratio, 4),
        branch_scores=branch_scores,
        winning_branch=winning_branch,
        raw_answer=raw_answer,
        optimized_answer=result.answer,
    )
