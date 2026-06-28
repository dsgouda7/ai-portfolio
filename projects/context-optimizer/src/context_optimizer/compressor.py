"""
Rolling LLM Compression Pipeline

Implements threshold-based compression with a rolling context window to avoid
context exhaustion. Each chunk is compressed individually using a local LLM,
then stored alongside raw data for optional detailed retrieval.

Key Design:
- Rolling window: compress one chunk at a time (no context limit hit)
- Threshold-based: only compress when chunk accumulation reaches target size
- Dual storage: compressed summary + raw data backing
- MCP-ready: provides both compressed and detailed retrieval options
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from context_optimizer.raw_index import RawIndex

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None


@dataclass
class CompressedChunk:
    """Result of LLM compression with dual storage."""

    chunk_id: str
    raw_text: str  # Original data (for fallback retrieval)
    compressed_summary: str  # LLM prose summary — fed to reasoning LLM
    entities: list[str]  # Extracted entities for filtering
    keywords: list[str]  # Key concepts for search
    metadata: dict[str, str | int]  # Source, timestamp, etc
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed / original
    index_text: str = ""  # Entity-dense stopword-stripped form embedded in ChromaDB.
    # If empty, compressed_summary is used as the fallback.


class CompressorLLM(Protocol):
    """Protocol for LLM backends that support compression."""

    def invoke(self, prompt: str) -> object:
        """Invoke the LLM with a compression prompt."""
        ...


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)."""
    return max(1, len(text) // 4)


# ── Per-chunk JSONL file logger ──────────────────────────────────────────────
# Enabled by setting COMPRESSOR_LOG_FILE=/path/to/chunks.jsonl
# Each line is a JSON record: {ts, chunk_id, label, elapsed_s, orig_tokens,
#   comp_tokens, ratio, error (optional)}
# The logger is initialised lazily and shared across all threads (Python's
# logging module serialises handler writes with an internal lock).

_chunk_logger: logging.Logger | None = None


def _get_chunk_logger() -> logging.Logger | None:
    """Return the module-level chunk logger, creating it on first call."""
    global _chunk_logger
    if _chunk_logger is not None:
        return _chunk_logger
    log_path = os.getenv("COMPRESSOR_LOG_FILE")
    if not log_path:
        return None
    logger = logging.getLogger("compressor.chunks")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSONL
        logger.addHandler(handler)
    _chunk_logger = logger
    return _chunk_logger


def _log_chunk(
    chunk_id: str,
    label: str,
    elapsed_s: float,
    orig_tokens: int,
    comp_tokens: int,
    ratio: float,
    error: str | None = None,
) -> None:
    """Append one JSONL record to the chunk log (no-op if log file not configured)."""
    logger = _get_chunk_logger()
    if logger is None:
        return
    record: dict = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "chunk_id": chunk_id,
        "label": label,
        "elapsed_s": round(elapsed_s, 3),
        "orig_tokens": orig_tokens,
        "comp_tokens": comp_tokens,
        "ratio": round(ratio, 4),
    }
    if error:
        record["error"] = error
    logger.info(json.dumps(record))


# ── Ollama parallelism gate ───────────────────────────────────────────────────

def _ollama_parallel_slots() -> int:
    """Return OLLAMA_NUM_PARALLEL as int (defaults to 1 when unset)."""
    return max(1, int(os.getenv("OLLAMA_NUM_PARALLEL", "1")))


def _effective_workers(requested: int, explicit_llm: bool) -> int:
    """
    Clamp *requested* worker count to what the backend can actually serve in
    parallel.

    Rules
    -----
    - Explicit LLM instance provided → trust the caller, no clamping.
    - Ollama backend (default) → clamp to ``OLLAMA_NUM_PARALLEL`` (default 1).
      Ollama serialises requests when only one model slot is loaded; running
      N workers just creates N queued requests that are processed one-at-a-time,
      giving no speedup while holding N threads idle.
    - Any other backend (Groq, etc.) → no clamping.

    To enable true parallelism with Ollama, either:
      - Set ``OLLAMA_NUM_PARALLEL=4`` before starting Ollama *and* pass
        ``--workers 4`` to the benchmark, **or**
      - Switch to a cloud backend via ``CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=groq``.
    """
    if explicit_llm:
        return requested  # caller supplied their own LLM — they know best
    provider = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "ollama").lower()
    if provider != "ollama":
        return requested  # cloud backends handle their own concurrency
    limit = _ollama_parallel_slots()
    if requested > limit:
        print(
            f"[ParallelCompressor] Ollama detected — clamping workers "
            f"{requested} → {limit}  "
            f"(set OLLAMA_NUM_PARALLEL={requested} to unlock full parallelism)"
        )
    return min(requested, limit)


def _build_local_llm(
    provider: str = "ollama", model: str | None = None
) -> CompressorLLM | None:
    """
    Build a local LLM for compression.

    Prefers lightweight models optimized for summarization:
    - Ollama: phi4:mini, qwen2.5-coder:7b, llama3.2:3b
    - Groq: llama-3.3-70b-versatile (fast inference)
    """
    selected_provider = os.getenv(
        "CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", provider
    ).lower()

    if selected_provider == "ollama" and ChatOllama is not None:
        model_name = model or os.getenv(
            "CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "qwen2.5-coder:7b"
        )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.1)

    if selected_provider == "groq" and ChatGroq is not None:
        model_name = model or os.getenv(
            "CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "llama-3.3-70b-versatile"
        )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable required for Groq compression"
            )
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.1)

    return None


# ── Stopword normalisation ──────────────────────────────────────────────────

# Curated set of English function words that carry no retrieval signal.
# Deliberately small: sentence-transformer models (all-MiniLM-L6-v2, nomic-embed)
# are robust to their removal, and stripping them shifts the stored embedding
# toward content words — exactly what ToT branch scoring needs.
# Stemming is intentionally excluded: morphological variants ("timeouts" / "timeout")
# are already collapsed by the LLM prompt below.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "not",
        "no",
        "if",
        "so",
        "than",
        "then",
        "when",
        "where",
        "which",
        "who",
        "how",
        "what",
        "there",
        "their",
        "they",
        "them",
        "we",
        "our",
        "you",
        "your",
    }
)


def _normalise_for_index(text: str) -> str:
    """Strip stopwords and normalise whitespace before storing in ChromaDB.

    Applied only to ``compressed_summary`` (the indexed field).  ``raw_text``
    is never touched — it remains the human-readable fallback in RawIndex.

    Token-level stripping: split on whitespace, lower-case each token,
    strip trailing punctuation before the stopword check so "is," / "is."
    are caught.  Original casing is preserved for the kept tokens so
    technical identifiers (``CosmosDB``, ``HTTP 504``) round-trip unchanged.
    """
    tokens = text.split()
    kept = [t for t in tokens if t.lower().rstrip(".,;:!?'\"") not in _STOPWORDS]
    return " ".join(kept)


# ── Compression prompt ───────────────────────────────────────────────────────

COMPRESSION_PROMPT_TEMPLATE = """You are a semantic index builder. Transform the input text into a
retrieval-optimized representation that a vector-search system will embed.

Write as dense entity phrases, NOT narrative prose. A downstream LLM will
read your output and reconstruct full meaning from it, so preserve every
technical signal verbatim — it does not need connective words to understand.

**Preserve exactly (never paraphrase or abbreviate):**
- Error codes, status codes, version numbers, metric values, thresholds
- Function / class / variable names, config keys, file paths, package names
- System, component, protocol, and service names
- Causal chains expressed as: "X caused Y", "X triggered Z", "X exceeded Y"
- Numeric relationships: latency values, percentages, counts

**Style rules:**
- Lead each clause with the primary entity (noun-first)
- Join clauses with semicolons, not conjunctions
- Omit: articles (the/a/an), copulas (is/are/was), filler adverbs, hedges
- Wrong: "The CosmosDB instance experienced a timeout due to exceeded RU limits"
- Right:  "CosmosDB RU limit exceeded; request timeout error 21012; AKS ingress 504 upstream; retry cascade triggered"

**Input Text:**
{text}

**Output Format (JSON):**
{{
  "summary": "Retrieval-optimized dense-phrase summary — entity-first, semicolon-separated, no filler",
  "entities": ["entity1", "entity2", "entity3"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "has_code": true,
  "has_math": false,
  "section": "section name if identifiable"
}}

Respond with ONLY valid JSON, no explanations."""


def compress_chunk_with_llm(
    text: str,
    chunk_id: str,
    metadata: dict[str, str | int] | None = None,
    llm: CompressorLLM | None = None,
    max_summary_tokens: int = 150,
    label: str = "",
) -> CompressedChunk:
    """
    Compress a single chunk using LLM with rolling context window.

    This operates on ONE chunk at a time, avoiding context exhaustion even
    for large documents. The LLM sees only the current chunk, not the full corpus.

    Args:
        text: Raw chunk text to compress
        chunk_id: Unique identifier for this chunk
        metadata: Optional metadata (source, timestamp, etc)
        llm: LLM backend (if None, uses environment config)
        max_summary_tokens: Target summary length

    Returns:
        CompressedChunk with both compressed and raw data
    """
    if llm is None:
        llm = _build_local_llm()
        if llm is None:
            # Fallback: no compression, return raw text as "summary"
            return CompressedChunk(
                chunk_id=chunk_id,
                raw_text=text,
                compressed_summary=text[:200],  # Truncate for fallback
                entities=[],
                keywords=[],
                metadata=metadata or {},
                original_tokens=_estimate_tokens(text),
                compressed_tokens=_estimate_tokens(text[:200]),
                compression_ratio=0.5,
            )

    # Build compression prompt for THIS chunk only (rolling window)
    prompt = COMPRESSION_PROMPT_TEMPLATE.format(
        text=text[:2000]
    )  # Limit input to ~500 tokens

    t_start = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        elapsed = time.perf_counter() - t_start
        result_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse JSON response
        try:
            parsed = json.loads(result_text)
            summary = parsed.get("summary", text[:600])
            entities = parsed.get("entities", [])
            keywords = parsed.get("keywords", [])
            # Extract structural metadata
            has_code = parsed.get("has_code", False)
            has_math = parsed.get("has_math", False)
            section = parsed.get("section", "")
            if metadata is None:
                metadata = {}
            metadata["has_code"] = has_code
            metadata["has_math"] = has_math
            metadata["section"] = section
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            summary = result_text[:600]
            entities = []
            keywords = []

        # Build index_text: entity-dense, stopword-stripped form used exclusively
        # by ChromaDB for embedding.  compressed_summary stays as readable prose
        # so the reasoning LLM (ToT aggregated path) gets coherent sentences.
        index_parts = summary
        if entities:
            index_parts = index_parts + "; " + "; ".join(entities)
        index_text = _normalise_for_index(index_parts)

        original_tokens = _estimate_tokens(text)
        compressed_tokens = _estimate_tokens(summary)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        _log_chunk(chunk_id, label, elapsed, original_tokens,
                   min(compressed_tokens, max_summary_tokens), ratio)

        return CompressedChunk(
            chunk_id=chunk_id,
            raw_text=text,
            compressed_summary=summary,  # prose — used by reasoning LLM
            index_text=index_text,  # entity-dense — embedded in ChromaDB
            entities=entities,
            keywords=keywords,
            metadata=metadata or {},
            original_tokens=original_tokens,
            compressed_tokens=min(compressed_tokens, max_summary_tokens),
            compression_ratio=ratio,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        _log_chunk(chunk_id, label, elapsed, _estimate_tokens(text), 0, 0.0,
                   error=str(e))
        # Fallback on any LLM error
        print(f"[Compression Warning] LLM failed for chunk {chunk_id}: {e}")
        return CompressedChunk(
            chunk_id=chunk_id,
            raw_text=text,
            compressed_summary=text[:200],
            entities=[],
            keywords=[],
            metadata=metadata or {},
            original_tokens=_estimate_tokens(text),
            compressed_tokens=_estimate_tokens(text[:200]),
            compression_ratio=0.5,
        )


def compress_corpus_rolling(
    corpus_lines: list[str],
    chunk_size_threshold: int = 512,
    chunk_overlap_tokens: int = 64,
    compression_batch_size: int = 10,
    llm: CompressorLLM | None = None,
    progress_callback: callable | None = None,
    raw_index: "RawIndex | None" = None,
    label: str = "",
) -> list[CompressedChunk]:
    """
    Compress a large corpus using a rolling window strategy with overlap.

    Process:
    1. Accumulate lines until threshold reached
    2. Compress the accumulated chunk
    3. Keep last ~12% of chunk as overlap for next chunk (preserves causality
       chains that span boundaries; boundary-entity coverage is now handled by
       the LLM entity-extraction step which appends entities to compressed_summary)
    4. Repeat until corpus exhausted

    This avoids context exhaustion by:
    - Never sending full corpus to LLM
    - Processing one chunk at a time
    - Using threshold-based batching

    Args:
        corpus_lines: List of raw text lines
        chunk_size_threshold: Accumulate lines until this token count
        compression_batch_size: Process N chunks before yielding (for progress)
        llm: LLM backend (if None, uses environment config)
        progress_callback: Optional function(chunk_idx, total) for progress tracking
        raw_index: Optional :class:`~context_optimizer.raw_index.RawIndex` instance.
            When provided, each chunk's raw text is written to the SQLite store
            **in a background thread** while the main thread waits on the LLM.
            Because a SQLite write (~1 ms) is ~500× faster than a typical LLM
            call, the indexing is effectively free (fully overlapped with I/O).
            When ``None`` (default), raw text is only stored in ChromaDB metadata
            (truncated to 4 000 chars).

    Returns:
        List of CompressedChunk objects with dual storage
    """
    if llm is None:
        llm = _build_local_llm()

    compressed_chunks: list[CompressedChunk] = []
    current_chunk_lines: list[str] = []
    current_chunk_tokens = 0
    chunk_idx = 0
    overlap_lines: list[str] = []  # Track overlap from previous chunk

    _pfx = f"[{label}] " if label else ""
    print(
        f"{_pfx}[Compressor] Starting rolling compression of {len(corpus_lines):,} lines..."
    )
    print(
        f"{_pfx}[Compressor] Threshold: {chunk_size_threshold} tokens, "
        f"Overlap: {chunk_overlap_tokens} tokens, Batch: {compression_batch_size}"
        + (f", RawIndex: {raw_index._db_path!r}" if raw_index is not None else "")
    )

    # Background thread pool for raw-text indexing.
    # max_workers=1 → a single dedicated writer thread; all SQLite writes from
    # that thread share one WAL-mode connection, so there is no contention.
    raw_futures: list[Future] = []  # type: ignore[type-arg]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw_idx") as raw_exe:

        def _maybe_index_raw(cid: str, text: str) -> None:
            if raw_index is not None:
                f = raw_exe.submit(raw_index.add, cid, text)
                raw_futures.append(f)

        for line_idx, line in enumerate(corpus_lines):
            line_tokens = _estimate_tokens(line)
            current_chunk_lines.append(line)
            current_chunk_tokens += line_tokens

            # Threshold reached: compress this chunk
            if current_chunk_tokens >= chunk_size_threshold:
                chunk_text = "\n".join(current_chunk_lines)
                chunk_id = f"chunk_{chunk_idx:06d}"

                # ── Step 1: submit raw-text write (non-blocking, ~1 ms) ──────
                _maybe_index_raw(chunk_id, chunk_text)

                # ── Step 2: LLM compression (~500 ms, overlaps with above) ───
                _t0 = time.perf_counter()
                compressed = compress_chunk_with_llm(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata={
                        "line_start": line_idx - len(current_chunk_lines) + 1,
                        "line_end": line_idx,
                        "source_lines": len(current_chunk_lines),
                    },
                    llm=llm,
                    label=label,
                )
                _elapsed = time.perf_counter() - _t0
                compressed_chunks.append(compressed)
                est_total = max(1, len(corpus_lines) // max(1, chunk_size_threshold // 4))
                print(
                    f"{_pfx}[Compressor] chunk {chunk_idx + 1}/{est_total} done"
                    f"  ratio={compressed.compression_ratio:.0%}"
                    f"  {_elapsed:.1f}s"
                )

                # Progress reporting
                if progress_callback and chunk_idx % compression_batch_size == 0:
                    progress_callback(chunk_idx, est_total)

                # Prepare overlap for next chunk (last ~25% of current chunk)
                overlap_lines = []
                overlap_tokens = 0
                for overlap_line in reversed(current_chunk_lines):
                    line_tokens = _estimate_tokens(overlap_line)
                    if overlap_tokens + line_tokens <= chunk_overlap_tokens:
                        overlap_lines.insert(0, overlap_line)
                        overlap_tokens += line_tokens
                    else:
                        break

                # Reset for next chunk with overlap
                current_chunk_lines = overlap_lines.copy()
                current_chunk_tokens = overlap_tokens
                chunk_idx += 1

        # Handle remaining lines (final partial chunk)
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunk_id = f"chunk_{chunk_idx:06d}"

            _maybe_index_raw(chunk_id, chunk_text)

            compressed = compress_chunk_with_llm(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    "line_start": len(corpus_lines) - len(current_chunk_lines),
                    "line_end": len(corpus_lines) - 1,
                    "source_lines": len(current_chunk_lines),
                },
                llm=llm,
                label=label,
            )
            compressed_chunks.append(compressed)

    # ThreadPoolExecutor.__exit__ calls shutdown(wait=True), so all raw_futures
    # are guaranteed to be complete by the time we reach here.

    total_original = sum(c.original_tokens for c in compressed_chunks)
    total_compressed = sum(c.compressed_tokens for c in compressed_chunks)
    avg_ratio = total_compressed / total_original if total_original > 0 else 1.0

    print(f"{_pfx}[Compressor] [OK] Compressed {len(compressed_chunks):,} chunks")
    print(
        f"{_pfx}[Compressor] ratio: {avg_ratio:.2%} ({total_original:,} => {total_compressed:,} tokens)"
    )

    return compressed_chunks


def compress_corpus_parallel(
    corpus_map: dict[str, list[str]],
    workers: int = 4,
    **rolling_kwargs,
) -> dict[str, list["CompressedChunk"]]:
    """
    Compress multiple independent corpora concurrently.

    Each corpus (e.g. one book) is compressed in its own worker thread by
    calling :func:`compress_corpus_rolling`.  The Ollama server queues the
    underlying LLM calls; set ``OLLAMA_NUM_PARALLEL=<workers>`` in your
    environment to allow Ollama to serve that many requests simultaneously
    (default is 1).

    Parameters
    ----------
    corpus_map:
        ``{corpus_id: lines}`` mapping.  Each value is the list of raw text
        lines that would normally be passed to ``compress_corpus_rolling``.
    workers:
        Maximum number of parallel compression threads.
    **rolling_kwargs:
        Passed verbatim to every ``compress_corpus_rolling`` call
        (e.g. ``chunk_size_threshold``, ``chunk_overlap_tokens``, ``llm``).

    Returns
    -------
    dict[corpus_id, list[CompressedChunk]]
        Preserves all keys from *corpus_map*; failed corpora return ``[]``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, list[CompressedChunk]] = {}
    total = len(corpus_map)

    effective = _effective_workers(workers, explicit_llm="llm" in rolling_kwargs)
    print(
        f"[ParallelCompressor] {total} corpus/a  |  "
        f"{effective} effective worker(s) (requested: {workers})  |  "
        f"log: {os.getenv('COMPRESSOR_LOG_FILE', 'disabled')}"
    )

    with ThreadPoolExecutor(
        max_workers=effective, thread_name_prefix="compressor"
    ) as exe:
        futures = {
            exe.submit(
                compress_corpus_rolling, lines, label=corpus_id, **rolling_kwargs
            ): corpus_id
            for corpus_id, lines in corpus_map.items()
        }
        done = 0
        for future in as_completed(futures):
            corpus_id = futures[future]
            done += 1
            try:
                chunks = future.result()
                results[corpus_id] = chunks
                print(
                    f"  [ParallelCompressor] [{done}/{total}] {corpus_id}: "
                    f"{len(chunks)} chunks"
                )
            except Exception as exc:
                results[corpus_id] = []
                print(
                    f"  [ParallelCompressor] [{done}/{total}] {corpus_id}: "
                    f"ERROR — {exc}"
                )

    total_chunks = sum(len(v) for v in results.values())
    print(
        f"[ParallelCompressor] Done — {total_chunks} total chunks across {total} corpus/a"
    )
    return results


if __name__ == "__main__":
    # Quick test
    test_corpus = [
        "System.TimeoutException at line 1042 in CosmosClient.ReadItemAsync",
        "Error code 21012: Connection timeout to primary replica",
        "Cascade failure detected in payment-service downstream",
        "Retry attempt 3/3 failed with same error",
        "Circuit breaker opened for cosmosdb-primary endpoint",
    ]

    compressed = compress_corpus_rolling(
        test_corpus,
        chunk_size_threshold=50,  # Small threshold for test
        compression_batch_size=1,
        progress_callback=lambda idx, total: print(f"  Progress: {idx}/{total}"),
    )

    for chunk in compressed:
        print(f"\n{chunk.chunk_id}:")
        print(f"  Original ({chunk.original_tokens} tokens): {chunk.raw_text[:100]}...")
        print(
            f"  Compressed ({chunk.compressed_tokens} tokens): {chunk.compressed_summary}"
        )
        print(f"  Entities: {chunk.entities}")
        print(f"  Keywords: {chunk.keywords}")
