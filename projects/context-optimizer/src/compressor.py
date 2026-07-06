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


def split_into_sub_chunks(text: str, sub_chunk_tokens: int = 200) -> list[str]:
    """
    Split *text* into sub-chunks of approximately *sub_chunk_tokens* each.

    Used by parent-child multi-vector indexing: the parent stores a compressed
    summary while children store raw sub-chunks embedded with exact vocabulary,
    so granular keywords that an LLM might scrub from a summary are preserved in
    the vector space.  When a child chunk matches a query, its ``parent_chunk_id``
    is used to fetch and return the parent summary.

    Splitting is sentence-boundary-aware (splits on ``.`` / ``!`` / ``?``).
    Token budget uses the same 4-chars/token heuristic as :func:`_estimate_tokens`.

    Parameters
    ----------
    text:
        Source text to split.
    sub_chunk_tokens:
        Target token budget per sub-chunk (default 200 ≈ 800 chars).

    Returns
    -------
    list[str]
        Ordered sub-chunk strings.  Always at least one element.
    """
    import re as _re

    if _estimate_tokens(text) <= sub_chunk_tokens:
        return [text]

    sentences = [
        s.strip() for s in _re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
    ]
    if not sentences:
        return [text]

    sub_chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)
        if current_tokens + sent_tokens > sub_chunk_tokens and current:
            sub_chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        sub_chunks.append(" ".join(current))

    return sub_chunks if sub_chunks else [text]


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
            f"{requested} -> {limit}  "
            f"(set OLLAMA_NUM_PARALLEL={requested} to unlock full parallelism)"
        )
    return min(requested, limit)


def _build_local_llm(
    provider: str | None = None, model: str | None = None
) -> "CompressorLLM | None":
    """
    Build a compressor LLM.  Explicit *provider* / *model* params take
    precedence over environment variables so CLI flags always win.

    Supported providers
    -------------------
    ``ollama``  Local Ollama instance (default).  Cheap, private.
                Env vars: ``OLLAMA_BASE_URL``, ``CONTEXT_OPTIMIZER_COMPRESSOR_MODEL``
    ``groq``    Groq cloud API (fast, free tier available).
                Env var: ``GROQ_API_KEY``, ``CONTEXT_OPTIMIZER_COMPRESSOR_MODEL``
    ``azure``   Azure OpenAI (any deployment — use a cheap model here, e.g.
                ``gpt-4o-mini``).
                Env vars: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
                           ``AZURE_OPENAI_API_VERSION`` (default 2024-02-01),
                           ``AZURE_COMPRESSOR_DEPLOYMENT`` (default gpt-4o-mini)
    """
    selected_provider = (
        provider or os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "ollama")
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
            raise ValueError("GROQ_API_KEY environment variable required for Groq")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.1)

    if selected_provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pip install langchain-openai  for Azure OpenAI support"
            ) from exc
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
        if not endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set for azure provider"
            )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        deployment = model or os.getenv("AZURE_COMPRESSOR_DEPLOYMENT", "gpt-4o-mini")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.1,
        )

    return None


def _build_reasoner_llm(
    provider: str | None = None, model: str | None = None
) -> "CompressorLLM | None":
    """
    Build a *reasoning* LLM — typically a larger, more capable model than the
    compressor.  Reads from separate ``CONTEXT_OPTIMIZER_REASONER_*`` env vars
    so compressor and reasoner can target different providers / deployments.

    Supported providers: same as :func:`_build_local_llm`.
    Azure env vars: ``AZURE_REASONER_DEPLOYMENT`` (default ``gpt-4o``).
    """
    selected_provider = (
        provider or os.getenv("CONTEXT_OPTIMIZER_REASONER_PROVIDER", "")
    ).lower()
    if not selected_provider:
        return None  # no reasoner configured — ToT returns raw snippets

    if selected_provider == "ollama" and ChatOllama is not None:
        model_name = model or os.getenv(
            "CONTEXT_OPTIMIZER_REASONER_MODEL", "llama3.2:3b"
        )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.0)

    if selected_provider == "groq" and ChatGroq is not None:
        model_name = model or os.getenv(
            "CONTEXT_OPTIMIZER_REASONER_MODEL", "llama-3.3-70b-versatile"
        )
        api_key = os.getenv("GROQ_API_KEY") or ""
        if not api_key:
            raise ValueError("GROQ_API_KEY required for Groq reasoner")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.0)

    if selected_provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install langchain-openai for Azure OpenAI") from exc
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
        if not endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set"
            )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        deployment = model or os.getenv("AZURE_REASONER_DEPLOYMENT", "gpt-4o")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.0,
        )

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

COMPRESSION_PROMPT_TEMPLATE = """You are a semantic index builder. Compress the input text into a dense, information-rich representation for vector search and retrieval.

**Goal:** Preserve maximum semantic content in minimum tokens. Target output: 25-40% of input length. Do NOT produce a one-sentence abstract.

**Always preserve:**
- Proper nouns: character names, place names, organization names, titles, dates
- Key actions and relationships: "X did Y", "X caused Z", "X is the Y of Z"
- Specific values: numbers, measurements, dates, counts, percentages
- Causal chains: what triggered what, why something happened
- Domain terms, technical vocabulary, named concepts

**Style:**
- Write noun-first dense phrases, semicolons between clauses
- Omit: articles (the/a/an), filler words, connective phrases with no information
- Capture multiple facts from across the passage — not just the opening sentence

**Examples:**
- Input (narrative): "Call me Ishmael. Having little money and nothing on shore, I decided to sail and see the sea. I signed aboard the Pequod, captained by the obsessed Ahab."
- Output: "Ishmael narrator; no money, nothing on shore; joins Pequod whaling ship; Captain Ahab obsessed"

- Input (technical): "The function raises ValueError when the input buffer exceeds 4096 bytes. This limit was introduced in v2.3 to prevent memory exhaustion on constrained devices."
- Output: "ValueError raised; input buffer >4096 bytes; limit since v2.3; prevents memory exhaustion constrained devices"

**Input Text:**
{text}

**Output (JSON only):**
{{
  "summary": "Dense, fact-preserving representation — 25-40% of input length, noun-first, semicolon-separated",
  "entities": ["entity1", "entity2"],
  "keywords": ["keyword1", "keyword2"],
  "has_code": false,
  "has_math": false,
  "section": "section name if identifiable"
}}

Respond with ONLY valid JSON, no explanations."""


# ── Extractive compression (no LLM required) ────────────────────────────────


def _extractive_compress(text: str, ratio: float = 0.35) -> str:
    """
    Select the most informative sentences using TF-IDF scoring.
    Returns the top *ratio* fraction of sentences in original order.
    Completely deterministic — zero LLM, zero external dependencies.
    """
    import re as _re

    sentences = [
        s.strip() for s in _re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
    ]
    if len(sentences) <= 2:
        return text

    _stop = frozenset(
        "a an the and or but in on at to for of with by from is are was were be been "
        "have has had do does did that this it he she they we you i his her its their "
        "our not so as if up out no about what which will would could should may might "
        "then than when where who whom whose how much many more most some any all".split()
    )

    freq: dict[str, int] = {}
    for s in sentences:
        for w in _re.findall(r"[a-z]+", s.lower()):
            if w not in _stop and len(w) >= 3:
                freq[w] = freq.get(w, 0) + 1
    if not freq:
        return text

    max_f = max(freq.values())

    def _score(s: str) -> float:
        words = [
            w
            for w in _re.findall(r"[a-z]+", s.lower())
            if w not in _stop and len(w) >= 3
        ]
        return sum(freq.get(w, 0) / max_f for w in words) / len(words) if words else 0.0

    scored = [(i, s, _score(s)) for i, s in enumerate(sentences)]
    n_keep = max(2, round(len(sentences) * ratio))
    top = sorted(scored, key=lambda x: x[2], reverse=True)[:n_keep]
    top.sort(key=lambda x: x[0])  # restore original order
    return " ".join(s for _, s, _ in top)


def compress_chunk_extractive(
    text: str,
    chunk_id: str,
    metadata: dict[str, str | int] | None = None,
    ratio: float = 0.35,
    label: str = "",
) -> "CompressedChunk":
    """
    Compress a chunk by extracting the most important sentences (no LLM).
    ~0.01s/chunk, deterministic, preserves exact source vocabulary so
    keyword recall on raw-text questions is high.
    """
    import re as _re

    t_start = time.perf_counter()
    summary = _extractive_compress(text, ratio=ratio)

    _stop = frozenset(
        "a an the and or but in on at to for of with by from is are was were".split()
    )
    freq: dict[str, int] = {}
    for w in _re.findall(r"[a-z]+", text.lower()):
        if w not in _stop and len(w) >= 4:
            freq[w] = freq.get(w, 0) + 1
    keywords = sorted(freq, key=lambda k: freq[k], reverse=True)[:15]
    entities = list(
        dict.fromkeys(
            w
            for w in _re.findall(r"\b[A-Z][a-z]{2,}\b", text)
            if w.lower() not in _stop
        )
    )[:15]

    orig = _estimate_tokens(text)
    comp = _estimate_tokens(summary)
    actual_ratio = comp / orig if orig else 1.0
    _log_chunk(chunk_id, label, time.perf_counter() - t_start, orig, comp, actual_ratio)

    return CompressedChunk(
        chunk_id=chunk_id,
        raw_text=text,
        compressed_summary=summary,
        index_text="",
        entities=entities,
        keywords=keywords,
        metadata=metadata or {},
        original_tokens=orig,
        compressed_tokens=comp,
        compression_ratio=actual_ratio,
    )


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
        _log_chunk(
            chunk_id,
            label,
            elapsed,
            original_tokens,
            min(compressed_tokens, max_summary_tokens),
            ratio,
        )

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
        _log_chunk(
            chunk_id, label, elapsed, _estimate_tokens(text), 0, 0.0, error=str(e)
        )
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
    strategy: str = "llm",  # "llm" | "extractive" | "raw_only"
    compressor_provider: str | None = None,
    compressor_model: str | None = None,
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
        compressor_provider: Provider override (``ollama``, ``groq``, ``azure``).
            Takes precedence over ``CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER`` env var.
        compressor_model: Model / deployment name override.

    Returns:
        List of CompressedChunk objects with dual storage
    """
    if llm is None and strategy == "llm":
        llm = _build_local_llm(provider=compressor_provider, model=compressor_model)

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

                # ── Step 2: Compression (~0.01s extractive | ~12s LLM) ───────
                _t0 = time.perf_counter()
                _meta = {
                    "line_start": line_idx - len(current_chunk_lines) + 1,
                    "line_end": line_idx,
                    "source_lines": len(current_chunk_lines),
                }
                if strategy == "extractive":
                    compressed = compress_chunk_extractive(
                        text=chunk_text, chunk_id=chunk_id, metadata=_meta, label=label
                    )
                else:
                    compressed = compress_chunk_with_llm(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata=_meta,
                        llm=llm,
                        label=label,
                    )
                _elapsed = time.perf_counter() - _t0
                compressed_chunks.append(compressed)
                est_total = max(
                    1, len(corpus_lines) // max(1, chunk_size_threshold // 4)
                )
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

            _meta_final = {
                "line_start": len(corpus_lines) - len(current_chunk_lines),
                "line_end": len(corpus_lines) - 1,
                "source_lines": len(current_chunk_lines),
            }
            if strategy == "extractive":
                compressed = compress_chunk_extractive(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata=_meta_final,
                    label=label,
                )
            else:
                compressed = compress_chunk_with_llm(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata=_meta_final,
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


def cluster_and_compress_corpus(
    corpus_lines: list[str],
    target_cluster_size: int = 50,
    sub_chunk_tokens: int = 200,
    llm: "CompressorLLM | None" = None,
    label: str = "",
    strategy: str = "llm",
    compressor_provider: str | None = None,
    compressor_model: str | None = None,
    raw_index: "RawIndex | None" = None,
) -> "list[CompressedChunk]":
    """
    K-Means cluster-then-summarize ingestion strategy.

    Motivation
    ----------
    A naive pipeline that compresses every 200-token sub-chunk individually
    makes O(N) LLM calls where N is the number of sub-chunks.  For a 2 GB
    corpus (≈500 M tokens → ≈2.5 M sub-chunks at 200 t/chunk) this is
    millions of LLM calls.

    This pipeline instead:
    1. Splits the corpus into raw sub-chunks (200 tokens each) — no LLM.
    2. Builds TF-IDF vectors over all sub-chunks — no LLM, < 1 s for 10 k docs.
    3. Groups sub-chunks by K-Means similarity into macro-clusters.
    4. Concatenates each cluster and compresses it as one unit (1 LLM call).

    Result: LLM calls reduced from N → N / target_cluster_size (≈ 95% savings
    with target_cluster_size=50).

    Requires ``scikit-learn`` (``pip install scikit-learn``).

    Parameters
    ----------
    corpus_lines:
        Raw corpus lines (same format as :func:`compress_corpus_rolling`).
    target_cluster_size:
        Average number of sub-chunks per cluster.  Default 50 sub-chunks/cluster
        ≈ 10 000 tokens of source text per compressed chunk.
    sub_chunk_tokens:
        Token budget for each sub-chunk before clustering.
    llm, strategy, compressor_provider, compressor_model:
        Passed to :func:`compress_chunk_with_llm` / :func:`compress_chunk_extractive`.
    raw_index:
        When provided, raw text of each cluster is stored here.
    label:
        Progress log prefix.

    Returns
    -------
    list[CompressedChunk]
        One chunk per cluster.  ``chunk_id`` = ``cluster_{k:04d}``.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,  # type: ignore[import]
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn required for cluster_and_compress_corpus. "
            "pip install scikit-learn"
        ) from exc

    _pfx = f"[{label}] " if label else ""
    full_text = "\n".join(corpus_lines)
    sub_chunks = split_into_sub_chunks(full_text, sub_chunk_tokens=sub_chunk_tokens)

    n_clusters = max(1, len(sub_chunks) // target_cluster_size)
    print(
        f"{_pfx}[ClusterCompressor] {len(sub_chunks):,} sub-chunks → "
        f"{n_clusters:,} clusters (target_cluster_size={target_cluster_size})"
    )

    if n_clusters == 1:
        # Corpus is small enough to treat as one cluster
        return compress_corpus_rolling(
            corpus_lines,
            llm=llm,
            label=label,
            strategy=strategy,
            compressor_provider=compressor_provider,
            compressor_model=compressor_model,
            raw_index=raw_index,
        )

    # ── Step 1: TF-IDF vectorisation (sparse, no LLM, no embeddings) ─────────
    # TfidfVectorizer returns scipy.sparse.csr_matrix — MiniBatchKMeans works
    # natively on sparse matrices so no densification is needed.
    # max_features=5000 keeps memory bounded: 50K docs × 5K features ≈ 200 MB sparse.
    print(f"{_pfx}[ClusterCompressor] Building TF-IDF vectors ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,   # log(1 + tf) instead of raw tf — compresses high-freq terms
    )
    tfidf_matrix = vectorizer.fit_transform(sub_chunks)

    # ── Step 2: Mini-batch K-Means clustering ─────────────────────────────────
    # MiniBatchKMeans processes random mini-batches each iteration instead of
    # the full matrix, giving 10–100× speedup vs KMeans on large corpora with
    # near-identical cluster quality for approximate groupings.
    #
    # batch_size: process 1024 docs per mini-batch (fits in L2 cache on most CPUs).
    # n_init=3:   try 3 random seeds (vs default 10 for KMeans); for grouping
    #             purposes, the best of 3 is indistinguishable from best of 10.
    # max_iter=100: enough for convergence on text data; TF-IDF vectors are
    #              already normalised so convergence is fast.
    print(
        f"{_pfx}[ClusterCompressor] Fitting MiniBatchKMeans(n_clusters={n_clusters}) ..."
    )
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3,
        max_iter=100,
        batch_size=min(1024, len(sub_chunks)),  # never larger than the dataset
    )
    labels = km.fit_predict(tfidf_matrix)

    # ── Step 3: Collect sub-chunks per cluster ────────────────────────────────
    clusters: dict[int, list[str]] = {k: [] for k in range(n_clusters)}
    for sub_idx, cluster_id in enumerate(labels):
        clusters[int(cluster_id)].append(sub_chunks[sub_idx])

    # ── Step 4: Compress each cluster (one LLM call per cluster) ─────────────
    if llm is None and strategy == "llm":
        llm = _build_local_llm(provider=compressor_provider, model=compressor_model)

    compressed_chunks: list[CompressedChunk] = []
    for k, cluster_texts in sorted(clusters.items()):
        cluster_text = "\n\n".join(cluster_texts)
        chunk_id = f"cluster_{k:04d}"
        meta: dict[str, str | int] = {"cluster_id": k, "sub_chunks": len(cluster_texts)}

        if raw_index is not None:
            raw_index.add(chunk_id, cluster_text)

        t0 = time.perf_counter()
        if strategy == "extractive":
            compressed = compress_chunk_extractive(
                text=cluster_text, chunk_id=chunk_id, metadata=meta, label=label
            )
        else:
            compressed = compress_chunk_with_llm(
                text=cluster_text,
                chunk_id=chunk_id,
                metadata=meta,
                llm=llm,
                label=label,
            )
        elapsed = time.perf_counter() - t0
        compressed_chunks.append(compressed)
        print(
            f"{_pfx}[ClusterCompressor] cluster {k + 1}/{n_clusters} done "
            f"({len(cluster_texts)} sub-chunks, {len(cluster_text):,} chars)  "
            f"{elapsed:.1f}s"
        )

    print(
        f"{_pfx}[ClusterCompressor] [OK] {len(compressed_chunks):,} cluster chunks "
        f"(saved {len(sub_chunks) - n_clusters:,} LLM calls vs. per-sub-chunk compression)"
    )
    return compressed_chunks


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
