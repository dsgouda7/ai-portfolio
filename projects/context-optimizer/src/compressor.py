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


# ── Fixed-length block summary prompt ────────────────────────────────────────
# Used by ingest_file_blocks for large-corpus ingestion.
# Unlike COMPRESSION_PROMPT_TEMPLATE (which targets 25-40% of input), this
# prompt enforces a HARD 150-200 word output budget regardless of input size.
# A 500 KB block (~131K tokens) gets the same 200-word summary as a 2 KB chunk.
# This keeps ChromaDB documents within the embedding model's 512-token window
# and makes per-query token cost predictable and cheap (~1,000 tokens for top-5).

BLOCK_SUMMARY_PROMPT = """You are a semantic index builder for a large-corpus retrieval system.

A block of text is given below. Your task: produce a FIXED-LENGTH DENSE SUMMARY of EXACTLY 150-200 words that becomes the sole search index entry for this entire block.

HARD RULES:
1. Output EXACTLY 150-200 words in the "summary" field. Count carefully.
2. Every word must carry information — zero filler, zero preamble.
3. Style: noun-first, semicolon-separated dense clauses.
   Good: "Einstein; general relativity 1915; field equations; light bends near mass"
   Bad:  "This passage discusses Einstein's work on relativity..."
4. Cover facts from THROUGHOUT the block, not just the opening sentences.
5. Preserve: proper nouns, dates, numbers, causal chains, named concepts.

Block text (may be large — summarise it all into 150-200 words):
{text}

Output ONLY valid JSON, no explanation:
{{
  "summary": "150-200 word dense noun-first fact summary; semicolons between clauses",
  "entities": ["name1", "name2"],
  "keywords": ["concept1", "concept2"]
}}"""


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
    max_summary_tokens: int = 0,
    label: str = "",
) -> "CompressedChunk":
    """
    Compress a chunk by extracting the most important sentences (no LLM).
    ~0.01s/chunk, deterministic, preserves exact source vocabulary so
    keyword recall on raw-text questions is high.

    Parameters
    ----------
    max_summary_tokens:
        When > 0, truncate the extracted summary to this many tokens after
        sentence selection.  Crucial for large blocks (e.g. 500 KB): without
        this cap the summary may still be 50K tokens, which exceeds the
        embedding model's context window and inflates per-query token cost.
        Recommended value for 500 KB blocks: 500.
    """
    import re as _re

    t_start = time.perf_counter()
    summary = _extractive_compress(text, ratio=ratio)

    # Cap summary length when max_summary_tokens is set.
    # We truncate at a sentence boundary to preserve readability.
    if max_summary_tokens > 0:
        cap_chars = max_summary_tokens * 4  # 4 chars ≈ 1 token
        if len(summary) > cap_chars:
            # Truncate at the last sentence boundary before the cap
            truncated = summary[:cap_chars]
            last_period = max(
                truncated.rfind(". "),
                truncated.rfind("? "),
                truncated.rfind("! "),
            )
            if last_period > cap_chars // 2:
                summary = truncated[: last_period + 1]
            else:
                summary = truncated

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


def ingest_file_blocks(
    source_path: "str | Path",
    block_size_bytes: int = 500_000,
    overlap_bytes: int = 0,
    block_index: "Any | None" = None,
    llm: "CompressorLLM | None" = None,
    strategy: str = "llm",
    label: str = "",
    compressor_provider: str | None = None,
    compressor_model: str | None = None,
) -> "list[CompressedChunk]":
    """
    Ingest a large file as fixed-size byte blocks with optional overlap.

    Architecture
    ------------
    Each block produces two representations:

    1. **Compressed summary** in ChromaDB — for the ``strategy="llm"`` path
       this is a fixed 150-200 word summary generated by a quantized local LLM
       (default: ``llama3.2:3b`` via Ollama).  The output budget is enforced
       in the prompt regardless of input size, so every block always produces
       a ~200-token summary that fits within the embedding model's context
       window.

    2. **File pointer** in ``BlockIndex`` — records ``(file_path, byte_start,
       byte_end)`` so raw text is read on demand without duplication.

    Overlap
    -------
    When *overlap_bytes* > 0 (recommended: 10% of *block_size_bytes*), the
    last *overlap_bytes* of each block are prepended as context to the next
    block's LLM input.  The file pointer records only the NEW bytes of each
    block (not the overlap), so there is no storage overhead.  Overlap helps
    when a concept spans a block boundary — the LLM sees both sides and the
    summary captures the full context.

    Parameters
    ----------
    source_path:
        Path to the text file to ingest.
    block_size_bytes:
        Target block size in bytes (default 500 KB).
    overlap_bytes:
        Bytes of the previous block to prepend as context for the next block's
        summary (default 0).  Set to ``block_size_bytes // 10`` (10% overlap)
        for better semantic continuity across block boundaries.
    block_index:
        A ``BlockIndex`` instance.  When provided, file pointers are written.
    llm:
        Pre-built LLM instance.  If ``None`` and ``strategy="llm"``, a local
        Ollama instance is created via ``_build_local_llm()``.
    strategy:
        ``"llm"``        — fixed 150-200 word dense summary via quantized LLM.
        ``"extractive"`` — TF-IDF sentence selection, capped at 500 tokens.
        ``"raw_only"``   — truncated raw text (no compression).
    """
    import re as _re
    from pathlib import Path as _Path

    source_path = _Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    file_size = source_path.stat().st_size
    source_name = _re.sub(r"[^a-z0-9]+", "_", source_path.stem.lower())[:30]
    _pfx = f"[{label}] " if label else ""

    if llm is None and strategy == "llm":
        llm = _build_local_llm(provider=compressor_provider, model=compressor_model)

    est_blocks = max(1, file_size // block_size_bytes)
    overlap_info = f"  overlap={overlap_bytes//1024} KB" if overlap_bytes > 0 else ""
    print(
        f"{_pfx}[BlockIngestor] {source_path.name}  "
        f"({file_size / 1_048_576:.1f} MB)  "
        f"block={block_size_bytes // 1024} KB{overlap_info}  "
        f"~{est_blocks} blocks  strategy={strategy}"
    )

    compressed_chunks: list[CompressedChunk] = []
    block_metas: list[tuple[str, str, int, int]] = []  # (block_id, path, start, end)

    byte_pos = 0
    block_idx = 0
    overlap_tail: bytes = b""  # tail of the previous block for context

    with open(source_path, "rb") as fh:
        while True:
            block_start = byte_pos
            raw_bytes = fh.read(block_size_bytes)
            if not raw_bytes:
                break

            # Align to next newline to avoid mid-word splits
            next_byte = fh.read(1)
            while next_byte and next_byte != b"\n":
                raw_bytes += next_byte
                next_byte = fh.read(1)
            if next_byte:
                raw_bytes += next_byte

            block_end = block_start + len(raw_bytes)
            byte_pos = block_end

            # The LLM input: previous overlap (for context) + current block
            llm_input_bytes = overlap_tail + raw_bytes
            block_text = llm_input_bytes.decode("utf-8", errors="replace")

            # Update overlap for next block (tail of CURRENT block, not input)
            overlap_tail = raw_bytes[-overlap_bytes:] if overlap_bytes > 0 else b""

            chunk_id = f"{source_name}_block_{block_idx:06d}"
            meta: dict[str, str | int] = {
                "source_file": str(source_path),
                "byte_start": block_start,  # unique range only
                "byte_end": block_end,
                "block_idx": block_idx,
            }

            t0 = time.perf_counter()

            if strategy == "llm":
                # Use fixed-length block summary prompt: enforces 150-200 word output
                # regardless of input size, then post-truncate to 200 tokens as safety net.
                if llm is None:
                    # Fallback if Ollama is not running
                    compressed = compress_chunk_extractive(
                        text=block_text[:8000],
                        chunk_id=chunk_id,
                        metadata=meta,
                        label=label,
                        max_summary_tokens=200,
                    )
                else:
                    # Build prompt: feed up to 8000 chars (~2000 tokens) to stay within
                    # context limits of small quantized models (llama3.2:3b = 128K ctx)
                    # but keeping the call fast.  The block is sampled strategically:
                    # first 4000 chars + last 4000 chars — captures intro and conclusion.
                    if len(block_text) > 8000:
                        text_sample = block_text[:4000] + "\n...\n" + block_text[-4000:]
                    else:
                        text_sample = block_text

                    prompt = BLOCK_SUMMARY_PROMPT.format(text=text_sample)
                    try:
                        resp = llm.invoke(prompt)
                        resp_text = (
                            resp.content if hasattr(resp, "content") else str(resp)
                        )
                        try:
                            parsed = json.loads(resp_text)
                            summary = parsed.get("summary", "")
                            entities = parsed.get("entities", [])
                            keywords = parsed.get("keywords", [])
                        except (json.JSONDecodeError, ValueError):
                            # LLM returned non-JSON — use raw response, truncated
                            summary = resp_text[:800]
                            entities = []
                            keywords = []

                        # Safety net: hard-cap at 200 tokens (~800 chars)
                        if len(summary) > 800:
                            cut = summary[:800]
                            last_semi = cut.rfind(";")
                            summary = cut[: last_semi + 1] if last_semi > 400 else cut

                        orig_tok = _estimate_tokens(block_text)
                        comp_tok = _estimate_tokens(summary)
                        ratio = comp_tok / orig_tok if orig_tok else 1.0
                        elapsed = time.perf_counter() - t0
                        _log_chunk(chunk_id, label, elapsed, orig_tok, comp_tok, ratio)

                        index_text = _normalise_for_index(
                            summary + ("; " + "; ".join(entities) if entities else "")
                        )
                        compressed = CompressedChunk(
                            chunk_id=chunk_id,
                            raw_text=block_text[:2000],  # store excerpt as raw fallback
                            compressed_summary=summary,
                            index_text=index_text,
                            entities=entities,
                            keywords=keywords,
                            metadata=meta,
                            original_tokens=orig_tok,
                            compressed_tokens=comp_tok,
                            compression_ratio=ratio,
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - t0
                        print(f"{_pfx}[BlockIngestor] WARNING block {block_idx}: LLM error — {exc}")
                        compressed = compress_chunk_extractive(
                            text=block_text[:8000],
                            chunk_id=chunk_id,
                            metadata=meta,
                            label=label,
                            max_summary_tokens=200,
                        )

            elif strategy == "extractive":
                compressed = compress_chunk_extractive(
                    text=block_text,
                    chunk_id=chunk_id,
                    metadata=meta,
                    label=label,
                    max_summary_tokens=500,
                )
            else:  # raw_only
                compressed = CompressedChunk(
                    chunk_id=chunk_id,
                    raw_text=block_text,
                    compressed_summary=block_text[:800],
                    index_text="",
                    entities=[],
                    keywords=[],
                    metadata=meta,
                    original_tokens=_estimate_tokens(block_text),
                    compressed_tokens=_estimate_tokens(block_text[:800]),
                    compression_ratio=1.0,
                )

            elapsed = time.perf_counter() - t0
            compressed_chunks.append(compressed)
            block_metas.append((chunk_id, str(source_path), block_start, block_end))

            print(
                f"{_pfx}[BlockIngestor] block {block_idx + 1}/{est_blocks} "
                f"({len(raw_bytes) / 1024:.0f} KB -> {compressed.compressed_tokens} tokens"
                f"{' [overlap]' if overlap_bytes and block_idx > 0 else ''})  "
                f"{elapsed:.2f}s"
            )
            block_idx += 1

    # Write file pointers to BlockIndex after all blocks processed
    if block_index is not None:
        block_index.add_many(block_metas)
        print(
            f"{_pfx}[BlockIngestor] BlockIndex: registered {len(block_metas)} file pointers"
        )

    total_orig = sum(c.original_tokens for c in compressed_chunks)
    total_comp = sum(c.compressed_tokens for c in compressed_chunks)
    ratio = total_comp / total_orig if total_orig else 1.0
    print(
        f"{_pfx}[BlockIngestor] [OK] {len(compressed_chunks)} blocks  "
        f"ratio={ratio:.1%} ({total_orig:,} -> {total_comp:,} tokens)"
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

