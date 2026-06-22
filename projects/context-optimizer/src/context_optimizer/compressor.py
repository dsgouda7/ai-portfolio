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

import os
from dataclasses import dataclass
from typing import Protocol

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
    compressed_summary: str  # LLM-compressed semantic index
    entities: list[str]  # Extracted entities for filtering
    keywords: list[str]  # Key concepts for search
    metadata: dict[str, str | int]  # Source, timestamp, etc
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed / original


class CompressorLLM(Protocol):
    """Protocol for LLM backends that support compression."""

    def invoke(self, prompt: str) -> object:
        """Invoke the LLM with a compression prompt."""
        ...


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)."""
    return max(1, len(text) // 4)


def _build_local_llm(provider: str = "ollama", model: str | None = None) -> CompressorLLM | None:
    """
    Build a local LLM for compression.

    Prefers lightweight models optimized for summarization:
    - Ollama: phi4:mini, qwen2.5-coder:7b, llama3.2:3b
    - Groq: llama-3.3-70b-versatile (fast inference)
    """
    selected_provider = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", provider).lower()

    if selected_provider == "ollama" and ChatOllama is not None:
        model_name = model or os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "qwen2.5-coder:7b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.1)

    if selected_provider == "groq" and ChatGroq is not None:
        model_name = model or os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "llama-3.3-70b-versatile")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required for Groq compression")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.1)

    return None


COMPRESSION_PROMPT_TEMPLATE = """You are a semantic compression agent. Your job is to distill this text into a dense summary that preserves key information for retrieval.

**Guidelines:**
- PRESERVE: technical terms, numbers, code snippets, formulas, error codes, metrics
- Extract main entities (people, systems, concepts, algorithms, functions)
- Keep key facts, relationships, causality, and actionable information
- Maintain structural context (section headers, list items, code blocks)
- Remove: filler phrases, background narrative, redundant phrasing
- Target: ~150 tokens (3-4 sentences with technical detail)

**Input Text:**
{text}

**Output Format (JSON):**
{{
  "summary": "Dense 3-4 sentence summary preserving technical terms, numbers, and key facts",
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
    prompt = COMPRESSION_PROMPT_TEMPLATE.format(text=text[:2000])  # Limit input to ~500 tokens

    try:
        response = llm.invoke(prompt)
        result_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON response
        import json
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

        original_tokens = _estimate_tokens(text)
        compressed_tokens = _estimate_tokens(summary)

        return CompressedChunk(
            chunk_id=chunk_id,
            raw_text=text,
            compressed_summary=summary,
            entities=entities,
            keywords=keywords,
            metadata=metadata or {},
            original_tokens=original_tokens,
            compressed_tokens=min(compressed_tokens, max_summary_tokens),
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
        )

    except Exception as e:
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
    chunk_overlap_tokens: int = 128,
    compression_batch_size: int = 10,
    llm: CompressorLLM | None = None,
    progress_callback: callable | None = None,
) -> list[CompressedChunk]:
    """
    Compress a large corpus using a rolling window strategy with overlap.

    Process:
    1. Accumulate lines until threshold reached
    2. Compress the accumulated chunk
    3. Keep last 25% of chunk as overlap for next chunk (preserves boundaries)
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

    print(f"[Compressor] Starting rolling compression of {len(corpus_lines):,} lines...")
    print(f"[Compressor] Threshold: {chunk_size_threshold} tokens, Overlap: {chunk_overlap_tokens} tokens, Batch: {compression_batch_size}")

    for line_idx, line in enumerate(corpus_lines):
        line_tokens = _estimate_tokens(line)
        current_chunk_lines.append(line)
        current_chunk_tokens += line_tokens

        # Threshold reached: compress this chunk
        if current_chunk_tokens >= chunk_size_threshold:
            chunk_text = "\n".join(current_chunk_lines)
            chunk_id = f"chunk_{chunk_idx:06d}"

            # Compress with rolling window (only this chunk, not full corpus)
            compressed = compress_chunk_with_llm(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    "line_start": line_idx - len(current_chunk_lines) + 1,
                    "line_end": line_idx,
                    "source_lines": len(current_chunk_lines),
                },
                llm=llm,
            )
            compressed_chunks.append(compressed)

            # Progress reporting
            if progress_callback and chunk_idx % compression_batch_size == 0:
                progress_callback(chunk_idx, len(corpus_lines) // chunk_size_threshold)

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

    # Handle remaining lines
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunk_id = f"chunk_{chunk_idx:06d}"
        compressed = compress_chunk_with_llm(
            text=chunk_text,
            chunk_id=chunk_id,
            metadata={
                "line_start": len(corpus_lines) - len(current_chunk_lines),
                "line_end": len(corpus_lines) - 1,
                "source_lines": len(current_chunk_lines),
            },
            llm=llm,
        )
        compressed_chunks.append(compressed)

    total_original = sum(c.original_tokens for c in compressed_chunks)
    total_compressed = sum(c.compressed_tokens for c in compressed_chunks)
    avg_ratio = total_compressed / total_original if total_original > 0 else 1.0

    print(f"[Compressor] [OK] Compressed {len(compressed_chunks):,} chunks")
    print(f"[Compressor] Compression ratio: {avg_ratio:.2%} ({total_original:,} => {total_compressed:,} tokens)")

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
        print(f"  Compressed ({chunk.compressed_tokens} tokens): {chunk.compressed_summary}")
        print(f"  Entities: {chunk.entities}")
        print(f"  Keywords: {chunk.keywords}")
