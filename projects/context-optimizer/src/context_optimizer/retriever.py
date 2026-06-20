"""
Dual-Storage Retriever with Compressed + Raw Data

Stores both compressed semantic summaries (for fast retrieval) and raw data
(for detailed inspection). The reasoning LLM can request detailed data when
compressed summaries are insufficient.

MCP Tool Contract:
- get_context: Returns compressed summaries (default, fast)
- get_context_details: Returns raw data for specific chunks (on-demand)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .compressor import CompressedChunk, compress_corpus_rolling


@dataclass
class CompressedRetrievalHit:
    """Search result with both compressed and raw data available."""
    chunk_id: str
    compressed_summary: str  # Serves to reasoning LLM by default
    entities: list[str]
    keywords: list[str]
    metadata: dict[str, str | int]
    relevance_score: float
    # Raw data available but not returned unless explicitly requested
    _raw_text: str  # Private - use get_details() to access


class DualStorageRetriever:
    """
    Retriever with dual storage: compressed for search, raw for fallback.

    Architecture:
    1. Index on compressed summaries + entities/keywords
    2. Search returns compressed by default (fast, low tokens)
    3. Reasoning LLM can request raw data for specific chunks if needed
    """

    def __init__(
        self,
        compressed_chunks: list[CompressedChunk],
        embedding_backend: str = "hash",
    ):
        self._chunks = compressed_chunks
        self._chunk_map = {chunk.chunk_id: chunk for chunk in compressed_chunks}
        self._embedding_backend = embedding_backend

        # Build search index on compressed summaries
        self._compressed_texts = [chunk.compressed_summary for chunk in compressed_chunks]
        self._entity_index = self._build_entity_index()
        self._keyword_index = self._build_keyword_index()

        print(f"[DualRetriever] Initialized with {len(compressed_chunks):,} chunks")
        print(f"[DualRetriever] Indexed {len(self._entity_index)} entities, {len(self._keyword_index)} keywords")

    def _build_entity_index(self) -> dict[str, list[str]]:
        """Build inverted index: entity -> chunk_ids"""
        index: dict[str, list[str]] = {}
        for chunk in self._chunks:
            for entity in chunk.entities:
                entity_lower = entity.lower()
                if entity_lower not in index:
                    index[entity_lower] = []
                index[entity_lower].append(chunk.chunk_id)
        return index

    def _build_keyword_index(self) -> dict[str, list[str]]:
        """Build inverted index: keyword -> chunk_ids"""
        index: dict[str, list[str]] = {}
        for chunk in self._chunks:
            for keyword in chunk.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in index:
                    index[keyword_lower] = []
                index[keyword_lower].append(chunk.chunk_id)
        return index

    def search_compressed(
        self,
        query: str,
        top_k: int = 5,
        entity_filter: list[str] | None = None,
    ) -> list[CompressedRetrievalHit]:
        """
        Search using compressed summaries. Returns compressed data only.

        This is the default retrieval mode - fast, low token count.
        Reasoning LLM receives compressed summaries and can request
        detailed data for specific chunks if needed.

        Args:
            query: Search query
            top_k: Number of results to return
            entity_filter: Optional list of entities to filter by

        Returns:
            List of hits with compressed summaries (raw data available via get_details)
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Score chunks by relevance
        scored_chunks: list[tuple[float, CompressedChunk]] = []

        for chunk in self._chunks:
            score = 0.0

            # Entity matching (high weight)
            for entity in chunk.entities:
                if entity.lower() in query_lower:
                    score += 2.0

            # Keyword matching (medium weight)
            for keyword in chunk.keywords:
                if keyword.lower() in query_lower:
                    score += 1.5

            # Summary text matching (lower weight)
            summary_tokens = set(chunk.compressed_summary.lower().split())
            overlap = len(query_tokens & summary_tokens)
            score += overlap * 0.5

            # Apply entity filter if specified
            if entity_filter:
                if not any(e.lower() in [ent.lower() for ent in chunk.entities] for e in entity_filter):
                    score *= 0.1  # Penalize non-matching entities

            scored_chunks.append((score, chunk))

        # Sort by score and take top-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:top_k]

        # Build retrieval hits (compressed only, raw hidden)
        hits: list[CompressedRetrievalHit] = []
        for score, chunk in top_chunks:
            hits.append(
                CompressedRetrievalHit(
                    chunk_id=chunk.chunk_id,
                    compressed_summary=chunk.compressed_summary,
                    entities=chunk.entities,
                    keywords=chunk.keywords,
                    metadata=chunk.metadata,
                    relevance_score=score,
                    _raw_text=chunk.raw_text,  # Available but not exposed by default
                )
            )

        return hits

    def get_chunk_details(self, chunk_id: str) -> str | None:
        """
        Retrieve raw data for a specific chunk.

        This is called by the reasoning LLM when compressed summary is
        insufficient. Returns full original text.

        Args:
            chunk_id: ID of chunk to retrieve

        Returns:
            Raw text of the chunk, or None if not found
        """
        chunk = self._chunk_map.get(chunk_id)
        if chunk is None:
            return None
        return chunk.raw_text

    def get_compression_stats(self) -> dict[str, int | float]:
        """Get statistics about compression efficiency."""
        total_original = sum(c.original_tokens for c in self._chunks)
        total_compressed = sum(c.compressed_tokens for c in self._chunks)
        avg_ratio = total_compressed / total_original if total_original > 0 else 1.0

        return {
            "total_chunks": len(self._chunks),
            "original_tokens": total_original,
            "compressed_tokens": total_compressed,
            "compression_ratio": avg_ratio,
            "savings_percent": (1 - avg_ratio) * 100,
        }


def build_mcp_tool_schemas() -> list[dict]:
    """
    Build MCP tool schemas for compressed + detailed retrieval.

    These are the tools exposed to the reasoning LLM:
    - get_context: Fast retrieval using compressed summaries
    - get_context_details: Detailed retrieval for specific chunks
    """
    return [
        {
            "name": "get_context",
            "description": (
                "Retrieve compressed semantic summaries matching your query. "
                "Returns concise summaries optimized for quick understanding. "
                "If you need more detail about specific chunks, use get_context_details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (entities, keywords, or natural language)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                    "entity_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: filter results to chunks containing these entities",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_context_details",
            "description": (
                "Retrieve full raw data for specific chunks when compressed summaries "
                "are insufficient. Use this when you need complete information to answer "
                "the user's question accurately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of chunk IDs to retrieve detailed data for",
                    },
                },
                "required": ["chunk_ids"],
            },
        },
    ]


def format_compressed_results(hits: list[CompressedRetrievalHit]) -> str:
    """
    Format compressed retrieval results for the reasoning LLM.

    Shows: compressed summaries, entities, keywords, relevance scores.
    Hides: raw data (available via get_context_details if needed).
    """
    if not hits:
        return "No relevant context found."

    lines: list[str] = [
        f"Found {len(hits)} relevant chunks (compressed summaries):",
        "",
    ]

    for idx, hit in enumerate(hits, 1):
        lines.append(f"**[{idx}] {hit.chunk_id}** (relevance: {hit.relevance_score:.2f})")
        lines.append(f"Summary: {hit.compressed_summary}")
        if hit.entities:
            lines.append(f"Entities: {', '.join(hit.entities)}")
        if hit.keywords:
            lines.append(f"Keywords: {', '.join(hit.keywords)}")
        if hit.metadata:
            meta_str = ', '.join(f"{k}={v}" for k, v in hit.metadata.items())
            lines.append(f"Metadata: {meta_str}")
        lines.append("")

    lines.append("💡 Tip: If you need more detail, use get_context_details([chunk_id, ...]) to retrieve raw data.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick integration test
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Create test corpus
    test_lines = [
        "CosmosDB timeout error 21012 in order-service at 2024-01-15T02:13:45Z",
        "Primary replica connection failed, retrying with secondary",
        "Payment service cascade failure detected downstream",
        "Circuit breaker opened for cosmosdb-primary endpoint",
        "Error persisted across 3 retry attempts",
    ]

    print("=== Compressing corpus with rolling window ===")
    compressed_chunks = compress_corpus_rolling(
        test_lines,
        chunk_size_threshold=40,
        chunk_overlap_tokens=10,  # 25% of 40
        compression_batch_size=1,
    )

    print("\n=== Building dual-storage retriever ===")
    retriever = DualStorageRetriever(compressed_chunks)

    print("\n=== Searching with compressed summaries ===")
    hits = retriever.search_compressed("CosmosDB timeout error", top_k=2)
    print(format_compressed_results(hits))

    if hits:
        print(f"\n=== Requesting detailed data for {hits[0].chunk_id} ===")
        details = retriever.get_chunk_details(hits[0].chunk_id)
        print(f"Raw data:\n{details}")

    print("\n=== Compression Statistics ===")
    stats = retriever.get_compression_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
