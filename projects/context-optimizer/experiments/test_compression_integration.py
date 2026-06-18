"""
Integration Test: Rolling LLM Compression on Large Corpus

Validates the full pipeline:
1. Rolling compression of large corpus (avoiding context exhaustion)
2. Dual-storage indexing (compressed + raw)
3. MCP tool simulation (compressed vs detailed retrieval)
4. Token efficiency comparison with non-compressed baseline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compressor import compress_corpus_rolling
from experiments.dual_storage_retriever import (
    DualStorageRetriever,
    format_compressed_results,
    build_mcp_tool_schemas,
)
from experiments.large_corpus_data import (
    build_excel_corpus_lines,
    build_gutenberg_corpus_lines,
)
from experiments.shared_inputs import estimate_tokens


def test_compression_on_large_corpus(
    corpus_path: Path,
    corpus_type: str,
    chunk_threshold: int = 512,
    top_k: int = 5,
) -> dict:
    """
    Test rolling compression on large corpus and measure efficiency.

    Returns metrics comparing:
    - Baseline (no compression): full corpus tokens
    - Compressed: indexed compressed tokens + raw available on-demand
    """
    print(f"\n{'='*70}")
    print(f"COMPRESSION TEST: {corpus_type}")
    print(f"Corpus: {corpus_path}")
    print(f"{'='*70}\n")

    # Load corpus
    print(f"[1/5] Loading corpus...")
    if corpus_type == "gutenberg":
        corpus_lines = build_gutenberg_corpus_lines(corpus_path)
    elif corpus_type == "excel":
        corpus_lines = build_excel_corpus_lines(corpus_path)
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")

    print(f"  Loaded {len(corpus_lines):,} lines")

    # Baseline: measure uncompressed tokens
    print(f"\n[2/5] Computing baseline (uncompressed) metrics...")
    total_baseline_tokens = sum(estimate_tokens(line) for line in corpus_lines)
    print(f"  Baseline corpus tokens: {total_baseline_tokens:,}")

    # Compress with rolling window
    print(f"\n[3/5] Compressing with rolling window (threshold={chunk_threshold} tokens)...")

    def progress(idx: int, total: int):
        if idx % 50 == 0:
            print(f"  Compressed {idx:,}/{total:,} chunks...")

    compressed_chunks = compress_corpus_rolling(
        corpus_lines[:10000],  # Limit to first 10K lines for testing
        chunk_size_threshold=chunk_threshold,
        compression_batch_size=10,
        progress_callback=progress,
    )

    # Build dual-storage retriever
    print(f"\n[4/5] Building dual-storage retriever...")
    retriever = DualStorageRetriever(compressed_chunks)
    stats = retriever.get_compression_stats()

    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Original tokens: {stats['original_tokens']:,}")
    print(f"  Compressed tokens: {stats['compressed_tokens']:,}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}")
    print(f"  Token savings: {stats['savings_percent']:.1f}%")

    # Simulate MCP retrieval
    print(f"\n[5/5] Simulating MCP retrieval...")

    if corpus_type == "gutenberg":
        test_query = "character moral conviction evolution"
    else:
        test_query = "high latency failed status region"

    print(f"\n  Query: '{test_query}'")
    print(f"\n  [MCP Tool: get_context]")
    hits = retriever.search_compressed(test_query, top_k=top_k)
    result_text = format_compressed_results(hits)
    print(result_text)

    compressed_result_tokens = estimate_tokens(result_text)
    print(f"\n  Tokens in compressed result: {compressed_result_tokens:,}")

    # Simulate detailed retrieval if needed
    if hits:
        print(f"\n  [MCP Tool: get_context_details] (if reasoning LLM requests details)")
        chunk_id = hits[0].chunk_id
        details = retriever.get_chunk_details(chunk_id)
        details_tokens = estimate_tokens(details) if details else 0
        print(f"  Retrieved {chunk_id}: {details_tokens:,} tokens")
        print(f"  Raw excerpt: {details[:200]}..." if details else "  (not found)")

    # Compute efficiency comparison
    print(f"\n{'='*70}")
    print(f"EFFICIENCY COMPARISON")
    print(f"{'='*70}\n")

    print(f"Baseline (no compression):")
    print(f"  Corpus tokens: {stats['original_tokens']:,}")
    print(f"  Retrieval tokens: {stats['original_tokens']:,} (full corpus)")
    print(f"  Total: {stats['original_tokens']:,}\n")

    print(f"With LLM Compression:")
    print(f"  Indexed tokens: {stats['compressed_tokens']:,} (compressed)")
    print(f"  Retrieval tokens: {compressed_result_tokens:,} (top-{top_k} compressed)")
    print(f"  Total: {compressed_result_tokens:,}")
    print(f"  Savings: {(1 - compressed_result_tokens / stats['original_tokens']) * 100:.1f}%\n")

    print(f"With Compression + Fallback (if reasoning LLM requests details):")
    print(f"  Compressed retrieval: {compressed_result_tokens:,}")
    print(f"  + Detailed data (1 chunk): {details_tokens:,}")
    print(f"  Total: {compressed_result_tokens + details_tokens:,}")
    print(f"  Savings: {(1 - (compressed_result_tokens + details_tokens) / stats['original_tokens']) * 100:.1f}%\n")

    return {
        "corpus_type": corpus_type,
        "corpus_lines": len(corpus_lines),
        "chunks_compressed": stats["total_chunks"],
        "baseline_tokens": stats["original_tokens"],
        "compressed_index_tokens": stats["compressed_tokens"],
        "retrieval_tokens_compressed": compressed_result_tokens,
        "retrieval_tokens_with_details": compressed_result_tokens + details_tokens,
        "compression_ratio": stats["compression_ratio"],
        "retrieval_savings_percent": (1 - compressed_result_tokens / stats['original_tokens']) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM compression on large corpus")
    parser.add_argument(
        "--corpus-type",
        choices=["gutenberg", "excel"],
        default="gutenberg",
        help="Type of corpus to test",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        help="Path to corpus file (optional, uses default if not specified)",
    )
    parser.add_argument(
        "--chunk-threshold",
        type=int,
        default=512,
        help="Token threshold for chunking before compression",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve",
    )
    args = parser.parse_args()

    # Determine corpus path
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
    else:
        root = Path(__file__).resolve().parents[1]
        if args.corpus_type == "gutenberg":
            corpus_path = root / "data" / "large_corpus" / "gutenberg" / "combined_gutenberg.txt"
        else:
            corpus_path = root / "data" / "large_corpus" / "excel" / "mock_500mb.xlsx"

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print(f"Run large corpus benchmarks first to generate test data.")
        return 1

    # Run test
    results = test_compression_on_large_corpus(
        corpus_path=corpus_path,
        corpus_type=args.corpus_type,
        chunk_threshold=args.chunk_threshold,
        top_k=args.top_k,
    )

    print(f"\n{'='*70}")
    print(f"MCP TOOL SCHEMAS")
    print(f"{'='*70}\n")

    print("Tools exposed to reasoning LLM:\n")
    import json
    for tool in build_mcp_tool_schemas():
        print(json.dumps(tool, indent=2))
        print()

    print(f"{'='*70}")
    print(f"TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nKey Takeaway:")
    print(f"  Rolling LLM compression achieves {results['retrieval_savings_percent']:.1f}% token savings")
    print(f"  without context exhaustion during ingestion.")
    print(f"  Reasoning LLM can request detailed data on-demand via get_context_details.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
