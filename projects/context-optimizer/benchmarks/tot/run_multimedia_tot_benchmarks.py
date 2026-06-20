"""
Multimedia ToT Benchmarks

Tests Tree-of-Thought retrieval on multimedia datasets (images, audio, video).
Saves results to temp markdown files for consolidation.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
from datetime import datetime

# Import multimedia data
from download_multimedia_data import get_multimedia_corpus, download_all_multimedia

# Import existing infrastructure
from compressor import compress_corpus_rolling, CompressedChunk
from dual_storage_retriever import DualStorageRetriever
from shared_inputs import estimate_tokens


TEMP_DIR = Path(__file__).parent / "temp_results"
TEMP_DIR.mkdir(exist_ok=True)


@dataclass
class MultimediaToTResult:
    """Result for multimedia ToT testing."""
    corpus_size: str
    media_type: str
    query: str

    # Single-path
    single_f1: float
    single_tokens: int
    single_latency_ms: float

    # Multi-perspective ToT
    tot_f1: float
    tot_tokens: int
    tot_latency_ms: float
    tot_perspectives: int
    tot_dedup_savings_pct: float

    # Improvements
    f1_improvement: float
    token_ratio: float
    latency_ratio: float

    # Metadata
    compressed_chunks: int
    original_lines: int


# Test queries by media type
MULTIMEDIA_QUERIES = {
    "images": [
        ("Find images of people in outdoor settings", ["people", "outdoor", "park", "street", "field"]),
        ("Show pictures with animals", ["dog", "cat", "bird", "animal"]),
        ("Images of urban environments", ["city", "street", "building", "metropolitan"]),
    ],
    "audio": [
        ("Explain neural networks", ["neural", "network", "deep", "learning"]),
        ("Discuss machine learning basics", ["machine", "learning", "supervised", "model"]),
        ("Talks about AI ethics", ["ethical", "AI", "artificial", "intelligence"]),
    ],
    "video": [
        ("Tutorial on deep learning", ["tutorial", "deep", "learning", "neural"]),
        ("Lectures about NLP", ["lecture", "NLP", "natural", "language", "processing"]),
        ("Demos of computer vision", ["demo", "computer", "vision", "image"]),
    ]
}


def calculate_simple_f1(chunks: List[CompressedChunk], keywords: List[str]) -> float:
    """Simple keyword-based F1."""
    if not chunks or not keywords:
        return 0.0

    text = " ".join([c.compressed_summary.lower() for c in chunks])

    found = sum(1 for kw in keywords if kw in text)
    recall = found / len(keywords)

    relevant_chunks = sum(1 for c in chunks if any(kw in c.compressed_summary.lower() for kw in keywords))
    precision = relevant_chunks / len(chunks) if chunks else 0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def single_path_retrieval(retriever, query: str, top_k: int = 6) -> tuple:
    """Baseline single-path retrieval."""
    start = time.time()
    results = retriever.search(query, top_k=top_k)
    latency = (time.time() - start) * 1000
    return results, latency


def multi_perspective_retrieval(retriever, query: str, num_perspectives: int = 3, top_k: int = 6) -> tuple:
    """ToT-style multi-perspective retrieval."""
    start = time.time()

    # Generate perspectives
    perspectives = [
        f"broad overview: {query}",
        f"specific details: {query}",
        f"related context: {query}"
    ][:num_perspectives]

    # Retrieve from each perspective
    all_chunks = []
    for persp in perspectives:
        chunks = retriever.search(persp, top_k=max(2, top_k // num_perspectives))
        all_chunks.extend(chunks)

    initial_count = len(all_chunks)

    # Deduplicate by chunk_id
    chunk_map = {}
    for chunk in all_chunks:
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk
        else:
            # Keep higher relevance score
            if chunk.relevance_score > chunk_map[chunk.chunk_id].relevance_score:
                chunk_map[chunk.chunk_id] = chunk

    unique_chunks = list(chunk_map.values())
    dedup_savings = ((initial_count - len(unique_chunks)) / initial_count * 100) if initial_count > 0 else 0

    # Re-rank and return top-k
    ranked = sorted(unique_chunks, key=lambda c: c.relevance_score, reverse=True)[:top_k]

    latency = (time.time() - start) * 1000
    return ranked, latency, len(unique_chunks), dedup_savings


def save_temp_markdown(results: List[MultimediaToTResult], media_type: str):
    """Save results to temp markdown file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multimedia_tot_{media_type}_{timestamp}.md"
    filepath = TEMP_DIR / filename

    with open(filepath, 'w') as f:
        f.write(f"# Multimedia ToT Results: {media_type.upper()}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Tests**: {len(results)}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Corpus Size | Query | F1 Δ | Token Ratio | Latency Ratio | Dedup % |\n")
        f.write("|-------------|-------|------|-------------|---------------|----------|\n")

        for r in results:
            f.write(f"| {r.corpus_size} | {r.query[:40]}... | {r.f1_improvement:+.3f} | {r.token_ratio:.2f}x | {r.latency_ratio:.2f}x | {r.tot_dedup_savings_pct:.1f}% |\n")

        # Averages by corpus size
        f.write("\n## Averages by Corpus Size\n\n")

        for size in ["100mb", "500mb", "1gb"]:
            size_results = [r for r in results if r.corpus_size == size]
            if not size_results:
                continue

            avg_f1 = sum(r.f1_improvement for r in size_results) / len(size_results)
            avg_tok = sum(r.token_ratio for r in size_results) / len(size_results)
            avg_lat = sum(r.latency_ratio for r in size_results) / len(size_results)
            avg_dedup = sum(r.tot_dedup_savings_pct for r in size_results) / len(size_results)

            f.write(f"### {size.upper()}\n\n")
            f.write(f"- **F1 Improvement**: {avg_f1:+.3f} ({avg_f1*100:+.1f}%)\n")
            f.write(f"- **Token Ratio**: {avg_tok:.2f}x\n")
            f.write(f"- **Latency Ratio**: {avg_lat:.2f}x\n")
            f.write(f"- **Deduplication**: {avg_dedup:.1f}%\n\n")

        # Detailed results
        f.write("\n## Detailed Results\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"### Test {i}: {r.corpus_size.upper()} - {r.query}\n\n")
            f.write(f"**Corpus**: {r.original_lines:,} lines → {r.compressed_chunks} chunks\n\n")

            f.write("**Single-Path**:\n")
            f.write(f"- F1: {r.single_f1:.3f}\n")
            f.write(f"- Tokens: {r.single_tokens:,}\n")
            f.write(f"- Latency: {r.single_latency_ms:.1f}ms\n\n")

            f.write("**Multi-Perspective ToT**:\n")
            f.write(f"- F1: {r.tot_f1:.3f}\n")
            f.write(f"- Tokens: {r.tot_tokens:,}\n")
            f.write(f"- Latency: {r.tot_latency_ms:.1f}ms\n")
            f.write(f"- Perspectives: {r.tot_perspectives}\n")
            f.write(f"- Deduplication: {r.tot_dedup_savings_pct:.1f}%\n\n")

            f.write("**Improvements**:\n")
            f.write(f"- Δ F1: {r.f1_improvement:+.3f} ({r.f1_improvement*100:+.1f}%)\n")
            f.write(f"- Token Ratio: {r.token_ratio:.2f}x\n")
            f.write(f"- Latency Ratio: {r.latency_ratio:.2f}x\n\n")

            verdict = "✅ PASS" if r.f1_improvement > 0.03 and r.token_ratio < 3.0 else "⚠️ REVIEW"
            f.write(f"**Verdict**: {verdict}\n\n")
            f.write("---\n\n")

    print(f"  ✓ Saved temp markdown: {filepath}")
    return filepath


def run_multimedia_tot_benchmarks():
    """Run multimedia ToT benchmarks."""

    print("=" * 100)
    print("MULTIMEDIA ToT BENCHMARKS")
    print("=" * 100)

    # Step 1: Download datasets
    print("\nStep 1: Preparing multimedia datasets...")
    datasets = download_all_multimedia()

    # Step 2: Initialize LLM
    print("\nStep 2: Initializing LLM...")
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")
        print("  ✓ Ollama initialized")
    except Exception as e:
        print(f"  ✗ Ollama failed: {e}")
        return

    # Step 3: Run benchmarks
    print("\nStep 3: Running benchmarks...")
    print("=" * 100)

    # Test configurations (reduced for speed)
    test_configs = [
        ("100mb", "images"),
        ("500mb", "audio"),
        ("1gb", "video"),
    ]

    all_results_by_media = {
        "images": [],
        "audio": [],
        "video": []
    }

    for size, media_type in test_configs:
        print(f"\n{'─' * 100}")
        print(f"Testing: {size.upper()} corpus, {media_type.upper()} domain")
        print('─' * 100)

        # Get corpus
        corpus = datasets[media_type][size]
        print(f"  Corpus: {len(corpus):,} lines")

        # Compress
        print(f"  Compressing...")
        try:
            start_time = time.time()
            compressed_chunks = compress_corpus_rolling(
                lines=corpus,
                llm=llm,
                chunk_threshold=512,
                max_summary_tokens=150,
                chunk_overlap_tokens=128
            )
            compress_time = time.time() - start_time
            print(f"  ✓ Compressed: {len(compressed_chunks)} chunks in {compress_time:.1f}s")
        except Exception as e:
            print(f"  ✗ Compression failed: {e}")
            continue

        # Initialize retriever
        try:
            retriever = DualStorageRetriever(compressed_chunks, embedding_backend="hash")
            print(f"  ✓ Retriever initialized")
        except Exception as e:
            print(f"  ✗ Retriever init failed: {e}")
            continue

        # Test queries
        queries = MULTIMEDIA_QUERIES.get(media_type, MULTIMEDIA_QUERIES["images"])

        for query, keywords in queries:
            print(f"\n  Query: {query}")

            try:
                # Single-path
                single_chunks, single_latency = single_path_retrieval(retriever, query)
                single_tokens = sum(estimate_tokens(c.compressed_summary) for c in single_chunks)
                single_f1 = calculate_simple_f1(single_chunks, keywords)

                # Multi-perspective ToT
                tot_chunks, tot_latency, tot_unique, dedup_savings = multi_perspective_retrieval(retriever, query)
                tot_tokens = sum(estimate_tokens(c.compressed_summary) for c in tot_chunks)
                tot_f1 = calculate_simple_f1(tot_chunks, keywords)

                # Calculate improvements
                f1_improvement = tot_f1 - single_f1
                token_ratio = tot_tokens / single_tokens if single_tokens > 0 else 0
                latency_ratio = tot_latency / single_latency if single_latency > 0 else 0

                print(f"    Single: F1={single_f1:.3f} | {single_tokens:,} tokens | {single_latency:.1f}ms")
                print(f"    ToT:    F1={tot_f1:.3f} | {tot_tokens:,} tokens | {tot_latency:.1f}ms")
                print(f"    Δ:      F1 {f1_improvement:+.3f} | Token {token_ratio:.2f}x | Dedup {dedup_savings:.1f}%")

                # Store result
                result = MultimediaToTResult(
                    corpus_size=size,
                    media_type=media_type,
                    query=query,
                    single_f1=single_f1,
                    single_tokens=single_tokens,
                    single_latency_ms=single_latency,
                    tot_f1=tot_f1,
                    tot_tokens=tot_tokens,
                    tot_latency_ms=tot_latency,
                    tot_perspectives=3,
                    tot_dedup_savings_pct=dedup_savings,
                    f1_improvement=f1_improvement,
                    token_ratio=token_ratio,
                    latency_ratio=latency_ratio,
                    compressed_chunks=len(compressed_chunks),
                    original_lines=len(corpus)
                )
                all_results_by_media[media_type].append(result)

            except Exception as e:
                print(f"    ✗ Query failed: {e}")
                continue

    # Step 4: Save results
    print("\n" + "=" * 100)
    print("SAVING RESULTS TO TEMP MARKDOWN FILES")
    print("=" * 100)

    saved_files = []
    for media_type, results in all_results_by_media.items():
        if results:
            filepath = save_temp_markdown(results, media_type)
            saved_files.append(filepath)

    # Also save JSON for programmatic access
    json_path = TEMP_DIR / f"multimedia_tot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "results_by_media": {
                media_type: [asdict(r) for r in results]
                for media_type, results in all_results_by_media.items()
            }
        }, f, indent=2)

    print(f"  ✓ Saved JSON: {json_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
    print(f"\nTemp results directory: {TEMP_DIR}")
    print(f"\nGenerated files:")
    for filepath in saved_files:
        print(f"  - {filepath.name}")
    print(f"  - {json_path.name}")

    total_tests = sum(len(results) for results in all_results_by_media.values())
    print(f"\nTotal tests: {total_tests}")

    return all_results_by_media


if __name__ == "__main__":
    results = run_multimedia_tot_benchmarks()
