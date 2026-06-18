"""
E2E Compression Experiment Runner

Runs compression benchmarks and displays results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("=" * 80)
print("E2E COMPRESSION EXPERIMENTS")
print("=" * 80)
print()

# Test 1: Gutenberg Compression
print("[1/2] Running Gutenberg compression benchmark...")
print("-" * 80)

try:
    from experiments.run_compression_benchmark import run_compression_benchmark
    from experiments.large_corpus_data import build_gutenberg_corpus_lines

    gutenberg_path = Path(__file__).resolve().parents[1] / "data" / "large_corpus" / "gutenberg" / "combined_gutenberg.txt"

    if gutenberg_path.exists():
        result = run_compression_benchmark("gutenberg", gutenberg_path, sample_size=5000)

        print(f"\n✅ Gutenberg Results:")
        print(f"   Corpus: {result.corpus_size_mb:.1f} MB, {result.corpus_lines:,} lines")
        print(f"   Baseline index: {result.baseline_index_tokens:,} tokens")
        print(f"   Compressed index: {result.compressed_index_tokens:,} tokens")
        print(f"   Compression ratio: {result.compression_ratio:.1%}")
        print(f"   Index savings: {(1 - result.compression_ratio) * 100:.1f}%")
        print()
        print(f"   Query baseline: {result.query_baseline_tokens:,} tokens")
        print(f"   Query compressed: {result.query_compressed_tokens:,} tokens")
        print(f"   Query + 1 detail: {result.query_compressed_with_details_tokens:,} tokens")
        print(f"   🎯 Retrieval savings: {result.retrieval_savings_percent:.1f}%")

    else:
        print(f"   ⚠️  Corpus not found at {gutenberg_path}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Excel Compression
print("[2/2] Running Excel compression benchmark...")
print("-" * 80)

try:
    excel_path = Path(__file__).resolve().parents[1] / "data" / "large_corpus" / "excel" / "mock_500mb.xlsx"

    if excel_path.exists():
        result = run_compression_benchmark("excel", excel_path, sample_size=5000)

        print(f"\n✅ Excel Results:")
        print(f"   Corpus: {result.corpus_size_mb:.1f} MB, {result.corpus_lines:,} lines")
        print(f"   Baseline index: {result.baseline_index_tokens:,} tokens")
        print(f"   Compressed index: {result.compressed_index_tokens:,} tokens")
        print(f"   Compression ratio: {result.compression_ratio:.1%}")
        print(f"   Index savings: {(1 - result.compression_ratio) * 100:.1f}%")
        print()
        print(f"   Query baseline: {result.query_baseline_tokens:,} tokens")
        print(f"   Query compressed: {result.query_compressed_tokens:,} tokens")
        print(f"   Query + 1 detail: {result.query_compressed_with_details_tokens:,} tokens")
        print(f"   🎯 Retrieval savings: {result.retrieval_savings_percent:.1f}%")

    else:
        print(f"   ⚠️  Corpus not found at {excel_path}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("COMPRESSION EXPERIMENTS COMPLETE")
print("=" * 80)
print()
print("📊 Check docs/experiments/EXPERIMENTS_CONSOLIDATED.md for full results")
