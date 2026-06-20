"""Quick test to validate compression pipeline"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("=" * 70)
print("QUICK COMPRESSION VALIDATION")
print("=" * 70)

# Test 1: Import modules
print("\n[1/3] Testing imports...")
try:
    from experiments.compressor import compress_chunk_with_llm, CompressedChunk
    from experiments.shared_inputs import estimate_tokens
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Compress a single chunk
print("\n[2/3] Testing single chunk compression...")
test_text = "CosmosDB timeout error 21012 in order-service. Primary replica failed after 3 retries."
try:
    chunk = compress_chunk_with_llm(
        text=test_text,
        chunk_id="test_001",
        metadata={"source": "test"},
        llm=None,  # Will use fallback
    )
    print(f"✓ Compression successful")
    print(f"  Original: {chunk.original_tokens} tokens")
    print(f"  Compressed: {chunk.compressed_tokens} tokens")
    print(f"  Ratio: {chunk.compression_ratio:.1%}")
    print(f"  Summary: {chunk.compressed_summary}")
    print(f"  Entities: {chunk.entities}")
    print(f"  Keywords: {chunk.keywords}")
except Exception as e:
    print(f"✗ Compression failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Rolling compression on multiple lines
print("\n[3/3] Testing rolling compression...")
test_lines = [
    "Error 21012: CosmosDB connection timeout in us-west-2 region.",
    "Payment service cascade failure detected at 02:13:45Z.",
    "Recovery initiated, failover to secondary completed successfully.",
]
try:
    from experiments.compressor import compress_corpus_rolling

    chunks = compress_corpus_rolling(
        test_lines,
        chunk_size_threshold=50,  # Low threshold for quick test
        compression_batch_size=1,
        llm=None,  # Fallback mode
    )
    print(f"✓ Rolling compression successful")
    print(f"  Input lines: {len(test_lines)}")
    print(f"  Output chunks: {len(chunks)}")

    total_original = sum(c.original_tokens for c in chunks)
    total_compressed = sum(c.compressed_tokens for c in chunks)
    print(f"  Total original tokens: {total_original}")
    print(f"  Total compressed tokens: {total_compressed}")
    print(f"  Overall ratio: {(total_compressed / total_original):.1%}")

except Exception as e:
    print(f"✗ Rolling compression failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nCompression pipeline is ready for large-scale experiments.")
