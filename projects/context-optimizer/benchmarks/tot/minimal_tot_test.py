"""
Minimal ToT Test - writes all output to file for debugging
"""

import sys
import time
from pathlib import Path

# Redirect all output to file
log_file = Path(__file__).parent / "minimal_test_log.txt"
sys.stdout = open(log_file, 'w', buffering=1)
sys.stderr = sys.stdout

print("=" * 80)
print("MINIMAL ToT TEST STARTING")
print("=" * 80)
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Working directory: {Path.cwd()}")
print()

try:
    # Step 1: Import dependencies
    print("[1/6] Importing dependencies...")
    from compressor import compress_corpus_rolling
    from dual_storage_retriever import DualStorageRetriever
    from shared_inputs import estimate_tokens
    print("  ✓ Imports successful")

    # Step 2: Initialize LLM
    print("\n[2/6] Initializing LLM...")
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")
        print("  ✓ Ollama initialized")
    except Exception as e:
        print(f"  ✗ Ollama failed: {e}")
        sys.exit(1)

    # Step 3: Create test corpus
    print("\n[3/6] Creating test corpus...")
    test_corpus = [
        "User authentication is a security process.",
        "The system validates user credentials.",
        "Rate limiting prevents abuse by limiting requests.",
        "Password hashing protects user passwords.",
        "Session tokens maintain user state."
    ] * 50  # 250 lines total
    print(f"  ✓ Created {len(test_corpus)} lines")

    # Step 4: Compress corpus
    print("\n[4/6] Compressing corpus...")
    start = time.time()
    compressed = compress_corpus_rolling(
        lines=test_corpus,
        llm=llm,
        chunk_threshold=512,
        max_summary_tokens=150,
        chunk_overlap_tokens=128
    )
    compress_time = time.time() - start
    print(f"  ✓ Compressed to {len(compressed)} chunks in {compress_time:.1f}s")

    # Step 5: Test retrieval
    print("\n[5/6] Testing retrieval...")
    retriever = DualStorageRetriever(compressed, embedding_backend="hash")

    query = "How does authentication work?"
    keywords = ["authentication", "user", "validate"]

    # Single-path
    start = time.time()
    single_results = retriever.search(query, top_k=3)
    single_latency = (time.time() - start) * 1000
    single_tokens = sum(estimate_tokens(c.compressed_summary) for c in single_results)

    # Count keywords
    single_text = " ".join([c.compressed_summary.lower() for c in single_results])
    single_keywords_found = sum(1 for kw in keywords if kw in single_text)
    single_f1 = single_keywords_found / len(keywords)  # Simplified F1

    print(f"  Single-path: {len(single_results)} chunks, {single_tokens} tokens, {single_latency:.1f}ms")
    print(f"    Keywords found: {single_keywords_found}/{len(keywords)} (F1={single_f1:.3f})")

    # Multi-perspective ToT
    perspectives = [
        f"broad overview: {query}",
        f"specific details: {query}",
        f"related context: {query}"
    ]

    start = time.time()
    all_chunks = []
    for persp in perspectives:
        chunks = retriever.search(persp, top_k=1)
        all_chunks.extend(chunks)

    # Deduplicate
    unique_chunks = {}
    for c in all_chunks:
        if c.chunk_id not in unique_chunks:
            unique_chunks[c.chunk_id] = c

    tot_results = list(unique_chunks.values())[:3]
    tot_latency = (time.time() - start) * 1000
    tot_tokens = sum(estimate_tokens(c.compressed_summary) for c in tot_results)
    dedup_pct = ((len(all_chunks) - len(unique_chunks)) / len(all_chunks) * 100) if all_chunks else 0

    tot_text = " ".join([c.compressed_summary.lower() for c in tot_results])
    tot_keywords_found = sum(1 for kw in keywords if kw in tot_text)
    tot_f1 = tot_keywords_found / len(keywords)

    print(f"  Multi-perspective ToT: {len(tot_results)} chunks, {tot_tokens} tokens, {tot_latency:.1f}ms")
    print(f"    Keywords found: {tot_keywords_found}/{len(keywords)} (F1={tot_f1:.3f})")
    print(f"    Deduplication: {dedup_pct:.1f}%")

    # Step 6: Calculate improvements
    print("\n[6/6] Calculating improvements...")
    f1_improvement = tot_f1 - single_f1
    token_ratio = tot_tokens / single_tokens if single_tokens > 0 else 0
    latency_ratio = tot_latency / single_latency if single_latency > 0 else 0

    print(f"  F1 improvement: {f1_improvement:+.3f} ({f1_improvement*100:+.1f}%)")
    print(f"  Token ratio: {token_ratio:.2f}x")
    print(f"  Latency ratio: {latency_ratio:.2f}x")

    # Save results
    print("\n" + "=" * 80)
    print("WRITING RESULTS")
    print("=" * 80)

    import json
    results = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_lines": len(test_corpus),
        "compressed_chunks": len(compressed),
        "compression_time_sec": compress_time,
        "query": query,
        "keywords": keywords,
        "single_path": {
            "chunks": len(single_results),
            "tokens": single_tokens,
            "latency_ms": single_latency,
            "f1": single_f1,
            "keywords_found": single_keywords_found
        },
        "tot_multi_perspective": {
            "chunks": len(tot_results),
            "tokens": tot_tokens,
            "latency_ms": tot_latency,
            "f1": tot_f1,
            "keywords_found": tot_keywords_found,
            "dedup_pct": dedup_pct
        },
        "improvements": {
            "f1_delta": f1_improvement,
            "f1_pct": f1_improvement * 100,
            "token_ratio": token_ratio,
            "latency_ratio": latency_ratio
        },
        "verdict": "PASS" if f1_improvement > 0 and token_ratio < 3.0 else "FAIL"
    }

    results_file = Path(__file__).parent / "MINIMAL_TOT_RESULTS.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {results_file}")
    print()
    print("=" * 80)
    print("TEST VERDICT:", results["verdict"])
    print("=" * 80)

    if results["verdict"] == "PASS":
        print("✅ ToT shows improvement with acceptable overhead")
    else:
        print("⚠️ ToT did not meet success criteria")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    sys.stdout.close()

print("Test completed")
