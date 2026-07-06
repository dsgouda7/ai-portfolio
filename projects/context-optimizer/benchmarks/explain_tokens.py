"""
Live demo: what's actually stored in ChromaDB for vanilla RAG vs optimized RAG,
and why the token count was wrong.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── 1. Show stored data from the completed benchmark ────────────────────────
print("=" * 65)
print("PART 1: What's in the completed benchmark JSON")
print("=" * 65)

results_path = Path(__file__).parent / "corpus_results.json"
if results_path.exists():
    d = json.loads(results_path.read_text())
    o = d.get("optimized_rag") or {}
    qrs = o.get("query_results", [])
    print(f"\nRun date   : {d['run_date']}")
    print(f"Strategy   : extractive (pre-fix — summary NOT capped to 500 tokens)")
    print(f"Questions  : {len(qrs)}")
    print(
        f"Index size : {o.get('index_size_chunks',0)} blocks  /  {o.get('index_size_mb',0):.0f} MB"
    )
    if qrs:
        print(f"\nFirst 3 queries (what was retrieved and token cost):")
        for q in qrs[:3]:
            snip = q.get("answer_snippet", "")
            print(f"\n  Query  : {q['query'][:55]}")
            print(f"  Snippet: {snip[:100]}")
            print(
                f"  Tokens : {q['tokens_used']:,}  ← each block summary was ~64K tokens"
            )
            print(f"  Recall : {q['kw_recall']:.0%}")
else:
    print("(corpus_results.json not found)")

# ─── 2. Build a tiny live demo showing the difference ────────────────────────
print("\n\n" + "=" * 65)
print("PART 2: Live ChromaDB entries — vanilla vs optimized")
print("=" * 65)

SAMPLE_TEXT = """\
The theory of general relativity, published by Albert Einstein in 1915,
describes gravity as a curvature of spacetime caused by mass and energy.
The famous field equations relate the geometry of spacetime to the
distribution of matter within it. The theory predicted light bending
around massive objects, confirmed during the 1919 solar eclipse by
Arthur Eddington. General relativity also predicted gravitational waves,
first directly detected by LIGO in February 2016, more than a century
after Einstein's prediction. The GPS satellite system requires relativistic
corrections of about 38 microseconds per day to maintain accuracy.
""".strip()

print(f"\nSample passage ({len(SAMPLE_TEXT.split())} words):")
print(f"  '{SAMPLE_TEXT[:80]}...'")

# ── Vanilla RAG: raw chunk stored as-is ──────────────────────────────────────
chunk_tokens = len(SAMPLE_TEXT) // 4
print(f"\nVANILLA RAG entry in ChromaDB:")
print(f"  Document  : (raw text, {chunk_tokens} tokens)")
print(f"  '{SAMPLE_TEXT[:120]}...'")
print(f"  Embedding : sentence-transformers embeds the raw 512-token text")
print(f"  Metadata  : {{source: enwik9, chunk_idx: 0}}")

# ── Optimized RAG: extractive (OLD, pre-fix) ──────────────────────────────────
from context_optimizer.compressor import _estimate_tokens, compress_chunk_extractive

# Simulate a 500KB block: repeat the passage many times
big_block = (SAMPLE_TEXT + "\n") * 300  # ~500KB
ext_chunk = compress_chunk_extractive(big_block, "blk_000", ratio=0.35)

print(f"\nOPTIMIZED RAG — extractive (OLD, pre-fix on a 500KB block):")
print(f"  Raw block  : {_estimate_tokens(big_block):,} tokens")
print(
    f"  Summary    : {ext_chunk.compressed_tokens:,} tokens  ← 35% of 500KB = still huge!"
)
print(f"  '{ext_chunk.compressed_summary[:120]}...'")
print(f"  *** THIS is why tokens/query = 322K: 5 blocks × ~64K tokens each ***")

# ── Optimized RAG: extractive (NEW, with 500-token cap) ──────────────────────
ext_capped = compress_chunk_extractive(
    big_block, "blk_000_cap", ratio=0.35, max_summary_tokens=500
)

print(f"\nOPTIMIZED RAG — extractive (FIXED, 500-token cap on same 500KB block):")
print(f"  Summary    : {ext_capped.compressed_tokens} tokens  ← capped")
print(f"  '{ext_capped.compressed_summary[:120]}...'")
print(
    f"  Tokens/query: 5 blocks × {ext_capped.compressed_tokens} = {5*ext_capped.compressed_tokens:,} tokens"
)

# ── Optimized RAG: LLM triple format (target) ────────────────────────────────
print(f"\nOPTIMIZED RAG — LLM triple format (target, pending run):")
example_triple = (
    "TOPIC:general_relativity;PERSON:Einstein;DATE:1915;"
    "CONCEPT:spacetime_curvature,field_equations;EVENT:1919_eclipse_confirmation;"
    "PERSON:Eddington;EVENT:LIGO_detection_2016;NUM:38_microseconds_GPS_correction;"
    "REL:Einstein->published->general_relativity_1915;"
    "CAUSE:mass_energy->EFFECT:spacetime_curvature"
)
print(
    f"  Summary    : {_estimate_tokens(example_triple)} tokens  ← LLM compresses whole block to ~100 tokens"
)
print(f"  '{example_triple}'")
print(
    f"  Tokens/query: 5 blocks × {_estimate_tokens(example_triple)} = {5*_estimate_tokens(example_triple):,} tokens"
)

# ── Where TF-IDF actually lives ───────────────────────────────────────────────
print(f"\n\n{'=' * 65}")
print("PART 3: Where TF-IDF is used (not retrieval)")
print("=" * 65)
print("""
TF-IDF in this codebase:
  compress_chunk_extractive() — compressor.py
  ┌─────────────────────────────────────────────────────┐
  │ Input: 500KB block of raw text                      │
  │ For each sentence: score = mean TF-IDF of its words │
  │ Keep top 35% of sentences by score                  │
  │ Output: ~64K token "summary" (before fix)           │
  │         ~500 token "summary" (after fix)            │
  └─────────────────────────────────────────────────────┘

  This is COMPRESSION, not retrieval.
  The output IS THEN EMBEDDED via sentence-transformers → ChromaDB.

Vanilla RAG has NO TF-IDF at all:
  raw chunk text → sentence-transformers → ChromaDB HNSW cosine search

The only "index" in vanilla RAG is the ChromaDB HNSW vector index,
which is a tree of 384-dimensional float32 vectors.  Nothing to do
with TF-IDF or inverted indexes.
""")

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SUMMARY: tokens sent to reasoning model per query (top-k=5)")
print("=" * 65)
print(f"  Vanilla RAG         : 5 × 512 =          2,560 tokens")
print(
    f"  Opt extractive (old): 5 × {ext_chunk.compressed_tokens:,} =  {5*ext_chunk.compressed_tokens:,} tokens  ← BUG"
)
print(
    f"  Opt extractive (fix): 5 × {ext_capped.compressed_tokens} =          {5*ext_capped.compressed_tokens:,} tokens"
)
print(
    f"  Opt LLM triple      : 5 × {_estimate_tokens(example_triple)} =            {5*_estimate_tokens(example_triple):,} tokens  ← TARGET"
)
