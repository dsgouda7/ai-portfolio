"""
Medium Corpus Compression -- compress and save to ChromaDB.

Compresses 25 000 lines (Gutenberg books + Wikipedia) and persists them in a
local ChromaDB collection for use by latency_comparison.py and
accuracy_benchmarks.py.  Run once; re-run only when you want to refresh the index.

LLM backend (defaults to Ollama -- runs locally, no API key):
    See llm_provider.py for full env-var reference.

    Quick start:
        ollama serve
        ollama pull qwen2.5-coder:7b
        python quick_compress_and_save.py
"""

import sys
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.cached_retriever import CachedChromaRetriever
from download_test_data import download_all_datasets
from llm_provider import build_compression_llm

CORPUS_SIZE     = 25_000
CHROMA_DIR      = Path(__file__).parent / "chroma_db"
COLLECTION      = "medium_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def quick_compress_and_save():
    """Compress medium corpus and persist chunks to ChromaDB."""

    print("=" * 80)
    print("QUICK COMPRESSION  --  Medium Corpus -> ChromaDB")
    print("=" * 80)

    # Step 1: LLM
    print("\n[1/4] Initialising compression LLM...")
    llm = build_compression_llm()

    # Step 2: Corpus
    print("\n[2/4] Loading corpus data...")
    corpus_samples = download_all_datasets()
    medium = corpus_samples["medium_500mb"]

    corpus_lines = []
    corpus_lines.extend(medium["books"])
    corpus_lines.extend(medium["code"])
    corpus_lines.extend(medium["wiki"])
    corpus_lines = corpus_lines[:CORPUS_SIZE]
    print(f"  Lines : {len(corpus_lines):,}")

    # Step 3: Compress
    print(f"\n[3/4] Compressing corpus (rolling window, 512-token chunks)...")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    chunks = compress_corpus_rolling(
        corpus_lines=corpus_lines,
        chunk_size_threshold=512,
        chunk_overlap_tokens=128,
        llm=llm,
    )

    total_orig = sum(c.original_tokens for c in chunks)
    total_comp = sum(c.compressed_tokens for c in chunks)
    ratio      = total_comp / total_orig if total_orig else 0
    print(f"  [OK] {len(chunks):,} chunks produced")
    print(f"  Compression ratio : {ratio:.3f}  ({total_orig:,} -> {total_comp:,} tokens)")

    # Step 4: Persist
    print(f"\n[4/4] Saving to ChromaDB  ({CHROMA_DIR})...")
    retriever = CachedChromaRetriever(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_model_name=EMBEDDING_MODEL,
        cache_size=1000,
        cache_threshold=0.85,
    )
    retriever.add_chunks(chunks)

    print("\n" + "=" * 80)
    print("COMPRESSION COMPLETE!")
    print("=" * 80)
    print(f"  Collection : {COLLECTION}")
    print(f"  Chunks     : {len(chunks):,}")
    print(f"  Storage    : {CHROMA_DIR}")
    print(f"  Embeddings : {EMBEDDING_MODEL}  (local CPU, ~90 MB)")
    print(f"\nNext steps:")
    print(f"  python latency_comparison.py")
    print(f"  python accuracy_benchmarks.py")
    print(f"  python run_benchmarks.py")

    return {"medium_chunks": len(chunks), "chroma_dir": str(CHROMA_DIR)}


if __name__ == "__main__":
    result = quick_compress_and_save()
    sys.exit(0 if result else 1)
