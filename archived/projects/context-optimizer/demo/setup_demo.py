"""
setup_demo.py — Build the demo index from the bundled corpus.

Run directly from the repo — no pip install needed:

    cd projects/context-optimizer/demo
    python setup_demo.py

Optional editable install (also fine):  pip install -e '../[hf]'

The script uses facebook/bart-large-cnn for summarization (~400 MB download on
first run, cached by HuggingFace thereafter). This produces coherent, readable
summaries that make the tree-traversal demo meaningful.

Override via env var:
    CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=google/flan-t5-large python setup_demo.py

For code-aware summarization the same model is used in the demo.
Production usage should set CONTEXT_OPTIMIZER_CODE_MODEL=Salesforce/codet5-base-codexglue-sum-python.
"""
from __future__ import annotations

# ── Dev-mode bootstrap ─────────────────────────────────────────────────────────
# Wire src/ as the context_optimizer package when running from the repo without
# a pip install.  This is a no-op if the package is already installed.
import importlib.util as _ilu
import sys
from pathlib import Path
_bs = Path(__file__).parent / "_bootstrap.py"
_bs_spec = _ilu.spec_from_file_location("_bootstrap", _bs)
_bs_mod  = _ilu.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(_bs_mod)
del _ilu, _bs, _bs_spec, _bs_mod  # keep namespace clean
# ─────────────────────────────────────────────────────────────────────────────

import os
import time

# ── Point to the demo's own lightweight model ─────────────────────────────────
os.environ.setdefault("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "hf")
os.environ.setdefault(
    "CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "facebook/bart-large-cnn"
)
# For the demo, use the same model for code files too (bart-large-cnn handles
# code reasonably well; swap to codet5-base-codexglue-sum-python for production).
os.environ.setdefault("CONTEXT_OPTIMIZER_CODE_MODEL", "facebook/bart-large-cnn")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEMO_DIR = Path(__file__).parent.resolve()
INDEX_DIR = DEMO_DIR / ".index"
CORPUS_DIR = DEMO_DIR / "corpus"

# Also ingest the project's own src/ as a code corpus — exercises the code path.
SRC_DIR = (DEMO_DIR.parent / "src").resolve()


def _check_deps() -> None:
    missing = []
    try:
        import context_optimizer  # noqa: F401
    except ImportError:
        missing.append("context-optimizer[hf]")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers torch (pip install 'context-optimizer[hf]')")
    if missing:
        print("[setup] Missing dependencies:")
        for m in missing:
            print(f"  pip install '{m}'")
        sys.exit(1)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-code", action="store_true",
        help="Skip ingesting the src/ code corpus (faster, prose-only demo)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Delete an existing index and rebuild from scratch",
    )
    args = ap.parse_args()

    _check_deps()

    from context_optimizer.ingest_corpus import ingest_directory
    from context_optimizer.raw_index import BlockIndex
    from context_optimizer.tree_index import TreeIndex, _auto_tree_depth
    from context_optimizer.compressor import _build_local_llm

    if args.force and INDEX_DIR.exists():
        import shutil
        shutil.rmtree(INDEX_DIR)
        print(f"[setup] Removed existing index: {INDEX_DIR}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    block_db = INDEX_DIR / "blocks.db"

    print(f"\n[setup] Index dir : {INDEX_DIR}")
    print(f"[setup] Corpus    : {CORPUS_DIR}")
    print(f"[setup] Provider  : {os.environ['CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER']}")
    print(f"[setup] Model     : {os.environ['CONTEXT_OPTIMIZER_COMPRESSOR_MODEL']}")
    print()

    block_index = BlockIndex(str(block_db))

    t0 = time.perf_counter()

    # ── Ingest prose corpus ───────────────────────────────────────────────────
    print("[setup] Ingesting prose corpus ...")
    prose_chunks = ingest_directory(
        directory=CORPUS_DIR / "prose",
        block_index=block_index,
        block_size_bytes=50_000,   # 50 KB — small blocks for a small corpus
        overlap_pct=10.0,
        verbose=True,
    )

    # ── Ingest extended corpus (if prepare_extended_corpus.py was run) ────────
    extended_chunks: list = []

    gutenberg_dir = CORPUS_DIR / "gutenberg"
    if gutenberg_dir.exists() and any(gutenberg_dir.iterdir()):
        print(f"\n[setup] Ingesting Gutenberg corpus ({gutenberg_dir}) ...")
        extended_chunks += ingest_directory(
            directory=gutenberg_dir,
            block_index=block_index,
            block_size_bytes=80_000,   # 80 KB — one chapter-sized block
            overlap_pct=5.0,
            verbose=True,
        )

    requests_dir = CORPUS_DIR / "requests-src"
    if requests_dir.exists() and any(requests_dir.glob("*.py")):
        print(f"\n[setup] Ingesting requests source code ({requests_dir}) ...")
        extended_chunks += ingest_directory(
            directory=requests_dir,
            block_index=block_index,
            block_size_bytes=40_000,   # 40 KB — fits most files in one block
            include_exts=[".py"],
            overlap_pct=5.0,
            verbose=True,
        )

    if extended_chunks:
        print(
            f"\n[setup] Extended corpus: {len(extended_chunks)} blocks "
            f"(Gutenberg + requests-src)"
        )

    # ── Ingest Django source (if prepare_extended_corpus.py was run) ──────────
    django_dir = CORPUS_DIR / "django-src"
    if django_dir.exists() and any(django_dir.rglob("*.py")):
        subsystem_count = sum(1 for d in django_dir.iterdir() if d.is_dir())
        print(
            f"\n[setup] Ingesting Django source ({django_dir})"
            f" — {subsystem_count} subsystems …"
        )
        django_chunks = ingest_directory(
            directory=django_dir,
            block_index=block_index,
            block_size_bytes=40_000,   # 40 KB — one module per block
            include_exts=[".py"],
            overlap_pct=5.0,
            verbose=True,
        )
        extended_chunks += django_chunks
        print(f"[setup] Django: {len(django_chunks)} blocks")

    # ── Ingest project src/ as code corpus (optional) ─────────────────────────
    code_chunks: list = []
    if SRC_DIR.exists() and not args.no_code:
        print(f"\n[setup] Ingesting code corpus ({SRC_DIR}) ...")
        code_chunks = ingest_directory(
            directory=SRC_DIR,
            block_index=block_index,
            block_size_bytes=50_000,
            include_exts=[".py"],
            overlap_pct=5.0,
            verbose=True,
        )

    all_chunks = prose_chunks + extended_chunks + code_chunks
    if not all_chunks:
        print("[setup] ERROR: no content extracted — check corpus path.")
        sys.exit(1)

    # ── Build Tree-of-Summaries ───────────────────────────────────────────────
    depth = _auto_tree_depth(n_blocks=len(all_chunks), cluster_size=4, top_k=4)
    print(
        f"\n[setup] {len(all_chunks)} blocks total"
        f" → depth={depth}, cluster_size=4"
    )

    llm = _build_local_llm()
    tree = TreeIndex(
        collection_name="demo_index",
        persist_directory=str(INDEX_DIR),
        block_index=block_index,
        depth=depth,
    )
    tree.build_from_chunks(all_chunks, cluster_size=4, llm=llm, label="demo")

    elapsed = time.perf_counter() - t0
    print(f"\n[setup] Done in {elapsed:.1f}s")
    print(
        f"[setup] Tree: {tree.block_count()} L1 blocks,"
        f" {tree.cluster_count()} clusters"
    )
    print(f"[setup] Index: {INDEX_DIR}")
    print()
    print("─" * 60)
    print("Next: python run_demo.py")
    print("Then open: http://localhost:8000")
    print("─" * 60)


if __name__ == "__main__":
    main()
