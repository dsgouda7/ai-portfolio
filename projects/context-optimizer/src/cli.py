"""
context-optimizer CLI

Sub-commands (new tree architecture)
-------------------------------------
build   Index your documents once (reads app_config.yaml automatically).
ask     Ask a question against the built index.
watch   Keep the index up-to-date as documents change.

Legacy sub-commands (flat RAG, kept for compatibility)
-------------------------------------------------------
ingest     Compress and index a corpus from a .txt file or directory.
query      Query a persisted index.
benchmark  Run benchmarks against local Docker services.

Examples (minimal — everything comes from app_config.yaml)::

    context-optimizer build
    context-optimizer ask "What is the return policy?"

Override corpus path or index dir at the command line::

    context-optimizer build --corpus ./my-docs --index ~/.co/my-index
    context-optimizer ask "Who wrote the refund clause?" --index ~/.co/my-index
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── App config loader ─────────────────────────────────────────────────────────

_DEFAULT_CONFIG_NAMES = ("app_config.yaml", "app_config.yml")


def _find_app_config() -> Path | None:
    """Walk from cwd upward looking for app_config.yaml."""
    search = [Path.cwd()] + list(Path.cwd().parents)
    for directory in search:
        for name in _DEFAULT_CONFIG_NAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _load_app_config(config_path: Path | None = None) -> dict:
    """Load app_config.yaml. Returns empty dict if not found."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}
    path = config_path or _find_app_config()
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text("utf-8")) or {}


def _apply_app_config(cfg: dict) -> None:
    """
    Apply app_config.yaml model settings to environment variables so
    _build_local_llm() and the tree index pick them up without extra plumbing.
    """
    comp = cfg.get("compressor", {})
    provider = comp.get("provider", "").lower()
    if provider:
        os.environ.setdefault("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", provider)
        provider_cfg = comp.get(provider, {})
        if "model" in provider_cfg:
            os.environ.setdefault("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", provider_cfg["model"])
        if "code_model" in provider_cfg:
            os.environ.setdefault("CONTEXT_OPTIMIZER_CODE_MODEL", provider_cfg["code_model"])
        if provider == "ollama" and "base_url" in provider_cfg:
            os.environ.setdefault("OLLAMA_BASE_URL", provider_cfg["base_url"])
        if provider == "hf" and "device" in provider_cfg:
            os.environ["CONTEXT_OPTIMIZER_HF_DEVICE"] = str(provider_cfg["device"])
        if provider == "azure_foundry":
            if "endpoint" in provider_cfg:
                os.environ.setdefault("AZURE_AI_FOUNDRY_ENDPOINT", provider_cfg["endpoint"])
            if "model" in provider_cfg:
                os.environ.setdefault("AZURE_AI_FOUNDRY_MODEL", provider_cfg["model"])

    reason = cfg.get("reasoning", {})
    ollama_r = reason.get("ollama", {})
    if "base_url" in ollama_r:
        os.environ.setdefault("OLLAMA_BASE_URL", ollama_r["base_url"])


# ── build / ask commands ──────────────────────────────────────────────────────


def _cmd_build(args: argparse.Namespace) -> None:
    """
    Index documents using the Tree-of-Summaries architecture.

    Reads corpus path, index dir, and model settings from app_config.yaml
    (or the file passed with --config). All settings can be overridden on
    the command line.
    """
    import time

    cfg = _load_app_config(getattr(args, "config_file", None))
    _apply_app_config(cfg)

    corpus_cfg = cfg.get("corpus", {})
    idx_cfg    = cfg.get("index", {})

    # Resolve paths: CLI > config > defaults
    corpus_path = Path(
        getattr(args, "corpus", None)
        or corpus_cfg.get("path", ".")
    ).expanduser().resolve()

    index_dir = Path(
        getattr(args, "index", None)
        or corpus_cfg.get("index_dir", "~/.co/index")
    ).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    block_mb      = float(getattr(args, "block_mb", None) or idx_cfg.get("block_mb", 0.5))
    cluster_size  = int(getattr(args, "cluster_size", None) or idx_cfg.get("cluster_size", 4))
    overlap_pct   = float(idx_cfg.get("overlap_pct", 10.0))

    print(f"[build] Corpus  : {corpus_path}")
    print(f"[build] Index   : {index_dir}")
    print(f"[build] Provider: {os.getenv('CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER', 'hf')}")
    print(f"[build] Model   : {os.getenv('CONTEXT_OPTIMIZER_COMPRESSOR_MODEL', 'facebook/bart-large-cnn')}")

    from context_optimizer.compressor import _build_local_llm, ingest_file_blocks
    from context_optimizer.raw_index import BlockIndex
    from context_optimizer.tree_index import TreeIndex, _auto_tree_depth

    block_db    = index_dir / "blocks.db"
    block_index = BlockIndex(str(block_db))

    t0 = time.perf_counter()

    if corpus_path.is_dir():
        # Multi-format directory ingestion
        from context_optimizer.ingest_corpus import ingest_directory
        chunks = ingest_directory(
            directory=corpus_path,
            block_index=block_index,
            block_size_bytes=int(block_mb * 1_048_576),
            overlap_pct=overlap_pct,
            verbose=True,
        )
    else:
        # Single file
        llm = _build_local_llm()
        chunks = ingest_file_blocks(
            source_path=corpus_path,
            block_size_bytes=int(block_mb * 1_048_576),
            overlap_bytes=int(block_mb * 1_048_576 * overlap_pct / 100),
            block_index=block_index,
            llm=llm,
            strategy="llm",
            label="build",
        )

    if not chunks:
        print("[build] No content extracted. Check corpus path and supported formats.")
        sys.exit(1)

    # Resolve depth from actual block count
    depth = _auto_tree_depth(n_blocks=len(chunks), cluster_size=cluster_size)
    print(f"[build] {len(chunks)} blocks -> depth={depth}, cluster_size={cluster_size}")

    llm = _build_local_llm()
    tree = TreeIndex(
        collection_name="app_index",
        persist_directory=str(index_dir),
        block_index=block_index,
        depth=depth,
    )
    tree.build_from_chunks(chunks, cluster_size=cluster_size, llm=llm, label="build")

    elapsed = time.perf_counter() - t0
    print(f"\n[build] Done in {elapsed:.1f}s")
    print(f"[build] L1={tree.block_count()} blocks   L2+={tree.cluster_count()} clusters")
    print(f"[build] Index saved to: {index_dir}")
    print(f'\n  Run: context-optimizer ask "your question here"')


def _cmd_ask(args: argparse.Namespace) -> None:
    """
    Answer a free-form question against the built index.

    Loads the tree index from the index directory and runs the reasoning
    agent. The agent navigates the tree, fetches raw blocks on demand,
    and returns an answer with file citations.
    """
    cfg = _load_app_config(getattr(args, "config_file", None))
    _apply_app_config(cfg)

    corpus_cfg = cfg.get("corpus", {})
    query_cfg  = cfg.get("query", {})
    reason_cfg = cfg.get("reasoning", {}).get("ollama", {})

    index_dir = Path(
        getattr(args, "index", None)
        or corpus_cfg.get("index_dir", "~/.co/index")
    ).expanduser().resolve()

    if not index_dir.exists():
        print(f"[ask] Index not found at {index_dir}")
        print("[ask] Run `context-optimizer build` first.")
        sys.exit(1)

    question = args.question
    top_k     = int(getattr(args, "top_k", None) or query_cfg.get("top_k", 5))
    max_rounds = int(query_cfg.get("max_rounds", 4))
    show_citations = query_cfg.get("show_citations", True)
    show_steps     = bool(getattr(args, "show_steps", False) or query_cfg.get("show_steps", False))

    from context_optimizer.raw_index import BlockIndex
    from context_optimizer.tree_index import TreeIndex
    from context_optimizer.tree_reasoner import TreeReasoningAgent

    block_index = BlockIndex(str(index_dir / "blocks.db"))
    tree = TreeIndex(
        collection_name="app_index",
        persist_directory=str(index_dir),
        block_index=block_index,
    )

    # Build reasoning LLM
    reasoning_model = (
        getattr(args, "model", None)
        or reason_cfg.get("model", "")
    )
    llm = None
    if reasoning_model:
        try:
            from langchain_ollama import ChatOllama  # type: ignore[import]
            base_url = reason_cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
            llm = ChatOllama(model=reasoning_model, base_url=base_url, temperature=0.0)
        except Exception as exc:
            print(f"[ask] Warning: could not load reasoning model ({exc}). Running retrieval-only.")

    agent = TreeReasoningAgent(
        tree=tree,
        llm=llm,
        top_clusters=max(1, top_k // 2),
        top_blocks_per_cluster=3,
        max_rounds=max_rounds,
    )

    result = agent.reason(question)

    print(f"\n{result.answer}")

    if show_citations and result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  {c}")

    if show_steps and result.steps:
        print(f"\nSteps taken: {len(result.steps)}")
        for s in result.steps:
            if s.action != "answer":
                print(f"  {s.action}({s.target_id})")

    print(f"\n({result.total_latency_ms:.0f} ms)")


# ── Legacy sub-command handlers ───────────────────────────────────────────────


def _cmd_ingest(args: argparse.Namespace) -> None:
    from context_optimizer.index import CorpusIndex

    src = Path(args.src)
    lines: list[str] = []

    if src.is_dir():
        txt_files = sorted(src.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {src}", file=sys.stderr)
            sys.exit(1)
        for f in txt_files:
            lines.extend(f.read_text(encoding="utf-8", errors="replace").splitlines())
        print(f"Loaded {len(lines):,} lines from {len(txt_files)} file(s) in {src}")
    elif src.is_file():
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"Loaded {len(lines):,} lines from {src}")
    else:
        print(f"Error: {src} does not exist", file=sys.stderr)
        sys.exit(1)

    index = CorpusIndex(
        compression_model=args.model,
        persist_dir=args.out,
        chunk_tokens=args.chunk_tokens,
        retrieval_strategy=args.strategy,
    )
    print(f"Compressing with model: {args.model} …")
    stats = index.ingest(lines, collection=args.collection)
    print(
        f"\nDone in {stats.elapsed_s:.1f}s — {stats.chunks} chunks\n"
        f"  Tokens:  {stats.original_tokens:,}  →  {stats.compressed_tokens:,}"
        f"  ({(1 - stats.compression_ratio) * 100:.1f}% reduction)\n"
        f"  Index:   {args.out}"
    )


def _cmd_query(args: argparse.Namespace) -> None:
    from context_optimizer.index import CorpusIndex

    index_dir = Path(args.index)
    if not index_dir.exists():
        print(f"Error: index directory '{index_dir}' not found", file=sys.stderr)
        sys.exit(1)

    # Re-open a persisted index — mark collection as ingested so query() works
    index = CorpusIndex(persist_dir=str(index_dir), retrieval_strategy=args.strategy)
    # Bootstrapping: lazily re-create retriever pointing at existing collection on disk
    from context_optimizer.cached_retriever import CachedChromaRetriever

    coll_dir = index_dir / args.collection
    if not coll_dir.exists():
        print(
            f"Error: collection '{args.collection}' not found under {index_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    index._retrievers[args.collection] = CachedChromaRetriever(
        collection_name=args.collection,
        persist_directory=str(coll_dir),
    )
    index._ingested_collections.add(args.collection)

    if args.strategy == "tot":
        from context_optimizer.tot_reasoner import ToTReasoner

        index._reasoners[args.collection] = ToTReasoner(
            retriever=index._retrievers[args.collection]
        )

    result = index.query(args.question, collection=args.collection, top_k=args.top_k)

    print(f"\nAnswer:\n{result.answer}")
    if result.branch_id:
        print(f"\n(branch: {result.branch_id})")
    print(
        f"\nEvidence — {len(result.evidence)} chunk(s)  "
        f"≈{result.tokens_used} tokens  {result.latency_ms:.0f} ms"
    )
    for i, snippet in enumerate(result.evidence[:5], 1):
        print(f"  [{i}] {snippet[:140]}{'…' if len(snippet) > 140 else ''}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """
    Run a Pipe A vs Pipe C comparison benchmark and print a metrics table.

    All corpus sizes run inline via the library's
    :func:`~context_optimizer.benchmark.compare` function — no Docker or
    external services needed.

    Corpus line counts used per size:
        small   — 500 lines   (fast, ~seconds)
        medium  — 5 000 lines (representative, ~1 min)
        large   — 25 000 lines (stress test, several minutes)
    """
    import json as _json

    corpus_lines = {"small": 500, "medium": 5_000, "large": 25_000}
    max_lines = corpus_lines[args.corpus]

    # ── Load corpus from test_data books, fallback to synthetic AKS logs ────
    from context_optimizer.benchmark import compare

    bench_dir = Path(__file__).parent.parent.parent / "benchmarks" / "tot" / "test_data"
    book_files = sorted(bench_dir.glob("books_*.txt")) if bench_dir.exists() else []

    if book_files:
        raw: list[str] = []
        for f in book_files:
            raw.extend(f.read_text(encoding="utf-8", errors="replace").splitlines())
            if len(raw) >= max_lines:
                break
        lines = raw[:max_lines]
        question = "Summarise the main themes or technical topics covered."
        corpus_label = f"{len(book_files)} book file(s)"
    else:
        # Synthetic AKS incident log — always available
        templates = [
            "2026-01-10T02:13:00Z ERROR order-service CosmosDB timeout substatus=21012 region=eastus2",
            "2026-01-10T02:13:01Z WARN  ingress-nginx upstream timed out client=10.42.7.19",
            "2026-01-10T02:13:02Z ERROR api-gateway HTTP 504 checkout p95=8.7s",
            "2026-01-10T02:13:03Z WARN  order-service CosmosDB retry ru_charge=128 partition=tenant-1",
            "2026-01-10T02:13:04Z ERROR payment-service CosmosDB cancellation timeout substatus=21012",
            "2026-01-10T02:13:05Z INFO  order-service request completed status=200 latency_ms=220",
        ]
        lines = [templates[i % len(templates)] for i in range(max_lines)]
        question = "What caused the CosmosDB timeout cascade?"
        corpus_label = "synthetic AKS incident log"

    print(f"\nCorpus: {corpus_label}  ({len(lines):,} lines)  [--corpus {args.corpus}]")
    print(f"Question: {question!r}")
    print("Running compare() locally …\n")

    result = compare(
        question=question,
        raw_corpus=lines,
        compression_model=args.model,
        collection="cli_benchmark",
    )

    print(result.summary())

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(_json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nMetrics saved -> {out_path}")


def _cmd_watch(args: argparse.Namespace) -> None:
    """
    Watch a directory and incrementally re-index changed files.

    On first run performs a full scan (only indexes files whose hash is
    not already in the registry).  Subsequent runs re-index only dirty
    files — typically 1-3 SLM calls per save.
    """
    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.raw_index import BlockIndex, FileRegistry
    from context_optimizer.watcher import CorpusWatcher, IncrementalIndexer

    watch_dir = Path(args.watch_dir).expanduser().resolve()
    index_dir = Path(args.index).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    block_db = index_dir / "blocks.db"
    block_index = BlockIndex(str(block_db))
    # FileRegistry shares the same SQLite file
    file_registry = FileRegistry(str(block_db))

    retriever = CachedChromaRetriever(
        collection_name=args.collection,
        persist_directory=str(index_dir / args.collection),
    )

    # Optionally load TreeIndex
    tree = None
    if args.tree:
        from context_optimizer.tree_index import TreeIndex

        tree = TreeIndex(
            collection_name=f"{args.collection}_tree",
            persist_directory=str(index_dir),
            block_index=block_index,
        )

    block_bytes = int(args.block_mb * 1024 * 1024)
    overlap_bytes = int(block_bytes * args.overlap_pct / 100)

    indexer = IncrementalIndexer(
        block_index=block_index,
        file_registry=file_registry,
        retriever=retriever,
        tree=tree,
        block_size_bytes=block_bytes,
        overlap_bytes=overlap_bytes,
        compressor_model=args.model,
        compressor_provider=getattr(args, "provider", None),
        verbose=True,
    )

    print(f"[watch] Directory : {watch_dir}")
    print(f"[watch] Index     : {index_dir}")
    print(f"[watch] Pattern   : {args.glob}")
    print(f"[watch] Block     : {args.block_mb} MB  overlap={args.overlap_pct}%")
    provider_tag = getattr(args, "provider", None) or "ollama"
    print(f"[watch] Provider  : {provider_tag}  model={args.model}")
    if args.tree:
        print("[watch] Tree-of-Summaries: enabled")
    print()

    if args.scan_only:
        # One-shot scan: re-index dirty files then exit
        results = indexer.scan_directory(watch_dir, glob=args.glob)
        total_new = sum(results.values())
        changed = sum(1 for n in results.values() if n > 0)
        print(
            f"[watch] Scan complete — {changed} files re-indexed, {total_new} blocks ingested"
        )
        return

    # Initial scan before starting the watcher
    print("[watch] Initial scan ...")
    indexer.scan_directory(watch_dir, glob=args.glob)

    watcher = CorpusWatcher(
        watch_dir=watch_dir,
        indexer=indexer,
        glob=args.glob,
        debounce_s=args.debounce,
    )
    watcher.run_forever()


# ── Argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-optimizer",
        description="Context Optimizer — LLM corpus compression and retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  context-optimizer ingest ./logs --model llama3.2:3b --out ./idx\n"
            "  context-optimizer query 'CosmosDB timeout' --index ./idx\n"
            "  context-optimizer benchmark --corpus medium --mode text\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # ── build (new tree architecture) ─────────────────────────────────────────
    p_build = sub.add_parser(
        "build",
        help="Index documents using Tree-of-Summaries (reads app_config.yaml)",
    )
    p_build.add_argument(
        "--corpus", default=None, metavar="PATH",
        help="Override corpus path from app_config.yaml",
    )
    p_build.add_argument(
        "--index", default=None, metavar="DIR",
        help="Override index directory from app_config.yaml",
    )
    p_build.add_argument(
        "--block-mb", type=float, default=None, dest="block_mb",
        help="Override block_mb from app_config.yaml",
    )
    p_build.add_argument(
        "--cluster-size", type=int, default=None, dest="cluster_size",
        help="Override cluster_size from app_config.yaml",
    )
    p_build.add_argument(
        "--config", default=None, metavar="FILE", dest="config_file",
        help="Path to app_config.yaml (default: auto-discover from cwd upward)",
    )

    # ── ask (new tree architecture) ────────────────────────────────────────────
    p_ask = sub.add_parser(
        "ask",
        help="Answer a question against the built index (reads app_config.yaml)",
    )
    p_ask.add_argument("question", help="Free-form question to answer")
    p_ask.add_argument(
        "--index", default=None, metavar="DIR",
        help="Override index directory from app_config.yaml",
    )
    p_ask.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Override reasoning model from app_config.yaml",
    )
    p_ask.add_argument(
        "--top-k", type=int, default=None, dest="top_k",
        help="Number of tree nodes retrieved per query",
    )
    p_ask.add_argument(
        "--show-steps", action="store_true", default=False, dest="show_steps",
        help="Print the agent's navigation steps alongside the answer",
    )
    p_ask.add_argument(
        "--config", default=None, metavar="FILE", dest="config_file",
        help="Path to app_config.yaml (default: auto-discover from cwd upward)",
    )

    # ── ingest (legacy flat RAG) ───────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Compress and index a corpus")
    p_ingest.add_argument("src", help="Path to a .txt file or directory of .txt files")
    p_ingest.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Model name for compression (default: llama3.2:3b)",
    )
    p_ingest.add_argument(
        "--provider",
        default=None,
        choices=["ollama", "hf", "groq", "azure"],
        help="LLM provider (default: ollama). Use 'hf' for local BART/T5 — faster on CPU, no Ollama needed.",
    )
    p_ingest.add_argument(
        "--out", required=True, metavar="DIR", help="Directory to persist the index"
    )
    p_ingest.add_argument(
        "--collection",
        default="default",
        help="Collection name within the index (default: default)",
    )
    p_ingest.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        dest="chunk_tokens",
        help="Token budget per compressed chunk (default: 512)",
    )
    p_ingest.add_argument(
        "--strategy",
        choices=["tot", "simple"],
        default="tot",
        help="Retrieval strategy (default: tot)",
    )

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Query a persisted index")
    p_query.add_argument("question", help="Natural-language query string")
    p_query.add_argument(
        "--index",
        required=True,
        metavar="DIR",
        help="Persisted index directory (created by 'ingest')",
    )
    p_query.add_argument(
        "--collection",
        default="default",
        help="Collection name to query (default: default)",
    )
    p_query.add_argument(
        "--top-k",
        type=int,
        default=6,
        dest="top_k",
        help="Evidence snippets to return (simple strategy, default: 6)",
    )
    p_query.add_argument(
        "--strategy",
        choices=["tot", "simple"],
        default="tot",
        help="Retrieval strategy (default: tot)",
    )

    # ── watch ─────────────────────────────────────────────────────────────────
    p_watch = sub.add_parser(
        "watch",
        help="Watch a directory and incrementally re-index changed files",
    )
    p_watch.add_argument(
        "watch_dir", metavar="DIR", help="Directory to watch (recursively)"
    )
    p_watch.add_argument(
        "--index",
        required=True,
        metavar="DIR",
        help="Persistent index directory (created on first run)",
    )
    p_watch.add_argument(
        "--glob",
        default="**/*.txt",
        metavar="PATTERN",
        help="File pattern to watch (default: **/*.txt)",
    )
    p_watch.add_argument(
        "--model",
        default="google/flan-t5-small",
        help="Model name for block compression (default: google/flan-t5-small with hf provider)",
    )
    p_watch.add_argument(
        "--provider",
        default="hf",
        choices=["ollama", "hf", "groq", "azure"],
        help="LLM provider (default: hf — facebook/bart-large-cnn, ~400 MB, 5-15x faster than Ollama on CPU)",
    )
    p_watch.add_argument(
        "--block-mb",
        type=float,
        default=0.1,
        dest="block_mb",
        help="Block size in MB (default: 0.1 = 100 KB)",
    )
    p_watch.add_argument(
        "--overlap-pct",
        type=float,
        default=10.0,
        dest="overlap_pct",
        help="Block overlap as %% of block size (default: 10)",
    )
    p_watch.add_argument(
        "--collection",
        default="default",
        help="ChromaDB collection name (default: default)",
    )
    p_watch.add_argument(
        "--tree",
        action="store_true",
        default=False,
        help="Also maintain a Tree-of-Summaries (L1+L2) index",
    )
    p_watch.add_argument(
        "--debounce",
        type=float,
        default=3.0,
        metavar="SECS",
        help="Seconds to wait before re-indexing after a change (default: 3)",
    )
    p_watch.add_argument(
        "--scan-only",
        action="store_true",
        default=False,
        dest="scan_only",
        help="Scan once for dirty files and exit (no continuous watching)",
    )

    # ── benchmark ────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Run benchmarks against Docker services")
    p_bench.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size for the benchmark run (default: small)",
    )
    p_bench.add_argument(
        "--mode",
        choices=["text", "image", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    p_bench.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Compression model for inline small-corpus benchmark (default: llama3.2:3b)",
    )
    p_bench.add_argument(
        "--json-out",
        default=None,
        metavar="FILE",
        dest="json_out",
        help="Write metrics JSON to FILE (small corpus only)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "build":     _cmd_build,
        "ask":       _cmd_ask,
        "ingest":    _cmd_ingest,
        "query":     _cmd_query,
        "watch":     _cmd_watch,
        "benchmark": _cmd_benchmark,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
