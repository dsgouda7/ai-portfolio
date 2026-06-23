"""
context-optimizer CLI

Sub-commands
------------
ingest     Compress and index a corpus from a .txt file or directory.
query      Query a persisted index.
benchmark  Run benchmarks against local Docker services.

Examples::

    context-optimizer ingest ./logs --model llama3.2:3b --out ./my_index
    context-optimizer query "what caused the CosmosDB timeout?" --index ./my_index
    context-optimizer benchmark --corpus medium --mode text
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Sub-command handlers ──────────────────────────────────────────────────────

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
        print(f"\nMetrics saved → {out_path}")


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

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Compress and index a corpus")
    p_ingest.add_argument("src", help="Path to a .txt file or directory of .txt files")
    p_ingest.add_argument("--model",        default="llama3.2:3b",
                          help="Ollama model for compression (default: llama3.2:3b)")
    p_ingest.add_argument("--out",          required=True, metavar="DIR",
                          help="Directory to persist the index")
    p_ingest.add_argument("--collection",   default="default",
                          help="Collection name within the index (default: default)")
    p_ingest.add_argument("--chunk-tokens", type=int, default=512, dest="chunk_tokens",
                          help="Token budget per compressed chunk (default: 512)")
    p_ingest.add_argument("--strategy",     choices=["tot", "simple"], default="tot",
                          help="Retrieval strategy (default: tot)")

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Query a persisted index")
    p_query.add_argument("question",    help="Natural-language query string")
    p_query.add_argument("--index",     required=True, metavar="DIR",
                         help="Persisted index directory (created by 'ingest')")
    p_query.add_argument("--collection", default="default",
                         help="Collection name to query (default: default)")
    p_query.add_argument("--top-k",     type=int, default=6, dest="top_k",
                         help="Evidence snippets to return (simple strategy, default: 6)")
    p_query.add_argument("--strategy",  choices=["tot", "simple"], default="tot",
                         help="Retrieval strategy (default: tot)")

    # ── benchmark ────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Run benchmarks against Docker services")
    p_bench.add_argument("--corpus", choices=["small", "medium", "large"], default="small",
                         help="Corpus size for the benchmark run (default: small)")
    p_bench.add_argument("--mode",   choices=["text", "image", "all"],    default="all",
                         help="Which benchmark to run (default: all)")
    p_bench.add_argument("--model",  default="llama3.2:3b",
                         help="Compression model for inline small-corpus benchmark (default: llama3.2:3b)")
    p_bench.add_argument("--json-out", default=None, metavar="FILE", dest="json_out",
                         help="Write metrics JSON to FILE (small corpus only)")

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    dispatch = {
        "ingest":     _cmd_ingest,
        "query":      _cmd_query,
        "benchmark":  _cmd_benchmark,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
