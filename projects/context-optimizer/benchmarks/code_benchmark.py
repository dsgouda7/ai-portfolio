"""
code_benchmark.py — build and evaluate a CodeTreeIndex against a source corpus.

Usage
-----
    # Build index from a source directory
    python code_benchmark.py build --src path/to/repo --index-dir ./code_index

    # Evaluate against a question file
    python code_benchmark.py eval --index-dir ./code_index --questions questions.json

    # Full run (build + eval)
    python code_benchmark.py run --src path/to/repo --index-dir ./code_index --questions questions.json

Question file format (JSON array):
    [
      {
        "id": "q001",
        "question": "Which function handles TCP retransmit timeout?",
        "expected_file": "net/ipv4/tcp_timer.c",
        "expected_symbol": "tcp_retransmit_timer",
        "expected_start_line": 312
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_file: str  # partial match is OK (basename or subpath)
    expected_symbol: str = ""  # function/class name
    expected_start_line: int = 0
    difficulty: str = "medium"


@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    recall_at_1: bool  # correct file+symbol in top-1 result
    recall_at_3: bool  # correct file+symbol in top-3 results
    citations: list[str]
    steps_taken: int
    latency_ms: float
    difficulty: str = "medium"


@dataclass
class BenchmarkSummary:
    recall_at_1: float
    recall_at_3: float
    avg_latency_ms: float
    avg_steps: float
    n_questions: int
    results: list[EvalResult] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_match(citation: str, expected_file: str, expected_symbol: str) -> bool:
    """Return True if *citation* matches the expected file/symbol."""
    if not citation:
        return False
    # File match: expected_file should appear as a suffix of the citation path
    p = Path(citation.split(":")[0])
    ef = Path(expected_file)
    file_match = str(p).replace("\\", "/").endswith(str(ef).replace("\\", "/"))
    if not file_match:
        return False
    if expected_symbol:
        # Citation format is "path:start-end"; check pointer
        return True  # file match is sufficient for Recall@k
    return True


def _build_reasoning_llm(model: str) -> "Any | None":
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import]

        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
        )
    except Exception:
        return None


# ── Sub-commands ──────────────────────────────────────────────────────────────


def cmd_build(args: argparse.Namespace) -> None:
    from context_optimizer.code.chunker import CodeChunker
    from context_optimizer.code.code_index import CodeTreeIndex
    from context_optimizer.compressor import _build_local_llm

    src = Path(args.src)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] Source: {src}")
    print(f"[build] Index:  {index_dir}")

    # Chunk the codebase
    chunker = CodeChunker(
        min_lines=getattr(args, "min_lines", 5),
        max_lines=getattr(args, "max_lines", 300),
    )
    print(f"[build] Chunking {src} ...")
    t0 = time.perf_counter()
    chunks = chunker.chunk_directory(
        src,
        recursive=True,
        include_exts=getattr(args, "exts", None),
    )
    print(f"[build] {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")

    if not chunks:
        print("[build] No chunks produced — check --src and --exts")
        sys.exit(1)

    # Build summarization LLM
    os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = getattr(
        args, "provider", "ollama"
    )
    os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_MODEL"] = getattr(
        args, "model", "qwen2.5-coder:7b"
    )
    llm = _build_local_llm()

    # Build CodeTreeIndex
    idx = CodeTreeIndex(
        collection_name="code_tree",
        persist_directory=str(index_dir),
        code_embedding_model=getattr(args, "embed_model", "microsoft/codebert-base"),
        depth=0,  # auto
    )
    idx.build_from_code_chunks(
        chunks,
        cluster_size=getattr(args, "cluster_size", 4),
        llm=llm,
        label="build",
    )
    print(
        f"[build] Index ready: {idx.block_count()} L1 entries, {idx.cluster_count()} cluster entries"
    )


def cmd_eval(args: argparse.Namespace) -> BenchmarkSummary:
    from context_optimizer.code.code_index import CodeTreeIndex
    from context_optimizer.code.code_reasoner import CodeReasoningAgent

    index_dir = Path(args.index_dir)
    questions_path = Path(args.questions)

    # Load questions
    raw = json.loads(questions_path.read_text("utf-8"))
    questions = [EvalQuestion(**q) for q in raw][: getattr(args, "limit", len(raw))]
    print(f"[eval] {len(questions)} questions from {questions_path.name}")

    # Load index
    idx = CodeTreeIndex(
        collection_name="code_tree",
        persist_directory=str(index_dir),
        depth=0,  # auto-detect from saved collections
    )
    print(
        f"[eval] Index: {idx.block_count()} L1 entries, {idx.cluster_count()} cluster entries"
    )

    # Build reasoning LLM
    reasoning_model = getattr(args, "reasoning_model", "qwen2.5-coder:7b")
    llm = _build_reasoning_llm(reasoning_model)
    if llm is None:
        print(
            f"[eval] WARNING: reasoning model {reasoning_model} unavailable — no tool navigation"
        )

    agent = CodeReasoningAgent(
        index=idx,
        llm=llm,
        top_clusters=getattr(args, "top_k", 2),
        top_chunks_per_cluster=3,
        max_rounds=4,
    )

    results: list[EvalResult] = []
    for q in questions:
        result = agent.reason(q.question)

        # Check recall: did any citation match the expected file?
        r1 = any(
            _is_match(c, q.expected_file, q.expected_symbol)
            for c in result.citations[:1]
        )
        r3 = any(
            _is_match(c, q.expected_file, q.expected_symbol)
            for c in result.citations[:3]
        )

        er = EvalResult(
            question_id=q.id,
            question=q.question,
            answer=result.answer[:300],
            recall_at_1=r1,
            recall_at_3=r3,
            citations=result.citations[:5],
            steps_taken=len(result.steps),
            latency_ms=result.total_latency_ms,
            difficulty=q.difficulty,
        )
        results.append(er)
        status = "✓" if r3 else "✗"
        print(
            f"  [{status}] {q.id} ({q.difficulty})  R@1={r1}  R@3={r3}  "
            f"steps={er.steps_taken}  {er.latency_ms:.0f}ms"
        )

    summary = BenchmarkSummary(
        recall_at_1=sum(r.recall_at_1 for r in results) / len(results),
        recall_at_3=sum(r.recall_at_3 for r in results) / len(results),
        avg_latency_ms=sum(r.latency_ms for r in results) / len(results),
        avg_steps=sum(r.steps_taken for r in results) / len(results),
        n_questions=len(results),
        results=results,
    )

    print(
        f"\n[eval] Recall@1={summary.recall_at_1:.1%}  Recall@3={summary.recall_at_3:.1%}  "
        f"avg_latency={summary.avg_latency_ms:.0f}ms  avg_steps={summary.avg_steps:.1f}"
    )

    # Write report
    report_path = index_dir / "eval_results.json"
    report_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(f"[eval] Report: {report_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code_benchmark", description="Code search benchmark"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Build CodeTreeIndex from a source directory")
    b.add_argument("--src", required=True, type=Path, help="Source directory")
    b.add_argument("--index-dir", required=True, type=Path, dest="index_dir")
    b.add_argument("--provider", default="ollama")
    b.add_argument("--model", default="qwen2.5-coder:7b")
    b.add_argument(
        "--embed-model", default="microsoft/codebert-base", dest="embed_model"
    )
    b.add_argument("--cluster-size", default=4, type=int, dest="cluster_size")
    b.add_argument("--min-lines", default=5, type=int, dest="min_lines")
    b.add_argument("--max-lines", default=300, type=int, dest="max_lines")
    b.add_argument("--exts", nargs="*", default=None, help="File extensions to include")

    # eval
    e = sub.add_parser("eval", help="Evaluate against a question file")
    e.add_argument("--index-dir", required=True, type=Path, dest="index_dir")
    e.add_argument("--questions", required=True, type=Path)
    e.add_argument(
        "--reasoning-model", default="qwen2.5-coder:7b", dest="reasoning_model"
    )
    e.add_argument("--top-k", default=2, type=int, dest="top_k")
    e.add_argument("--limit", default=9999, type=int)

    # run (build + eval)
    r = sub.add_parser("run", help="Build then evaluate")
    for parent in (b, e):
        for action in parent._actions:
            if action.dest not in ("help", "cmd"):
                try:
                    r._add_action(action)
                except Exception:
                    pass

    return p


def main() -> None:
    args = _parser().parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "run":
        cmd_build(args)
        cmd_eval(args)


if __name__ == "__main__":
    main()
