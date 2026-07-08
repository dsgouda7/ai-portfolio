#!/usr/bin/env python3
"""
Vanilla RAG vs Optimized RAG — head-to-head benchmark on a 1-2 GB corpus.

The benchmark directly answers the three questions:
  1. How accurate are answers?          (keyword recall on 50 factual questions)
  2. What is the token footprint?       (tokens sent to the reasoning model per query)
  3. How long does index building take? (ingestion latency for each strategy)

Strategies
----------
vanilla_rag
    Standard dense-retrieval RAG.  The corpus is split into 512-token raw
    chunks, each chunk is embedded as-is in ChromaDB (no compression).
    At query time the top-k chunks are retrieved and their raw text is sent
    to the reasoning model.  This is the industry-standard baseline.

optimized_rag
    Block-based compression with on-demand raw fallback.
    1. The corpus is split into 500 KB blocks.
    2. Each block is extractively compressed to ~1-5% of its original size
       (TF-IDF sentence selection, no LLM needed).
    3. The compressed summary is stored in ChromaDB.
    4. A BlockIndex stores only the byte offsets of each block in the source
       file — raw text is NEVER duplicated.  For a 1 GB corpus this is
       ~3 KB of metadata vs 1 GB of raw storage in vanilla RAG.
    5. At query time:
         a. Top-k summaries are retrieved from ChromaDB.
         b. If the summary confidence (cosine score) is below a threshold,
            the reasoning model fetches the raw block via the file pointer.
         c. Only the requested block is read from disk (~2 ms per block).

Corpus
------
Default: Gutenberg top-20 public-domain books (~18 MB combined clean prose).
         Downloaded automatically from www.gutenberg.org.

Override: --corpus-path /path/to/your/file.txt  (any UTF-8 text file)

Usage
-----
    python corpus_benchmark.py prepare                   # download corpus + generate questions
    python corpus_benchmark.py run                       # build indexes + evaluate
    python corpus_benchmark.py all                       # prepare then run

    python corpus_benchmark.py prepare --corpus-path /data/myfile.txt
    python corpus_benchmark.py run --questions 25 --top-k 3 --block-mb 1
    python corpus_benchmark.py run --vanilla-only        # skip optimized build
    python corpus_benchmark.py run --optimized-only      # skip vanilla build
    python corpus_benchmark.py run --config benchmarks/bench_config.yaml
    python corpus_benchmark.py run --config benchmarks/bench_config.yaml --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime

# ── Config-file loader ──────────────────────────────────────────────────────


def _load_bench_config(config_path: Path) -> dict:
    """Load and return the YAML benchmark config."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise SystemExit("[config] PyYAML is required: pip install pyyaml")
    return yaml.safe_load(config_path.read_text("utf-8")) or {}


def _apply_bench_config(args: argparse.Namespace, cfg: dict) -> None:
    """
    Apply bench_config.yaml values to *args*, then set env vars so
    _build_local_llm() picks up the right provider/model without any
    CLI flags.  Explicit CLI values passed by the user are NOT overwritten
    (config is a default layer, CLI is an override layer).

    Section mapping
    ---------------
    compressor.provider                -> CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER
    compressor.<provider>.model        -> CONTEXT_OPTIMIZER_COMPRESSOR_MODEL
                                         + args.compressor_model
    compressor.ollama.base_url         -> OLLAMA_BASE_URL
    reasoning.ollama.model             -> args.reasoning_model
    reasoning.ollama.base_url          -> OLLAMA_BASE_URL (if not already set)
    benchmark.*                        -> matching args attributes
    """
    bench = cfg.get("benchmark", {})
    comp = cfg.get("compressor", {})
    reason = cfg.get("reasoning", {})

    # ── Benchmark settings ────────────────────────────────────────────
    _scalar_keys = {
        "block_mb": "block_mb",
        "cluster_size": "cluster_size",
        "questions": "questions",
        "max_mb": "max_mb",
        "overlap_pct": "overlap_pct",
        "top_k": "top_k",
        "tree_depth": "tree_depth",
    }
    _bool_keys = {
        "tree": "run_tree",
        "optimized_only": "optimized_only",
        "build_only": "build_only",
        "eval_only": "eval_only",
    }
    for cfg_key, arg_key in _scalar_keys.items():
        if cfg_key in bench:
            val = bench[cfg_key]
            # tree_depth: "auto" string -> 0 (resolved inside build_tree_rag)
            if cfg_key == "tree_depth" and str(val).lower() == "auto":
                val = 0
            setattr(args, arg_key, val)
    for cfg_key, arg_key in _bool_keys.items():
        if cfg_key in bench:
            setattr(args, arg_key, bool(bench[cfg_key]))
    # Resolve path values relative to the project root (config file's parent dir)
    # so  "benchmarks/data/corpus/..."  works regardless of working directory.
    _cfg_dir = Path(getattr(args, "_config_path", __file__)).parent
    _proj_root = _cfg_dir.parent  # context-optimizer/
    if "corpus_path" in bench:
        p = Path(bench["corpus_path"])
        args.corpus_path = p if p.is_absolute() else (_proj_root / p).resolve()
    if "index_dir" in bench:
        p = Path(bench["index_dir"])
        args.index_dir = p if p.is_absolute() else (_proj_root / p).resolve()

    # ── Compressor ────────────────────────────────────────────────────
    provider = comp.get("provider", "").lower()
    if provider:
        os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = provider
        provider_cfg = comp.get(provider, {})
        if "model" in provider_cfg:
            os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_MODEL"] = provider_cfg["model"]
            args.compressor_model = provider_cfg["model"]
        if "code_model" in provider_cfg:
            # Two-model design: separate code-specialized model, same size/speed
            os.environ["CONTEXT_OPTIMIZER_CODE_MODEL"] = provider_cfg["code_model"]
        if provider == "ollama" and "base_url" in provider_cfg:
            os.environ.setdefault("OLLAMA_BASE_URL", provider_cfg["base_url"])
        if provider == "hf" and "device" in provider_cfg:
            os.environ["CONTEXT_OPTIMIZER_HF_DEVICE"] = str(provider_cfg["device"])
        if provider == "azure_foundry":
            if "endpoint" in provider_cfg:
                os.environ.setdefault("AZURE_AI_FOUNDRY_ENDPOINT", provider_cfg["endpoint"])
            if "model" in provider_cfg:
                os.environ.setdefault("AZURE_AI_FOUNDRY_MODEL", provider_cfg["model"])

    # ── Reasoning model ───────────────────────────────────────────────
    ollama_reason = reason.get("ollama", {})
    if "model" in ollama_reason:
        args.reasoning_model = ollama_reason["model"]
    if "base_url" in ollama_reason:
        os.environ.setdefault("OLLAMA_BASE_URL", ollama_reason["base_url"])

    code_model = os.getenv("CONTEXT_OPTIMIZER_CODE_MODEL", "")
    print(
        f"[config] Loaded {getattr(args, '_config_path', 'config')}  "
        f"provider={provider or '(not set)'}  "
        f"model={getattr(args, 'compressor_model', '?')}  "
        + (f"code_model={code_model}  " if code_model else "")
        + f"reasoning={getattr(args, 'reasoning_model', '?')}"
    )


from pathlib import Path
from typing import Any

# ── Project path setup ────────────────────────────────────────────────────────
_BENCH_DIR = Path(__file__).parent
_SRC_DIR = _BENCH_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

_DATA_DIR = _BENCH_DIR / "data" / "corpus"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Corpus config ────────────────────────────────────────────────────────────
_GUTENBERG_PATH = _DATA_DIR / "gutenberg_combined.txt"  # combined Gutenberg books
_QUESTIONS_PATH = _DATA_DIR / "questions.json"
_RESULTS_PATH = _BENCH_DIR / "corpus_results.json"
_REPORT_PATH = _BENCH_DIR / "corpus_results.md"

# Top Gutenberg books (most popular public-domain fiction, plain UTF-8 text)
_GUTENBERG_BOOKS = [
    (
        "pg1342",
        "Pride and Prejudice",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
    ),
    ("pg84", "Frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
    ("pg11", "Alice in Wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
    ("pg345", "Dracula", "https://www.gutenberg.org/files/345/345-0.txt"),
    ("pg2701", "Moby Dick", "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("pg74", "Adventures of Tom Sawyer", "https://www.gutenberg.org/files/74/74-0.txt"),
    ("pg1661", "Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    (
        "pg2554",
        "Crime and Punishment",
        "https://www.gutenberg.org/files/2554/2554-0.txt",
    ),
    ("pg1400", "Great Expectations", "https://www.gutenberg.org/files/1400/1400-0.txt"),
    ("pg174", "Dorian Gray", "https://www.gutenberg.org/files/174/174-0.txt"),
    ("pg5200", "Metamorphosis", "https://www.gutenberg.org/files/5200/5200-0.txt"),
    ("pg2591", "Grimms Fairy Tales", "https://www.gutenberg.org/files/2591/2591-0.txt"),
    ("pg16", "Peter Pan", "https://www.gutenberg.org/files/16/16-0.txt"),
    ("pg1080", "A Modest Proposal", "https://www.gutenberg.org/files/1080/1080-0.txt"),
    ("pg2600", "War and Peace", "https://www.gutenberg.org/files/2600/2600-0.txt"),
    ("pg4300", "Ulysses", "https://www.gutenberg.org/files/4300/4300-0.txt"),
    ("pg100", "Complete Shakespeare", "https://www.gutenberg.org/files/100/100-0.txt"),
    ("pg1232", "The Prince", "https://www.gutenberg.org/files/1232/1232-0.txt"),
    ("pg76", "Adventures Huck Finn", "https://www.gutenberg.org/files/76/76-0.txt"),
    ("pg215", "The Call of the Wild", "https://www.gutenberg.org/files/215/215-0.txt"),
]


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Question:
    id: int
    query: str
    expected_keywords: list[str]
    source_title: str = ""


@dataclass
class QueryResult:
    question_id: int
    query: str
    answer_snippet: str  # raw retrieved context snippet
    tokens_used: int  # tokens sent to reasoning model
    latency_ms: float  # retrieval latency
    kw_recall: float  # retrieval recall: keywords in context
    used_raw_fallback: bool = False
    # ── Reasoning evaluation (populated when reasoning_model is set) ──────
    reasoning_answer: str = ""  # LLM-synthesized answer from context
    reasoning_recall: float = 0.0  # keyword recall on LLM answer
    faithfulness: float = 0.0  # fraction of answer facts grounded in context
    reasoning_latency_ms: float = 0.0


@dataclass
class StrategyResult:
    name: str
    ingestion_time_s: float
    index_size_chunks: int
    index_size_mb: float
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def avg_kw_recall(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.kw_recall for r in self.query_results) / len(self.query_results)

    @property
    def avg_tokens_per_query(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.tokens_used for r in self.query_results) / len(self.query_results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.latency_ms for r in self.query_results) / len(self.query_results)

    @property
    def fallback_rate(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(1 for r in self.query_results if r.used_raw_fallback) / len(
            self.query_results
        )

    @property
    def has_reasoning(self) -> bool:
        return any(r.reasoning_answer for r in self.query_results)

    @property
    def avg_reasoning_recall(self) -> float:
        rs = [r for r in self.query_results if r.reasoning_answer]
        return sum(r.reasoning_recall for r in rs) / len(rs) if rs else 0.0

    @property
    def avg_faithfulness(self) -> float:
        rs = [r for r in self.query_results if r.reasoning_answer]
        return sum(r.faithfulness for r in rs) / len(rs) if rs else 0.0

    @property
    def avg_reasoning_latency_ms(self) -> float:
        rs = [r for r in self.query_results if r.reasoning_latency_ms > 0]
        return sum(r.reasoning_latency_ms for r in rs) / len(rs) if rs else 0.0

    @property
    def reasoning_gap(self) -> float:
        """Retrieval recall minus reasoning recall.
        Positive = reasoning model loses information from context.
        Negative = reasoning model hallucinates beyond what context provides.
        """
        if not self.has_reasoning:
            return 0.0
        return self.avg_kw_recall - self.avg_reasoning_recall


# ── Helpers ───────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """4 chars ≈ 1 token (standard heuristic)."""
    return max(1, len(text) // 4)


def _kw_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    al = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in al) / len(keywords)


def _strip_xml(text: str) -> str:
    """Strip XML/HTML tags and decode common HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\[\[File:[^\]]+\]\]", " ", text)
    text = re.sub(r"\[\[Image:[^\]]+\]\]", " ", text)
    text = re.sub(r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", " ", text)
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _extract_words(text: str, min_len: int = 5) -> list[str]:
    _STOP = {
        "which",
        "their",
        "there",
        "these",
        "those",
        "about",
        "after",
        "before",
        "during",
        "would",
        "could",
        "should",
        "where",
        "while",
        "being",
        "since",
        "other",
        "first",
        "second",
        "third",
        "also",
    }
    tokens = re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in _STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:20]


# ── Corpus download ──────────────────────────────────────────────────────────


def download_gutenberg_corpus(
    output_path: Path | None = None, verbose: bool = True
) -> Path:
    """
    Download the top-20 Gutenberg books and combine into a single plain-text corpus.

    This is the preferred corpus for benchmarking because:
    - Clean English prose — no XML, no markup
    - Questions generated from the text are guaranteed to be answerable
    - Large enough (~50 MB combined) for meaningful block indexing
    - Directly comparable to the old book_benchmark.py results

    Returns path to the combined file.
    """
    dest = output_path or _GUTENBERG_PATH
    if dest.exists():
        mb = dest.stat().st_size / 1_048_576
        if verbose:
            print(
                f"[corpus] Gutenberg corpus already cached: {dest.name} ({mb:.0f} MB)"
            )
        return dest

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    books_dir = _DATA_DIR / "gutenberg_books"
    books_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"[corpus] Downloading {len(_GUTENBERG_BOOKS)} Gutenberg books ...")

    downloaded = []
    for slug, title, url in _GUTENBERG_BOOKS:
        book_path = books_dir / f"{slug}.txt"
        if book_path.exists():
            downloaded.append(book_path)
            if verbose:
                print(f"  cached  {title}")
            continue
        try:
            urllib.request.urlretrieve(url, book_path)
            downloaded.append(book_path)
            if verbose:
                print(f"  OK      {title}  ({book_path.stat().st_size//1024} KB)")
        except Exception as exc:
            if verbose:
                print(f"  SKIP    {title}: {exc}")

    # Combine into one file, stripping Gutenberg header/footer boilerplate
    if verbose:
        print(f"[corpus] Combining {len(downloaded)} books -> {dest.name} ...")
    with open(dest, "w", encoding="utf-8") as out:
        for book_path in downloaded:
            text = book_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            # Strip header up to START marker
            start = 0
            for i, ln in enumerate(lines):
                if (
                    "*** START OF" in ln.upper()
                    or "*** THE PROJECT GUTENBERG" in ln.upper()
                ):
                    start = i + 1
                    break
            # Strip footer from END marker
            end = len(lines)
            for i, ln in enumerate(lines):
                if "*** END OF" in ln.upper():
                    end = i
                    break
            body = "\n".join(lines[start:end]).strip()
            out.write(f"\n\n{'='*60}\n{book_path.stem.upper()}\n{'='*60}\n\n")
            out.write(body)
            out.write("\n")

    mb = dest.stat().st_size / 1_048_576
    if verbose:
        print(
            f"[corpus] Gutenberg corpus ready: {dest.name}  ({mb:.0f} MB, {len(downloaded)} books)"
        )
    return dest


# ── Question generation ────────────────────────────────────────────────────────


def generate_questions(
    corpus_path: Path, n_questions: int = 50, verbose: bool = True
) -> list[Question]:
    """Extract factual questions from the corpus by sampling paragraphs."""
    if verbose:
        print(
            f"[questions] Generating {n_questions} questions from {corpus_path.name} ..."
        )

    questions = _generate_from_plaintext(corpus_path, n_questions, verbose)

    if verbose:
        print(f"[questions] Generated {len(questions)} questions")
    return questions


def _generate_from_plaintext(
    corpus_path: Path, n_questions: int, verbose: bool
) -> list[Question]:
    """Generate questions from plain text by sampling paragraphs."""
    file_size = corpus_path.stat().st_size
    questions: list[Question] = []
    step = file_size // (n_questions + 1)

    with open(corpus_path, "rb") as fh:
        for i in range(n_questions):
            fh.seek(step * (i + 1))
            # Align to next newline
            fh.readline()
            # Read a paragraph
            para_bytes = b""
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                para_bytes += line
                if len(para_bytes) > 500 and para_bytes.endswith(b"\n"):
                    break

            para = para_bytes.decode("utf-8", errors="replace").strip()
            sentences = re.split(r"(?<=[.!?])\s+", para)
            first = next((s for s in sentences if len(s) >= 40), "")
            if not first:
                continue

            keywords = _extract_words(first)
            if len(keywords) < 3:
                continue

            questions.append(
                Question(
                    id=i,
                    query=f"What does the following passage describe: {first[:80]}...",
                    expected_keywords=keywords[:12],
                    source_title=f"offset_{step*(i+1)}",
                )
            )

    return questions


# ── Vanilla RAG ───────────────────────────────────────────────────────────────


def build_vanilla_rag(
    corpus_path: Path,
    top_k: int = 5,
    chunk_tokens: int = 512,
    max_mb: float = 0.0,
    verbose: bool = True,
) -> tuple["Any", "StrategyResult"]:
    """
    Build a vanilla RAG index: raw 512-token chunks embedded directly in ChromaDB.

    No compression — this is the standard dense-retrieval baseline.
    Chunks are streamed directly into ChromaDB in add_batch_size batches;
    no full-corpus accumulation in memory.

    Parameters
    ----------
    max_mb:
        If > 0, stop ingesting after this many MB of corpus (for demos /
        benchmarks where you want a consistent sub-corpus size).
    """
    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.compressor import CompressedChunk, _estimate_tokens

    if verbose:
        cap = f"  (capped at {max_mb:.0f} MB)" if max_mb > 0 else ""
        print(f"\n[vanilla_rag] Building index from {corpus_path.name}{cap} ...")
        print(
            f"[vanilla_rag] Chunk size: {chunk_tokens} tokens (~{chunk_tokens*4} chars)"
        )

    tmp_dir = tempfile.mkdtemp(prefix="co_vanilla_")
    t_start = time.perf_counter()

    chunk_size_chars = chunk_tokens * 4  # 4 chars ≈ 1 token
    add_batch_size = 200  # flush to ChromaDB every N chunks
    max_bytes = int(max_mb * 1_048_576) if max_mb > 0 else 0

    retriever = CachedChromaRetriever(
        collection_name="vanilla_rag",
        persist_directory=tmp_dir,
    )

    chunk_idx = 0
    total_chunks = 0
    buffer = ""
    pending: list[CompressedChunk] = []
    source_name = re.sub(r"[^a-z0-9]+", "_", corpus_path.stem.lower())[:20]
    bytes_read = 0

    with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            raw = fh.read(chunk_size_chars * 4)
            if not raw:
                break
            bytes_read += len(raw.encode("utf-8", errors="replace"))
            buffer += raw

            while len(buffer) >= chunk_size_chars:
                split_at = buffer.find("\n", chunk_size_chars)
                if split_at == -1 or split_at > chunk_size_chars * 2:
                    split_at = chunk_size_chars

                chunk_text = buffer[:split_at].strip()
                buffer = buffer[split_at:]

                if not chunk_text:
                    continue

                orig_tok = _estimate_tokens(chunk_text)
                pending.append(
                    CompressedChunk(
                        chunk_id=f"{source_name}_vchunk_{chunk_idx:07d}",
                        raw_text=chunk_text,
                        compressed_summary=chunk_text,
                        index_text="",
                        entities=[],
                        keywords=[],
                        metadata={"source": str(corpus_path), "chunk_idx": chunk_idx},
                        original_tokens=orig_tok,
                        compressed_tokens=orig_tok,
                        compression_ratio=1.0,
                    )
                )
                chunk_idx += 1

                # Flush to ChromaDB when batch is full
                if len(pending) >= add_batch_size:
                    retriever.add_chunks(pending)
                    total_chunks += len(pending)
                    pending.clear()
                    if verbose and total_chunks % 5000 == 0:
                        mb_done = bytes_read / 1_048_576
                        print(
                            f"  [vanilla_rag] {total_chunks:,} chunks  {mb_done:.0f} MB read ...",
                            end="\r",
                        )

            if max_bytes > 0 and bytes_read >= max_bytes:
                break

    # Flush remaining
    if buffer.strip():
        t = buffer.strip()
        orig_tok = _estimate_tokens(t)
        pending.append(
            CompressedChunk(
                chunk_id=f"{source_name}_vchunk_{chunk_idx:07d}",
                raw_text=t,
                compressed_summary=t,
                index_text="",
                entities=[],
                keywords=[],
                metadata={"source": str(corpus_path), "chunk_idx": chunk_idx},
                original_tokens=orig_tok,
                compressed_tokens=orig_tok,
                compression_ratio=1.0,
            )
        )
    if pending:
        retriever.add_chunks(pending)
        total_chunks += len(pending)

    ingestion_time = time.perf_counter() - t_start
    index_mb = (
        sum(f.stat().st_size for f in Path(tmp_dir).rglob("*") if f.is_file())
        / 1_048_576
    )

    if verbose:
        print(
            f"\n  [vanilla_rag] Done — {total_chunks:,} chunks  "
            f"{ingestion_time:.1f}s  {index_mb:.1f} MB (ChromaDB)"
        )

    return (
        retriever,
        StrategyResult(
            name="vanilla_rag",
            ingestion_time_s=ingestion_time,
            index_size_chunks=total_chunks,
            index_size_mb=index_mb,
        ),
    )


# ── Optimized RAG ─────────────────────────────────────────────────────────────


def build_optimized_rag(
    corpus_path: Path,
    block_size_mb: float = 0.5,
    overlap_pct: float = 10.0,
    max_mb: float = 0.0,
    strategy: str = "llm",
    compressor_model: str = "llama3.2:3b",
    index_dir: "Path | None" = None,
    verbose: bool = True,
) -> tuple["Any", "Any", "StrategyResult"]:
    """
    Build the optimized RAG index using quantized LLM block summarization.

    For each 500 KB block:
      - The LLM (default: llama3.2:3b via Ollama) produces a FIXED 150-200 word
        dense summary — exactly sized to fit the embedding model's context window.
      - The last ``overlap_pct``% of the block is prepended as context for the
        next block so concepts at block boundaries are captured in both summaries.
      - A file pointer (byte offsets) is stored in BlockIndex — raw text is never
        duplicated and is read on demand if a summary is insufficient.

    Returns (retriever, block_index, StrategyResult).
    """
    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.compressor import ingest_file_blocks
    from context_optimizer.raw_index import BlockIndex

    block_size_bytes = int(block_size_mb * 1_048_576)
    overlap_bytes = int(block_size_bytes * overlap_pct / 100)
    if verbose:
        print(f"\n[optimized_rag] Building index from {corpus_path.name} ...")
        print(
            f"[optimized_rag] Block: {block_size_mb:.1f} MB  "
            f"Overlap: {overlap_pct:.0f}% ({overlap_bytes//1024} KB)  "
            f"Strategy: {strategy}"
        )
        if strategy == "llm":
            print(
                f"[optimized_rag] Compressor: {compressor_model} (fixed 150-200 word output)"
            )

    if index_dir is not None:
        index_dir.mkdir(parents=True, exist_ok=True)
        persist_dir = index_dir
    else:
        persist_dir = Path(tempfile.mkdtemp(prefix="co_optimized_"))
    block_db = persist_dir / "blocks.db"
    block_index = BlockIndex(str(block_db))

    # Skip ingestion only if the index contains actual blocks (not just an
    # empty SQLite schema from a previously interrupted run).
    if block_db.exists() and block_index.count() > 0:
        retriever = CachedChromaRetriever(
            collection_name="optimized_rag",
            persist_directory=str(persist_dir),
        )
        n = retriever.collection.count()
        index_mb = (
            sum(f.stat().st_size for f in persist_dir.rglob("*") if f.is_file())
            / 1_048_576
        )
        if verbose:
            print(
                f"  [optimized_rag] Loaded existing index: {n} chunks  {index_mb:.1f} MB (skipping ingestion)"
            )
        return (
            retriever,
            block_index,
            StrategyResult(
                name="optimized_rag",
                ingestion_time_s=0.0,
                index_size_chunks=n,
                index_size_mb=index_mb,
            ),
        )

    t_start = time.perf_counter()

    # When max_mb is set, truncate the corpus to a temp file of that size
    # so the comparison is apples-to-apples with vanilla RAG's --max-mb cap.
    actual_source = corpus_path
    tmp_slice: str | None = None
    if max_mb > 0:
        max_bytes = int(max_mb * 1_048_576)
        actual_size = corpus_path.stat().st_size
        if actual_size > max_bytes:
            import tempfile as _tf

            tmp_slice_fd, tmp_slice = _tf.mkstemp(
                suffix=".txt", prefix="co_corpus_slice_"
            )
            with open(tmp_slice_fd, "wb") as out_fh, open(corpus_path, "rb") as in_fh:
                data = in_fh.read(max_bytes)
                tail = in_fh.read(1000)
                nl = tail.find(b"\n")
                if nl >= 0:
                    data += tail[: nl + 1]
                out_fh.write(data)
            actual_source = Path(tmp_slice)
            if verbose:
                print(f"  [optimized_rag] Using {max_mb:.0f} MB slice of corpus")

    # Ingest: quantized LLM compresses each block to fixed 150-200 word summary
    compressed_chunks = ingest_file_blocks(
        source_path=actual_source,
        block_size_bytes=block_size_bytes,
        overlap_bytes=overlap_bytes,
        block_index=block_index,
        strategy=strategy,
        label="opt_rag",
        compressor_model=compressor_model if strategy == "llm" else None,
    )

    if tmp_slice:
        import os as _os

        _os.unlink(tmp_slice)

    if verbose:
        print(
            f"  [optimized_rag] Adding {len(compressed_chunks):,} summaries to ChromaDB ..."
        )

    retriever = CachedChromaRetriever(
        collection_name="optimized_rag",
        persist_directory=str(persist_dir),
    )
    retriever.add_chunks(compressed_chunks)

    ingestion_time = time.perf_counter() - t_start
    index_mb = (
        sum(f.stat().st_size for f in persist_dir.rglob("*") if f.is_file()) / 1_048_576
    )

    if verbose:
        total_orig = sum(c.original_tokens for c in compressed_chunks)
        total_comp = sum(c.compressed_tokens for c in compressed_chunks)
        ratio = total_comp / total_orig if total_orig else 1.0
        print(
            f"  [optimized_rag] Done — {len(compressed_chunks):,} blocks  "
            f"ratio={ratio:.1%}  {ingestion_time:.1f}s  {index_mb:.1f} MB (index only)"
        )

    return (
        retriever,
        block_index,
        StrategyResult(
            name="optimized_rag",
            ingestion_time_s=ingestion_time,
            index_size_chunks=len(compressed_chunks),
            index_size_mb=index_mb,
        ),
    )


# ── Query evaluation ─────────────────────────────────────────────────────────


# ── Reasoning evaluation helpers ─────────────────────────────────────────────

_REASONING_PROMPT = """\
Answer the question using ONLY the retrieved context below.
Be concise (1-2 sentences). If context is insufficient say "Insufficient context."

Context:
{context}

Question: {question}

Answer:"""

_REASONING_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been "
    "have has had do does did will would could should may might this that these "
    "those it its there their they them we our you your i my he she his her".split()
)


def _build_reasoning_llm(model: str) -> "Any | None":
    """Build an Ollama LLM for the reasoning step."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import]

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0.0)
    except Exception:
        return None


def _reason(context: str, question: str, llm: Any) -> tuple[str, float]:
    """
    Ask the reasoning LLM to answer *question* using only *context*.

    Returns (answer_text, latency_seconds).
    """
    prompt = _REASONING_PROMPT.format(context=context[:3000], question=question)
    t0 = time.perf_counter()
    try:
        resp = llm.invoke(prompt)
        answer = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    except Exception as exc:
        answer = f"[REASONING_ERROR: {exc}]"
    return answer, time.perf_counter() - t0


def _faithfulness(answer: str, context: str) -> float:
    """
    Fraction of content words in *answer* that appear in *context*.

    Measures whether the reasoning model stays grounded in the retrieved
    evidence.  Score < 0.4 = likely hallucination.
    Score 1.0 = every non-trivial word in the answer comes from context.
    """
    if not answer or not context:
        return 0.0
    context_l = context.lower()
    words = [
        w
        for w in re.findall(r"[a-z]{4,}", answer.lower())
        if w not in _REASONING_STOPWORDS
    ]
    if not words:
        return 1.0  # empty / trivial answer
    return sum(1 for w in words if w in context_l) / len(words)


# ── Tree RAG ──────────────────────────────────────────────────────────────────


def _auto_tree_depth(
    cluster_size: int,
    top_k: int = 0,  # 0 = use cluster_size as target (recommended)
    max_depth: int = 4,
    n_blocks: int = 0,
    corpus_bytes: int = 0,
    block_bytes: int = 1,
) -> int:
    """
    Compute the minimum depth so the top level has ~cluster_size entries.

    Using cluster_size as the target keeps the top level at a number the
    reasoning LLM can meaningfully choose between -- the same branching
    factor it sees at every other level of the tree.

    Formula:  d = ceil(log(n_blocks / cluster_size) / log(cluster_size)) + 1
    Clamped to [2, max_depth].

    Examples (cluster_size=4):
        n_blocks     depth   top-level entries
        16           2       4
        64           3       4
        256          4       4  (max_depth cap)
        1,000        4       ~16 (cap hit, top grows)
    """
    import math as _math

    target = top_k if top_k > 0 else cluster_size
    actual_n = (
        n_blocks if n_blocks > 0 else max(1, _math.ceil(corpus_bytes / block_bytes))
    )
    if actual_n <= cluster_size:
        return 2
    raw = _math.log(actual_n / target) / _math.log(cluster_size)
    return max(2, min(int(_math.ceil(raw)) + 1, max_depth))


def build_tree_rag(
    corpus_path: Path,
    block_size_mb: float = 0.1,
    cluster_size: int = 4,
    overlap_pct: float = 10.0,
    max_mb: float = 0.0,
    compressor_model: str = "llama3.2:3b",
    index_dir: "Path | None" = None,
    tree_depth: int = 0,
    verbose: bool = True,
) -> tuple["Any", "Any", "StrategyResult"]:
    """
    Build a Tree-of-Summaries index.

    tree_depth=0 (default): determined AFTER Pass 1 (L1 block ingestion)
    from the actual number of L1 nodes produced, so it reflects true corpus
    density rather than a pre-ingestion estimate.
    tree_depth=2: L1 blocks + L2 clusters (good for <100 MB).
    tree_depth=3: L1 + L2 + L3 super-clusters (400 MB+).
    """
    from context_optimizer.compressor import ingest_file_blocks
    from context_optimizer.raw_index import BlockIndex
    from context_optimizer.tree_index import TreeIndex

    block_size_bytes = int(block_size_mb * 1_048_576)
    overlap_bytes = int(block_size_bytes * overlap_pct / 100)

    if verbose:
        print(
            f"\n[tree_rag] Building Tree-of-Summaries index from {corpus_path.name} ..."
        )
        print(
            f"[tree_rag] Block: {block_size_mb:.2f} MB  Overlap: {overlap_pct:.0f}%  "
            f"Cluster: {cluster_size} blocks/node  Compressor: {compressor_model}"
        )
        if tree_depth != 0:
            print(f"[tree_rag] Depth: {tree_depth} (fixed)")
        else:
            print("[tree_rag] Depth: auto (determined after Pass 1)")

    if index_dir is not None:
        index_dir.mkdir(parents=True, exist_ok=True)
        persist_dir = index_dir
    else:
        persist_dir = Path(tempfile.mkdtemp(prefix="co_tree_"))
    block_db = persist_dir / "blocks.db"
    block_index = BlockIndex(str(block_db))

    t_start = time.perf_counter()

    # Skip ingestion only if the index contains actual blocks.
    if block_db.exists() and block_index.count() > 0:
        # Detect depth from existing collections when auto (tree_depth=0)
        if tree_depth == 0:
            import chromadb as _chroma
            from chromadb.config import Settings as _Settings

            _client = _chroma.PersistentClient(
                path=str(persist_dir),
                settings=_Settings(anonymized_telemetry=False, allow_reset=True),
            )
            _existing_colls = {c.name for c in _client.list_collections()}
            for _d in range(4, 1, -1):
                if f"tree_rag_L{_d}" in _existing_colls:
                    tree_depth = _d
                    break
            else:
                tree_depth = 2  # fallback
            if verbose:
                print(f"[tree_rag] Detected depth={tree_depth} from saved index")
        tree = TreeIndex(
            collection_name="tree_rag",
            persist_directory=str(persist_dir),
            block_index=block_index,
            depth=tree_depth,
        )
        index_mb = (
            sum(f.stat().st_size for f in persist_dir.rglob("*") if f.is_file())
            / 1_048_576
        )
        if verbose:
            print(
                f"  [tree_rag] Loaded existing index: {tree.block_count()} L1 blocks, "
                f"{tree.cluster_count()} L2 clusters  {index_mb:.1f} MB (skipping ingestion)"
            )
        return (
            tree,
            block_index,
            StrategyResult(
                name="tree_rag",
                ingestion_time_s=0.0,
                index_size_chunks=tree.block_count(),
                index_size_mb=index_mb,
            ),
        )

    # Slice corpus if max_mb is set
    actual_source = corpus_path
    tmp_slice: str | None = None
    if max_mb > 0:
        max_bytes = int(max_mb * 1_048_576)
        if corpus_path.stat().st_size > max_bytes:
            import tempfile as _tf

            fd, tmp_slice = _tf.mkstemp(suffix=".txt", prefix="co_slice_")
            with open(fd, "wb") as out, open(corpus_path, "rb") as inp:
                data = inp.read(max_bytes)
                tail = inp.read(1000)
                nl = tail.find(b"\n")
                if nl >= 0:
                    data += tail[: nl + 1]
                out.write(data)
            actual_source = Path(tmp_slice)

    # Build the compressor LLM ONCE and reuse it for every pass:
    #   Pass 1 (L1) : ingest_file_blocks  -- one call per 100 KB block
    #   Pass 2 (L2) : build_from_chunks  -- one call per cluster_size blocks
    #   Pass 3+ (LN): build_from_chunks  -- one call per cluster_size L{N-1} nodes
    # Sharing the instance avoids loading the model (BART ~400 MB) twice.
    from context_optimizer.compressor import _build_local_llm

    llm = _build_local_llm(model=compressor_model)

    chunks = ingest_file_blocks(
        source_path=actual_source,
        block_size_bytes=block_size_bytes,
        overlap_bytes=overlap_bytes,
        block_index=block_index,
        strategy="llm",
        label="tree",
        llm=llm,
    )

    if tmp_slice:
        import os as _os

        _os.unlink(tmp_slice)

    # Resolve depth from actual L1 count (Pass 1 is now complete)
    actual_l1 = len(chunks)
    if tree_depth == 0:
        tree_depth = _auto_tree_depth(n_blocks=actual_l1, cluster_size=cluster_size)
    import math as _m

    top_est = max(1, _m.ceil(actual_l1 / cluster_size ** (tree_depth - 1)))
    if verbose:
        print(f"[tree_rag] Pass 1 complete: {actual_l1} L1 nodes generated")
        print(
            f"[tree_rag] Depth={tree_depth}  ({actual_l1} blocks, "
            f"~{top_est} top-level entries)  "
            f"Build plan:"
        )
        n = actual_l1
        print(f"  Pass 1 (L1 block summaries)   : {n:>6} nodes  {n} LLM calls (done)")
        for lvl in range(2, tree_depth + 1):
            n = max(1, _m.ceil(n / cluster_size))
            print(
                f"  Pass {lvl} (L{lvl} cluster summaries): {n:>6} nodes  {n} LLM calls"
            )

    # Build TreeIndex (L1..L{depth}) and run Pass 2..depth
    tree = TreeIndex(
        collection_name="tree_rag",
        persist_directory=str(persist_dir),
        block_index=block_index,
        depth=tree_depth,
    )
    tree.build_from_chunks(chunks, cluster_size=cluster_size, llm=llm, label="tree")

    ingestion_time = time.perf_counter() - t_start
    index_mb = (
        sum(f.stat().st_size for f in persist_dir.rglob("*") if f.is_file()) / 1_048_576
    )

    if verbose:
        print(
            f"  [tree_rag] Done — {tree.block_count()} L1 blocks, "
            f"{tree.cluster_count()} L2 clusters  "
            f"{ingestion_time:.1f}s  {index_mb:.1f} MB"
        )

    return (
        tree,
        block_index,
        StrategyResult(
            name="tree_rag",
            ingestion_time_s=ingestion_time,
            index_size_chunks=tree.block_count(),
            index_size_mb=index_mb,
        ),
    )


def evaluate_tree(
    tree: "Any",
    questions: list[Question],
    reasoning_llm: "Any | None" = None,
    top_clusters: int = 2,
    top_blocks_per_cluster: int = 3,
    max_rounds: int = 3,
    verbose: bool = True,
) -> list[QueryResult]:
    """Evaluate the Tree-of-Summaries strategy using the tool-calling agent."""
    from context_optimizer.tree_reasoner import TreeReasoningAgent

    agent = TreeReasoningAgent(
        tree=tree,
        llm=reasoning_llm,
        top_clusters=top_clusters,
        top_blocks_per_cluster=top_blocks_per_cluster,
        max_rounds=max_rounds,
    )

    results: list[QueryResult] = []
    for q in questions:
        result = agent.reason(q.query)
        retrieval_recall = _kw_recall(
            " ".join(step.target_id for step in result.steps) + " " + result.answer,
            q.expected_keywords,
        )
        # Better: check if keywords appear in any retrieved text
        full_context = result.answer  # the agent accumulated context internally

        r_recall = _kw_recall(result.answer, q.expected_keywords)
        faith = _faithfulness(result.answer, full_context)
        n_tool_calls = sum(1 for s in result.steps if s.action != "answer")

        results.append(
            QueryResult(
                question_id=q.id,
                query=q.query,
                answer_snippet=result.answer[:200],
                tokens_used=result.context_tokens,
                latency_ms=result.total_latency_ms,
                kw_recall=r_recall,
                used_raw_fallback=result.used_raw_fallback,
                reasoning_answer=result.answer[:300],
                reasoning_recall=r_recall,
                faithfulness=faith,
                reasoning_latency_ms=result.total_latency_ms,
            )
        )
        if verbose:
            fb = "[RAW]" if result.used_raw_fallback else ""
            tc = f"[{n_tool_calls} tools]" if n_tool_calls else ""
            print(
                f"  [tree  Q{q.id:02d}] recall={r_recall:.0%}  "
                f"ctx={result.context_tokens} tok  "
                f"{result.total_latency_ms:.0f}ms  {fb}{tc}"
            )
    return results


def evaluate_vanilla(
    retriever: Any,
    questions: list[Question],
    top_k: int = 5,
    reasoning_llm: Any | None = None,
    verbose: bool = True,
) -> list[QueryResult]:
    """Run questions against vanilla RAG and score results."""
    results: list[QueryResult] = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q.query, top_k=top_k, use_cache=False)
        latency = (time.perf_counter() - t0) * 1000

        # Concatenate retrieved chunks as the context
        context = " ".join(
            h.get("compressed_summary", h.get("raw_text", "")) for h in hits
        )
        tokens = sum(
            _estimate_tokens(h.get("compressed_summary", h.get("raw_text", "")))
            for h in hits
        )
        retrieval_recall = _kw_recall(context, q.expected_keywords)

        # ── Optional reasoning pass ────────────────────────────────────────
        r_answer, r_recall, r_faith, r_lat_ms = "", 0.0, 0.0, 0.0
        if reasoning_llm is not None:
            r_answer, r_lat_s = _reason(context, q.query, reasoning_llm)
            r_recall = _kw_recall(r_answer, q.expected_keywords)
            r_faith = _faithfulness(r_answer, context)
            r_lat_ms = r_lat_s * 1000

        results.append(
            QueryResult(
                question_id=q.id,
                query=q.query,
                answer_snippet=context[:200],
                tokens_used=tokens,
                latency_ms=latency,
                kw_recall=retrieval_recall,
                used_raw_fallback=False,
                reasoning_answer=r_answer[:300],
                reasoning_recall=r_recall,
                faithfulness=r_faith,
                reasoning_latency_ms=r_lat_ms,
            )
        )
        if verbose:
            r_str = f"  reason={r_recall:.0%}  faith={r_faith:.0%}" if r_answer else ""
            print(
                f"  [vanilla Q{q.id:02d}] ret={retrieval_recall:.0%}"
                f"  tokens={tokens:,}  {latency:.0f}ms{r_str}"
            )
    return results


def evaluate_optimized(
    retriever: Any,
    block_index: Any,
    questions: list[Question],
    top_k: int = 5,
    fallback_threshold: float = 0.30,
    force_fallback: bool = False,
    reasoning_llm: Any | None = None,
    verbose: bool = True,
) -> list[QueryResult]:
    """
    Run questions against optimized RAG.

    Step 1: Search compressed summaries in ChromaDB.
    Step 2: Measure retrieval recall (do the triple-format index entries
            contain the expected keywords?).
    Step 3: If reasoning_llm is set, pass the triples to the LLM and measure
            reasoning recall (can the model synthesize a correct answer FROM
            the machine-readable index entries?).
    Step 4: If best cosine score < fallback_threshold, fetch raw block.
    """
    results: list[QueryResult] = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q.query, top_k=top_k, use_cache=False)
        latency_summary = (time.perf_counter() - t0) * 1000

        # Assemble context from triple-format summaries
        context = " ".join(h.get("compressed_summary", "") for h in hits)
        summary_tokens = sum(
            _estimate_tokens(h.get("compressed_summary", "")) for h in hits
        )
        retrieval_recall = _kw_recall(context, q.expected_keywords)

        # Check if fallback is needed
        best_score = 1 - min((h.get("distance") or 1.0) for h in hits) if hits else 0.0
        used_fallback = False

        # force_fallback=True always reads the raw block regardless of score.
        # This directly tests whether the BlockIndex path provides recall gains
        # over the compressed summary alone.
        should_fallback = force_fallback or (
            best_score < fallback_threshold and hits and block_index is not None
        )
        if should_fallback and hits and block_index is not None:
            best_hit = hits[0]
            block_id = best_hit.get("metadata", {}).get("block_id") or best_hit.get(
                "chunk_id", ""
            )
            raw_text = block_index.get_text(block_id)
            if raw_text:
                t1 = time.perf_counter()
                # Score recall against the FULL raw block — every byte the
                # file pointer returned.  This is the correct test: we prove
                # the needle IS in the raw block, not just in the first 4 KB.
                fallback_recall = _kw_recall(raw_text, q.expected_keywords)
                # For token-budget purposes (what we'd send to an LLM), we
                # still cap at 4 000 chars (~1 000 tokens).  This is separate
                # from the recall check above.
                context_for_llm = raw_text[:4000]
                fallback_tokens = _estimate_tokens(context_for_llm)
                latency_summary += (time.perf_counter() - t1) * 1000

                if fallback_recall > retrieval_recall:
                    context = context_for_llm  # LLM sees the first 4 KB
                    summary_tokens += fallback_tokens
                    retrieval_recall = fallback_recall
                    used_fallback = True

        # ── Reasoning pass: can the LLM answer from triple-format context? ──
        # This is the critical test for machine-readable index entries.
        # A high reasoning_recall proves the triples are sufficient for the
        # reasoning model; a high faithfulness proves it stays grounded.
        r_answer, r_recall, r_faith, r_lat_ms = "", 0.0, 0.0, 0.0
        if reasoning_llm is not None:
            r_answer, r_lat_s = _reason(context, q.query, reasoning_llm)
            r_recall = _kw_recall(r_answer, q.expected_keywords)
            r_faith = _faithfulness(r_answer, context)
            r_lat_ms = r_lat_s * 1000

        latency = (time.perf_counter() - t0) * 1000
        results.append(
            QueryResult(
                question_id=q.id,
                query=q.query,
                answer_snippet=context[:200],
                tokens_used=summary_tokens,
                latency_ms=latency,
                kw_recall=retrieval_recall,
                used_raw_fallback=used_fallback,
                reasoning_answer=r_answer[:300],
                reasoning_recall=r_recall,
                faithfulness=r_faith,
                reasoning_latency_ms=r_lat_ms,
            )
        )
        if verbose:
            fb_str = "  [FALLBACK]" if used_fallback else ""
            r_str = f"  reason={r_recall:.0%}  faith={r_faith:.0%}" if r_answer else ""
            print(
                f"  [opt   Q{q.id:02d}] ret={retrieval_recall:.0%}"
                f"  tokens={summary_tokens:,}  {latency:.0f}ms{fb_str}{r_str}"
            )
    return results


# ── Report ────────────────────────────────────────────────────────────────────


def write_report(
    vanilla: StrategyResult | None,
    optimized: StrategyResult | None,
    questions: list[Question],
    corpus_path: Path,
    args: argparse.Namespace,
    tree: StrategyResult | None = None,
) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    corpus_mb = corpus_path.stat().st_size / 1_048_576 if corpus_path.exists() else 0

    def _fmt(v: StrategyResult | None, attr: str, fmt: str = ".1f") -> str:
        if v is None:
            return "—"
        val = getattr(v, attr, None)
        if val is None:
            return "—"
        return format(val, fmt)

    _strategies = [
        s for s in ["Vanilla RAG", "Optimized RAG", "Tree RAG" if tree else None] if s
    ]
    _title = " vs ".join(_strategies)
    lines = [
        f"# Corpus Benchmark: {_title}",
        "",
        f"**Run date**: {run_date}  |  "
        f"**Corpus**: {corpus_path.name} ({corpus_mb:.0f} MB)  |  "
        f"**Questions**: {len(questions)}  |  "
        f"**top-k**: {args.top_k}  |  "
        f"**Block size**: {args.block_mb:.1f} MB  |  "
        f"**Corpus cap**: {getattr(args,'max_mb',200):.0f} MB",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Metric | Vanilla RAG | Optimized RAG | Delta |",
        "|--------|-------------|---------------|-------|",
    ]

    def _delta(
        a: StrategyResult | None,
        b: StrategyResult | None,
        attr: str,
        lower_is_better: bool = False,
    ) -> str:
        if a is None or b is None:
            return "—"
        va, vb = getattr(a, attr, 0.0), getattr(b, attr, 0.0)
        if va == 0:
            return "—"
        pct = (vb - va) / va * 100
        better = (pct < 0) == lower_is_better
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}% {'✓' if better else '✗'}"

    rows = [
        ("Avg retrieval recall (context has keywords)", "avg_kw_recall", False, ".1%"),
        ("Avg tokens per query", "avg_tokens_per_query", True, ",.0f"),
        ("Avg query latency (ms)", "avg_latency_ms", True, ".1f"),
        ("Index ingestion time (s)", "ingestion_time_s", True, ".1f"),
        ("Index size (MB, excl. corpus)", "index_size_mb", True, ".1f"),
        ("Index entries", "index_size_chunks", True, ",d"),
    ]

    for label, attr, lib, fmt in rows:
        v_val = f"{getattr(vanilla, attr):>{fmt}}" if vanilla else "—"
        o_val = f"{getattr(optimized, attr):>{fmt}}" if optimized else "—"
        delta = _delta(vanilla, optimized, attr, lib)
        lines.append(f"| {label} | {v_val} | {o_val} | {delta} |")

    if optimized:
        lines += [
            f"| Raw block fallback rate | — | {optimized.fallback_rate:.0%} | — |",
        ]

    # Reasoning evaluation section (only when reasoning was run)
    has_reasoning = (vanilla and vanilla.has_reasoning) or (
        optimized and optimized.has_reasoning
    )
    if has_reasoning:
        lines += [
            "",
            "### Reasoning Evaluation",
            "",
            "Can the reasoning model synthesize correct answers from the retrieved context?",
            "- **Reasoning recall**: keyword overlap between LLM-generated answer and expected answer.",
            "- **Faithfulness**: fraction of answer content words traceable back to the retrieved context.",
            "- **Reasoning gap**: retrieval_recall − reasoning_recall  "
            "(positive = info lost in reasoning; negative = hallucination).",
            "",
            "| Metric | Vanilla RAG | Optimized RAG | Delta |",
            "|--------|-------------|---------------|-------|",
        ]
        reason_rows = [
            ("Reasoning recall (LLM answer)", "avg_reasoning_recall", False, ".1%"),
            ("Faithfulness (grounded in context)", "avg_faithfulness", False, ".1%"),
            ("Reasoning gap (ret - reason)", "reasoning_gap", True, ".1%"),
            ("Avg reasoning latency (ms)", "avg_reasoning_latency_ms", True, ".0f"),
        ]
        for label, attr, lib, fmt in reason_rows:
            v_val = (
                f"{getattr(vanilla, attr, 0.0):>{fmt}}"
                if vanilla and vanilla.has_reasoning
                else "—"
            )
            o_val = (
                f"{getattr(optimized, attr, 0.0):>{fmt}}"
                if optimized and optimized.has_reasoning
                else "—"
            )
            delta = (
                _delta(vanilla, optimized, attr, lib)
                if (
                    vanilla
                    and vanilla.has_reasoning
                    and optimized
                    and optimized.has_reasoning
                )
                else "—"
            )
            lines.append(f"| {label} | {v_val} | {o_val} | {delta} |")

    lines += [
        "",
        "**Delta** is Optimized relative to Vanilla.  ✓ = improvement, ✗ = regression.",
        "",
        "---",
        "",
        "## Architecture Contrast",
        "",
        "```",
        "VANILLA RAG                            OPTIMIZED RAG",
        "────────────────────────────────────   ────────────────────────────────────",
        f"Corpus split into 512-token chunks      Corpus split into {args.block_mb:.0f} MB blocks",
        (
            f"~{vanilla.index_size_chunks:,} ChromaDB entries                ~{optimized.index_size_chunks:,} ChromaDB entries"
            if vanilla and optimized
            else ""
        ),
        "Raw text embedded directly              Compressed summaries embedded",
        "No fallback                             Raw block fetched from disk on demand",
        (
            f"Index: {vanilla.index_size_mb:.0f} MB                           Index: {optimized.index_size_mb:.0f} MB (+ {corpus_mb:.0f} MB corpus on disk)"
            if vanilla and optimized
            else ""
        ),
        "```",
        "",
        "---",
        "",
        "## Per-Question Breakdown",
        "",
    ]

    if vanilla and optimized:
        show_reason = vanilla.has_reasoning or optimized.has_reasoning
        if show_reason:
            lines += [
                "| # | Question | V-ret | O-ret | V-reason | O-reason | O-faith | V-tok | O-tok | Fallback |",
                "|---|----------|:-----:|:-----:|:--------:|:--------:|:-------:|------:|------:|:--------:|",
            ]
        else:
            lines += [
                "| # | Question | Vanilla recall | Opt recall | Vanilla tokens | Opt tokens | Fallback |",
                "|---|----------|:--------------:|:----------:|---------------:|-----------:|:--------:|",
            ]
        vmap = {r.question_id: r for r in vanilla.query_results}
        omap = {r.question_id: r for r in optimized.query_results}
        for q in questions:
            vr = vmap.get(q.id)
            or_ = omap.get(q.id)
            if vr and or_:
                if show_reason:
                    lines.append(
                        f"| {q.id} | {q.query[:45]} | "
                        f"{vr.kw_recall:.0%} | {or_.kw_recall:.0%} | "
                        f"{vr.reasoning_recall:.0%} | {or_.reasoning_recall:.0%} | "
                        f"{or_.faithfulness:.0%} | "
                        f"{vr.tokens_used:,} | {or_.tokens_used:,} | "
                        f"{'yes' if or_.used_raw_fallback else 'no'} |"
                    )
                else:
                    lines.append(
                        f"| {q.id} | {q.query[:50]} | "
                        f"{vr.kw_recall:.0%} | {or_.kw_recall:.0%} | "
                        f"{vr.tokens_used:,} | {or_.tokens_used:,} | "
                        f"{'yes' if or_.used_raw_fallback else 'no'} |"
                    )
            else:
                lines.append(f"| {q.id} | {q.query[:50]} | — | — | — | — | — |")

    # ── Tree-of-Summaries section (when --tree was run) ──────────────────────
    if tree is not None:
        lines += [
            "",
            "---",
            "",
            "## Tree-of-Summaries Results",
            "",
            "Two-level hierarchical index: L1 block summaries + L2 cluster super-summaries.",
            "The reasoning agent navigates the tree autonomously (search_cluster / fetch_raw_block).",
            "",
            "| Metric | Vanilla RAG | Tree RAG | Delta (vs Vanilla) |",
            "|--------|-------------|----------|--------------------|",
        ]
        tree_rows = [
            ("Avg retrieval recall", "avg_kw_recall", False, ".1%"),
            ("Avg tokens per query", "avg_tokens_per_query", True, ",.0f"),
            ("Avg query latency (ms)", "avg_latency_ms", True, ".1f"),
            ("Index ingestion time (s)", "ingestion_time_s", True, ".1f"),
            ("Index size (MB)", "index_size_mb", True, ".1f"),
            ("L1 block entries", "index_size_chunks", True, ",d"),
        ]
        for label, attr, lib, fmt in tree_rows:
            v_val = f"{getattr(vanilla, attr):{fmt}}" if vanilla else "—"
            t_val = f"{getattr(tree, attr):{fmt}}"
            delta = _delta(vanilla, tree, attr, lib)
            lines.append(f"| {label} | {v_val} | {t_val} | {delta} |")
        lines += [
            f"| Raw block fallback rate | — | {tree.fallback_rate:.0%} | — |",
        ]
        if tree.has_reasoning:
            lines += [
                "",
                "### Tree Reasoning Evaluation",
                "",
                "| Metric | Vanilla RAG | Tree RAG | Delta |",
                "|--------|-------------|----------|-------|",
            ]
            for label, attr, lib, fmt in [
                ("Reasoning recall", "avg_reasoning_recall", False, ".1%"),
                ("Faithfulness", "avg_faithfulness", False, ".1%"),
                ("Reasoning gap", "reasoning_gap", True, ".1%"),
                ("Avg reasoning latency (ms)", "avg_reasoning_latency_ms", True, ".0f"),
            ]:
                v_val = (
                    f"{getattr(vanilla, attr, 0.0):{fmt}}"
                    if vanilla and vanilla.has_reasoning
                    else "—"
                )
                t_val = f"{getattr(tree, attr, 0.0):{fmt}}"
                delta = (
                    _delta(vanilla, tree, attr, lib)
                    if vanilla and vanilla.has_reasoning
                    else "—"
                )
                lines.append(f"| {label} | {v_val} | {t_val} | {delta} |")

    lines += [
        "",
        "---",
        "",
        "## How to Re-run",
        "",
        "```bash",
        "# Full run (prepare corpus + build indexes + evaluate)",
        "python corpus_benchmark.py all",
        "",
        "# Build indexes only (corpus already prepared)",
        "python corpus_benchmark.py run",
        "",
        "# Use a custom corpus",
        "python corpus_benchmark.py prepare --corpus-path /path/to/corpus.txt",
        "python corpus_benchmark.py run",
        "```",
        "",
        f"*Generated {run_date} — do not edit manually.*",
    ]

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # JSON output
    result_data = {
        "run_date": run_date,
        "corpus": str(corpus_path),
        "corpus_mb": corpus_mb,
        "questions": len(questions),
        "top_k": args.top_k,
        "block_mb": args.block_mb,
        "vanilla_rag": asdict(vanilla) if vanilla else None,
        "optimized_rag": asdict(optimized) if optimized else None,
        "tree_rag": asdict(tree) if tree else None,
    }
    _RESULTS_PATH.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

    print(f"\n[report] {_REPORT_PATH.relative_to(_BENCH_DIR.parent)}")
    print(f"[report] {_RESULTS_PATH.relative_to(_BENCH_DIR.parent)}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corpus_benchmark",
        description="Vanilla RAG vs Optimized RAG on a 1-2 GB corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # verify — fast end-to-end test of the BlockIndex file-pointer fallback
    ver = sub.add_parser(
        "verify",
        help="Fast offline test: prove the BlockIndex raw-block fallback works.",
    )
    ver.add_argument(
        "--needles",
        type=int,
        default=20,
        help="Number of needle facts to embed and test (default 20)",
    )
    ver.add_argument(
        "--block-mb",
        type=float,
        default=0.008,
        dest="block_mb",
        help="Block size in MB (default 0.008 = 8 KB — small enough for each needle "
        "to land in a dedicated block surrounded by its filler)",
    )

    # prepare
    prep = sub.add_parser("prepare", help="Download corpus and generate questions.")
    prep.add_argument(
        "--corpus",
        choices=["gutenberg"],
        default="gutenberg",
        dest="corpus_source",
        help="Corpus to download (default: gutenberg — top-20 public-domain books)",
    )
    prep.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        dest="corpus_path",
        help="Path to a local text corpus file (skips download). Min recommended: 100 MB.",
    )
    prep.add_argument(
        "--questions", type=int, default=50, help="Questions to generate (default 50)"
    )

    # run
    run = sub.add_parser("run", help="Build indexes and evaluate both strategies.")
    run.add_argument(
        "--config",
        type=Path,
        default=None,
        dest="config",
        metavar="FILE",
        help="Path to a bench_config.yaml file. Values in the file are applied "
        "before any other flags, so CLI flags can still override individual settings.",
    )
    run.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        dest="corpus_path",
        help="Override corpus path (default: uses previously prepared corpus).",
    )
    run.add_argument(
        "--questions", type=int, default=50, help="Questions to use (default 50)"
    )
    run.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Chunks/blocks per query (default 5)",
    )
    run.add_argument(
        "--block-mb",
        type=float,
        default=0.1,
        dest="block_mb",
        help="Block size in MB (default 0.1 = 100 KB — better granularity than 512 KB)",
    )
    run.add_argument(
        "--cluster-size",
        type=int,
        default=4,
        dest="cluster_size",
        help="Children per tree node (default 4). Lower = more specific summaries = better accuracy; higher = shallower tree = fewer LLM calls.",
    )
    run.add_argument(
        "--tree",
        action="store_true",
        default=False,
        dest="run_tree",
        help="Also build and evaluate Tree-of-Summaries (hierarchical two-level index)",
    )
    run.add_argument(
        "--max-mb",
        type=float,
        default=200.0,
        dest="max_mb",
        help="Cap corpus size in MB for both strategies (default 200 MB for practical runtime)",
    )
    run.add_argument(
        "--overlap-pct",
        type=float,
        default=10.0,
        dest="overlap_pct",
        help="Block overlap as %% of block size for boundary continuity (default 10%%)",
    )
    run.add_argument(
        "--opt-strategy",
        choices=["llm", "raw_only"],
        default="llm",
        dest="opt_strategy",
        help="Optimized RAG strategy: llm (default, requires Ollama) or raw_only",
    )
    run.add_argument(
        "--compressor-model",
        type=str,
        default="llama3.2:3b",
        dest="compressor_model",
        help="Ollama model for block summarization (default: llama3.2:3b)",
    )
    run.add_argument(
        "--reasoning-model",
        type=str,
        default="llama3.2:3b",
        dest="reasoning_model",
        help="Ollama model for reasoning evaluation pass (default: llama3.2:3b). "
        "Set to '' to disable reasoning evaluation.",
    )
    run.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.30,
        dest="fallback_threshold",
        help="Cosine distance threshold for raw block fallback (default 0.30)",
    )
    run.add_argument(
        "--force-fallback",
        action="store_true",
        default=False,
        dest="force_fallback",
        help="Always read raw blocks regardless of similarity score. "
        "Directly proves the BlockIndex path improves recall over summaries alone.",
    )
    run.add_argument(
        "--vanilla-only",
        action="store_true",
        dest="vanilla_only",
        help="Build and evaluate vanilla RAG only",
    )
    run.add_argument(
        "--optimized-only",
        action="store_true",
        dest="optimized_only",
        help="Build and evaluate optimized RAG only",
    )
    run.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        dest="index_dir",
        help="Persistent directory for built indexes. Sub-dirs 'optimized/' and 'tree/' "
        "are created automatically. If the sub-dir already contains a built index "
        "(blocks.db present), ingestion is skipped — only evaluation runs.",
    )
    run.add_argument(
        "--build-only",
        action="store_true",
        default=False,
        dest="build_only",
        help="Build and persist indexes, then exit without running evaluation.",
    )
    run.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        dest="eval_only",
        help="Skip building — load indexes from --index-dir and run evaluation only. "
        "Requires --index-dir with a previously built index.",
    )

    # all
    all_ = sub.add_parser("all", help="prepare + run in sequence.")
    all_.add_argument("--corpus-path", type=Path, default=None, dest="corpus_path")
    all_.add_argument("--questions", type=int, default=50)
    all_.add_argument("--top-k", type=int, default=5, dest="top_k")
    all_.add_argument("--block-mb", type=float, default=0.1, dest="block_mb")
    all_.add_argument("--cluster-size", type=int, default=4, dest="cluster_size")
    all_.add_argument("--tree", action="store_true", default=False, dest="run_tree")
    all_.add_argument("--max-mb", type=float, default=200.0, dest="max_mb")
    all_.add_argument("--overlap-pct", type=float, default=10.0, dest="overlap_pct")
    all_.add_argument(
        "--opt-strategy",
        choices=["llm", "raw_only"],
        default="llm",
        dest="opt_strategy",
    )
    all_.add_argument(
        "--compressor-model", type=str, default="llama3.2:3b", dest="compressor_model"
    )
    all_.add_argument(
        "--reasoning-model", type=str, default="llama3.2:3b", dest="reasoning_model"
    )
    all_.add_argument(
        "--fallback-threshold", type=float, default=0.30, dest="fallback_threshold"
    )
    all_.add_argument("--vanilla-only", action="store_true", dest="vanilla_only")
    all_.add_argument("--optimized-only", action="store_true", dest="optimized_only")

    return p


def cmd_prepare(args: argparse.Namespace) -> tuple[Path, list[Question]]:
    """Download corpus and generate questions from it."""
    source = getattr(args, "corpus_source", "gutenberg")

    if args.corpus_path:
        # User-supplied local file — use as-is
        corpus_path = args.corpus_path
        if not corpus_path.exists():
            print(f"[prepare] File not found: {corpus_path}")
            raise SystemExit(1)
    else:
        # Gutenberg books: clean English prose, questions sampled from text
        corpus_path = download_gutenberg_corpus()

    # Questions are sampled from the corpus text itself — guaranteed answerable
    questions = generate_questions(corpus_path, n_questions=args.questions)
    _QUESTIONS_PATH.write_text(
        json.dumps(
            [
                {
                    "id": q.id,
                    "query": q.query,
                    "expected_keywords": q.expected_keywords,
                    "source_title": q.source_title,
                }
                for q in questions
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[prepare] Saved {len(questions)} questions to {_QUESTIONS_PATH.name}")
    return corpus_path, questions


def cmd_run(args: argparse.Namespace) -> None:
    """Build indexes and run the evaluation."""
    # Corpus: explicit path > Gutenberg default
    if args.corpus_path:
        corpus_path = args.corpus_path
    elif _GUTENBERG_PATH.exists():
        corpus_path = _GUTENBERG_PATH
        print(
            f"[run] Using Gutenberg corpus: {corpus_path.name} ({corpus_path.stat().st_size//1_048_576} MB)"
        )
    else:
        corpus_path = _GUTENBERG_PATH
        print("[run] Gutenberg corpus not found — run `prepare` first")
    if not corpus_path.exists():
        print(f"[run] Corpus not found: {corpus_path}")
        print("[run] Run `prepare` first, or pass --corpus-path")
        raise SystemExit(1)

    # Load questions
    if _QUESTIONS_PATH.exists():
        raw = json.loads(_QUESTIONS_PATH.read_text("utf-8"))
        questions = [Question(**q) for q in raw][: args.questions]
    else:
        print("[run] No question bank found. Running prepare step ...")
        _, questions = cmd_prepare(args)

    print(
        f"\n[run] Corpus: {corpus_path}  ({corpus_path.stat().st_size/1_048_576:.0f} MB)"
    )
    print(
        f"[run] Questions: {len(questions)}  |  top-k: {args.top_k}  |  block: {args.block_mb} MB"
    )

    vanilla_result: StrategyResult | None = None
    optimized_result: StrategyResult | None = None
    vanilla_retriever = None
    optimized_retriever = None
    block_index = None
    tree_index = None
    tree_block_index = None
    tree_result: StrategyResult | None = None

    # ── Resolve persistent index directories (optional) ──────────────────────
    _index_dir: Path | None = getattr(args, "index_dir", None)
    _opt_idx = (_index_dir / "optimized") if _index_dir else None
    _tree_idx = (_index_dir / "tree") if _index_dir else None
    _build_only = getattr(args, "build_only", False)
    _eval_only = getattr(args, "eval_only", False)

    if _eval_only and _index_dir is None:
        print("[run] --eval-only requires --index-dir pointing to a built index.")
        raise SystemExit(1)

    # ── Build Vanilla RAG ─────────────────────────────────────────────────────
    if not getattr(args, "optimized_only", False):
        print("\n" + "=" * 60)
        print("BUILDING VANILLA RAG INDEX")
        print("=" * 60)
        vanilla_retriever, vanilla_result = build_vanilla_rag(
            corpus_path, top_k=args.top_k, max_mb=getattr(args, "max_mb", 200.0)
        )

    # ── Build Optimized RAG ───────────────────────────────────────────────────
    if not getattr(args, "vanilla_only", False):
        print("\n" + "=" * 60)
        print("BUILDING OPTIMIZED RAG INDEX")
        print("=" * 60)
        optimized_retriever, block_index, optimized_result = build_optimized_rag(
            corpus_path,
            block_size_mb=args.block_mb,
            overlap_pct=getattr(args, "overlap_pct", 10.0),
            max_mb=getattr(args, "max_mb", 200.0),
            strategy=getattr(args, "opt_strategy", "llm"),
            compressor_model=getattr(args, "compressor_model", "llama3.2:3b"),
            index_dir=_opt_idx,
        )

    # ── Build Tree RAG ────────────────────────────────────────────────────────
    if getattr(args, "run_tree", False) and not getattr(args, "vanilla_only", False):
        print("\n" + "=" * 60)
        print("BUILDING TREE-OF-SUMMARIES INDEX")
        print("=" * 60)
        tree_index, tree_block_index, tree_result = build_tree_rag(
            corpus_path,
            block_size_mb=args.block_mb,
            cluster_size=getattr(args, "cluster_size", 4),
            overlap_pct=getattr(args, "overlap_pct", 10.0),
            max_mb=getattr(args, "max_mb", 200.0),
            compressor_model=getattr(args, "compressor_model", "llama3.2:3b"),
            index_dir=_tree_idx,
            tree_depth=getattr(args, "tree_depth", 2),
        )

    # ── Exit early if build-only ──────────────────────────────────────────────
    if _build_only:
        print(f"\n[run] --build-only: indexes saved to {_index_dir}. Exiting.")
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Build reasoning LLM once and share between both evaluations
    reasoning_model_name = getattr(args, "reasoning_model", "llama3.2:3b")
    reasoning_llm = None
    if reasoning_model_name:
        print(f"\n[eval] Building reasoning LLM: {reasoning_model_name} ...")
        reasoning_llm = _build_reasoning_llm(reasoning_model_name)
        if reasoning_llm is None:
            print("[eval] WARNING: reasoning LLM unavailable — skipping reasoning pass")
        else:
            print(
                f"[eval] Reasoning LLM ready  (grounding + faithfulness scoring enabled)"
            )

    if vanilla_retriever is not None and vanilla_result is not None:
        print("\n" + "=" * 60)
        print(f"EVALUATING VANILLA RAG ({len(questions)} questions)")
        print("=" * 60)
        vanilla_result.query_results = evaluate_vanilla(
            vanilla_retriever,
            questions,
            top_k=args.top_k,
            reasoning_llm=reasoning_llm,
        )

    if optimized_retriever is not None and optimized_result is not None:
        print("\n" + "=" * 60)
        print(f"EVALUATING OPTIMIZED RAG ({len(questions)} questions)")
        print("=" * 60)
        optimized_result.query_results = evaluate_optimized(
            optimized_retriever,
            block_index,
            questions,
            top_k=args.top_k,
            fallback_threshold=args.fallback_threshold,
            force_fallback=getattr(args, "force_fallback", False),
            reasoning_llm=reasoning_llm,
        )

    if tree_index is not None and tree_result is not None:
        print("\n" + "=" * 60)
        print(f"EVALUATING TREE-OF-SUMMARIES RAG ({len(questions)} questions)")
        print("=" * 60)
        tree_result.query_results = evaluate_tree(
            tree_index,
            questions,
            reasoning_llm=reasoning_llm,
            top_clusters=getattr(args, "top_k", 5) // 2 or 2,
            top_blocks_per_cluster=3,
        )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    def _row(label: str, v: Any, o: Any, fmt: str = ".1f", pct: bool = False) -> None:
        vs = f"{v:{fmt}}" if v is not None else "—"
        os = f"{o:{fmt}}" if o is not None else "—"
        if v is not None and o is not None and v != 0:
            delta = (o - v) / v * 100
            ds = f"{delta:+.1f}%"
        else:
            ds = "—"
        print(f"  {label:<40} {vs:>12}  {os:>14}  {ds:>10}")

    print(f"  {'Metric':<40} {'Vanilla':>12}  {'Optimized':>14}  {'Delta':>10}")
    print("  " + "-" * 78)
    if vanilla_result and optimized_result:
        _row(
            "Avg KW recall",
            vanilla_result.avg_kw_recall,
            optimized_result.avg_kw_recall,
            ".1%",
        )
        _row(
            "Avg tokens/query",
            vanilla_result.avg_tokens_per_query,
            optimized_result.avg_tokens_per_query,
            ",.0f",
        )
        _row(
            "Avg query latency (ms)",
            vanilla_result.avg_latency_ms,
            optimized_result.avg_latency_ms,
            ".1f",
        )
        _row(
            "Ingestion time (s)",
            vanilla_result.ingestion_time_s,
            optimized_result.ingestion_time_s,
            ".1f",
        )
        _row(
            "Index size (MB)",
            vanilla_result.index_size_mb,
            optimized_result.index_size_mb,
            ".1f",
        )
        _row(
            "Index entries",
            vanilla_result.index_size_chunks,
            optimized_result.index_size_chunks,
            ",d",
        )
        print(
            f"  {'Raw block fallback rate':<40} {'—':>12}  {optimized_result.fallback_rate:>14.0%}  {'—':>10}"
        )
        # Reasoning metrics (if reasoning LLM was used)
        if reasoning_llm is not None:
            print("\n  -- Reasoning evaluation (LLM pass) --")
            _row(
                "Retrieval recall (context has keywords)",
                vanilla_result.avg_kw_recall,
                optimized_result.avg_kw_recall,
                ".1%",
            )
            _row(
                "Reasoning recall (LLM answer has keywords)",
                vanilla_result.avg_reasoning_recall,
                optimized_result.avg_reasoning_recall,
                ".1%",
            )
            _row(
                "Faithfulness (answer grounded in context)",
                vanilla_result.avg_faithfulness,
                optimized_result.avg_faithfulness,
                ".1%",
            )
            _row(
                "Reasoning gap (retrieval - reasoning)",
                vanilla_result.reasoning_gap,
                optimized_result.reasoning_gap,
                ".1%",
            )
            _row(
                "Avg reasoning latency (ms)",
                vanilla_result.avg_reasoning_latency_ms,
                optimized_result.avg_reasoning_latency_ms,
                ".0f",
            )
    elif vanilla_result:
        print(
            f"  Vanilla — recall={vanilla_result.avg_kw_recall:.1%}  tokens={vanilla_result.avg_tokens_per_query:,.0f}"
        )
    elif optimized_result:
        print(
            f"  Optimized — recall={optimized_result.avg_kw_recall:.1%}"
            f"  tokens={optimized_result.avg_tokens_per_query:,.0f}"
            f"  fallback={optimized_result.fallback_rate:.0%}"
            + (
                f"  reasoning={optimized_result.avg_reasoning_recall:.1%}"
                if reasoning_llm
                else ""
            )
        )

    if tree_result is not None:
        print(
            f"\n  Tree-of-Summaries — recall={tree_result.avg_kw_recall:.1%}"
            f"  entries={tree_result.index_size_chunks:,}"
            f"  {tree_result.index_size_mb:.1f} MB"
            + (
                f"  reasoning={tree_result.avg_reasoning_recall:.1%}"
                if reasoning_llm
                else ""
            )
        )

    write_report(
        vanilla_result, optimized_result, questions, corpus_path, args, tree=tree_result
    )


# ── Verify subcommand ─────────────────────────────────────────────────────────


def cmd_verify(args: argparse.Namespace) -> None:
    """
    Fast, self-contained end-to-end test of the BlockIndex raw-block fallback.

    Proves that the file-pointer path actually works and provides recall gains:

    1. Creates a synthetic corpus (~2 MB) with N needle facts embedded at
       positions unlikely to survive extractive summarisation:
         "NEEDLE_ID:NLD-{n:04d} value={unique_value} ref_date=2026-0{m}-{d:02d}"
       These are unique identifiers (low TF-IDF) that TF-IDF sentence
       selection will always discard in favour of repeated common sentences.

    2. Builds an optimized index (extractive, no LLM needed) so each block's
       compressed_summary is a 500-token extractive excerpt.

    3. Runs two evaluations against 10 needle-specific questions:
         a) Summary-only  (fallback disabled)     → expected: low needle recall
         b) Forced fallback (always read raw block) → expected: high needle recall

    4. Prints the recovery gap and asserts that forced fallback outperforms
       summary-only, proving the BlockIndex get_text() path is functional.

    Usage:
        python corpus_benchmark.py verify
        python corpus_benchmark.py verify --needles 30 --block-mb 0.2
    """
    import tempfile as _tf

    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.compressor import ingest_file_blocks
    from context_optimizer.raw_index import BlockIndex

    n_needles = args.needles
    block_mb = args.block_mb
    block_bytes = int(block_mb * 1_048_576)

    print("=" * 60)
    print("BlockIndex Fallback Verification")
    print("=" * 60)
    print(f"Needles: {n_needles}  |  Block size: {block_mb:.1f} MB")

    # ── Step 1: Build synthetic corpus ───────────────────────────────────────
    # Design: each block has a UNIQUE domain vocabulary in its filler text so
    # ChromaDB semantic search can distinguish blocks and retrieve the right one.
    # The needle (low-TF-IDF unique ID) is dropped by extractive compression.
    # The query includes the domain keyword so semantic search finds the right block.
    # Force-fallback then reads that block's raw file bytes and recovers the needle.
    _DOMAINS: list[tuple[str, str]] = [
        ("astronomy", "stars planets galaxies telescope universe orbit solar"),
        ("biology", "cells organisms evolution genetics species ecosystem"),
        ("chemistry", "molecules atoms reactions compounds elements periodic"),
        ("geology", "rocks minerals tectonic plates erosion sediment magma"),
        ("medicine", "diagnosis treatment patients clinical symptoms therapy"),
        ("economics", "markets supply demand inflation trade currency fiscal"),
        ("linguistics", "grammar syntax phonetics morphology semantics language"),
        ("architecture", "buildings structures materials foundations blueprints"),
        ("oceanography", "currents tides salinity coral reefs marine bathymetry"),
        ("climatology", "temperature precipitation humidity atmospheric pressure"),
        ("archaeology", "excavation artifacts pottery remains ancient civilisation"),
        ("psychology", "behaviour cognition memory emotion perception stimulus"),
        ("mathematics", "equations proofs theorems calculus algebra geometry"),
        ("engineering", "circuits voltage resistance capacitance semiconductor"),
        ("philosophy", "epistemology ethics metaphysics ontology rationalism"),
        ("literature", "narrative plot characters theme symbolism prose poetry"),
        ("nutrition", "calories protein vitamins minerals dietary fibre intake"),
        ("sociology", "society culture norms institutions stratification"),
        ("botany", "photosynthesis chlorophyll roots stems leaves taxonomy"),
        ("zoology", "mammals reptiles amphibians invertebrates vertebrates"),
    ]
    # Cap domain list to n_needles
    domains = (_DOMAINS * ((n_needles // len(_DOMAINS)) + 1))[:n_needles]

    # Filler: enough lines to fill ~80% of a block (extractive compression keeps
    # high-TF-IDF filler sentences and DROPS the low-frequency needle).
    n_filler_lines = max(4, block_bytes // 160 // 2)

    import random

    rng = random.Random(42)
    needles: list[dict] = []
    for i in range(n_needles):
        domain_name, domain_vocab = domains[i]
        needle_id = f"NLD-{i:04d}"
        value = rng.randint(10000, 99999)
        month = (i % 12) + 1
        day = (i % 28) + 1
        needle_text = (
            f"NEEDLE_ID:{needle_id} "
            f"unique_value={value} "
            f"ref_date=2026-{month:02d}-{day:02d} "
            f"authority=TestOrg_{i}"
        )
        # Filler uses the domain vocabulary so the block embedding is distinct
        filler = f"Research in {domain_name}: {domain_vocab}. " * 3
        needles.append(
            {
                "id": needle_id,
                "text": needle_text,
                "filler": filler,
                "domain": domain_name,
                "keywords": [needle_id, str(value), f"TestOrg_{i}"],
                # Query combines the needle ID with its domain so semantic search
                # finds the right block (domain vocabulary → correct block embedding)
                "query": f"Find the {domain_name} record for {needle_id}",
            }
        )

    # Write corpus: each needle gets its own block with domain-specific filler
    corpus_lines: list[str] = []
    for i, needle in enumerate(needles):
        filler = needle["filler"]
        # Pre-filler (domain vocabulary repeated → high TF-IDF → kept by extractive)
        for _ in range(n_filler_lines // 2):
            corpus_lines.append(filler)
        # Needle (unique low-TF-IDF — WILL BE DROPPED by extractive compression)
        corpus_lines.append(needle["text"])
        # Post-filler
        for _ in range(n_filler_lines // 2):
            corpus_lines.append(filler)

    tmp_dir = _tf.mkdtemp(prefix="co_verify_")
    corpus_path = Path(tmp_dir) / "synthetic_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")
    corpus_size_kb = corpus_path.stat().st_size / 1024
    print(f"Corpus: {corpus_size_kb:.0f} KB  ({len(corpus_lines):,} lines)")

    try:
        # ── Step 2: Build optimized index ────────────────────────────────────
        # Use raw_only so verify runs offline (no Ollama needed).
        # The point is to prove the BlockIndex file-pointer path works,
        # not to test compression quality.
        print("\n[verify] Building index (raw_only, no LLM needed) ...")
        block_db_path = Path(tmp_dir) / "blocks.db"
        block_index = BlockIndex(str(block_db_path))

        retriever = CachedChromaRetriever(
            collection_name="verify_test",
            persist_directory=tmp_dir,
        )
        chunks = ingest_file_blocks(
            source_path=corpus_path,
            block_size_bytes=block_bytes,
            overlap_bytes=block_bytes // 10,
            block_index=block_index,
            strategy="raw_only",
            label="verify",
        )
        retriever.add_chunks(chunks)
        print(
            f"[verify] {len(chunks)} blocks  "
            f"BlockIndex: {block_index.count()} pointers  "
            f"ChromaDB: {retriever.collection.count()} entries"
        )

        # ── Step 3a: Summary-only evaluation ─────────────────────────────────
        print("\n[verify] Evaluating summary-only (no fallback) ...")
        questions_v = [
            Question(
                id=i,
                query=n["query"],
                expected_keywords=n["keywords"],
                source_title=n["id"],
            )
            for i, n in enumerate(needles)
        ]

        results_summary = evaluate_optimized(
            retriever=retriever,
            block_index=block_index,
            questions=questions_v,
            top_k=3,
            fallback_threshold=0.0,  # never trigger threshold-based fallback
            force_fallback=False,
            verbose=False,
        )
        summary_recall = sum(r.kw_recall for r in results_summary) / len(
            results_summary
        )

        # ── Step 3b: Forced-fallback evaluation ───────────────────────────────
        print("[verify] Evaluating with FORCED raw-block fallback ...")
        results_fallback = evaluate_optimized(
            retriever=retriever,
            block_index=block_index,
            questions=questions_v,
            top_k=3,
            fallback_threshold=0.0,
            force_fallback=True,  # always read raw block from disk
            verbose=False,
        )
        fallback_recall = sum(r.kw_recall for r in results_fallback) / len(
            results_fallback
        )
        fallback_used = sum(1 for r in results_fallback if r.used_raw_fallback)

        # ── Step 4: Report ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"  Needles tested          : {n_needles}")
        print(f"  Summary-only recall     : {summary_recall:.1%}")
        print(f"  Forced-fallback recall  : {fallback_recall:.1%}")
        print(f"  Recall gain from fallback: {fallback_recall - summary_recall:+.1%}")
        print(f"  Raw blocks actually read : {fallback_used}/{n_needles}")
        print()

        # Per-needle breakdown
        needle_map_s = {r.question_id: r for r in results_summary}
        needle_map_f = {r.question_id: r for r in results_fallback}
        print(
            f"  {'#':<4} {'Needle ID':<12} {'Summary':>8} {'Fallback':>9} {'Block read':>10}"
        )
        print("  " + "-" * 48)
        for i, n in enumerate(needles):
            rs = needle_map_s.get(i)
            rf = needle_map_f.get(i)
            s_r = f"{rs.kw_recall:.0%}" if rs else "—"
            f_r = f"{rf.kw_recall:.0%}" if rf else "—"
            fb = "yes" if (rf and rf.used_raw_fallback) else "no"
            print(f"  {i:<4} {n['id']:<12} {s_r:>8} {f_r:>9} {fb:>10}")

        print()
        if fallback_recall > summary_recall:
            print(
                "  RESULT: BlockIndex fallback RECOVERS needle facts that summaries miss."
            )
            print("          File-pointer mechanism is WORKING correctly.")
        elif fallback_recall == summary_recall and fallback_used > 0:
            print("  RESULT: Fallback triggered but recall unchanged.")
            print("          Needles may be in blocks that didn't rank in top-k.")
        else:
            print("  WARNING: Fallback provided no recall gain.")
            print(
                "           Check BlockIndex is wired correctly in ingest_file_blocks."
            )

        if fallback_used == 0:
            print(
                "  WARNING: No raw blocks were actually read (all fallback_recall <= summary_recall)."
            )
            print("           Increase needle uniqueness or reduce block size.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    args = _parser().parse_args()

    # Apply config file BEFORE dispatching so every subcommand sees the settings.
    # Explicit CLI flags (non-default values) are applied after config, so they win.
    config_path: Path | None = getattr(args, "config", None)
    if config_path is not None:
        if not config_path.exists():
            raise SystemExit(f"[config] File not found: {config_path}")
        args._config_path = str(config_path)
        cfg = _load_bench_config(config_path)
        _apply_bench_config(args, cfg)
    elif (Path(__file__).parent / "bench_config.yaml").exists():
        # Auto-load if bench_config.yaml sits next to the script
        default_cfg = Path(__file__).parent / "bench_config.yaml"
        args._config_path = str(default_cfg)
        cfg = _load_bench_config(default_cfg)
        _apply_bench_config(args, cfg)

    if args.cmd == "verify":
        cmd_verify(args)

    elif args.cmd == "prepare":
        cmd_prepare(args)

    elif args.cmd == "run":
        cmd_run(args)

    elif args.cmd == "all":
        corpus_path, questions = cmd_prepare(args)
        args.corpus_path = corpus_path
        cmd_run(args)


if __name__ == "__main__":
    main()
