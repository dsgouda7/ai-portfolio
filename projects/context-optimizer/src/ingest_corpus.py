"""
ingest_corpus — multi-file, multi-format corpus ingestion with parallel extraction.

Architecture
------------
Extraction (I/O bound, per format) runs in a thread pool.
BART / Ollama compression (CPU bound, single model) runs serially after extraction.

Usage
-----
    from context_optimizer.ingest_corpus import ingest_directory

    chunks = ingest_directory(
        directory=Path("my_docs/"),
        block_index=block_idx,
        task_model_map={"code": ("ollama", "qwen2.5-coder:7b")},
    )
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_optimizer.compressor import CompressedChunk
    from context_optimizer.raw_index import BlockIndex


# ── Task model resolution ──────────────────────────────────────────────────────


def resolve_task_model(
    task: str,
    task_model_map: dict[str, tuple[str, str]] | None = None,
    default_provider: str = "ollama",
    default_model: str = "qwen2.5:3b",
) -> tuple[str, str, bool]:
    """
    Return ``(provider, model, is_code)`` for *task*.

    Two-model design — no per-task spaghetti:
    - ``is_code=True``   -> code-specialized model (qwen2.5-coder:3b or equivalent)
    - ``is_code=False``  -> default prose/doc model (qwen2.5:3b or equivalent)

    Priority:  task_model_map  >  env vars  >  defaults.
    """
    is_code = task == "code"

    if task_model_map and task in task_model_map:
        p, m = task_model_map[task]
        return p, m, is_code

    global_provider = os.getenv(
        "CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", default_provider
    )
    if is_code:
        model = os.getenv("CONTEXT_OPTIMIZER_CODE_MODEL") or os.getenv(
            "CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "qwen2.5-coder:3b"
        )
    else:
        model = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", default_model)

    return global_provider, model, is_code


# ── Parallel text extraction ───────────────────────────────────────────────────


def _extract_one(path: Path, router: Any) -> tuple[Path, str, str]:
    """Extract text from *path*. Returns (path, text, task)."""
    task = router.task_for(path)
    text = router.extract(path)
    return path, text, task


def extract_corpus_parallel(
    files: list[Path],
    max_workers: int = 4,
    verbose: bool = True,
) -> list[tuple[Path, str, str]]:
    """
    Extract text from all *files* in parallel (I/O bound).

    Returns list of ``(path, text, task)`` tuples.
    Each format category runs in its own thread.
    """
    from context_optimizer.extractors import FormatRouter

    router = FormatRouter()

    results: list[tuple[Path, str, str]] = []
    failed = 0

    if verbose:
        print(f"[ingest] Extracting {len(files)} files with {max_workers} workers ...")

    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="extractor"
    ) as pool:
        futures = {pool.submit(_extract_one, f, router): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                p, text, task = future.result()
                if text.strip():
                    results.append((p, text, task))
                elif verbose:
                    print(f"  [ingest] SKIP {p.name} — empty after extraction")
            except Exception as exc:
                failed += 1
                if verbose:
                    print(f"  [ingest] ERROR {path.name}: {exc}")
            if verbose and i % 50 == 0:
                print(f"  [ingest] {i}/{len(files)} files extracted ...")

    if verbose:
        print(
            f"[ingest] Extraction done: {len(results)} files OK, "
            f"{failed} failed, {len(files) - len(results) - failed} empty"
        )
    return results


# ── Main entry point ──────────────────────────────────────────────────────────


def ingest_directory(
    directory: Path,
    block_index: "BlockIndex | None" = None,
    block_size_bytes: int = 102_400,  # 100 KB default
    overlap_pct: float = 10.0,
    strategy: str = "llm",
    task_model_map: dict[str, tuple[str, str]] | None = None,
    max_extract_workers: int = 4,
    recursive: bool = True,
    include_exts: list[str] | None = None,
    exclude_exts: list[str] | None = None,
    label: str = "",
    verbose: bool = True,
) -> "list[CompressedChunk]":
    """
    Ingest all supported files under *directory* into compressed chunks.

    Parameters
    ----------
    directory:
        Root directory to ingest.
    block_index:
        BlockIndex to register file pointers.  None = no raw-fallback.
    block_size_bytes:
        Target block size for BART summarization.  Default 100 KB.
    overlap_pct:
        Overlap between adjacent blocks (%).
    strategy:
        ``"llm"`` (BART / Ollama) or ``"raw_only"`` (no compression).
    task_model_map:
        ``{"task_name": ("provider", "model")}`` overrides.
        If None, falls back to env vars or global compressor config.
    max_extract_workers:
        Threads for parallel extraction (I/O bound phase).
    recursive:
        Scan sub-directories recursively.
    include_exts / exclude_exts:
        Extension filters (e.g. ``[".pdf", ".docx"]``).
    """
    from context_optimizer.compressor import _build_local_llm, ingest_file_blocks
    from context_optimizer.extractors import FormatRouter

    router = FormatRouter()
    directory = Path(directory)

    # ── Scan ──────────────────────────────────────────────────────────────────
    scanned = router.scan_directory(directory, recursive=recursive)

    if include_exts:
        inc = {e.lower() for e in include_exts}
        scanned = [(p, t) for p, t in scanned if p.suffix.lower() in inc]
    if exclude_exts:
        exc = {e.lower() for e in exclude_exts}
        scanned = [(p, t) for p, t in scanned if p.suffix.lower() not in exc]

    files = [p for p, _ in scanned]
    if not files:
        if verbose:
            print(f"[ingest] No supported files found in {directory}")
        return []

    if verbose:
        from collections import Counter

        ext_counts = Counter(p.suffix.lower() for p in files)
        print(f"[ingest] Found {len(files)} files in {directory}")
        for ext, n in sorted(ext_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {ext}: {n}")

    # ── Parallel extraction ────────────────────────────────────────────────────
    extracted = extract_corpus_parallel(
        files, max_workers=max_extract_workers, verbose=verbose
    )

    # ── Per-task model selection ───────────────────────────────────────────────
    # Group files by task, build one LLM per task
    from collections import defaultdict

    task_files: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, text, task in extracted:
        task_files[task].append((path, text))

    if verbose:
        for task, items in sorted(task_files.items()):
            provider, model, is_code = resolve_task_model(task, task_model_map)
            label = f"{model} [code]" if is_code else model
            print(f"  Task '{task}': {len(items)} files -> {provider}/{label}")

    # ── Compress each task group ───────────────────────────────────────────────
    all_chunks: list[CompressedChunk] = []
    overlap_bytes = int(block_size_bytes * overlap_pct / 100)

    for task, items in task_files.items():
        provider, model, is_code = resolve_task_model(task, task_model_map)
        # Set global env so _build_local_llm resolves the right provider
        os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER"] = provider
        os.environ["CONTEXT_OPTIMIZER_COMPRESSOR_MODEL"] = model
        llm = _build_local_llm(is_code=is_code)

        if verbose:
            print(
                f"\n[ingest] Compressing {len(items)} '{task}' files with {model} ..."
            )

        for path, text in items:
            # Write text to a temp file so ingest_file_blocks can read it
            fd, tmp_path = tempfile.mkstemp(suffix=path.suffix, prefix="co_ingest_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                chunks = ingest_file_blocks(
                    source_path=tmp_path,
                    block_size_bytes=block_size_bytes,
                    overlap_bytes=overlap_bytes,
                    block_index=block_index,
                    llm=llm,
                    strategy=strategy,
                    label=f"{label}/{path.name}" if label else path.name,
                )
                all_chunks.extend(chunks)
            finally:
                os.unlink(tmp_path)

    if verbose:
        print(f"\n[ingest] Done — {len(all_chunks)} chunks from {len(extracted)} files")

    return all_chunks
