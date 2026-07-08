"""
watcher.py — Incremental corpus watcher for context-optimizer.

Watches a directory for file changes and re-indexes only the files (and
within each file, only the blocks) that actually changed.

Architecture
------------
::

    FileSystem events (watchdog)
           │
           ▼  debounce 3 s
    DirtyQueue (set of Path)
           │
           ▼
    FileRegistry.is_dirty(path)?   ──── no ──→ skip (0 SLM calls)
           │ yes
           ▼
    BlockIndex.delete_blocks_for_file(path)
    ChromaDB: delete stale vectors by block_id
           │
           ▼
    ingest_file_blocks(path, ...)   ── SLM calls for new/changed blocks only
           │
           ▼
    TreeIndex.rebuild_cluster(cluster_id)   ── 1 SLM call per dirty cluster
           │
           ▼
    FileRegistry.upsert(path, new_hash, block_ids)

Cost model (typical incremental commit)
----------------------------------------
* 1-line edit in a 100 KB file  → 1 block re-ingested + 1 cluster rebuild
                                → 2 SLM calls total
* Append 500 KB to a large file → 5 new blocks + 1 cluster rebuild
                                → 6 SLM calls total
* No content change             → 0 SLM calls (hash check is microseconds)

Usage
-----
::

    from context_optimizer.watcher import CorpusWatcher

    watcher = CorpusWatcher(
        watch_dir="./my-docs",
        index_dir="~/.co/indexes/my-docs",
        compressor_model="llama3.2:3b",
        glob="**/*.txt",          # which files to track
        debounce_s=3.0,
    )
    watcher.start()   # non-blocking — spawns background thread
    ...
    watcher.stop()

    # Or as a blocking CLI watch loop:
    watcher.run_forever()
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_optimizer.raw_index import BlockIndex, FileRegistry
    from context_optimizer.tree_index import TreeIndex


# ── Incremental re-indexer ────────────────────────────────────────────────────


class IncrementalIndexer:
    """
    Handles the re-index logic for a single changed file.

    Separated from the watcher so it can be called directly in tests or
    from a CI pipeline without needing a file-system event loop.
    """

    def __init__(
        self,
        block_index: "BlockIndex",
        file_registry: "FileRegistry",
        retriever: Any,                 # CachedChromaRetriever — duck-typed
        tree: "TreeIndex | None" = None,
        llm: Any | None = None,
        block_size_bytes: int = 100 * 1024,
        overlap_bytes: int = 10 * 1024,
        compressor_model: str = "facebook/bart-large-cnn",
        compressor_provider: str | None = "hf",
        verbose: bool = True,
    ) -> None:
        self.block_index = block_index
        self.file_registry = file_registry
        self.retriever = retriever
        self.tree = tree
        self.llm = llm
        self.block_size_bytes = block_size_bytes
        self.overlap_bytes = overlap_bytes
        self.compressor_model = compressor_model
        self.compressor_provider = compressor_provider
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[watcher] {msg}", flush=True)

    def remove_file(self, path: Path) -> None:
        """Remove all index data for a deleted file."""
        entry = self.file_registry.get(path)
        if entry is None:
            return
        old_block_ids = entry["block_ids"]
        if old_block_ids:
            # Delete vectors from ChromaDB
            try:
                self.retriever.collection.delete(ids=old_block_ids)
            except Exception:
                pass
            # Delete from BlockIndex
            self.block_index.delete_blocks_for_file(path)
            # Rebuild affected tree clusters
            if self.tree is not None:
                self._rebuild_dirty_clusters(old_block_ids)
        self.file_registry.delete(path)
        self._log(f"removed  {path.name}  ({len(old_block_ids)} blocks purged)")

    def reindex_file(self, path: Path) -> int:
        """
        Re-index a single file.  Returns the number of new blocks ingested.

        Steps
        -----
        1. Quick hash check — bail early if nothing changed.
        2. Delete stale block vectors from ChromaDB + BlockIndex rows.
        3. Call ``ingest_file_blocks`` → LLM compression for new blocks.
        4. Add compressed chunks to ChromaDB retriever.
        5. Rebuild affected L2 tree clusters (1 SLM call each).
        6. Update FileRegistry with new hash + block_ids.
        """
        from context_optimizer.raw_index import FileRegistry

        if not path.exists():
            self.remove_file(path)
            return 0

        if not self.file_registry.is_dirty(path):
            self._log(f"unchanged {path.name} — skipped")
            return 0

        self._log(f"dirty     {path.name} — re-indexing ...")
        t0 = time.perf_counter()

        # Remove stale data
        old_entry = self.file_registry.get(path)
        old_block_ids: list[str] = old_entry["block_ids"] if old_entry else []
        if old_block_ids:
            try:
                self.retriever.collection.delete(ids=old_block_ids)
            except Exception:
                pass
            self.block_index.delete_blocks_for_file(path)

        # Re-ingest
        from context_optimizer.compressor import ingest_file_blocks
        llm = self.llm
        if llm is None:
            from context_optimizer.compressor import _build_local_llm
            llm = _build_local_llm(
                provider=self.compressor_provider,
                model=self.compressor_model,
            )

        chunks = ingest_file_blocks(
            source_path=path,
            block_size_bytes=self.block_size_bytes,
            overlap_bytes=self.overlap_bytes,
            block_index=self.block_index,
            llm=llm,
            strategy="llm",
            label="watch",
        )

        # Add to retriever (ChromaDB)
        if chunks:
            self.retriever.add(
                ids=[c.chunk_id for c in chunks],
                documents=[c.compressed_summary for c in chunks],
                metadatas=[{"file_path": str(path), "block_id": c.chunk_id} for c in chunks],
            )

        new_block_ids = [c.chunk_id for c in chunks]

        # Rebuild affected tree clusters
        if self.tree is not None and chunks:
            self._rebuild_dirty_clusters(new_block_ids)

        # Update registry
        content_hash = FileRegistry.hash_file(path)
        self.file_registry.upsert(
            file_path=path,
            content_hash=content_hash,
            file_size=path.stat().st_size,
            block_ids=new_block_ids,
        )

        elapsed = time.perf_counter() - t0
        self._log(
            f"indexed   {path.name}  {len(chunks)} blocks  {elapsed:.1f}s"
            + (f"  (was {len(old_block_ids)} blocks)" if old_block_ids else "")
        )
        return len(chunks)

    def _rebuild_dirty_clusters(self, block_ids: list[str]) -> None:
        """Identify which L2 clusters contain *block_ids* and rebuild them."""
        if self.tree is None:
            return
        dirty_clusters: set[str] = set()
        for bid in block_ids:
            cid = self.tree.cluster_for_block(bid)
            if cid:
                dirty_clusters.add(cid)
        for cid in dirty_clusters:
            self._log(f"rebuild   L2 cluster {cid}")
            self.tree.rebuild_cluster(cid, llm=self.llm)

    def scan_directory(self, watch_dir: Path, glob: str = "**/*.txt") -> dict[str, int]:
        """
        Scan all matching files in *watch_dir* and re-index any that are dirty.

        Returns a summary dict: ``{path_str: blocks_ingested}``.
        Silently skips unchanged files (hash match → 0 SLM calls).
        """
        results: dict[str, int] = {}
        files = sorted(watch_dir.glob(glob))
        if not files:
            self._log(f"no files matching {glob!r} in {watch_dir}")
            return results

        # Detect deleted files (in registry but no longer on disk)
        indexed = set(self.file_registry.all_indexed_paths())
        on_disk = {str(f) for f in files}
        for gone in indexed - on_disk:
            self.remove_file(Path(gone))

        # Re-index dirty files
        for f in files:
            n = self.reindex_file(f)
            results[str(f)] = n

        return results


# ── Watchdog-based file-system event handler ──────────────────────────────────


class _ChangeHandler:
    """
    Minimal watchdog-compatible event handler.

    Puts changed paths onto a queue; the watcher thread drains it with
    debouncing so rapid saves (e.g. formatter on save) collapse into one
    re-index call.
    """

    def __init__(self, dirty_queue: "queue.Queue[Path]", glob_suffix: str) -> None:
        self._queue = dirty_queue
        self._suffix = glob_suffix.lstrip("*")  # e.g. "**/*.txt" → ".txt"

    def _accept(self, path_str: str) -> bool:
        return not self._suffix or path_str.endswith(self._suffix)

    def dispatch(self, event: Any) -> None:
        src = getattr(event, "src_path", None)
        dst = getattr(event, "dest_path", None)  # renames
        for p in filter(None, [src, dst]):
            if self._accept(p):
                self._queue.put(Path(p))


# ── CorpusWatcher ─────────────────────────────────────────────────────────────


class CorpusWatcher:
    """
    High-level watcher: monitors *watch_dir* and triggers incremental
    re-indexing whenever files matching *glob* change.

    Parameters
    ----------
    watch_dir:
        Directory to monitor (recursively).
    indexer:
        A pre-configured :class:`IncrementalIndexer` instance.
    glob:
        File pattern to watch, e.g. ``"**/*.txt"`` or ``"**/*.md"``.
    debounce_s:
        Seconds to wait for more events before triggering a re-index.
        Prevents thrashing during bulk saves / formatter runs.
    verbose:
        Print status messages.

    Example
    -------
    ::

        watcher = CorpusWatcher(
            watch_dir=Path("./docs"),
            indexer=IncrementalIndexer(...),
            glob="**/*.txt",
        )
        watcher.run_forever()   # blocks; Ctrl-C to stop
    """

    def __init__(
        self,
        watch_dir: Path,
        indexer: IncrementalIndexer,
        glob: str = "**/*.txt",
        debounce_s: float = 3.0,
        verbose: bool = True,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.indexer = indexer
        self.glob = glob
        self.debounce_s = debounce_s
        self.verbose = verbose
        self._dirty: queue.Queue[Path] = queue.Queue()
        self._stop_event = threading.Event()
        self._observer: Any = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[watcher] {msg}", flush=True)

    def start(self) -> None:
        """Start watching in a background thread (non-blocking)."""
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except ImportError:
            raise RuntimeError(
                "watchdog is required for file watching.\n"
                "Install with:  pip install watchdog"
            )

        handler = _ChangeHandler(self._dirty, self.glob)

        # Wrap our minimal handler in watchdog's base class
        class _WDHandler(FileSystemEventHandler):
            def __init__(self_, inner: _ChangeHandler) -> None:
                super().__init__()
                self_._inner = inner

            def on_any_event(self_, event: Any) -> None:
                self_._inner.dispatch(event)

        self._observer = Observer()
        self._observer.schedule(
            _WDHandler(handler), str(self.watch_dir), recursive=True
        )
        self._observer.start()

        self._worker = threading.Thread(
            target=self._drain_loop, daemon=True, name="co-watcher"
        )
        self._worker.start()
        self._log(f"watching {self.watch_dir} (glob={self.glob!r}, debounce={self.debounce_s}s)")

    def stop(self) -> None:
        """Stop the watcher threads."""
        self._stop_event.set()
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def run_forever(self) -> None:
        """Start watching and block until Ctrl-C."""
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._log("stopping ...")
        finally:
            self.stop()

    def _drain_loop(self) -> None:
        """
        Worker thread: collect dirty paths, debounce, then re-index.

        Collapses multiple rapid events for the same file into one
        re-index call (common with editors that do atomic save via rename).
        """
        while not self._stop_event.is_set():
            # Wait for the first dirty path
            try:
                path = self._dirty.get(timeout=0.5)
            except queue.Empty:
                continue

            # Collect any additional paths that arrive within debounce window
            pending: set[Path] = {path}
            deadline = time.monotonic() + self.debounce_s
            while time.monotonic() < deadline:
                try:
                    pending.add(self._dirty.get_nowait())
                except queue.Empty:
                    time.sleep(0.1)

            for p in sorted(pending):
                try:
                    self.indexer.reindex_file(p)
                except Exception as exc:
                    self._log(f"ERROR re-indexing {p.name}: {exc}")


# ── TreeIndex cluster helpers (monkey-patch if missing) ───────────────────────
# These methods are referenced by IncrementalIndexer but may not exist in
# older versions of tree_index.py — we add them conditionally.


def _ensure_tree_watcher_methods() -> None:
    """
    Add ``cluster_for_block`` and ``rebuild_cluster`` to TreeIndex if absent.

    This keeps watcher.py self-contained; tree_index.py doesn't need to know
    about the watcher.
    """
    try:
        from context_optimizer.tree_index import TreeIndex  # type: ignore
    except ImportError:
        return

    if hasattr(TreeIndex, "cluster_for_block"):
        return  # already patched or natively implemented

    def cluster_for_block(self: Any, block_id: str) -> str | None:
        """Return the L2 cluster_id that contains *block_id*, or None."""
        try:
            results = self._l1.get(ids=[block_id], include=["metadatas"])
            metas = results.get("metadatas", [[]])[0] if results else []
            if metas:
                return metas[0].get("cluster_id")
        except Exception:
            pass
        return None

    def rebuild_cluster(self: Any, cluster_id: str, llm: Any | None = None) -> None:
        """Recompute the L2 super-summary for *cluster_id*."""
        try:
            # Fetch all L1 summaries in this cluster
            results = self._l1.get(
                where={"cluster_id": cluster_id},
                include=["documents"],
            )
            docs = results.get("documents", []) or []
            if not docs:
                return
            combined = "\n".join(docs)

            # Regenerate L2 summary
            if llm is not None:
                from context_optimizer.tree_index import _L2_PROMPT  # type: ignore
                response = llm.invoke(_L2_PROMPT.format(summaries=combined[:4000]))
                new_summary = getattr(response, "content", str(response)).strip()[:800]
            else:
                new_summary = combined[:800]

            # Update L2 collection
            self._l2.upsert(
                ids=[cluster_id],
                documents=[new_summary],
                metadatas=[{"cluster_id": cluster_id, "block_count": len(docs)}],
            )
        except Exception:
            pass

    TreeIndex.cluster_for_block = cluster_for_block  # type: ignore[attr-defined]
    TreeIndex.rebuild_cluster = rebuild_cluster  # type: ignore[attr-defined]


_ensure_tree_watcher_methods()
