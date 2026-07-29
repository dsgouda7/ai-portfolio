"""
RawIndex — SQLite + FTS5 raw-content store for context-optimizer.

Design
------
ChromaDB is optimised for *vector similarity* search over **compressed**
summaries.  RawIndex fills the complementary role:

* **O(1) lookup by chunk_id** — primary-key select, ~0.1 ms per call.
* **FTS5 full-text search** — BM25-ranked keyword search over the original,
  *un-truncated* text without touching the embedding layer.
* **Parallel-write friendly** — thread-local SQLite connections and WAL
  journal mode let the compressor write raw chunks from a background thread
  *while* the main thread is blocked on an LLM inference call (typically
  ~500 ms vs. ~1 ms for a SQLite write — free concurrency).

Parallel ingestion pattern (used by ``compress_corpus_rolling``)::

    with ThreadPoolExecutor(max_workers=1) as exe:
        for chunk_text, chunk_id in chunks:
            if raw_index:
                exe.submit(raw_index.add, chunk_id, chunk_text)   # ~1 ms
            compressed = compress_chunk_with_llm(chunk_text, ...)  # ~500 ms

The executor's ``__exit__`` joins all futures, so the index is fully
committed before ``compress_corpus_rolling`` returns.

Usage::

    from context_optimizer.raw_index import RawIndex

    # Persistent (survives restarts)
    idx = RawIndex("./my_index/raw.db")
    idx.add("chunk_000000", "The quick brown fox…")
    raw = idx.get("chunk_000000")  # "The quick brown fox…"

    # Full-text search
    hits = idx.search("brown fox", top_k=3)
    for hit in hits:
        print(hit.chunk_id, hit.rank, hit.raw_text[:80])

    # In-memory (tests, ephemeral pipelines)
    idx = RawIndex(":memory:")
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import NamedTuple


class RawHit(NamedTuple):
    """Single FTS5 search result from :meth:`RawIndex.search`."""

    chunk_id: str
    raw_text: str
    #: BM25 relevance score — more negative == better match (SQLite FTS5 convention).
    rank: float


# ── RawIndex ─────────────────────────────────────────────────────────────────


class RawIndex:
    """
    Thread-safe SQLite + FTS5 store for chunk raw text.

    Two connection strategies depending on the database path:

    * ``:memory:`` — a single shared ``sqlite3.Connection`` (``check_same_thread=False``)
      so that all threads see the same in-memory data.
    * File-backed — thread-local connections in WAL journal mode, allowing
      one background writer and many concurrent readers without lock contention.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Pass ``":memory:"`` for a
        fully in-process, non-persistent store (useful in unit tests and
        one-shot pipelines where persistence is not required).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        self._schema_lock = threading.Lock()

        if self._is_memory:
            # Single shared connection for :memory: — SQLite in-memory databases
            # are per-connection, so thread-local storage would give each thread
            # an isolated empty DB.  A single connection with check_same_thread=False
            # lets all threads share the same in-memory store.
            self._shared_conn: sqlite3.Connection | None = self._make_conn()
            self._local = None  # not used for :memory:
        else:
            # File-backed: thread-local connections for WAL-mode parallelism.
            self._shared_conn = None
            self._local = threading.local()

        self._init_schema()

    # ── Connection management ────────────────────────────────────────────────

    def _make_conn(self) -> sqlite3.Connection:
        """Open a new SQLite connection with recommended PRAGMA settings."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # durable yet fast
        conn.execute("PRAGMA foreign_keys=OFF")
        return conn

    def _connect(self) -> sqlite3.Connection:
        """
        Return the appropriate SQLite connection for the current thread.

        * ``:memory:`` — always returns the single shared connection.
        * File-backed — returns (or creates) a thread-local connection.
        """
        if self._is_memory:
            assert self._shared_conn is not None
            return self._shared_conn
        # File-backed path: thread-local connection
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_conn()
            # Schema may already exist on disk; IF NOT EXISTS keeps this idempotent.
            self._apply_schema(conn)
            self._local.conn = conn  # type: ignore[union-attr]
        return conn

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id  TEXT PRIMARY KEY,
            raw_text  TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                raw_text,
                content='chunks',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );
        CREATE TRIGGER IF NOT EXISTS chunks_ai
            AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, raw_text)
                VALUES (new.rowid, new.raw_text);
            END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad
            AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, raw_text)
                VALUES ('delete', old.rowid, old.raw_text);
            END;
        CREATE TRIGGER IF NOT EXISTS chunks_au
            AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, raw_text)
                VALUES ('delete', old.rowid, old.raw_text);
                INSERT INTO chunks_fts(rowid, raw_text)
                VALUES (new.rowid, new.raw_text);
            END;
    """

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables / triggers if they do not yet exist (idempotent)."""
        conn.executescript(self._SCHEMA_SQL)
        conn.commit()

    def _init_schema(self) -> None:
        """Initialise schema on the primary connection (called once in __init__)."""
        with self._schema_lock:
            self._apply_schema(self._connect())

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, chunk_id: str, raw_text: str) -> None:
        """
        Insert or replace a single chunk.

        Thread-safe: multiple threads can call this concurrently; WAL mode
        serialises writes without blocking readers.

        Parameters
        ----------
        chunk_id:
            Unique identifier, e.g. ``"chunk_000042"``.
        raw_text:
            Original, un-compressed text for this chunk.
        """
        conn = self._connect()
        # INSERT OR REPLACE triggers DELETE then INSERT, which correctly
        # updates the FTS5 content table via the au/ad/ai triggers.
        conn.execute(
            "INSERT OR REPLACE INTO chunks(chunk_id, raw_text) VALUES (?, ?)",
            (chunk_id, raw_text),
        )
        conn.commit()

    def add_many(self, pairs: list[tuple[str, str]]) -> None:
        """
        Batch-insert a list of ``(chunk_id, raw_text)`` pairs in a single
        transaction (much faster than repeated :meth:`add` calls for large
        batches).
        """
        conn = self._connect()
        # Process one by one to trigger FTS5 triggers correctly per row
        for chunk_id, raw_text in pairs:
            conn.execute(
                "INSERT OR REPLACE INTO chunks(chunk_id, raw_text) VALUES (?, ?)",
                (chunk_id, raw_text),
            )
        conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def get(self, chunk_id: str) -> str | None:
        """
        Fetch the raw text for a chunk by its ID.

        Latency: ~0.1 ms (indexed primary-key lookup).

        Returns
        -------
        str | None
            Raw text if found; ``None`` if the chunk is not in the index.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT raw_text FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return row[0] if row else None

    def search(self, query: str, top_k: int = 5) -> list[RawHit]:
        """
        Full-text search over raw chunk content using FTS5 (BM25 ranking).

        Parameters
        ----------
        query:
            Free-text search string.  Supports FTS5 syntax: phrase queries
            (``"exact phrase"``), boolean operators (``AND``, ``OR``, ``NOT``),
            and prefix queries (``word*``).  Plain space-separated words are
            treated as an implicit AND.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RawHit]
            Results sorted by relevance (best match first).  BM25 ``rank``
            values are negative (SQLite convention); more negative = better.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.raw_text, bm25(chunks_fts) AS rank
                  FROM chunks_fts
                  JOIN chunks c ON chunks_fts.rowid = c.rowid
                 WHERE chunks_fts MATCH ?
                 ORDER BY rank
                 LIMIT ?
                """,
                (query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 query — return empty rather than crashing the pipeline
            return []
        return [RawHit(chunk_id=r[0], raw_text=r[1], rank=r[2]) for r in rows]

    def count(self) -> int:
        """Return the total number of chunks stored in the index."""
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Close the calling thread's connection (or the shared connection for
        ``:memory:`` databases).
        """
        if self._is_memory:
            if self._shared_conn is not None:
                self._shared_conn.close()
                self._shared_conn = None
        else:
            conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"RawIndex(db={self._db_path!r}, chunks={self.count()})"

    def __enter__(self) -> "RawIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── BlockIndex ────────────────────────────────────────────────────────────────


class BlockPointer(NamedTuple):
    """Metadata for a single on-disk block."""

    block_id: str
    file_path: str
    byte_start: int
    byte_end: int

    @property
    def size_bytes(self) -> int:
        return self.byte_end - self.byte_start


class BlockIndex:
    """
    File-pointer index for large-corpus block-based ingestion.

    Stores ``(block_id → file_path, byte_start, byte_end)`` in SQLite but
    **never copies the raw text** — it is read on demand by seeking in the
    original source file.

    Benefits vs RawIndex for large corpora
    ---------------------------------------
    * A 2 GB corpus produces ~1 MB of SQLite metadata instead of ~2 GB.
    * No data duplication: original files stay on disk as-is.
    * Random read of one 500 KB block: ~2 ms (OS file cache); no SQLite
      page-reads required.

    Usage::

        idx = BlockIndex("./corpus.blockindex.db")
        idx.add_block("enwik9_block_000000", "/data/enwik9", 0, 500_000)

        # Read the raw text of a block on demand
        text = idx.get_text("enwik9_block_000000")

        # Inspect metadata without reading the file
        ptr = idx.get_meta("enwik9_block_000000")
        print(ptr.file_path, ptr.byte_start, ptr.byte_end)
    """

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS blocks (
            block_id   TEXT PRIMARY KEY,
            file_path  TEXT NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end   INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_blocks_file ON blocks(file_path);
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        self._lock = threading.Lock()

        if self._is_memory:
            self._shared_conn: sqlite3.Connection | None = self._make_conn()
            self._local = None
        else:
            self._shared_conn = None
            self._local = threading.local()

        self._init_schema()

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connect(self) -> sqlite3.Connection:
        if self._is_memory:
            assert self._shared_conn is not None
            return self._shared_conn
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_conn()
            conn.executescript(self._SCHEMA_SQL)
            conn.commit()
            self._local.conn = conn  # type: ignore[union-attr]
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._connect().executescript(self._SCHEMA_SQL)
            self._connect().commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def add_block(
        self,
        block_id: str,
        file_path: str | Path,
        byte_start: int,
        byte_end: int,
    ) -> None:
        """Register a block's file location.  No file I/O; metadata only."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO blocks(block_id, file_path, byte_start, byte_end)"
            " VALUES (?, ?, ?, ?)",
            (block_id, str(file_path), byte_start, byte_end),
        )
        conn.commit()

    def add_many(
        self,
        blocks: list[tuple[str, str | Path, int, int]],
    ) -> None:
        """
        Batch-insert ``(block_id, file_path, byte_start, byte_end)`` tuples
        in one transaction.
        """
        conn = self._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO blocks(block_id, file_path, byte_start, byte_end)"
            " VALUES (?, ?, ?, ?)",
            [(bid, str(fp), bs, be) for bid, fp, bs, be in blocks],
        )
        conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_meta(self, block_id: str) -> BlockPointer | None:
        """Return the pointer metadata without reading the file."""
        row = (
            self._connect()
            .execute(
                "SELECT block_id, file_path, byte_start, byte_end"
                "  FROM blocks WHERE block_id = ?",
                (block_id,),
            )
            .fetchone()
        )
        return BlockPointer(*row) if row else None

    def get_text(
        self,
        block_id: str,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> str | None:
        """
        Read the block's raw bytes from disk and decode to a string.

        Seeks directly to ``byte_start`` — only the block's bytes are read,
        not the entire file.

        Returns
        -------
        str | None
            Decoded block text, or ``None`` if the block_id is not found or
            the source file is missing / unreadable.
        """
        ptr = self.get_meta(block_id)
        if ptr is None:
            return None
        try:
            with open(ptr.file_path, "rb") as fh:
                fh.seek(ptr.byte_start)
                raw_bytes = fh.read(ptr.byte_end - ptr.byte_start)
            return raw_bytes.decode(encoding, errors=errors)
        except (OSError, IOError):
            return None

    def count(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

    def all_ids(self) -> list[str]:
        rows = (
            self._connect()
            .execute("SELECT block_id FROM blocks ORDER BY block_id")
            .fetchall()
        )
        return [r[0] for r in rows]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def blocks_for_file(self, file_path: str | Path) -> list[str]:
        """Return all block_ids that belong to *file_path*."""
        rows = (
            self._connect()
            .execute(
                "SELECT block_id FROM blocks WHERE file_path = ?",
                (str(file_path),),
            )
            .fetchall()
        )
        return [r[0] for r in rows]

    def delete_blocks_for_file(self, file_path: str | Path) -> int:
        """Remove all block pointer rows for *file_path*.  Returns count deleted."""
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM blocks WHERE file_path = ?", (str(file_path),)
        )
        conn.commit()
        return cur.rowcount

    def __repr__(self) -> str:
        return f"BlockIndex(db={self._db_path!r}, blocks={self.count()})"

    def __enter__(self) -> "BlockIndex":
        return self

    def __exit__(self, *_: object) -> None:
        if self._is_memory and self._shared_conn:
            self._shared_conn.close()


# ── FileRegistry ──────────────────────────────────────────────────────────────


class FileRegistry:
    """
    Tracks which files have been indexed and their content hashes.

    Stored alongside BlockIndex in the same SQLite DB (separate table).
    Used by the incremental watcher to detect which files need re-indexing.

    Schema
    ------
    ``file_registry(file_path, content_hash, file_size, indexed_at, block_ids)``

    ``block_ids`` is a JSON list of block_ids produced during the last index
    run for this file — used to delete stale ChromaDB vectors before
    re-ingesting.
    """

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS file_registry (
            file_path    TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            file_size    INTEGER NOT NULL,
            indexed_at   REAL NOT NULL,
            block_ids    TEXT NOT NULL DEFAULT '[]'
        );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        self._lock = threading.Lock()
        if self._is_memory:
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._shared_conn.executescript(self._SCHEMA_SQL)
            self._shared_conn.commit()
            self._local = None
        else:
            self._shared_conn = None
            self._local = threading.local()
            self._init_schema()

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connect(self) -> sqlite3.Connection:
        if self._is_memory:
            assert self._shared_conn is not None
            return self._shared_conn
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_conn()
            conn.executescript(self._SCHEMA_SQL)
            conn.commit()
            self._local.conn = conn  # type: ignore[union-attr]
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._connect().executescript(self._SCHEMA_SQL)
            self._connect().commit()

    # ── Hash helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def hash_file(path: Path) -> str:
        """xxHash-64 (fast, non-cryptographic) of the file contents.

        Falls back to MD5 if ``xxhash`` is not installed.
        """
        try:
            import xxhash  # type: ignore
            h = xxhash.xxh64()
        except ImportError:
            import hashlib
            h = hashlib.md5()  # noqa: S324  — not used for security

        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, file_path: str | Path) -> dict | None:
        """Return the registry entry for *file_path*, or None if not indexed."""
        import json as _json
        row = (
            self._connect()
            .execute(
                "SELECT file_path, content_hash, file_size, indexed_at, block_ids"
                "  FROM file_registry WHERE file_path = ?",
                (str(file_path),),
            )
            .fetchone()
        )
        if row is None:
            return None
        return {
            "file_path": row[0],
            "content_hash": row[1],
            "file_size": row[2],
            "indexed_at": row[3],
            "block_ids": _json.loads(row[4]),
        }

    def is_dirty(self, path: Path) -> bool:
        """Return True if *path* is new, deleted, or its hash has changed."""
        if not path.exists():
            return False  # deleted — handled separately
        entry = self.get(path)
        if entry is None:
            return True  # never indexed
        if entry["file_size"] != path.stat().st_size:
            return True  # quick size check before hashing
        return entry["content_hash"] != self.hash_file(path)

    def all_indexed_paths(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT file_path FROM file_registry"
        ).fetchall()
        return [r[0] for r in rows]

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        file_path: str | Path,
        content_hash: str,
        file_size: int,
        block_ids: list[str],
    ) -> None:
        import json as _json
        import time as _time
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO file_registry"
            "(file_path, content_hash, file_size, indexed_at, block_ids)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                str(file_path),
                content_hash,
                file_size,
                _time.time(),
                _json.dumps(block_ids),
            ),
        )
        conn.commit()

    def delete(self, file_path: str | Path) -> None:
        conn = self._connect()
        conn.execute(
            "DELETE FROM file_registry WHERE file_path = ?", (str(file_path),)
        )
        conn.commit()

    def count(self) -> int:
        return self._connect().execute(
            "SELECT COUNT(*) FROM file_registry"
        ).fetchone()[0]
