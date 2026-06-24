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
