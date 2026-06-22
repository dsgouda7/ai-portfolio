"""
Vector Store Service — ChromaDB HNSW + in-memory semantic cache.

Dual-storage design (see ARCHITECTURE.md §4):
  - Compressed Index: embedded summaries, fast semantic HNSW search.
  - Raw Vault:        original text stored in ChromaDB metadata (NOT indexed).
                      Retrieved on-demand via chunk_id — no re-embedding needed.

Two-tier retrieval (see ARCHITECTURE.md §5-6):
  - Tier 1: In-memory LRU semantic cache (exact-string + cosine similarity) → 0-2 ms
  - Tier 2: ChromaDB HNSW search on cache miss → 10-50 ms

Endpoints
---------
POST /store
    Body: { "collection": "default", "chunks": [{...}, ...] }
    Embeds each compressed_summary via the embedding service, then stores
    both the vector (indexed) and raw_text (metadata pointer, not indexed).

GET  /search?q=...&top_k=5&collection=default
    Returns top_k compressed summaries + metadata (no raw text).
    Cache checked first; ChromaDB queried on miss.

GET  /chunks/{chunk_id}?collection=default
    Returns the raw_text for a specific chunk (pointer-model fetch).
    Does NOT re-embed — reads straight from ChromaDB metadata.

GET  /health
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import httpx
import chromadb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

EMBEDDING_URL      = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8002")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR",   "/data/chroma_db")
CACHE_SIZE         = int(os.getenv("CACHE_SIZE",        "1000"))
CACHE_THRESHOLD    = float(os.getenv("CACHE_THRESHOLD", "0.85"))

app = FastAPI(
    title="Context Optimizer — Vector Store Service",
    description="ChromaDB HNSW + semantic cache with raw-text pointer model",
    version="0.1.0",
)


# ── ChromaDB client (lazy init per collection) ────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_collections: dict[str, Any] = {}


def _get_collection(name: str) -> Any:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if name not in _collections:
        _collections[name] = _chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[name]


# ── In-memory semantic cache ─────────────────────────────────────────────────

class _SemanticCache:
    """Simple LRU cache with exact-string fast path + cosine similarity fallback."""

    def __init__(self, capacity: int = 1000, threshold: float = 0.85) -> None:
        self._cap   = capacity
        self._thr   = threshold
        self._store: OrderedDict[str, tuple[list[float], list[dict]]] = OrderedDict()

    def get(self, query: str, query_vec: list[float] | None = None) -> list[dict] | None:
        # Fast path: exact match
        if query in self._store:
            self._store.move_to_end(query)
            return self._store[query][1]
        # Slow path: cosine similarity
        if query_vec is not None:
            for key, (cached_vec, cached_results) in self._store.items():
                if _cosine(query_vec, cached_vec) >= self._thr:
                    self._store.move_to_end(key)
                    return cached_results
        return None

    def put(self, query: str, query_vec: list[float], results: list[dict]) -> None:
        if query in self._store:
            self._store.move_to_end(query)
        self._store[query] = (query_vec, results)
        if len(self._store) > self._cap:
            self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)


_cache = _SemanticCache(capacity=CACHE_SIZE, threshold=CACHE_THRESHOLD)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    denom = na * nb
    return dot / denom if denom > 0 else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{EMBEDDING_URL}/embed", json={"texts": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ── Request / response models ─────────────────────────────────────────────────

class ChunkInput(BaseModel):
    chunk_id:           str
    raw_text:           str
    compressed_summary: str
    entities:           list[str] = []
    keywords:           list[str] = []
    metadata:           dict[str, Any] = {}
    original_tokens:    int = 0
    compressed_tokens:  int = 0
    compression_ratio:  float = 0.0


class StoreRequest(BaseModel):
    collection: str = "default"
    chunks:     list[ChunkInput]


class StoreResponse(BaseModel):
    stored:     int
    collection: str


class SearchResult(BaseModel):
    chunk_id:           str
    compressed_summary: str
    relevance_score:    float
    entities:           list[str]
    metadata:           dict[str, Any]
    cache_hit:          bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "cache_size": _cache.size}


@app.post("/store", response_model=StoreResponse)
async def store(req: StoreRequest) -> StoreResponse:
    if not req.chunks:
        raise HTTPException(status_code=400, detail="chunks must be non-empty")

    col = _get_collection(req.collection)

    # Embed all compressed summaries in one batch call
    summaries = [c.compressed_summary for c in req.chunks]
    vectors   = await _embed(summaries)

    ids        = [c.chunk_id for c in req.chunks]
    metadatas  = [
        {
            "raw_text":           c.raw_text,         # pointer model — stored, not indexed
            "entities":           ",".join(c.entities),
            "keywords":           ",".join(c.keywords),
            "original_tokens":    c.original_tokens,
            "compressed_tokens":  c.compressed_tokens,
            "compression_ratio":  c.compression_ratio,
            **c.metadata,
        }
        for c in req.chunks
    ]

    col.upsert(ids=ids, embeddings=vectors, documents=summaries, metadatas=metadatas)
    return StoreResponse(stored=len(req.chunks), collection=req.collection)


@app.get("/search", response_model=list[SearchResult])
async def search(
    q:          str = Query(..., description="Search query"),
    top_k:      int = Query(5,   description="Number of results"),
    collection: str = Query("default"),
) -> list[SearchResult]:
    # Embed the query
    q_vec = (await _embed([q]))[0]

    # Cache check
    cached = _cache.get(q, q_vec)
    if cached is not None:
        for r in cached:
            r["cache_hit"] = True
        return [SearchResult(**r) for r in cached]

    # ChromaDB HNSW search
    col   = _get_collection(collection)
    qres  = col.query(query_embeddings=[q_vec], n_results=min(top_k, col.count()))

    results: list[dict] = []
    for i, doc in enumerate(qres["documents"][0]):
        meta     = qres["metadatas"][0][i]
        distance = qres["distances"][0][i] if qres.get("distances") else 0.0
        results.append({
            "chunk_id":           qres["ids"][0][i],
            "compressed_summary": doc,
            "relevance_score":    round(1.0 - distance, 4),
            "entities":           [e for e in meta.get("entities", "").split(",") if e],
            "metadata":           {k: v for k, v in meta.items() if k not in ("raw_text", "entities", "keywords")},
            "cache_hit":          False,
        })

    _cache.put(q, q_vec, results)
    return [SearchResult(**r) for r in results]


@app.get("/chunks/{chunk_id}", response_model=dict)
def get_chunk(
    chunk_id:   str,
    collection: str = Query("default"),
) -> dict[str, Any]:
    """
    Return the raw_text for a specific chunk (pointer-model read).
    No re-embedding — fetches directly from ChromaDB metadata.
    """
    col = _get_collection(collection)
    result = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"chunk_id '{chunk_id}' not found")

    meta = result["metadatas"][0]
    return {
        "chunk_id":           chunk_id,
        "compressed_summary": result["documents"][0],
        "raw_text":           meta.get("raw_text", ""),
        "entities":           [e for e in meta.get("entities", "").split(",") if e],
        "metadata":           {k: v for k, v in meta.items() if k not in ("raw_text", "entities", "keywords")},
    }


@app.get("/stats")
def stats(collection: str = Query("default")) -> dict[str, Any]:
    col = _get_collection(collection)
    return {
        "collection":    collection,
        "chunk_count":   col.count(),
        "cache_entries": _cache.size,
        "cache_capacity": CACHE_SIZE,
    }
