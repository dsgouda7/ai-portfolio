"""
Ingestion Service — rolling-window LLM compression.

Receives a raw corpus (list of text lines), compresses it into
CompressedChunks using the rolling-window pipeline, then:
  1. Calls the embedding service (POST /embed) to generate vectors.
  2. Calls the vector-store service (POST /store) to persist each chunk.

POST /ingest
    Body: { "corpus": ["line1", "line2", ...], "collection": "default" }
    Returns: { "chunks_produced": N, "compression_ratio": 0.14, "time_s": 12.3 }

GET /health
    Returns: { "status": "ok" }
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from context_optimizer.compressor import compress_corpus_rolling, CompressedChunk

EMBEDDING_URL  = os.getenv("EMBEDDING_SERVICE_URL",  "http://embedding:8002")
VECTOR_URL     = os.getenv("VECTOR_STORE_URL",        "http://vector-store:8003")

app = FastAPI(
    title="Context Optimizer — Ingestion Service",
    description="Rolling-window LLM compression pipeline",
    version="0.1.0",
)


# ── Request / response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    corpus:     list[str]
    collection: str = "default"


class IngestResponse(BaseModel):
    chunks_produced:    int
    original_tokens:    int
    compressed_tokens:  int
    compression_ratio:  float
    time_s:             float
    collection:         str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    if not req.corpus:
        raise HTTPException(status_code=400, detail="corpus must be a non-empty list of strings")

    t0     = time.perf_counter()
    chunks = compress_corpus_rolling(req.corpus)
    elapsed = time.perf_counter() - t0

    orig_tokens = sum(c.original_tokens for c in chunks)
    comp_tokens = sum(c.compressed_tokens for c in chunks)

    # Send each chunk to vector-store (embedding is handled inside vector-store
    # so this service stays stateless and dependency-light).
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "collection": req.collection,
            "chunks": [_chunk_to_dict(c) for c in chunks],
        }
        resp = await client.post(f"{VECTOR_URL}/store", json=payload)
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"vector-store /store returned {resp.status_code}: {resp.text[:200]}",
            )

    return IngestResponse(
        chunks_produced=len(chunks),
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        compression_ratio=comp_tokens / max(orig_tokens, 1),
        time_s=round(elapsed, 2),
        collection=req.collection,
    )


def _chunk_to_dict(c: CompressedChunk) -> dict[str, Any]:
    return {
        "chunk_id":           c.chunk_id,
        "raw_text":           c.raw_text,
        "compressed_summary": c.compressed_summary,
        "entities":           c.entities,
        "keywords":           c.keywords,
        "metadata":           c.metadata,
        "original_tokens":    c.original_tokens,
        "compressed_tokens":  c.compressed_tokens,
        "compression_ratio":  c.compression_ratio,
    }
