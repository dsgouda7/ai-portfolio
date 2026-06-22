"""
Embedding Service — sentence-transformers vectorization.

Converts text strings into dense float vectors suitable for semantic search.
This is the *embedding* step — not LLM tokenization.  The distinction matters:
  - LLM tokenization  → splits text into subword tokens for a language model's
                         vocabulary (e.g. BPE, WordPiece).
  - Embedding         → maps text to a point in a continuous vector space so
                         that semantically similar texts are close together.

We use sentence-transformers (all-MiniLM-L6-v2 by default) running on CPU.
This model produces 384-dimensional vectors with ~50 ms latency per query.
Swap the model via EMBEDDING_MODEL env var (see ARCHITECTURE.md §8).

POST /embed
    Body:    { "texts": ["text 1", "text 2", ...] }
    Returns: { "embeddings": [[...], [...]], "model": "all-MiniLM-L6-v2", "dims": 384 }

GET /health
    Returns: { "status": "ok", "model": "all-MiniLM-L6-v2" }
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(
    title="Context Optimizer — Embedding Service",
    description="sentence-transformers dense-vector embedding (all-MiniLM-L6-v2)",
    version="0.1.0",
)


# ── Lazy model loader ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    """Load the model once and cache it for the lifetime of the process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


# ── Request / response models ─────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dims: int


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": EMBEDDING_MODEL}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    model  = _get_model()
    vecs   = model.encode(req.texts, show_progress_bar=False, convert_to_numpy=True)
    result = [v.tolist() for v in vecs]

    return EmbedResponse(
        embeddings=result,
        model=EMBEDDING_MODEL,
        dims=len(result[0]) if result else 0,
    )
