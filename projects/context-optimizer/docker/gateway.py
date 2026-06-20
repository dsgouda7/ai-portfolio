"""
Context Optimizer API Gateway

FastAPI-based REST API for context compression service.
Enables deployment as a microservice/AI gateway.

Features:
- Compression endpoint
- Retrieval endpoint
- Health checks
- Metrics
- Rate limiting (future)
- Caching (future)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

# Import core components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.retriever import DualStorageRetriever


app = FastAPI(
    title="Context Optimizer API",
    description="LLM Context Compression and Retrieval Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CompressionRequest(BaseModel):
    """Request to compress a corpus."""
    lines: List[str] = Field(..., description="Text lines to compress")
    chunk_threshold: int = Field(512, description="Token threshold for chunking")
    max_summary_tokens: int = Field(150, description="Max tokens per summary")
    chunk_overlap_tokens: int = Field(128, description="Overlap between chunks")
    llm_backend: str = Field("ollama", description="LLM backend: 'ollama' or 'groq'")
    llm_model: str = Field("qwen2.5-coder:7b", description="Model to use")

    class Config:
        json_schema_extra = {
            "example": {
                "lines": ["Line 1 text", "Line 2 text", "Line 3 text"],
                "chunk_threshold": 512,
                "max_summary_tokens": 150,
                "chunk_overlap_tokens": 128,
                "llm_backend": "ollama",
                "llm_model": "qwen2.5-coder:7b"
            }
        }


class CompressionResponse(BaseModel):
    """Response from compression."""
    chunk_count: int
    original_tokens: int
    compressed_tokens: int
    reduction_pct: float
    compression_time_sec: float
    chunks: List[Dict[str, Any]]


class RetrievalRequest(BaseModel):
    """Request to retrieve compressed chunks."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(6, description="Number of chunks to retrieve")
    min_relevance: float = Field(0.0, description="Minimum relevance threshold")


class RetrievalResponse(BaseModel):
    """Response from retrieval."""
    query: str
    chunks_found: int
    retrieval_time_ms: float
    chunks: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_sec: float


# Global state
compression_cache: Dict[str, Any] = {}
retriever_instance: Optional[DualStorageRetriever] = None
start_time = time.time()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Context Optimizer API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
        uptime_sec=time.time() - start_time
    )


@app.post("/compress", response_model=CompressionResponse)
async def compress_corpus(request: CompressionRequest):
    """
    Compress a text corpus.

    Takes a list of text lines and returns compressed chunks.
    """
    try:
        # Initialize LLM
        if request.llm_backend == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=request.llm_model,
                base_url="http://localhost:11434"
            )
        elif request.llm_backend == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=request.llm_model,
                api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid LLM backend")

        # Compress
        start = time.time()
        compressed_chunks = compress_corpus_rolling(
            lines=request.lines,
            llm=llm,
            chunk_threshold=request.chunk_threshold,
            max_summary_tokens=request.max_summary_tokens,
            chunk_overlap_tokens=request.chunk_overlap_tokens
        )
        compression_time = time.time() - start

        # Calculate stats
        from shared_inputs import estimate_tokens
        original_tokens = sum(estimate_tokens(line) for line in request.lines)
        compressed_tokens = sum(estimate_tokens(c.compressed_summary) for c in compressed_chunks)
        reduction_pct = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0

        # Store retriever for later queries
        global retriever_instance
        retriever_instance = DualStorageRetriever(compressed_chunks, embedding_backend="hash")

        return CompressionResponse(
            chunk_count=len(compressed_chunks),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            reduction_pct=reduction_pct,
            compression_time_sec=compression_time,
            chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "compressed_summary": c.compressed_summary,
                    "token_count": estimate_tokens(c.compressed_summary)
                }
                for c in compressed_chunks[:10]  # Limit response size
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_chunks(request: RetrievalRequest):
    """
    Retrieve compressed chunks based on query.

    Requires prior call to /compress to initialize retriever.
    """
    global retriever_instance

    if retriever_instance is None:
        raise HTTPException(
            status_code=400,
            detail="No compressed corpus available. Call /compress first."
        )

    try:
        start = time.time()
        chunks = retriever_instance.search(request.query, top_k=request.top_k)
        retrieval_time = (time.time() - start) * 1000

        # Filter by relevance
        filtered_chunks = [
            c for c in chunks
            if c.relevance_score >= request.min_relevance
        ]

        return RetrievalResponse(
            query=request.query,
            chunks_found=len(filtered_chunks),
            retrieval_time_ms=retrieval_time,
            chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "compressed_summary": c.compressed_summary,
                    "relevance_score": c.relevance_score
                }
                for c in filtered_chunks
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Get service metrics.

    Future: Prometheus-compatible metrics.
    """
    return {
        "uptime_sec": time.time() - start_time,
        "compression_requests": 0,  # TODO: Add counter
        "retrieval_requests": 0,    # TODO: Add counter
        "cached_corpora": len(compression_cache)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
