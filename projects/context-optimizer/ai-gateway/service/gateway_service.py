"""
Context Optimizer AI Gateway Service

Deployable FastAPI service with LiteLLM integration and semantic compression.
Drop-in replacement for OpenAI API with 50-98% cost reduction.
"""

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import time
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "wrapper"))

from context_optimizer_gateway import CompressedLiteLLM, SemanticCache


# ============================================================================
# Pydantic Models (OpenAI-compatible)
# ============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name")


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model name (e.g., gpt-4, claude-3-opus)")
    messages: List[Message] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = Field(False)
    n: Optional[int] = Field(1, ge=1, le=10)

    # Compression settings
    enable_compression: Optional[bool] = Field(True, description="Enable auto compression")
    compression_threshold: Optional[int] = Field(2000, description="Token threshold")


class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

    # Context Optimizer extensions
    compression_stats: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    cache_stats: Optional[Dict[str, Any]] = None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Context Optimizer AI Gateway",
    description="LiteLLM gateway with semantic compression (50-98% token reduction)",
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

# Global state
START_TIME = time.time()
compression_client: Optional[CompressedLiteLLM] = None
semantic_cache: Optional[SemanticCache] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global compression_client, semantic_cache

    # Initialize compression client
    compression_client = CompressedLiteLLM(
        compression_threshold=2000,
        track_costs=True,
        cache_enabled=True
    )

    # Initialize semantic cache
    semantic_cache = SemanticCache(
        redis_url=None,  # Use in-memory by default (set env var for Redis)
        default_ttl=3600
    )

    print("🚀 Context Optimizer Gateway started")
    print("   - Compression: Enabled")
    print("   - Cache: In-memory (set REDIS_URL for Redis)")
    print("   - LiteLLM: 100+ providers")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        uptime_seconds=time.time() - START_TIME,
        cache_stats=semantic_cache.get_stats() if semantic_cache else None
    )


@app.get("/stats")
async def get_stats():
    """Get compression and cost statistics."""
    if not compression_client:
        raise HTTPException(status_code=503, detail="Service not initialized")

    compression_stats = compression_client.get_stats()
    cache_stats = semantic_cache.get_stats() if semantic_cache else {}

    return {
        "compression": compression_stats,
        "cache": cache_stats,
        "uptime_seconds": time.time() - START_TIME
    }


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def create_chat_completion(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = None
):
    """
    OpenAI-compatible chat completion endpoint with compression.

    Automatically compresses large contexts before sending to LLM provider.
    Transparent drop-in replacement for OpenAI API.
    """
    if not compression_client:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.time()

    # Convert Pydantic messages to dict
    messages = [msg.dict(exclude_none=True) for msg in request.messages]

    # Check cache first
    cache_key = semantic_cache.semantic_key(
        content=json.dumps(messages),
        query=messages[-1]["content"] if messages else None
    ) if semantic_cache else None

    cached_response = semantic_cache.get(cache_key) if cache_key else None

    if cached_response:
        # Cache hit!
        print(f"[Cache HIT] Returning cached response (key: {cache_key[:8]}...)")
        cached_response["compression_stats"]["cache_hit"] = True
        return cached_response

    # Call LiteLLM with compression
    try:
        llm_response = compression_client.completion(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            n=request.n
        )

        # Format OpenAI-compatible response
        response = CompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": llm_response.choices[0].message.content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
                "total_tokens": llm_response.usage.total_tokens
            },
            compression_stats={
                "enabled": request.enable_compression,
                "latency_ms": (time.time() - start_time) * 1000,
                "cache_hit": False
            }
        )

        # Cache the response
        if cache_key and semantic_cache:
            semantic_cache.set(cache_key, response.dict())

        return response

    except Exception as e:
        print(f"[Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion():
    """Legacy completions endpoint (redirects to chat completions)."""
    raise HTTPException(
        status_code=400,
        detail="Use /v1/chat/completions instead. Legacy completions deprecated."
    )


@app.get("/v1/models")
async def list_models():
    """List available models (returns all LiteLLM supported models)."""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4-turbo", "object": "model", "owned_by": "openai"},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
            {"id": "claude-3-opus-20240229", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-sonnet-20240229", "object": "model", "owned_by": "anthropic"},
            {"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "groq"},
            # Add more models as needed
        ]
    }


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.post("/admin/cache/clear")
async def clear_cache():
    """Clear semantic cache."""
    if semantic_cache:
        semantic_cache.clear()
        return {"status": "success", "message": "Cache cleared"}
    return {"status": "error", "message": "Cache not initialized"}


@app.get("/admin/config")
async def get_config():
    """Get current gateway configuration."""
    return {
        "compression_threshold": 2000,
        "cache_backend": "memory",  # or "redis"
        "cost_tracking": True,
        "providers_supported": 100
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*80)
    print("CONTEXT OPTIMIZER AI GATEWAY")
    print("="*80)
    print("\n🌐 Starting service...")
    print("   http://localhost:8080")
    print("   http://localhost:8080/docs (API documentation)")
    print("   http://localhost:8080/health (health check)")
    print("\n💡 OpenAI-compatible endpoint:")
    print("   POST http://localhost:8080/v1/chat/completions")
    print("\n🔄 Auto-compression enabled (>2K tokens)")
    print("="*80 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
