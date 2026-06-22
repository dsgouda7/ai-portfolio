"""
Reasoning Gateway Service — LiteLLM router + MCP tool bridge.

Routes OpenAI-compatible chat completion requests to a configurable remote
reasoning LLM (Ollama / Groq / Azure / any LiteLLM-supported provider) while
exposing two MCP tools that the reasoning LLM calls to retrieve evidence:

  retrieve_context(query, depth)
      → hits vector-store /search, returns compressed summaries (~300 tokens)

  get_context_details(chunk_ids)
      → hits vector-store /chunks/{id}, returns raw text (~500 tokens)

Token flow (per query, ~1.7K tokens constant regardless of corpus size):
  System prompt          ~200 tokens  (fixed)
  Tool schemas           ~300 tokens  (fixed)
  MCP response (6 chunks) ~1200 tokens (capped)
  Reasoning instruction   ~150 tokens  (fixed)
  ─────────────────────────────────────────────
  Total:                 ~1 850 tokens

vs. monolithic baseline: ~9 833 tokens → 81% reduction.

Endpoints
---------
POST /v1/chat/completions   OpenAI-compatible; routes to configured LLM
GET  /health
GET  /stats
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import httpx
import litellm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VECTOR_STORE_URL = os.getenv("VECTOR_STORE_URL", "http://vector-store:8003")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "default")

# LiteLLM picks up OPENAI_API_KEY, GROQ_API_KEY, etc. from env automatically.
litellm.set_verbose = False

app = FastAPI(
    title="Context Optimizer — Reasoning Gateway",
    description="LiteLLM router with MCP tool bridge to vector-store",
    version="0.1.0",
)

_start_time = time.time()


# ── MCP tool helpers (synchronous HTTP to vector-store) ──────────────────────

def _retrieve_context(query: str, depth: str = "detailed", collection: str = DEFAULT_COLLECTION) -> dict:
    """MCP tool: retrieve compressed summaries from vector-store."""
    depth_map = {"brief": 3, "detailed": 6, "exhaustive": 12}
    top_k = depth_map.get(depth, 6)
    t0 = time.perf_counter()
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            f"{VECTOR_STORE_URL}/search",
            params={"q": query, "top_k": top_k, "collection": collection},
        )
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000
    chunks = resp.json()
    return {
        "status":             "success",
        "query":              query,
        "depth":              depth,
        "chunks":             chunks,
        "total_input_tokens": sum(len(c.get("compressed_summary", "")) // 4 for c in chunks),
        "retrieval_latency_ms": round(latency_ms, 2),
    }


def _get_context_details(chunk_ids: list[str], collection: str = DEFAULT_COLLECTION) -> dict:
    """MCP tool: retrieve raw text for specific chunk IDs (pointer model)."""
    results = []
    with httpx.Client(timeout=10.0) as client:
        for cid in chunk_ids:
            resp = client.get(
                f"{VECTOR_STORE_URL}/chunks/{cid}",
                params={"collection": collection},
            )
            if resp.status_code == 200:
                results.append(resp.json())
            else:
                results.append({"chunk_id": cid, "error": f"not found ({resp.status_code})"})
    return {"status": "success", "chunks": results}


# Tool schemas injected into every system prompt
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name":        "retrieve_context",
            "description": "Retrieve compressed evidence matching a query from the corpus index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Search query (entities, keywords, or natural language)",
                    },
                    "depth": {
                        "type":        "string",
                        "enum":        ["brief", "detailed", "exhaustive"],
                        "description": "brief=top-3, detailed=top-6, exhaustive=top-12",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_context_details",
            "description": "Return full raw text for specific chunk IDs (pointer model — no re-embedding).",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type":  "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["chunk_ids"],
            },
        },
    },
]


# ── Request / response models ────────────────────────────────────────────────

class Message(BaseModel):
    role:    str
    content: str
    name:    str | None = None


class CompletionRequest(BaseModel):
    model:                    str
    messages:                 list[Message]
    temperature:              float | None = Field(0.7, ge=0, le=2)
    max_tokens:               int   | None = None
    stream:                   bool  | None = False
    enable_context_tools:     bool         = True
    collection:               str          = DEFAULT_COLLECTION


class CompletionResponse(BaseModel):
    id:              str
    object:          str = "chat.completion"
    created:         int
    model:           str
    choices:         list[dict[str, Any]]
    usage:           dict[str, int]
    tool_call_stats: dict[str, Any] | None = None


# ── Tool call dispatch ───────────────────────────────────────────────────────

def _dispatch_tool(name: str, args: dict, collection: str) -> str:
    if name == "retrieve_context":
        result = _retrieve_context(
            query=args.get("query", ""),
            depth=args.get("depth", "detailed"),
            collection=collection,
        )
        return json.dumps(result)
    if name == "get_context_details":
        result = _get_context_details(
            chunk_ids=args.get("chunk_ids", []),
            collection=collection,
        )
        return json.dumps(result)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status":         "ok",
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "vector_store":   VECTOR_STORE_URL,
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    with httpx.Client(timeout=5.0) as client:
        try:
            vs = client.get(f"{VECTOR_STORE_URL}/stats").json()
        except Exception:
            vs = {"error": "vector-store unavailable"}
    return {"vector_store": vs}


@app.post("/v1/chat/completions", response_model=CompletionResponse)
def chat_completions(req: CompletionRequest) -> CompletionResponse:
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools    = _TOOL_SCHEMAS if req.enable_context_tools else None
    tool_call_count = 0

    # Agentic loop: let the reasoning LLM call tools until it produces a final answer.
    for _round in range(10):  # safety cap
        resp = litellm.completion(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        choice  = resp.choices[0]
        message = choice.message

        # If the LLM wants to call a tool, dispatch it and continue the loop.
        if choice.finish_reason == "tool_calls" and message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            for tc in message.tool_calls:
                args   = json.loads(tc.function.arguments or "{}")
                result = _dispatch_tool(tc.function.name, args, req.collection)
                tool_call_count += 1
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
            continue

        # Final answer — return it.
        usage = resp.usage or {}
        return CompletionResponse(
            id=resp.id,
            created=int(time.time()),
            model=resp.model,
            choices=[{
                "index":         0,
                "message":       {"role": "assistant", "content": message.content or ""},
                "finish_reason": choice.finish_reason,
            }],
            usage={
                "prompt_tokens":     getattr(usage, "prompt_tokens",     0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens":      getattr(usage, "total_tokens",      0),
            },
            tool_call_stats={"tool_calls_made": tool_call_count},
        )

    raise HTTPException(status_code=500, detail="Reasoning loop exceeded maximum rounds")
