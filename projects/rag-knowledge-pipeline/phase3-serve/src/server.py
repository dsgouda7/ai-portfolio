"""FastAPI composition root for local and Databricks RAG serving modes."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from starlette.concurrency import run_in_threadpool


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config_loader import load_config
from shared.contract_adapters import AuthorizationContext
from shared.logging_config import get_logger, setup_logging

from .models import ChatCompletionRequest, ChatCompletionResponse, HealthResponse, QueryRequest, QueryResponse
from .rag_pipeline import RAGPipeline


logger = get_logger(__name__)
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    config = load_config(os.getenv("RAG_CONFIG_PATH", "config.yaml"))
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    rag_pipeline = RAGPipeline(config)
    yield
    rag_pipeline = None


app = FastAPI(
    title="Riverside RAG Knowledge Pipeline API",
    description="Provider-neutral RAG serving over local ChromaDB or Databricks AI Search",
    version="0.2.0",
    lifespan=lifespan,
)


def _require_pipeline() -> RAGPipeline:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


def _authorization(
    pipeline: RAGPipeline,
    tenant_id: str | None,
    region: str | None,
    classifications: str | None,
    principal_id: str | None,
    group_ids: str | None,
) -> AuthorizationContext:
    if pipeline.mode == "local":
        return AuthorizationContext("local", "local", ("public",))
    if not tenant_id or not region or not classifications:
        raise HTTPException(
            status_code=401,
            detail="Remote mode requires trusted tenant, region, and classification headers",
        )
    return AuthorizationContext(
        tenant_id=tenant_id,
        region=region,
        classifications=tuple(value.strip() for value in classifications.split(",") if value.strip()),
        principal_ids=(principal_id,) if principal_id else (),
        group_ids=tuple(value.strip() for value in (group_ids or "").split(",") if value.strip()),
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {"message": "Riverside RAG Knowledge Pipeline API", "version": "0.2.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    status = _require_pipeline().get_status()
    return HealthResponse(status="healthy", **{key: status[key] for key in ("mode", "retrieval", "generation")})


@app.post("/query", response_model=QueryResponse, tags=["Legacy"])
async def query_rag(
    request: QueryRequest,
    tenant_id: str | None = Header(default=None, alias="X-Riverside-Tenant-ID"),
    region: str | None = Header(default=None, alias="X-Riverside-Region"),
    classifications: str | None = Header(default=None, alias="X-Riverside-Classifications"),
    principal_id: str | None = Header(default=None, alias="X-Riverside-Principal-ID"),
    group_ids: str | None = Header(default=None, alias="X-Riverside-Group-IDs"),
) -> QueryResponse:
    pipeline = _require_pipeline()
    authorization = _authorization(pipeline, tenant_id, region, classifications, principal_id, group_ids)
    try:
        result = await run_in_threadpool(
            pipeline.query,
            request.question,
            authorization=authorization,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        return QueryResponse(**result)
    except Exception as exc:
        logger.error("Legacy query failed", extra={"failure_type": type(exc).__name__})
        raise HTTPException(status_code=502, detail="RAG backend failed") from exc


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Riverside v1"])
async def chat_completions(
    request: ChatCompletionRequest,
    tenant_id: str | None = Header(default=None, alias="X-Riverside-Tenant-ID"),
    region: str | None = Header(default=None, alias="X-Riverside-Region"),
    classifications: str | None = Header(default=None, alias="X-Riverside-Classifications"),
    principal_id: str | None = Header(default=None, alias="X-Riverside-Principal-ID"),
    group_ids: str | None = Header(default=None, alias="X-Riverside-Group-IDs"),
) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented by this serving project")
    pipeline = _require_pipeline()
    authorization = _authorization(pipeline, tenant_id, region, classifications, principal_id, group_ids)
    retrieval = request.retrieval
    try:
        result = await run_in_threadpool(
            pipeline.complete,
            [message.model_dump(exclude_none=True) for message in request.messages],
            authorization=authorization,
            retrieval_enabled=retrieval.enabled if retrieval else True,
            top_k=retrieval.top_k if retrieval else None,
            search_type=retrieval.search_type if retrieval else None,
            retrieval_filters=(
                retrieval.filters.model_dump(exclude_none=True)
                if retrieval and retrieval.filters
                else None
            ),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            max_input_tokens=request.max_input_tokens,
        )
        return ChatCompletionResponse.model_validate(pipeline.contract_response(result, request.model))
    except Exception as exc:
        logger.error("Chat completion failed", extra={"failure_type": type(exc).__name__})
        raise HTTPException(status_code=502, detail="RAG backend failed") from exc


@app.post("/reset", tags=["Legacy"])
async def reset_history() -> dict[str, str]:
    _require_pipeline().reset_history()
    return {"message": "Chat history cleared"}


@app.get("/status", tags=["Health"])
async def get_status() -> dict[str, str]:
    return _require_pipeline().get_status()
