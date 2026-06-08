"""
Phase 3: RAG Query Server
FastAPI application serving RAG queries

This server provides REST endpoints for querying the RAG pipeline
built in Phases 1 and 2.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config_loader import load_config
from shared.logging_config import setup_logging, get_logger

from models import QueryRequest, QueryResponse, HealthResponse
from rag_pipeline import RAGPipeline


# Global pipeline instance
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global rag_pipeline

    # Startup: Initialize RAG pipeline
    logger = get_logger(__name__)
    logger.info("Starting RAG server...")

    try:
        config = load_config("config.yaml")
        log_level = config.get("logging", {}).get("level", "INFO")
        setup_logging(level=log_level)

        rag_pipeline = RAGPipeline(config)
        logger.info("RAG pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG server")


# Create FastAPI app
app = FastAPI(
    title="Databricks RAG API",
    description="Production RAG pipeline with Delta Lake and ChromaDB",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Databricks RAG API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the RAG pipeline components.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    status = rag_pipeline.get_status()

    return HealthResponse(
        status="healthy",
        vector_store=status["vectorstore"],
        llm=status["llm"]
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.

    Args:
        request: Query request with question and optional parameters

    Returns:
        Generated answer with metadata
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag_pipeline.query(
            question=request.question,
            temperature=request.temperature
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.post("/reset", tags=["RAG"])
async def reset_history():
    """
    Reset chat history.

    Clears the conversation history for the RAG pipeline.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    rag_pipeline.reset_history()
    return {"message": "Chat history cleared"}


@app.get("/status", tags=["Health"])
async def get_status():
    """
    Get detailed pipeline status.

    Returns configuration and component status.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    return rag_pipeline.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
