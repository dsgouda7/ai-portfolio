"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(
        ...,
        description="The question to ask the RAG system",
        min_length=1,
        max_length=1000
    )

    top_k: Optional[int] = Field(
        default=6,
        description="Number of documents to retrieve",
        ge=1,
        le=20
    )

    temperature: Optional[float] = Field(
        default=0.1,
        description="LLM temperature for generation",
        ge=0.0,
        le=2.0
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str = Field(
        ...,
        description="The generated answer"
    )

    question: str = Field(
        ...,
        description="The original question"
    )

    sources_count: int = Field(
        ...,
        description="Number of source documents used"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_store: str
    llm: str
