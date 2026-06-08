"""Unit tests for RAG server."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "mode": "local",
        "local": {
            "vector_store_path": "./data/chroma_db",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "llm": {
            "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "max_tokens": 256,
            "temperature": 0.1
        },
        "retrieval": {
            "search_type": "mmr",
            "top_k": 6,
            "lambda_mult": 0.25
        },
        "logging": {
            "level": "INFO"
        }
    }


def test_query_request_validation():
    """Test QueryRequest model validation."""
    from phase3_serve.src.models import QueryRequest

    # Valid request
    valid_request = QueryRequest(question="What is machine learning?")
    assert valid_request.question == "What is machine learning?"
    assert valid_request.top_k == 6  # default

    # Invalid: empty question
    with pytest.raises(Exception):
        QueryRequest(question="")


def test_health_response_model():
    """Test HealthResponse model."""
    from phase3_serve.src.models import HealthResponse

    response = HealthResponse(
        status="healthy",
        vector_store="1000 vectors",
        llm="initialized"
    )

    assert response.status == "healthy"
    assert "vectors" in response.vector_store


@pytest.mark.skipif(
    not Path("../../data/chroma_db").exists(),
    reason="ChromaDB not available"
)
def test_server_health_endpoint():
    """Test FastAPI health endpoint (requires data)."""
    from phase3_serve.src.server import app

    client = TestClient(app)
    response = client.get("/health")

    # May fail if pipeline not initialized, but endpoint should respond
    assert response.status_code in [200, 503]
