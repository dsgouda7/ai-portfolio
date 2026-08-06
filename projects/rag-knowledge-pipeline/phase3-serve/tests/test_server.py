from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.models import ChatCompletionRequest, HealthResponse, QueryRequest
from src.server import _authorization


class Pipeline:
    def __init__(self, mode: str):
        self.mode = mode


def test_legacy_request_defaults_are_preserved() -> None:
    request = QueryRequest(question="What is retrieval augmentation?")
    assert request.top_k == 6
    assert request.temperature == 0.1


def test_v1_request_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "riverside-editor",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_input_tokens": 1024,
                "max_tokens": 128,
                "stream": False,
                "unknown": True,
            }
        )


def test_local_authorization_is_fixed_to_local_public_scope() -> None:
    authorization = _authorization(Pipeline("local"), "ignored", "ignored", "restricted", None, None)
    assert authorization.tenant_id == "local"
    assert authorization.classifications == ("public",)


def test_remote_authorization_requires_trusted_headers() -> None:
    with pytest.raises(HTTPException) as error:
        _authorization(Pipeline("remote"), None, None, None, None, None)
    assert error.value.status_code == 401


def test_health_response_names_selected_providers() -> None:
    response = HealthResponse(
        status="healthy",
        mode="remote",
        retrieval="Databricks index main.rag.index",
        generation="apim endpoint configured",
    )
    assert response.mode == "remote"
