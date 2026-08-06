from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from endpoint_client import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamEvent
from rag_orchestrator import ACL, OrchestratorConfig, RAGOrchestrator, RetrievedChunk, SearchQuery, TrustedAuthContext


class FakeSearchIndex:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        self.queries.append(query)
        return self.results


class FakeEndpointClient:
    def __init__(self, response: ChatCompletionResponse) -> None:
        self.response = response
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.requests.append(request)
        return self.response

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionStreamEvent]:
        self.requests.append(request)
        if False:
            yield

    async def close(self) -> None:
        return None


class FakeTelemetry:
    def __init__(self) -> None:
        self.retrieval_events: list[dict[str, object]] = []
        self.generation_events: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []

    async def retrieval_finished(self, **values: object) -> None:
        self.retrieval_events.append(values)

    async def generation_finished(self, **values: object) -> None:
        self.generation_events.append(values)

    async def failed(self, **values: object) -> None:
        self.failures.append(values)


NOW = datetime(2026, 8, 5, 12, 1, tzinfo=UTC)


def _auth() -> TrustedAuthContext:
    return TrustedAuthContext(
        tenant_id="tenant-a",
        principal_id="editor-1",
        group_ids=frozenset({"editors"}),
        tenant_tier="premium",
    )


def _request(*, retrieval_enabled: bool = True, top_k: int = 20) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "riverside-editor",
            "messages": [{"role": "user", "content": "What does Aria see?"}],
            "max_input_tokens": 1024,
            "max_tokens": 128,
            "stream": False,
            "retrieval": {"enabled": retrieval_enabled, "top_k": top_k, "search_type": "hybrid"},
        }
    )


def _chunk(
    chunk_id: str,
    *,
    tenant_id: str = "tenant-a",
    visibility: str = "restricted",
    principals: frozenset[str] = frozenset(),
    groups: frozenset[str] = frozenset({"editors"}),
    deletion_status: str = "active",
) -> RetrievedChunk:
    return RetrievedChunk.model_validate(
        {
            "tenant_id": tenant_id,
            "chunk_id": chunk_id,
            "document_id": f"document-{chunk_id}",
            "source_uri": f"https://content.example/{chunk_id}.txt",
            "source_version": "2026-08-05T100000Z",
            "content_hash": "3" * 64,
            "title": f"Title {chunk_id}",
            "content": f"Authorized content for {chunk_id}.",
            "score": 0.95,
            "acl": {
                "visibility": visibility,
                "principals": principals,
                "groups": groups,
            },
            "region": "eastus2",
            "classification": "confidential",
            "index_version": "1.0.0",
            "deletion_status": deletion_status,
            "indexed_at": NOW,
        }
    )


def _response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-riverside0001",
            "created": 1785931260,
            "model": "riverside-editor",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            "citations": [],
            "trace": {"trace_id": "0123456789abcdef0123456789abcdef", "retry_count": 0},
            "deployment": {
                "environment": "staging",
                "release_id": "riverside-editor-2026-08-05",
                "model_alias": "riverside-editor",
                "deployment_name": "riverside-staging-green",
                "deployment_slot": "green",
                "region": "eastus2",
                "runtime": "azureml-riverside-runtime",
                "runtime_version": "1.0.0",
                "index_version": "1.0.0",
                "deployed_at": "2026-08-05T12:00:00Z",
                "source_commit": "0123456789abcdef0123456789abcdef01234567",
            },
        }
    )


@pytest.mark.asyncio
async def test_trusted_auth_bounds_retrieval_and_citation_lineage() -> None:
    authorized = _chunk("chunk-authorized")
    unauthorized = _chunk("chunk-denied", groups=frozenset({"reviewers"}))
    index = FakeSearchIndex([authorized, unauthorized, _chunk("chunk-overflow")])
    endpoint = FakeEndpointClient(_response("Aria sees harbor lights.<cite:chunk-authorized>"))
    telemetry = FakeTelemetry()
    orchestrator = RAGOrchestrator(
        index,
        endpoint,
        config=OrchestratorConfig(default_top_k=2, max_top_k=2),
        telemetry=telemetry,
        clock=lambda: NOW,
    )

    response = await orchestrator.complete(_request(top_k=20), _auth())

    assert index.queries[0].top_k == 2
    assert index.queries[0].auth == _auth()
    assert "tenant-a" not in str(index.queries[0].filters)
    context = endpoint.requests[0].messages[0].content
    assert "chunk-authorized" in context
    assert "chunk-denied" not in context
    assert response.choices[0].message.content == "Aria sees harbor lights."
    assert [citation.chunk_id for citation in response.citations] == ["chunk-authorized"]
    assert not hasattr(response.citations[0], "content")
    assert response.citations[0].answer_spans[0].end <= len(response.choices[0].message.content)
    assert telemetry.retrieval_events == [
        {
            "tenant_tier": "premium",
            "requested_top_k": 2,
            "result_count_bucket": "1",
            "outcome": "success",
        }
    ]


@pytest.mark.asyncio
async def test_cross_tenant_deleted_and_restricted_results_are_excluded() -> None:
    results = [
        _chunk("cross-tenant", tenant_id="tenant-b"),
        _chunk("deleted", deletion_status="deleted"),
        _chunk("restricted", groups=frozenset({"reviewers"})),
    ]
    index = FakeSearchIndex(results)
    endpoint = FakeEndpointClient(_response("This must be discarded."))
    orchestrator = RAGOrchestrator(index, endpoint, clock=lambda: NOW)

    response = await orchestrator.complete(_request(top_k=3), _auth())

    assert response.citations == []
    assert response.choices[0].message.refusal == "I cannot answer from the authorized sources available."
    assert all("cross-tenant" not in message.content for message in endpoint.requests[0].messages)


@pytest.mark.asyncio
async def test_unknown_or_missing_citation_markers_force_refusal() -> None:
    index = FakeSearchIndex([_chunk("chunk-authorized")])
    endpoint = FakeEndpointClient(_response("Unsupported answer.<cite:invented-chunk>"))
    orchestrator = RAGOrchestrator(index, endpoint, clock=lambda: NOW)

    response = await orchestrator.complete(_request(), _auth())

    assert response.citations == []
    assert response.choices[0].message.refusal is not None


@pytest.mark.asyncio
async def test_any_uncited_claim_forces_refusal() -> None:
    index = FakeSearchIndex([_chunk("chunk-authorized")])
    endpoint = FakeEndpointClient(
        _response("Supported claim.<cite:chunk-authorized> Uncited claim.")
    )
    orchestrator = RAGOrchestrator(index, endpoint, clock=lambda: NOW)

    response = await orchestrator.complete(_request(), _auth())

    assert response.citations == []
    assert response.choices[0].message.refusal is not None


@pytest.mark.asyncio
async def test_disabled_retrieval_bypasses_index_without_rewriting_response() -> None:
    original = _response("Direct response.")
    index = FakeSearchIndex([])
    endpoint = FakeEndpointClient(original)
    orchestrator = RAGOrchestrator(index, endpoint)

    response = await orchestrator.complete(_request(retrieval_enabled=False), _auth())

    assert response is original
    assert index.queries == []
    assert endpoint.requests == [_request(retrieval_enabled=False)]
