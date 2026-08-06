from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from endpoint_client import ChatCompletionRequest, DeploymentMetadata, EndpointClientError, EndpointClientConfig
from endpoint_client.client import APIMGatewayClient, FoundryOpenAIClient, create_endpoint_client


@dataclass(frozen=True)
class FakeAccessToken:
    token: str = "managed-identity-token"


class FakeCredential:
    def __init__(self) -> None:
        self.scopes: list[str] = []
        self.closed = False

    async def get_token(self, *scopes: str, **kwargs: object) -> FakeAccessToken:
        self.scopes.extend(scopes)
        return FakeAccessToken()

    async def close(self) -> None:
        self.closed = True


class FailingCredential(FakeCredential):
    async def get_token(self, *scopes: str, **kwargs: object) -> FakeAccessToken:
        raise RuntimeError("credential provider included sensitive diagnostic details")


def deployment_metadata() -> DeploymentMetadata:
    return DeploymentMetadata.model_validate(
        {
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
        }
    )


def _request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="riverside-editor",
        messages=[{"role": "user", "content": "What happens?"}],
        max_input_tokens=256,
        max_tokens=64,
        stream=stream,
    )


def _config(provider: str, **overrides: Any) -> EndpointClientConfig:
    values: dict[str, Any] = {
        "provider": provider,
        "endpoint_url": "https://endpoint.example",
        "token_scope": "api://riverside/.default",
        "max_retries": 0,
        "deployment_metadata": deployment_metadata(),
    }
    values.update(overrides)
    return EndpointClientConfig(**values)


def _openai_response(content: str = "Grounded answer.") -> dict[str, Any]:
    return {
        "id": "chatcmpl-provider0001",
        "created": 1785931260,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


@pytest.mark.asyncio
async def test_apim_client_uses_bearer_identity_and_normalizes_response() -> None:
    credential = FakeCredential()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer managed-identity-token"
        assert request.headers["traceparent"].startswith("00-")
        return httpx.Response(200, json=_openai_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(_config("apim"), credential=credential, http_client=http_client)

    response = await client.complete(_request())

    assert isinstance(client, APIMGatewayClient)
    assert response.model == "riverside-editor"
    assert response.usage.total_tokens == 13
    assert response.deployment.release_id == "riverside-editor-2026-08-05"
    assert credential.scopes == ["api://riverside/.default"]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_azure_ml_client_sets_deployment_header() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["azureml-model-deployment"] == "riverside-green"
        return httpx.Response(200, json={"output": "Azure ML answer", "usage": {"input_tokens": 7, "output_tokens": 2}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(
        _config("azure_ml", route="/score", azureml_deployment="riverside-green"),
        credential=FakeCredential(),
        http_client=http_client,
    )

    response = await client.complete(_request())

    assert response.choices[0].message.content == "Azure ML answer"
    assert response.usage.total_tokens == 9
    await http_client.aclose()


@pytest.mark.asyncio
async def test_foundry_client_uses_deployment_route_and_openai_payload() -> None:
    observed: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        assert request.url.path == "/openai/deployments/riverside-managed/chat/completions"
        assert request.url.params["api-version"] == "2024-10-21"
        return httpx.Response(200, json=_openai_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(
        _config("foundry", foundry_deployment="riverside-managed"),
        credential=FakeCredential(),
        http_client=http_client,
    )

    await client.complete(_request())

    assert isinstance(client, FoundryOpenAIClient)
    assert "max_input_tokens" not in observed
    assert "retrieval" not in observed
    await http_client.aclose()


@pytest.mark.asyncio
async def test_retry_after_is_bounded_and_retry_count_is_reported() -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "9999"})
        return httpx.Response(200, json=_openai_response())

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(
        _config("apim", max_retries=1),
        credential=FakeCredential(),
        http_client=http_client,
        sleep=fake_sleep,
    )

    response = await client.complete(_request())

    assert delays == [300.0]
    assert response.trace.retry_count == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_backend_body_is_not_exposed_in_safe_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="credential=secret internal-resource=/subscriptions/123")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(
        _config("apim"), credential=FakeCredential(), http_client=http_client
    )

    with pytest.raises(EndpointClientError) as caught:
        await client.complete(_request())

    assert caught.value.envelope.error.code == "backend_failure"
    assert "secret" not in str(caught.value)
    assert "subscriptions" not in str(caught.value)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_credential_failure_is_normalized_without_provider_details() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    client = create_endpoint_client(
        _config("apim"), credential=FailingCredential(), http_client=http_client
    )

    with pytest.raises(EndpointClientError) as caught:
        await client.complete(_request())

    assert caught.value.envelope.error.code == "unauthorized"
    assert "sensitive" not in str(caught.value)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_streaming_sse_events_are_normalized() -> None:
    body = (
        'data: {"id":"chatcmpl-stream0001","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-stream0001","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_endpoint_client(
        _config("apim"), credential=FakeCredential(), http_client=http_client
    )

    events = [event async for event in client.stream(_request(stream=True))]

    assert events[0].choices[0].delta.content == "Hello"
    assert events[-1].choices[0].finish_reason == "stop"
    assert events[-1].usage is not None and events[-1].usage.total_tokens == 3
    await http_client.aclose()
