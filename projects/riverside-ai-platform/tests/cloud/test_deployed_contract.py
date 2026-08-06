from __future__ import annotations

import os

import httpx
import pytest
from azure.identity.aio import DefaultAzureCredential

from endpoint_client import ChatCompletionResponse


_VALID_REQUEST = {
    "model": "riverside-editor",
    "messages": [{"role": "user", "content": "Summarize the authorized source."}],
    "max_input_tokens": 1024,
    "max_tokens": 128,
    "temperature": 0.1,
    "top_p": 1.0,
    "stream": False,
    "retrieval": {"enabled": True, "top_k": 6, "search_type": "hybrid"},
}


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"cloud test requires {name}")
    return value


@pytest.fixture
async def gateway_client():
    base_url = _required("RIVERSIDE_CLOUD_GATEWAY_URL")
    token_scope = _required("RIVERSIDE_CLOUD_GATEWAY_SCOPE")
    credential = DefaultAzureCredential(managed_identity_client_id=os.getenv("AZURE_CLIENT_ID"))
    token = await credential.get_token(token_scope)
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token.token}"},
        timeout=120,
    ) as client:
        yield client
    await credential.close()


@pytest.fixture
async def backend_client():
    async with httpx.AsyncClient(
        base_url=_required("RIVERSIDE_CLOUD_BACKEND_URL"),
        timeout=30,
    ) as client:
        yield client


@pytest.mark.cloud
async def test_deployed_health_and_readiness(backend_client: httpx.AsyncClient) -> None:
    health = await backend_client.get("/health")
    readiness = await backend_client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"


@pytest.mark.cloud
async def test_deployed_non_streaming_v1_contract(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        json=_VALID_REQUEST,
    )

    assert response.status_code == 200
    completion = ChatCompletionResponse.model_validate(response.json())
    assert completion.deployment.environment == _required("RIVERSIDE_ENVIRONMENT")
    assert completion.deployment.region == _required("AZURE_LOCATION")
    assert completion.deployment.deployment_name == _required("RIVERSIDE_ACTIVE_DEPLOYMENT_NAME")


@pytest.mark.cloud
async def test_direct_backend_rejects_forged_trusted_headers(
    backend_client: httpx.AsyncClient,
) -> None:
    response = await backend_client.post(
        "/v1/chat/completions",
        json=_VALID_REQUEST,
        headers={
            "X-Riverside-Tenant-ID": "forged-tenant",
            "X-Riverside-Actor-ID": "forged-actor",
            "X-Riverside-Tenant-Tier": "premium",
            "X-Riverside-Group-IDs": "administrators",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.cloud
async def test_direct_backend_rejects_invalid_bearer_token(
    backend_client: httpx.AsyncClient,
) -> None:
    response = await backend_client.post(
        "/v1/chat/completions",
        json=_VALID_REQUEST,
        headers={
            "Authorization": "Bearer not-a-valid-token",
            "X-Riverside-Tenant-ID": "forged-tenant",
            "X-Riverside-Actor-ID": "forged-actor",
            "X-Riverside-Tenant-Tier": "premium",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.cloud
async def test_gateway_rejects_request_over_one_mebibyte(
    gateway_client: httpx.AsyncClient,
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        content=b"x" * 1_048_577,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code in {400, 413}
