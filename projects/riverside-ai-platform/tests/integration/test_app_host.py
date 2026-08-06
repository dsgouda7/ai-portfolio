from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from app.config import PlatformConfig, load_platform_config
from app.main import create_app
from app.runtime import ApplicationRuntime
from endpoint_client import ChatCompletionRequest, ChatCompletionResponse
from rag_orchestrator import TrustedAuthContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment() -> dict[str, str]:
    return {
        "RIVERSIDE_RELEASE_MANIFEST_URI": "https://artifacts.example.invalid/dev/release.json",
        "RIVERSIDE_GATEWAY_BASE_URL": "https://dev.gateway.example.invalid",
        "RIVERSIDE_SERVING_ENDPOINT_NAME": "riverside-dev",
        "RIVERSIDE_BLUE_DEPLOYMENT_NAME": "riverside-dev-blue",
        "RIVERSIDE_GREEN_DEPLOYMENT_NAME": "riverside-dev-green",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://dev.otel.example.invalid",
        "RIVERSIDE_EVALUATION_REPORT_URI": "https://artifacts.example.invalid/dev/report.json",
    }


class FakeOrchestrator:
    def __init__(self, response: ChatCompletionResponse) -> None:
        self.response = response
        self.auth: TrustedAuthContext | None = None

    async def complete(
        self,
        request: ChatCompletionRequest,
        auth: TrustedAuthContext,
    ) -> ChatCompletionResponse:
        self.auth = auth
        return self.response


class FakeBackendIdentity:
    async def authenticate(self, authorization: str | None) -> None:
        if authorization != "Bearer backend-token":
            from app.auth import BackendAuthenticationError

            raise BackendAuthenticationError("invalid test backend token")


class FakeMeasurement:
    def __enter__(self) -> FakeMeasurement:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def stage(self, _: str) -> FakeMeasurement:
        return self

    def complete(self, **_: object) -> None:
        return None

    def record_first_token(self) -> None:
        return None


class FakeRequestTelemetry:
    def request_span(self) -> FakeMeasurement:
        return FakeMeasurement()


class FakeTelemetryFactory:
    def for_tenant_tier(self, _: str) -> FakeRequestTelemetry:
        return FakeRequestTelemetry()


class FakeRuntime:
    def __init__(self, config: PlatformConfig, response: ChatCompletionResponse) -> None:
        self.config = config
        self.orchestrator = FakeOrchestrator(response)
        self.backend_identity = FakeBackendIdentity()
        self.telemetry = FakeTelemetryFactory()
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    async def close(self) -> None:
        self.ready = False


def _runtime() -> FakeRuntime:
    profile = load_platform_config(PROJECT_ROOT / "config" / "dev.yaml", environ=_environment())
    response = ChatCompletionResponse.model_validate_json(
        (PROJECT_ROOT / "tests" / "fixtures" / "valid" / "app-chat-completion-response.json").read_text(
            encoding="utf-8"
        )
    )
    return FakeRuntime(profile, response)


def _request() -> dict[str, Any]:
    return json.loads(
        (PROJECT_ROOT / "tests" / "fixtures" / "valid" / "app-chat-completion-request.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.integration
def test_health_readiness_and_non_streaming_chat_contract() -> None:
    runtime = _runtime()
    application = create_app(runtime=cast(ApplicationRuntime, runtime))
    with TestClient(application) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").json() == {"status": "ready"}
        response = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={
                "Authorization": "Bearer backend-token",
                "X-Riverside-Tenant-ID": "tenant-a",
                "X-Riverside-Actor-ID": "editor-1",
                "X-Riverside-Tenant-Tier": "premium",
                "X-Riverside-Group-IDs": "editors,reviewers",
            },
        )

    assert response.status_code == 200
    ChatCompletionResponse.model_validate(response.json())
    assert runtime.orchestrator.auth is not None
    assert runtime.orchestrator.auth.group_ids == frozenset({"editors", "reviewers"})


@pytest.mark.integration
def test_missing_trusted_auth_context_returns_v1_error() -> None:
    application = create_app(runtime=cast(ApplicationRuntime, _runtime()))
    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"Authorization": "Bearer backend-token"},
        )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


@pytest.mark.integration
def test_backend_rejects_request_larger_than_apim_bound() -> None:
    application = create_app(runtime=cast(ApplicationRuntime, _runtime()))
    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={"Content-Length": "1048577", "Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.integration
def test_backend_counts_body_when_content_length_understates_size() -> None:
    application = create_app(runtime=cast(ApplicationRuntime, _runtime()))
    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"x" * 1_048_577,
            headers={"Content-Length": "2", "Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.integration
def test_streaming_request_uses_v1_sse_envelopes() -> None:
    request = _request()
    request["stream"] = True
    application = create_app(runtime=cast(ApplicationRuntime, _runtime()))
    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            json=request,
            headers={
                "Authorization": "Bearer backend-token",
                "X-Riverside-Tenant-ID": "tenant-a",
                "X-Riverside-Actor-ID": "editor-1",
                "X-Riverside-Tenant-Tier": "premium",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")
