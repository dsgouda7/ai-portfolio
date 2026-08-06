from __future__ import annotations

import pytest
from pydantic import ValidationError

from endpoint_client.config import EndpointClientConfig, EndpointProvider
from endpoint_client.models import ChatCompletionRequest, Usage


def test_environment_selects_a_validated_provider() -> None:
    config = EndpointClientConfig.from_environment(
        {
            "RIVERSIDE_ENDPOINT_PROVIDER": "apim",
            "RIVERSIDE_ENDPOINT_URL": "https://gateway.example",
            "RIVERSIDE_ENDPOINT_TOKEN_SCOPE": "api://riverside-gateway/.default",
            "RIVERSIDE_ENDPOINT_TIMEOUT_SECONDS": "25",
            "RIVERSIDE_ENDPOINT_MAX_RETRIES": "3",
            "AZURE_CLIENT_ID": "managed-identity-client-id",
        }
    )

    assert config.provider is EndpointProvider.APIM
    assert config.timeout_seconds == 25
    assert config.max_retries == 3
    assert config.managed_identity_client_id == "managed-identity-client-id"


@pytest.mark.parametrize("key", ["RIVERSIDE_ENDPOINT_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"])
def test_environment_rejects_api_key_authentication(key: str) -> None:
    with pytest.raises(ValueError, match="API-key authentication is not supported"):
        EndpointClientConfig.from_environment(
            {
                "RIVERSIDE_ENDPOINT_PROVIDER": "apim",
                "RIVERSIDE_ENDPOINT_URL": "https://gateway.example",
                "RIVERSIDE_ENDPOINT_TOKEN_SCOPE": "api://riverside-gateway/.default",
                key: "not-a-real-secret",
            }
        )


def test_foundry_requires_deployment_and_https() -> None:
    with pytest.raises(ValidationError):
        EndpointClientConfig(
            provider="foundry",
            endpoint_url="http://foundry.example",
            token_scope="https://cognitiveservices.azure.com/.default",
        )


def test_request_and_usage_enforce_runtime_invariants() -> None:
    with pytest.raises(ValidationError, match="stop entries must be unique"):
        ChatCompletionRequest(
            model="riverside-editor",
            messages=[{"role": "user", "content": "Question"}],
            max_input_tokens=100,
            max_tokens=20,
            stream=False,
            stop=["END", "END"],
        )
    with pytest.raises(ValidationError, match="total_tokens"):
        Usage(prompt_tokens=10, completion_tokens=4, total_tokens=15)
