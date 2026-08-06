"""Generation providers for local OpenAI-compatible and Azure-hosted endpoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlencode


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    retry_count: int = 0
    deployment: Mapping[str, Any] | None = None


class GenerationProvider(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int,
    ) -> GenerationResult: ...

    def status(self) -> str: ...


class HttpGenerationProvider:
    def __init__(self, config: Mapping[str, Any], *, azure_contract: bool) -> None:
        import httpx

        endpoint = config["endpoint"]
        self._http = httpx.Client(timeout=float(config.get("timeout_seconds", 30)))
        self._endpoint = endpoint
        self._model_alias = str(config["model_alias"])
        self._max_retries = int(config.get("max_retries", 2))
        self._azure_contract = azure_contract
        self._credential = None
        if azure_contract:
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential(
                managed_identity_client_id=os.getenv("AZURE_CLIENT_ID")
            )

    def _url(self) -> str:
        base = str(self._endpoint["url"]).rstrip("/")
        provider = self._endpoint.get("provider")
        if provider == "foundry":
            deployment = self._endpoint.get("foundry_deployment")
            if not deployment:
                raise ValueError("Foundry generation requires foundry_deployment")
            query = urlencode({"api-version": self._endpoint.get("foundry_api_version", "2024-10-21")})
            return f"{base}/openai/deployments/{deployment}/chat/completions?{query}"
        return f"{base}{self._endpoint['route']}"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._azure_contract:
            token = self._credential.get_token(str(self._endpoint["token_scope"]))
            headers["authorization"] = f"Bearer {token.token}"
        else:
            token_env_var = self._endpoint.get("bearer_token_env_var")
            if token_env_var and os.getenv(str(token_env_var)):
                headers["authorization"] = f"Bearer {os.environ[str(token_env_var)]}"
        deployment = self._endpoint.get("azureml_deployment")
        if deployment:
            headers["azureml-model-deployment"] = str(deployment)
        return headers

    def _payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model_alias,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._endpoint.get("provider") == "foundry":
            payload.pop("model")
        if self._azure_contract and self._endpoint.get("provider") != "foundry":
            payload["max_input_tokens"] = max_input_tokens
            payload["retrieval"] = {"enabled": False, "top_k": 1}
        return payload

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int,
    ) -> GenerationResult:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
        )
        for retry_count in range(self._max_retries + 1):
            response = self._http.post(self._url(), headers=self._headers(), json=payload)
            if response.status_code in {408, 429, 500, 502, 503, 504} and retry_count < self._max_retries:
                time.sleep(min(4.0, 0.25 * (2**retry_count)))
                continue
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage", {})
            choice = body["choices"][0]
            return GenerationResult(
                content=str(choice["message"]["content"]),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                finish_reason=str(choice.get("finish_reason") or "stop"),
                retry_count=retry_count,
                deployment=body.get("deployment"),
            )
        raise RuntimeError("Generation retry loop exhausted")

    def status(self) -> str:
        return f"{self._endpoint.get('provider', 'openai-compatible')} endpoint configured"


class LegacyWatsonxProvider:
    def __init__(self, config: Mapping[str, Any]) -> None:
        from langchain_ibm import WatsonxLLM

        legacy = config["watsonx_legacy"]
        self._llm = WatsonxLLM(
            model_id=str(legacy["model_id"]),
            url=str(legacy["url"]),
            project_id=str(legacy["project_id"]),
            params={
                "max_new_tokens": int(config["max_tokens"]),
                "temperature": float(config.get("temperature", 0.1)),
            },
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int,
    ) -> GenerationResult:
        prompt = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
        content = str(self._llm.invoke(prompt))
        return GenerationResult(
            content=content,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
        )

    def status(self) -> str:
        return "legacy Watsonx provider configured"


def build_generation_provider(config: Mapping[str, Any]) -> GenerationProvider:
    generation = config["serving"]["generation"]
    provider = generation["provider"]
    if provider == "openai_compatible":
        return HttpGenerationProvider(generation, azure_contract=False)
    if provider == "azure_endpoint":
        return HttpGenerationProvider(generation, azure_contract=True)
    if provider == "watsonx_legacy":
        return LegacyWatsonxProvider(generation)
    raise ValueError(f"Unsupported generation provider: {provider}")
