from __future__ import annotations

import asyncio
import secrets
from abc import ABC
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from .auth import AsyncTokenCredential, create_default_credential
from .config import EndpointClientConfig, EndpointProvider
from .errors import EndpointClientError, error_from_status, parse_retry_after, transport_error
from .models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamEvent
from .normalization import NormalizationContext, iter_sse_json, normalize_response, normalize_stream_event


class AsyncEndpointClient(Protocol):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionStreamEvent]: ...

    async def close(self) -> None: ...


Sleep = Callable[[float], Awaitable[None]]


class BaseEndpointClient(ABC):
    def __init__(
        self,
        config: EndpointClientConfig,
        credential: AsyncTokenCredential,
        http_client: httpx.AsyncClient,
        *,
        owns_credential: bool = False,
        owns_http_client: bool = False,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._config = config
        self._credential = credential
        self._http = http_client
        self._owns_credential = owns_credential
        self._owns_http_client = owns_http_client
        self._sleep = sleep

    def _url(self) -> str:
        return f"{self._config.endpoint_url.rstrip('/')}{self._config.route}"

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = stream
        return payload

    async def _headers(self, trace_id: str) -> dict[str, str]:
        try:
            token = await self._credential.get_token(self._config.token_scope)
        except Exception:
            raise error_from_status(
                401,
                trace_id=trace_id,
                retry_count=0,
                deployment=self._config.deployment_metadata,
            ) from None
        span_id = secrets.token_hex(8)
        headers = {
            "authorization": f"Bearer {token.token}",
            "content-type": "application/json",
            "traceparent": f"00-{trace_id}-{span_id}-01",
        }
        if self._config.azureml_deployment:
            headers["azureml-model-deployment"] = self._config.azureml_deployment
        return headers

    def _context(self, request: ChatCompletionRequest, trace_id: str, retry_count: int) -> NormalizationContext:
        return NormalizationContext(
            model_alias=request.model,
            trace_id=trace_id,
            retry_count=retry_count,
            deployment=self._config.deployment_metadata,
        )

    def _retry_delay(self, retry_count: int, retry_after: int | None) -> float:
        return float(retry_after) if retry_after is not None else min(4.0, 0.25 * (2**retry_count))

    async def _request(self, request: ChatCompletionRequest, trace_id: str) -> tuple[httpx.Response, int]:
        for retry_count in range(self._config.max_retries + 1):
            try:
                response = await self._http.post(
                    self._url(),
                    headers=await self._headers(trace_id),
                    json=self._payload(request, stream=False),
                    timeout=self._config.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                if retry_count >= self._config.max_retries:
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    ) from None
                await self._sleep(self._retry_delay(retry_count, None))
                continue
            if response.is_success:
                return response, retry_count
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            retryable = response.status_code == 429 or response.status_code in {408, 500, 502, 503, 504}
            if retryable and retry_count < self._config.max_retries:
                await response.aclose()
                await self._sleep(self._retry_delay(retry_count, retry_after))
                continue
            raise error_from_status(
                response.status_code,
                trace_id=trace_id,
                retry_count=retry_count,
                retry_after_seconds=retry_after,
                deployment=self._config.deployment_metadata,
            )
        raise AssertionError("retry loop exhausted")

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        trace_id = secrets.token_hex(16)
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                response, retry_count = await self._request(request, trace_id)
                try:
                    payload = response.json()
                except ValueError:
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    ) from None
                if not isinstance(payload, Mapping):
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    )
                try:
                    return normalize_response(payload, self._context(request, trace_id, retry_count))
                except ValidationError:
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    ) from None
        except TimeoutError:
            raise transport_error(
                "timeout",
                trace_id=trace_id,
                retry_count=self._config.max_retries,
                deployment=self._config.deployment_metadata,
            ) from None

    async def _open_stream(self, request: ChatCompletionRequest, trace_id: str) -> tuple[httpx.Response, int]:
        for retry_count in range(self._config.max_retries + 1):
            prepared = self._http.build_request(
                "POST",
                self._url(),
                headers=await self._headers(trace_id),
                json=self._payload(request, stream=True),
                timeout=self._config.timeout_seconds,
            )
            try:
                response = await self._http.send(prepared, stream=True)
            except (httpx.TimeoutException, httpx.NetworkError):
                if retry_count >= self._config.max_retries:
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    ) from None
                await self._sleep(self._retry_delay(retry_count, None))
                continue
            if response.is_success:
                return response, retry_count
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            retryable = response.status_code == 429 or response.status_code in {408, 500, 502, 503, 504}
            await response.aclose()
            if retryable and retry_count < self._config.max_retries:
                await self._sleep(self._retry_delay(retry_count, retry_after))
                continue
            raise error_from_status(
                response.status_code,
                trace_id=trace_id,
                retry_count=retry_count,
                retry_after_seconds=retry_after,
                deployment=self._config.deployment_metadata,
            )
        raise AssertionError("retry loop exhausted")

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionStreamEvent]:
        trace_id = secrets.token_hex(16)
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                response, retry_count = await self._open_stream(request, trace_id)
                context = self._context(request, trace_id, retry_count)
                try:
                    async for payload in iter_sse_json(response.aiter_lines()):
                        yield normalize_stream_event(payload, context)
                except (ValueError, ValidationError):
                    raise transport_error(
                        "backend_failure",
                        trace_id=trace_id,
                        retry_count=retry_count,
                        deployment=self._config.deployment_metadata,
                    ) from None
        except TimeoutError:
            raise transport_error(
                "timeout",
                trace_id=trace_id,
                retry_count=self._config.max_retries,
                deployment=self._config.deployment_metadata,
            ) from None
        finally:
            if response is not None:
                await response.aclose()

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()
        if self._owns_credential:
            await self._credential.close()


class AzureMLDirectClient(BaseEndpointClient):
    pass


class APIMGatewayClient(BaseEndpointClient):
    pass


class FoundryOpenAIClient(BaseEndpointClient):
    def _url(self) -> str:
        deployment = self._config.foundry_deployment
        route = f"/openai/deployments/{deployment}/chat/completions"
        return f"{self._config.endpoint_url.rstrip('/')}{route}?{urlencode({'api-version': self._config.foundry_api_version})}"

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop is not None:
            payload["stop"] = request.stop
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload


def create_endpoint_client(
    config: EndpointClientConfig,
    *,
    credential: AsyncTokenCredential | None = None,
    http_client: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> AsyncEndpointClient:
    owns_credential = credential is None
    owns_http_client = http_client is None
    resolved_credential = credential or create_default_credential(config.managed_identity_client_id)
    resolved_http_client = http_client or httpx.AsyncClient()
    client_type = {
        EndpointProvider.AZURE_ML: AzureMLDirectClient,
        EndpointProvider.APIM: APIMGatewayClient,
        EndpointProvider.FOUNDRY: FoundryOpenAIClient,
    }[config.provider]
    return client_type(
        config,
        resolved_credential,
        resolved_http_client,
        owns_credential=owns_credential,
        owns_http_client=owns_http_client,
        sleep=sleep,
    )
