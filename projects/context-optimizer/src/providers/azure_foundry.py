"""
Azure AI Foundry provider for context-optimizer.

Wraps the azure-ai-inference SDK to provide a CompressorLLM-compatible
interface for models deployed on Azure AI Foundry (Serverless API or
Managed Compute endpoints).

Recommended models (bang-for-buck on Azure AI Foundry serverless):
  phi-4-mini   3.8B  ~$0.0003/1K tokens  best efficiency
  phi-4        14B   ~$0.001/1K tokens   highest quality
  mistral-small-3.1  24B  strong at summarization

Config
------
Set the following environment variables (or pass to the constructor):
  AZURE_AI_FOUNDRY_ENDPOINT   - e.g. https://project.eastus2.models.ai.azure.com
  AZURE_AI_FOUNDRY_API_KEY    - your Foundry API key
  AZURE_AI_FOUNDRY_MODEL      - model name (default: phi-4-mini)

Usage via _build_local_llm (compressor.py)
-----------------------------------------
    provider = "azure_foundry"
    # Set env vars above, then:
    llm = _build_local_llm()
    response = llm.invoke("Summarize: ...")
    print(response.content)
"""
from __future__ import annotations

import os
import threading
from typing import Any


class _Response:
    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


class AzureFoundryLLM:
    """
    Drop-in compressor-LLM backed by Azure AI Foundry inference API.

    Parameters
    ----------
    endpoint:
        Azure AI Foundry endpoint URL.
        Pattern: ``https://<host>.<region>.models.ai.azure.com``
    api_key:
        Azure AI Foundry API key.  Alternatively, set
        ``AZURE_AI_FOUNDRY_API_KEY`` env var.
    model:
        Model deployment name (e.g. ``phi-4-mini``).
    max_tokens:
        Maximum tokens in the summary output (default 300).
    temperature:
        Sampling temperature (default 0.1 for deterministic summaries).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "phi-4-mini",
        max_tokens: int = 300,
        temperature: float = 0.1,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: Any = None
        self._lock = threading.Lock()

    # ── Lazy client init ──────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from azure.ai.inference import ChatCompletionsClient  # type: ignore[import]
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "azure-ai-inference is required for the azure_foundry provider.\n"
                "Install with:  pip install azure-ai-inference"
            ) from exc

        self._client = ChatCompletionsClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )
        return self._client

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, prompt: str) -> _Response:
        """
        Send *prompt* to Azure AI Foundry and return the completion.

        Returns a ``_Response`` with a ``.content`` attribute, matching
        the interface expected by ``ingest_file_blocks``.
        """
        with self._lock:
            client = self._get_client()
            try:
                from azure.ai.inference.models import UserMessage  # type: ignore[import]
                response = client.complete(
                    messages=[UserMessage(content=prompt)],
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                return _Response(response.choices[0].message.content or "")
            except Exception as exc:
                return _Response(f"[AZURE_FOUNDRY_ERROR: {exc}]")

    def __repr__(self) -> str:
        return f"AzureFoundryLLM(model={self._model!r}, endpoint={self._endpoint!r})"


def build(
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> AzureFoundryLLM:
    """
    Build an :class:`AzureFoundryLLM` from explicit args or env vars.

    Environment variables (used when args are None):
      AZURE_AI_FOUNDRY_ENDPOINT
      AZURE_AI_FOUNDRY_API_KEY
      AZURE_AI_FOUNDRY_MODEL      (default: phi-4-mini)
    """
    ep  = endpoint or os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", "")
    key = api_key  or os.environ.get("AZURE_AI_FOUNDRY_API_KEY",  "")
    mdl = model    or os.environ.get("AZURE_AI_FOUNDRY_MODEL", "phi-4-mini")

    if not ep:
        raise ValueError(
            "Azure AI Foundry endpoint is required. "
            "Set AZURE_AI_FOUNDRY_ENDPOINT or pass endpoint= to build()."
        )
    if not key:
        raise ValueError(
            "Azure AI Foundry API key is required. "
            "Set AZURE_AI_FOUNDRY_API_KEY or pass api_key= to build()."
        )

    return AzureFoundryLLM(
        endpoint=ep,
        api_key=key,
        model=mdl,
        max_tokens=max_tokens,
        temperature=temperature,
    )
