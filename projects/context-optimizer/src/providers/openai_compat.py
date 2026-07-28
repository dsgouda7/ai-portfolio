"""
Generic OpenAI-compatible LLM builder — zero mandatory dependencies.

Covers any endpoint that speaks the OpenAI Chat Completions API:
  - OpenAI           (base_url="https://api.openai.com/v1")
  - Groq             (base_url="https://api.groq.com/openai/v1")
  - Azure OpenAI     (base_url="https://<resource>.openai.azure.com/openai")
  - Anyscale, Fireworks, Together, any local vLLM/LM Studio server

Priority:
  1. openai package  (official SDK, streaming, retries)  [optional]
  2. litellm         (universal wrapper)                  [optional]
  3. _DirectLLM      (urllib only, always available)      [built-in fallback]
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class _DirectLLM:
    """
    Zero-dependency client for any OpenAI-compatible chat completions endpoint.

    Uses only Python stdlib — no openai, no litellm, no LangChain.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str) -> object:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible API error {exc.code}: {body}"
            ) from exc
        content = data["choices"][0]["message"]["content"]
        return type("_Msg", (), {"content": content})()


def build(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> object:
    """
    Build an OpenAI-compatible LLM instance.

    Falls back gracefully through the three tiers so the library works
    with no optional extras installed.
    """
    # Tier 1: openai package
    try:
        from openai import OpenAI  # type: ignore[import]

        client = OpenAI(base_url=base_url, api_key=api_key)

        class _OpenAIWrapper:
            def invoke(self, prompt: str) -> object:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return type("_Msg", (), {"content": resp.choices[0].message.content})()

        return _OpenAIWrapper()
    except ImportError:
        pass

    # Tier 2: litellm
    try:
        import litellm  # type: ignore[import]

        class _LiteLLMWrapper:
            def invoke(self, prompt: str) -> object:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return type("_Msg", (), {"content": resp.choices[0].message.content})()

        return _LiteLLMWrapper()
    except ImportError:
        pass

    # Tier 3: zero-dep direct HTTP
    return _DirectLLM(base_url, api_key, model, temperature, max_tokens)
