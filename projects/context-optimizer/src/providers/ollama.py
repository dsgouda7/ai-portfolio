"""
Ollama LLM builder — zero external dependency required.

Priority:
  1. langchain-ollama  (richer streaming, tool-calling)  [optional]
  2. litellm           (universal wrapper)               [optional]
  3. _OllamaDirectLLM  (urllib only, always available)   [built-in fallback]
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class _OllamaDirectLLM:
    """
    Zero-dependency Ollama client using stdlib urllib.

    Calls the Ollama ``/api/generate`` endpoint directly — no LangChain,
    no litellm, no extra packages required.  Any machine with Ollama
    running can use this.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_predict: int = 300,
    ) -> None:
        self._model = model
        self._base = base_url.rstrip("/")
        self._temperature = temperature
        self._num_predict = num_predict

    def invoke(self, prompt: str) -> object:
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._num_predict,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base}. "
                "Is Ollama running? (`ollama serve`)"
            ) from exc
        content = data.get("response", "")
        return type("_Msg", (), {"content": content})()


def build(
    model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    num_predict: int = 300,
) -> object:
    """
    Build an Ollama-compatible LLM instance.

    Falls back gracefully through the three tiers above so the library
    works even without any optional extras installed.
    """
    # Tier 1: langchain-ollama (richest; streaming, tool-calling)
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import]

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
        )
    except ImportError:
        pass

    # Tier 2: litellm (universal wrapper — handles model aliases, retries)
    try:
        import os
        import litellm  # type: ignore[import]

        class _LiteLLMOllama:
            def __init__(self, m: str, url: str) -> None:
                self._model = f"ollama/{m}"
                self._base = url

            def invoke(self, prompt: str) -> object:
                os.environ.setdefault("OLLAMA_API_BASE", self._base)
                resp = litellm.completion(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = resp.choices[0].message.content
                return type("_Msg", (), {"content": content})()

        return _LiteLLMOllama(model, base_url)
    except ImportError:
        pass

    # Tier 3: zero-dep direct HTTP — always available
    return _OllamaDirectLLM(model, base_url, temperature, num_predict)
