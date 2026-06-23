"""
LiteLLM multi-provider builder.

Supports any provider LiteLLM understands (Groq, OpenAI, Azure, Bedrock …).
Model strings follow LiteLLM conventions:

    groq/llama-3.3-70b
    openai/gpt-4o-mini
    azure/gpt-4o
    bedrock/meta.llama3-70b-instruct-v1:0

Requires litellm:

    pip install 'context-optimizer[litellm]'
"""
from __future__ import annotations

from typing import Any


def build(model: str, **completion_kwargs: Any):
    """
    Build a LiteLLM-backed LLM instance.

    Parameters
    ----------
    model:
        LiteLLM model string, e.g. ``"groq/llama-3.3-70b"``.
    **completion_kwargs:
        Passed verbatim to ``litellm.completion()`` (temperature, max_tokens …).
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for multi-provider support. "
            "Install with: pip install 'context-optimizer[litellm]'"
        ) from None

    class _LiteLLMWrapper:
        def __init__(self, m: str, kw: dict) -> None:
            self._model  = m
            self._kwargs = kw

        def invoke(self, prompt: str) -> object:
            resp = litellm.completion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **self._kwargs,
            )
            content = resp.choices[0].message.content
            return type("_Msg", (), {"content": content})()

    return _LiteLLMWrapper(model, completion_kwargs)
