"""
Ollama LLM builder.

Returns a ``CompressorLLM``-compatible object that talks to a local
Ollama instance (default: http://localhost:11434).

Requires either ``langchain-ollama`` (preferred) or ``litellm``:

    pip install 'context-optimizer[ollama]'
"""
from __future__ import annotations


def build(
    model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
):
    """
    Build an Ollama LLM instance.

    Tries ``langchain_ollama.ChatOllama`` first; falls back to a thin
    ``litellm`` wrapper if langchain-ollama is not installed.
    """
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url)
    except ImportError:
        pass

    try:
        import os
        import litellm

        class _LiteLLMOllama:
            def __init__(self, m: str, url: str) -> None:
                self._model = f"ollama/{m}"
                self._base  = url

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

    raise ImportError(
        "No Ollama provider available. "
        "Install one of: pip install langchain-ollama  OR  pip install litellm"
    )
