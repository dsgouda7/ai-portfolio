"""
Azure OpenAI provider for context-optimizer.

Builds a LangChain ``AzureChatOpenAI`` instance that satisfies the
``CompressorLLM`` protocol used by the compressor and the ToT reasoner.

Required environment variables
-------------------------------
AZURE_OPENAI_ENDPOINT      e.g. https://my-resource.openai.azure.com/
AZURE_OPENAI_API_KEY       Your Azure OpenAI API key
AZURE_OPENAI_API_VERSION   API version (default: 2024-02-01)

Optional environment variables
-------------------------------
AZURE_COMPRESSOR_DEPLOYMENT   Deployment for the cheap compressor (default: gpt-4o-mini)
AZURE_REASONER_DEPLOYMENT     Deployment for the heavier reasoner  (default: gpt-4o)

Usage
-----
    from context_optimizer.providers.azure import build_compressor, build_reasoner

    compressor_llm = build_compressor()          # gpt-4o-mini by default
    reasoner_llm   = build_reasoner()            # gpt-4o by default

    # Or with explicit deployment names:
    compressor_llm = build_compressor(deployment="gpt-4o-mini")
    reasoner_llm   = build_reasoner(deployment="gpt-4o")

    # Or pass either directly to the pipeline:
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.tot_reasoner import ToTReasoner

    chunks = compress_corpus_rolling(
        lines, strategy="llm", compressor_provider="azure", compressor_model="gpt-4o-mini"
    )
    reasoner = ToTReasoner(retriever=retriever, llm=build_reasoner())
"""
from __future__ import annotations

import os


def _get_azure_chat(deployment: str, temperature: float = 0.1):
    """Return an AzureChatOpenAI instance; raises ImportError / ValueError on bad config."""
    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Install langchain-openai to use the Azure provider:\n"
            "  pip install langchain-openai"
        ) from exc

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        raise ValueError(
            "Azure provider requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "environment variables to be set."
        )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
    )


def build_compressor(deployment: str | None = None):
    """
    Return an Azure-backed compressor LLM.

    Uses a cheap, fast model (``gpt-4o-mini`` by default).  Override via
    ``AZURE_COMPRESSOR_DEPLOYMENT`` or the *deployment* argument.
    """
    dep = deployment or os.getenv("AZURE_COMPRESSOR_DEPLOYMENT", "gpt-4o-mini")
    return _get_azure_chat(dep, temperature=0.1)


def build_reasoner(deployment: str | None = None):
    """
    Return an Azure-backed reasoning LLM.

    Uses a heavier, more capable model (``gpt-4o`` by default).  Override via
    ``AZURE_REASONER_DEPLOYMENT`` or the *deployment* argument.
    """
    dep = deployment or os.getenv("AZURE_REASONER_DEPLOYMENT", "gpt-4o")
    return _get_azure_chat(dep, temperature=0.0)
