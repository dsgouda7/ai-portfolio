"""
LLM Provider for local compression benchmarks.

Callable stubs — every model role is resolved from environment variables with
Ollama defaults.  Swap any model without touching code.

Environment variables — COMPRESSION LLM:
    CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER  ollama (default) | groq
    CONTEXT_OPTIMIZER_COMPRESSOR_MODEL     llama3.2:1b (default) — fast summariser, ~600 MB
    OLLAMA_BASE_URL                        http://localhost:11434 (default)
    GROQ_API_KEY                           required only when provider=groq

Environment variables — REASONING / JUDGE LLM:
    CONTEXT_OPTIMIZER_REASONING_MODEL      mistral:7b (default) — strong reasoning, ~4 GB Q4_K_M
    (provider and base_url inherit from compressor env vars)
    Note: judge reuses the reasoning LLM — build_reasoning_llm() for both roles.

Environment variables — EMBEDDING MODEL:
    CONTEXT_OPTIMIZER_EMBEDDING_BACKEND    sentence-transformers (default) | ollama
    CONTEXT_OPTIMIZER_EMBEDDING_MODEL      nomic-embed-text (when backend=ollama)
                                           all-MiniLM-L6-v2 (when backend=sentence-transformers)

Quick start (all local):
    ollama serve
    ollama pull llama3.2:1b    # compression
    ollama pull mistral:7b     # reasoning + judge
    python run_experiments.py
"""

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path so compressor is importable from any working dir
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from context_optimizer.compressor import _build_local_llm  # noqa: E402


def build_compression_llm():
    """
    Build the compression LLM (used during write-time summarisation).
    Default: llama3.2:1b via Ollama — fast, ~600 MB, good enough for chunk-level summarisation.
    Override: CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=llama3.2:3b for higher fidelity.
    """
    provider = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "ollama").lower()
    # llama3.2:1b: ~600 MB, fast summariser — good enough for chunk-level compression.
    # Upgrade to llama3.2:3b or phi3:mini for higher fidelity at extra cost.
    model_env = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "llama3.2:1b")

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(
            f"  [Compression LLM] Provider: Ollama  |  Model: {model_env}  |  URL: {base_url}"
        )
    elif provider == "groq":
        print(f"  [Compression LLM] Provider: Groq (cloud)  |  Model: {model_env}")
    else:
        print(f"  [Compression LLM] Provider: {provider}  |  Model: {model_env}")

    llm = _build_local_llm(provider=provider, model=model_env)

    if llm is None:
        print(
            f"\n  [WARN] Could not initialise compression LLM for provider '{provider}'."
        )
        print(f"  [WARN] Compression will fall back to truncation.")
        if provider == "ollama":
            print(f"  [HINT] ollama serve  &&  ollama pull {model_env}")
        return None

    try:
        llm.invoke("ping")
        print(f"  [OK] Compression LLM connected")
    except Exception as exc:
        print(f"  [WARN] Compression LLM health-check failed: {exc}")

    return llm


def build_reasoning_llm():
    """
    Build the reasoning LLM (used at query-time to answer questions and as judge).
    Default: mistral:7b via Ollama — strong general reasoning, ~4 GB Q4_K_M.
    Override: CONTEXT_OPTIMIZER_REASONING_MODEL=<model>
    """
    provider = os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER", "ollama").lower()
    # mistral:7b: strong general reasoning on CPU, ~4 GB (Q4_K_M quantisation).
    # Alternatives: phi3:medium (14B, excellent CPU perf), qwen2.5:7b, llama3.1:8b.
    model_env = os.getenv("CONTEXT_OPTIMIZER_REASONING_MODEL", "mistral:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if provider == "ollama":
        print(
            f"  [Reasoning  LLM] Provider: Ollama  |  Model: {model_env}  |  URL: {base_url}"
        )
    else:
        print(f"  [Reasoning  LLM] Provider: {provider}  |  Model: {model_env}")

    llm = _build_local_llm(provider=provider, model=model_env)

    if llm is None:
        print(
            f"\n  [WARN] Could not initialise reasoning LLM for provider '{provider}'."
        )
        if provider == "ollama":
            print(f"  [HINT] ollama serve  &&  ollama pull {model_env}")
        return None

    try:
        llm.invoke("ping")
        print(f"  [OK] Reasoning LLM connected")
    except Exception as exc:
        print(f"  [WARN] Reasoning LLM health-check failed: {exc}")

    return llm


def get_embedding_config() -> dict:
    """
    Return the active embedding backend config.

    Returns dict with keys:
        backend   : "sentence-transformers" | "ollama"
        model     : model name
        base_url  : Ollama URL (only relevant for ollama backend)
    """
    backend = os.getenv(
        "CONTEXT_OPTIMIZER_EMBEDDING_BACKEND", "sentence-transformers"
    ).lower()
    if backend == "ollama":
        model = os.getenv("CONTEXT_OPTIMIZER_EMBEDDING_MODEL", "nomic-embed-text")
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  [Embedding]      Backend: Ollama  |  Model: {model}  |  URL: {url}")
        return {"backend": "ollama", "model": model, "base_url": url}
    else:
        model = os.getenv("CONTEXT_OPTIMIZER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"  [Embedding]      Backend: sentence-transformers  |  Model: {model}")
        return {"backend": "sentence-transformers", "model": model, "base_url": None}

    return llm
