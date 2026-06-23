"""
Provider configuration for the voice assistant.
Set env vars to switch between local (Ollama / faster-whisper / edge-tts)
and cloud (Azure AI Foundry / Azure Speech) providers.
"""

import os

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
# "ollama"  → local Ollama (default)
# "azure"   → Azure AI Foundry (OpenAI-compatible)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()

# "local"   → faster-whisper running on CPU/GPU (default)
# "azure"   → Azure Cognitive Services Speech SDK
STT_PROVIDER: str = os.getenv("STT_PROVIDER", "local").lower()

# "local"   → edge-tts (Microsoft Edge neural voices, no API key required)
# "azure"   → Azure Cognitive Services Speech SDK
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "local").lower()

# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "mistral")

# faster-whisper model size: tiny | base | small | medium | large-v3
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
# "cpu" or "cuda"
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
# "int8" | "float16" | "float32"
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Default edge-tts voice (used when caller sends "" or "default")
DEFAULT_LOCAL_VOICE: str = os.getenv("DEFAULT_LOCAL_VOICE", "en-US-AriaNeural")

# ---------------------------------------------------------------------------
# Azure AI Foundry (cloud LLM)
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION: str = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
)
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ---------------------------------------------------------------------------
# Azure Cognitive Services Speech (cloud STT + TTS)
# ---------------------------------------------------------------------------
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "eastus")
DEFAULT_AZURE_VOICE: str = os.getenv("DEFAULT_AZURE_VOICE", "en-US-AriaNeural")
