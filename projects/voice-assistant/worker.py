"""
Provider implementations for STT, LLM, and TTS.

Providers are selected at runtime via environment variables (see config.py).

STT providers:
  local  → faster-whisper (runs entirely on your machine)
  azure  → Azure Cognitive Services Speech SDK

LLM providers:
  ollama → Ollama running locally (OpenAI-compatible endpoint)
  azure  → Azure AI Foundry (same openai client, different endpoint/auth)

TTS providers:
  local  → edge-tts (Microsoft Edge neural voices — no API key required)
  azure  → Azure Cognitive Services Speech SDK
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from functools import lru_cache
from typing import Optional

import config

# ---------------------------------------------------------------------------
# Lazy imports — only import what the selected provider actually needs
# so the app can start without every optional package installed.
# ---------------------------------------------------------------------------


def _require(package: str, install_hint: str) -> None:
    """Raise a clear ImportError when an optional dependency is missing."""
    raise ImportError(
        f"Package '{package}' is required for the selected provider. "
        f"Install it with: {install_hint}"
    )


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        _require("faster-whisper", "pip install faster-whisper")
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )


def _stt_local(audio_binary: bytes) -> str:
    """Transcribe audio using faster-whisper (local)."""
    model = _get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_binary)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path)
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return text or "null"


def _stt_azure(audio_binary: bytes) -> str:
    """Transcribe audio using Azure Cognitive Services Speech SDK."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        _require(
            "azure-cognitiveservices-speech",
            "pip install azure-cognitiveservices-speech",
        )

    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY,
        region=config.AZURE_SPEECH_REGION,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_binary)
        tmp_path = tmp.name
    try:
        audio_cfg = speechsdk.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_cfg
        )
        result = recognizer.recognize_once()
    finally:
        os.unlink(tmp_path)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    return "null"


def speech_to_text(audio_binary: bytes) -> str:
    """Route to the configured STT provider."""
    if config.STT_PROVIDER == "azure":
        return _stt_azure(audio_binary)
    return _stt_local(audio_binary)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _get_llm_client():
    """Return an openai-compatible client for the configured LLM provider."""
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        _require("openai", "pip install openai")

    if config.LLM_PROVIDER == "azure":
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    # Ollama exposes an OpenAI-compatible API at /v1
    from openai import OpenAI

    return OpenAI(
        base_url=config.OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama ignores this value but the client requires it
    )


def process_message(user_message: str) -> str:
    """Send user_message to the LLM and return the response text."""
    client = _get_llm_client()
    model = (
        config.AZURE_OPENAI_DEPLOYMENT
        if config.LLM_PROVIDER == "azure"
        else config.OLLAMA_LLM_MODEL
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator. "
                    "Translate the user's English text into Spanish. "
                    "Reply ONLY with the translation — no explanations, no formatting, no extra text."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


async def _synth_edge_tts(text: str, voice: str) -> bytes:
    """Synthesise speech with edge-tts and return raw audio bytes (MP3)."""
    try:
        import edge_tts
    except ImportError:
        _require("edge-tts", "pip install edge-tts")

    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _tts_local(text: str, voice: str) -> bytes:
    """Convert text to speech using edge-tts (local, no API key needed)."""
    effective_voice = (
        voice if voice and voice != "default" else config.DEFAULT_LOCAL_VOICE
    )
    return asyncio.run(_synth_edge_tts(text, effective_voice))


def _tts_azure(text: str, voice: str) -> bytes:
    """Convert text to speech using Azure Cognitive Services."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        _require(
            "azure-cognitiveservices-speech",
            "pip install azure-cognitiveservices-speech",
        )

    effective_voice = (
        voice if voice and voice != "default" else config.DEFAULT_AZURE_VOICE
    )

    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY,
        region=config.AZURE_SPEECH_REGION,
    )
    speech_config.speech_synthesis_voice_name = effective_voice

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_cfg
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure TTS failed: {result.cancellation_details}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def text_to_speech(text: str, voice: str = "") -> bytes:
    """Route to the configured TTS provider."""
    if config.TTS_PROVIDER == "azure":
        return _tts_azure(text, voice)
    return _tts_local(text, voice)
