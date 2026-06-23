"""
Dry-run tests for the voice-assistant server.

Run while the server is up:
    python test_dry_run.py

Each test prints PASS / FAIL with details.
No Ollama required for STT and TTS tests (silence → empty transcript is valid).
The /process-message test is skipped automatically when Ollama is not running.
"""

import base64
import importlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave

# Ensure UTF-8 output on Windows consoles (cp1252 can't encode arrows, ellipsis etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"


def _make_silent_wav(duration_s: int = 1, sample_rate: int = 16000) -> bytes:
    """Return a minimal valid WAV with silence."""
    samples = b"\x00\x00" * (sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples)
    return buf.getvalue()


def _post(path: str, body: bytes, content_type: str) -> tuple[int, bytes]:
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_get_index():
    print("TEST  GET /")
    req = urllib.request.Request(BASE + "/", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        status = resp.status
        body = resp.read().decode()
    assert status == 200, f"Expected 200, got {status}"
    assert "Voice" in body or "voice" in body, "index.html doesn't look right"
    print("PASS  GET / -> 200, HTML served\n")


def test_speech_to_text():
    print("TEST  POST /speech-to-text  (1 s silent WAV)")
    wav = _make_silent_wav()
    print(
        f"      Sending {len(wav)} bytes to STT ... (model loads on first call, may take ~30 s)"
    )
    status, body = _post("/speech-to-text", wav, "application/octet-stream")
    data = json.loads(body)
    assert status == 200, f"Expected 200, got {status}\n{body}"
    assert "text" in data, f"Missing 'text' key in response: {data}"
    # Silence -> empty / "null" is fine
    print(f"PASS  POST /speech-to-text -> 200, text={data['text']!r}\n")


def test_tts_only():
    """
    Hit /process-message with a pre-canned message but mock the LLM by
    temporarily checking the TTS path works via a direct Python call.
    This avoids needing Ollama for TTS validation.
    """
    print("TEST  TTS pipeline (direct Python call, no network LLM needed)")
    sys.path.insert(0, ".")
    import worker  # noqa: PLC0415

    audio = worker.text_to_speech("Hola mundo", "en-US-AriaNeural")
    assert (
        len(audio) > 1000
    ), f"TTS returned suspiciously small audio: {len(audio)} bytes"
    b64 = base64.b64encode(audio).decode()
    assert len(b64) > 0
    print(f"PASS  TTS returned {len(audio)} bytes of audio (edge-tts)\n")


def _detect_ollama_model() -> str | None:
    """Return the name of the first non-embedding Ollama model, or None."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        for m in models:
            if "embed" not in m.lower():
                return m
        return models[0] if models else None
    except Exception:
        return None


def test_process_message_skip_if_no_ollama():
    print("TEST  LLM + TTS pipeline (worker direct call)")

    model = _detect_ollama_model()
    if model is None:
        print("SKIP  Ollama not running or no models pulled -- skipping LLM test\n")
        return

    print(f"      Detected Ollama model: {model!r}")

    # Patch config + clear LRU cache so worker uses the detected model
    sys.path.insert(0, ".")
    import config as _cfg
    import worker as _wk

    os.environ["OLLAMA_LLM_MODEL"] = model
    _cfg.OLLAMA_LLM_MODEL = model  # patch live value so process_message picks it up

    print("      Calling process_message('Good morning') ...")
    response_text = _wk.process_message("Good morning")
    assert response_text, "LLM returned empty response"
    print(f"      LLM -> {response_text!r}")

    print("      Synthesising TTS response ...")
    audio = _wk.text_to_speech(response_text, "en-US-AriaNeural")
    assert len(audio) > 500, f"TTS too small: {len(audio)} bytes"
    print(f"PASS  LLM + TTS -> text={response_text!r}, audio={len(audio)} bytes\n")


if __name__ == "__main__":
    failures = []
    for fn in [
        test_get_index,
        test_speech_to_text,
        test_tts_only,
        test_process_message_skip_if_no_ollama,
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc}\n")
            failures.append(fn.__name__)

    if failures:
        print(f"--- {len(failures)} test(s) failed: {failures} ---")
        sys.exit(1)
    else:
        print("--- All tests passed ---")
