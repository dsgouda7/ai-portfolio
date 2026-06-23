# Voice Assistant

A Flask voice assistant that converts speech → LLM response → speech. Ported from IBM Watson/WatsonX to a provider-agnostic architecture that supports local Ollama models and Azure AI Foundry.

## Architecture

```
browser audio → POST /speech-to-text
                    └── STT provider ──► text
                                           │
               POST /process-message ──► LLM provider ──► response text
                                                               │
                                         TTS provider ──► audio bytes (base64)
```

| Component | Local (default) | Cloud |
|-----------|----------------|-------|
| STT | `faster-whisper` | Azure Cognitive Services Speech |
| LLM | Ollama (OpenAI-compatible API) | Azure AI Foundry |
| TTS | `edge-tts` (no API key needed) | Azure Cognitive Services Speech |

## Quick Start (Local / Ollama)

```powershell
# Windows
.\setup.ps1
.\.venv\Scripts\python server.py
```

```bash
# macOS / Linux
./setup.sh
.venv/bin/python server.py
```

The server starts on `http://localhost:8000`.

## Switching providers

All provider selection is done via environment variables. Copy `.env.example` to `.env` and edit:

### Use a different Ollama model

```env
OLLAMA_LLM_MODEL=llama3.2:3b    # lighter, faster
OLLAMA_LLM_MODEL=gemma3:12b     # higher quality
```

Pull the model first: `ollama pull llama3.2:3b`

### Switch to Azure AI Foundry (LLM)

```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Switch to Azure Speech (STT + TTS)

```env
STT_PROVIDER=azure
TTS_PROVIDER=azure
AZURE_SPEECH_KEY=<key>
AZURE_SPEECH_REGION=eastus
```

Then install the Azure Speech SDK:

```bash
pip install azure-cognitiveservices-speech
```

### Full Azure stack (no local models required)

```env
LLM_PROVIDER=azure
STT_PROVIDER=azure
TTS_PROVIDER=azure
```

## Available Ollama models

| Model | Size | Notes |
|-------|------|-------|
| `mistral` | ~4 GB | Default — good quality, fast |
| `llama3.2:3b` | ~2 GB | Lightweight fallback |
| `llama3.1:8b` | ~5 GB | Higher quality |
| `gemma3:4b` | ~3 GB | Google's model |

## Available edge-tts voices (local TTS)

Set `DEFAULT_LOCAL_VOICE` in `.env`, or pass `voice` in the request JSON.

Popular English voices:
- `en-US-AriaNeural` (female, default)
- `en-US-GuyNeural` (male)
- `en-US-JennyNeural` (female)
- `en-GB-SoniaNeural` (British female)

Full list: https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462

## API

### `POST /speech-to-text`

Body: raw audio bytes (WAV)

Response:
```json
{ "text": "hello world" }
```

### `POST /process-message`

```json
{
  "userMessage": "Good morning",
  "voice": "en-US-AriaNeural"
}
```

Response:
```json
{
  "responseText": "Buenos días",
  "responseSpeech": "<base64-encoded audio>"
}
```

## Templates

Place your `index.html` in the `templates/` directory. The server renders it at `GET /`.

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `azure` |
| `STT_PROVIDER` | `local` | `local` or `azure` |
| `TTS_PROVIDER` | `local` | `local` or `azure` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API URL |
| `OLLAMA_LLM_MODEL` | `mistral` | Model name to use |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `DEFAULT_LOCAL_VOICE` | `en-US-AriaNeural` | Default edge-tts voice |
| `AZURE_OPENAI_ENDPOINT` | — | Azure AI Foundry endpoint URL |
| `AZURE_OPENAI_API_KEY` | — | Azure AI Foundry API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment name |
| `AZURE_SPEECH_KEY` | — | Azure Speech Services key |
| `AZURE_SPEECH_REGION` | `eastus` | Azure Speech region |
