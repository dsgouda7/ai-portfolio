# 🎙️ Voice Assistant

A containerized voice-enabled conversational AI assistant powered by **HuggingFace Transformers**. Supports both text and speech input/output with a modern web interface.

## 🌟 Features

- **🗣️ Speech-to-Text**: Converts voice input to text using OpenAI Whisper (tiny model)
- **🤖 AI Conversation**: Generates intelligent responses using Microsoft DialoGPT
- **🔊 Text-to-Speech**: Converts text responses back to speech using Microsoft SpeechT5
- **🌐 Web Interface**: Modern, responsive UI for easy interaction
- **🐳 Dockerized**: Fully containerized for consistent deployment
- **📦 Lightweight Models**: Uses optimized HuggingFace models suitable for CPU inference
- **🆓 100% Free**: All models are open-source and free to use

## 📋 Prerequisites

- **Docker Desktop** (automatically installed by setup scripts if not present)
- **Windows**: PowerShell 5.1+ or PowerShell Core
- **macOS/Linux**: Bash shell
- **System Requirements**:
  - 4GB+ RAM (8GB+ recommended)
  - 5GB+ disk space for models
  - Internet connection for initial model download

## 🚀 Quick Start

### Windows (PowerShell)

```powershell
# 1. Run setup (installs Docker if needed)
.\setup.ps1

# 2. Start the application
.\run.ps1
```

### macOS / Linux (Bash)

```bash
# 1. Make scripts executable
chmod +x setup.sh run.sh

# 2. Run setup (installs Docker if needed)
./setup.sh

# 3. Start the application
./run.sh
```

The application will:
1. Build the Docker image (first run only)
2. Download AI models (~2GB, first run only)
3. Start the Flask server
4. Open your browser to `http://localhost:5000`

**First run takes 5-10 minutes** due to model downloads. Subsequent runs start in ~30 seconds.

## 🎯 Usage

### Web Interface

1. **Text Mode**:
   - Type your message in the text box
   - Click "💬 Send Text"
   - View the AI's response

2. **Speech Mode**:
   - Click "🎤 Record Speech"
   - Speak your message
   - Click "⏹️ Stop Recording"
   - View transcription, text response, and listen to audio response

### API Endpoints

The app exposes REST APIs for programmatic access:

#### Text Input
```bash
curl -X POST http://localhost:5000/api/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

#### Speech Input
```bash
curl -X POST http://localhost:5000/api/speech \
  -F "audio=@recording.webm"
```

#### Model Status
```bash
curl http://localhost:5000/api/status
```

## 🏗️ Architecture

### Project Structure
```
voice_assistant/
├── controllers.py          # Flask app with REST endpoints
├── model_manager.py        # HuggingFace model management
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── setup.ps1 / setup.sh   # Setup scripts
├── run.ps1 / run.sh       # Run scripts
├── models/                # Cached models (auto-created)
└── cache/                 # HuggingFace cache (auto-created)
```

### Components

#### 1. **Model Manager** (`model_manager.py`)
- Manages three HuggingFace models:
  - **STT**: `openai/whisper-tiny.en` (~150MB)
  - **LLM**: `microsoft/DialoGPT-small` (~350MB)
  - **TTS**: `microsoft/speecht5_tts` (~500MB)
- Lazy loading (downloads on first use)
- GPU acceleration if available (falls back to CPU)

#### 2. **Controllers** (`controllers.py`)
- Flask web server with REST APIs
- Handles text and audio input/output
- Serves embedded web interface
- CORS-enabled for external requests

#### 3. **Docker Container**
- Python 3.11 slim base image
- Pre-installed audio libraries (ffmpeg, portaudio)
- Persistent model caching via volumes
- Health checks for reliability

### Data Flow

```
┌─────────────┐
│ User Input  │
│ (Text/Audio)│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Flask Server   │
│  (controllers)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐        ┌──────────────┐
│ Model Manager   │───────▶│ Whisper STT  │ (if audio)
│                 │        └──────────────┘
│                 │               │
│                 │               ▼
│                 │        ┌──────────────┐
│                 │───────▶│ DialoGPT LLM │ (text generation)
│                 │        └──────────────┘
│                 │               │
│                 │               ▼
│                 │        ┌──────────────┐
│                 │───────▶│ SpeechT5 TTS │ (if audio output)
└─────────────────┘        └──────────────┘
       │
       ▼
┌─────────────┐
│  Response   │
│ (Text/Audio)│
└─────────────┘
```

## 🔧 Advanced Usage

### Custom Port

```powershell
# Windows
.\run.ps1 -Port 8080

# macOS/Linux
./run.sh --port 8080
```

### Skip Docker Build

```powershell
# Windows (rebuild manually first)
.\run.ps1 -NoBuild

# macOS/Linux
./run.sh --no-build
```

### Run Without Opening Browser

```powershell
# Windows
.\run.ps1 -NoBrowser

# macOS/Linux
./run.sh --no-browser
```

### View Live Logs

```bash
docker logs -f voice-assistant-app
```

### Manual Container Management

```bash
# Stop the app
docker stop voice-assistant-app

# Restart the app
docker restart voice-assistant-app

# Remove the container
docker rm -f voice-assistant-app

# Rebuild image from scratch
docker build --no-cache -t voice-assistant .
```

## 🛠️ Development

### Local Testing (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python controllers.py
```

### Modify Models

Edit `model_manager.py` to use different HuggingFace models:

```python
self.models_config = {
    'stt': {
        'name': 'openai/whisper-base.en',  # Larger, more accurate
        ...
    },
    'llm': {
        'name': 'facebook/blenderbot-400M-distill',  # Alternative
        ...
    },
    ...
}
```

### Add New Endpoints

Edit `controllers.py` to add custom routes:

```python
@app.route('/api/custom')
def custom_endpoint():
    # Your logic here
    return jsonify({'result': 'success'})
```

## 📊 Model Information

| Model | Size | Purpose | Speed (CPU) |
|-------|------|---------|-------------|
| **Whisper Tiny** | ~150MB | Speech recognition | ~1-2s per 10s audio |
| **DialoGPT Small** | ~350MB | Conversational AI | ~2-3s per response |
| **SpeechT5 TTS** | ~500MB | Speech synthesis | ~3-5s per sentence |

**Total disk usage**: ~2GB (including dependencies)

## 🔒 Security Notes

- No API keys required (all models run locally)
- No data sent to external servers
- Models cached locally in `./models` and `./cache`
- CORS enabled by default (restrict in production)

## 🐛 Troubleshooting

### Docker Not Starting
```bash
# Check Docker daemon
docker info

# Restart Docker Desktop
# Windows: Search for "Docker Desktop" and restart
# macOS: Open Docker Desktop from Applications
# Linux: sudo systemctl restart docker
```

### Models Not Downloading
```bash
# Check logs
docker logs voice-assistant-app

# Manually trigger download
curl -X POST http://localhost:5000/api/init-models

# Check disk space
df -h  # Linux/macOS
Get-PSDrive C | Select-Object Used,Free  # Windows PowerShell
```

### Port Already in Use
```powershell
# Use a different port
.\run.ps1 -Port 5001
```

### Microphone Not Working
- Ensure browser has microphone permissions
- Check system microphone settings
- Try a different browser (Chrome/Edge recommended)

### Container Won't Stop
```bash
# Force remove
docker rm -f voice-assistant-app

# Clean up all stopped containers
docker container prune
```

## 📈 Performance Tips

1. **GPU Acceleration**: If you have an NVIDIA GPU, install `nvidia-docker` for 5-10x faster inference
2. **Larger Models**: Use bigger models for better quality (requires more RAM)
3. **Persistent Cache**: Keep `./models` and `./cache` folders to avoid re-downloading
4. **RAM**: Allocate at least 4GB to Docker (8GB+ for larger models)

## 🤝 Contributing

Feel free to:
- Report bugs via issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project uses open-source models and libraries:
- **Whisper**: MIT License (OpenAI)
- **DialoGPT**: MIT License (Microsoft)
- **SpeechT5**: MIT License (Microsoft)
- **Flask**: BSD License

## 🙏 Acknowledgments

- **HuggingFace** for the Transformers library and model hosting
- **OpenAI** for Whisper speech recognition
- **Microsoft** for DialoGPT and SpeechT5 models

## 📚 Related Projects

See also:
- [playground/rag-agents](../playground/rag-agents) - RAG and agentic AI examples
- [exercises/03-ai](../../exercises/03-ai) - AI/ML exercise templates

---

**Built with ❤️ using HuggingFace Transformers**
