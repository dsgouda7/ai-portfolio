# Video Enhancer AI

AI-powered video upscaling and audio enhancement service that converts low-resolution videos to 4K with improved audio quality using open-source HuggingFace models.

## Features

- **4K Video Upscaling**: Converts low-res (720p/1080p) videos to 4K (3840×2160) using state-of-the-art super-resolution models
- **Audio Enhancement**: Improves audio quality with noise reduction and spectral enhancement
- **GPU Acceleration**: Automatically detects and utilizes NVIDIA GPUs for faster processing
- **CPU Fallback**: Uses optimized CPU models when GPU is not available
- **Parallel Processing**: Video and audio processed simultaneously for optimal performance
- **Dockerized**: Fully containerized for consistent deployment across platforms
- **REST API**: Simple HTTP endpoints for video enhancement
- **No API Keys Required**: 100% local processing with open-source models

## Models Used

### Video Enhancement

| Hardware | Model | Description |
|----------|-------|-------------|
| **GPU** | `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | High-quality 4x super-resolution optimized for real-world images |
| **CPU** | `caidas/swin2SR-classical-sr-x4-64` | Lightweight 4x upscaler suitable for CPU inference |

### Audio Enhancement

| Hardware | Model | Description |
|----------|-------|-------------|
| **GPU/CPU** | `speechbrain/metricgan-plus-voicebank` | Spectral mask enhancement for noise reduction and clarity improvement |

## Prerequisites

- **Docker Desktop** (automatically installed by setup scripts if not present)
- **System Requirements**:
  - **For GPU**: NVIDIA GPU with CUDA support, 4GB+ VRAM
  - **For CPU**: 8GB+ RAM, multi-core processor recommended
  - **Disk Space**: 10GB+ for models and processing
  - **Internet**: Required for initial model download (~2-3GB)

## Quick Start

### Windows (PowerShell)

```powershell
# 1. Run setup (installs Docker if needed)
.\setup.ps1

# 2. Start the service
.\run.ps1
```

### macOS / Linux (Bash)

```bash
# 1. Make scripts executable
chmod +x setup.sh run.sh

# 2. Run setup (installs Docker if needed)
./setup.sh

# 3. Start the service
./run.sh
```

The service will:
1. Build the Docker image (first run only - ~5 minutes)
2. Download AI models (~2-3GB, first run only)
3. Start the Flask server
4. Open your browser to `http://localhost:5000`

**First run takes 10-15 minutes** due to model downloads. Subsequent runs start in ~60 seconds.

## Usage

### REST API

#### Check Status

```bash
curl http://localhost:5000/api/status
```

Response:
```json
{
  "video": {
    "device": "cuda",
    "model_loaded": true
  },
  "audio": {
    "device": "cuda",
    "model_loaded": true
  }
}
```

#### Enhance Video

```bash
curl -X POST http://localhost:5000/api/enhance \
  -F "video=@input_video.mp4"
```

Response:
```json
{
  "status": "success",
  "output_file": "enhanced_input_video.mp4",
  "message": "Video enhanced to 4K with improved audio"
}
```

#### Download Enhanced Video

```bash
curl http://localhost:5000/api/download/enhanced_input_video.mp4 \
  --output enhanced_video.mp4
```

### Python Example

```python
import requests

# Upload and enhance video
with open('input_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/enhance',
        files={'video': f}
    )

result = response.json()
output_filename = result['output_file']

# Download enhanced video
response = requests.get(
    f'http://localhost:5000/api/download/{output_filename}'
)

with open('enhanced_output.mp4', 'wb') as f:
    f.write(response.content)

print(f"Enhanced video saved: enhanced_output.mp4")
```

## Architecture

### Project Structure

```
video_enhancer_ai/
├── app.py                  # Flask REST API server
├── video_processor.py      # Video upscaling logic with GPU detection
├── audio_processor.py      # Audio enhancement logic with GPU detection
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── setup.ps1 / setup.sh   # Environment setup scripts
├── run.ps1 / run.sh       # Docker run scripts
├── models/                # Cached models (auto-created)
├── cache/                 # HuggingFace cache (auto-created)
├── uploads/               # Temporary input files (auto-created)
└── outputs/               # Enhanced output files (auto-created)
```

### Processing Pipeline

```
1. Upload video → /app/uploads
2. Extract audio track (ffmpeg)
3. Parallel processing:
   ├─ Video: Upscale frames to 4K (Swin2SR)
   └─ Audio: Enhance quality (MetricGAN+)
4. Merge enhanced video + audio (ffmpeg)
5. Save to /app/outputs
6. Return download URL
```

### GPU Detection Logic

```python
# Automatic hardware detection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load appropriate model
if device == "cuda":
    model = load_gpu_optimized_model()  # High-quality, fast
else:
    model = load_cpu_optimized_model()   # Good quality, slower
```

## Performance

### Expected Processing Times

| Input Resolution | Hardware | Processing Time (per minute of video) |
|-----------------|----------|---------------------------------------|
| 720p → 4K | NVIDIA RTX 3080 | ~2-3 minutes |
| 1080p → 4K | NVIDIA RTX 3080 | ~3-5 minutes |
| 720p → 4K | CPU (8-core) | ~20-30 minutes |
| 1080p → 4K | CPU (8-core) | ~30-45 minutes |

**Note**: First-time processing includes model loading overhead (~30-60 seconds).

## Advanced Usage

### Custom Port

```powershell
# Windows
.\run.ps1 -Port 8080

# macOS/Linux
./run.sh --port 8080
```

### Skip Docker Build

```powershell
# Windows
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
docker logs -f video-enhancer-app
```

### Manual Container Management

```bash
# Stop the service
docker stop video-enhancer-app

# Restart the service
docker restart video-enhancer-app

# Remove the container
docker rm -f video-enhancer-app

# Rebuild image from scratch
docker build --no-cache -t video-enhancer-ai .
```

## Development

### Local Testing (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python app.py
```

### Model Configuration

Edit `video_processor.py` or `audio_processor.py` to use different models:

```python
self.model_config = {
    'gpu': {
        'name': 'your-preferred-model',  # Change model
        ...
    }
}
```

## Troubleshooting

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
docker logs video-enhancer-app

# Check disk space
df -h  # Linux/macOS
Get-PSDrive C | Select-Object Used,Free  # Windows PowerShell

# Clear cache and retry
rm -rf models/ cache/
./run.sh
```

### Out of Memory Errors

**GPU**:
- Reduce batch size in `video_processor.py`
- Use smaller model variant
- Close other GPU-intensive applications

**CPU**:
- Reduce resolution target (e.g., 1440p instead of 4K)
- Process shorter video segments
- Increase system swap/pagefile

### Port Already in Use

```powershell
# Use a different port
.\run.ps1 -Port 5001
```

### Container Won't Stop

```bash
# Force remove
docker rm -f video-enhancer-app

# Clean up all stopped containers
docker container prune
```

## Supported Formats

### Input Video Formats
- MP4, AVI, MKV, MOV, FLV, WMV, WebM
- Any format supported by OpenCV and FFmpeg

### Output Format
- MP4 (H.264 video + AAC audio)
- 3840×2160 resolution (4K)
- Original frame rate preserved

### Audio Formats
- Any audio track in input video
- Output: 16kHz mono WAV (enhanced), then converted to AAC

## Performance Tips

1. **GPU Acceleration**: Use a modern NVIDIA GPU for 10-20x faster processing
2. **Batch Processing**: Process multiple videos sequentially to amortize model loading time
3. **Persistent Cache**: Keep `./models` and `./cache` folders to avoid re-downloading
4. **RAM**: Allocate at least 8GB to Docker for CPU processing
5. **Disk Space**: Ensure 3x the input video size is available for temporary files

## Security Notes

- No API keys required (all models run locally)
- No data sent to external servers
- Models cached locally in `./models` and `./cache`
- Temporary files stored in `./uploads` and `./outputs`
- CORS enabled by default (restrict in production if needed)

## License

This project uses open-source models and libraries:
- **Swin2SR**: Apache 2.0 License
- **SpeechBrain**: Apache 2.0 License
- **FFmpeg**: LGPL/GPL License
- **Flask**: BSD License
- **PyTorch**: BSD License

## Acknowledgments

- **HuggingFace** for the Transformers library and model hosting
- **Microsoft Research** for Swin2SR models
- **SpeechBrain** for audio enhancement models
- **FFmpeg** for video/audio processing

## Related Projects

See also:
- [ibm_genai/voice_assistant](../ibm_genai/voice_assistant) - Voice-enabled conversational AI

---

**Built with open-source AI models from HuggingFace**
