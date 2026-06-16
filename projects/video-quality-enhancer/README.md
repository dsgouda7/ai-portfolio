# Video Quality Enhancer

## Problem statement

**Can open-source super-resolution and audio-denoising models running entirely locally upscale a consumer-quality video to 4K with improved audio, without any paid API or cloud GPU?**

Consumer video is often shot at 720p or 1080p on older hardware. Professional upscaling services are either expensive or require cloud uploads. This project builds a local service that applies state-of-the-art super-resolution (Swin2SR) and spectral audio enhancement (MetricGAN+) to arbitrary video files, with full GPU acceleration when available and a CPU fallback for machines without CUDA.

**Constraints we set for ourselves:**
- No paid APIs — 100% local processing with HuggingFace open-source models
- GPU acceleration where available; CPU fallback that actually completes in reasonable time
- Drop-in REST API: the same interface whether running on a workstation or a server

**Result:** 4× upscaling (720p → 2880p, 1080p → 4K) with Swin2SR; noise-reduced audio via MetricGAN+. First run takes 10–15 min for model downloads; subsequent runs start in ~60 seconds.

## Models

| Component | Model | Why |
|---|---|---|
| **Video (GPU)** | `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | BSRGAN-trained for real-world degradation; better on compressed consumer video than classical SR |
| **Video (CPU)** | `caidas/swin2SR-classical-sr-x4-64` | Lighter variant; acceptable quality at CPU-only speeds |
| **Audio** | `speechbrain/metricgan-plus-voicebank` | Spectral masking; removes background noise, improves vocal clarity |

All models are downloaded from HuggingFace Hub on first run and cached locally.

## How it works

1. Upload a video via REST API or the web UI
2. Video frames are extracted, processed through the SR model in batches, and reassembled
3. Audio track is extracted, enhanced through MetricGAN+, and merged back
4. Video and audio are processed in parallel threads to minimise wall-clock time
5. Enhanced file is available for download at `/api/download/{filename}`

## Quick start

```powershell
# Windows
.\setup.ps1   # installs Docker Desktop if not present
.\run.ps1     # builds image, downloads models (~2-3 GB), starts server
```

```bash
# macOS / Linux
./setup.sh && ./run.sh
```

Then open `http://localhost:5001` or use the REST API directly.

## API

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/status` | GET | Returns device (cuda/cpu) and model-loaded status |
| `POST /api/enhance` | POST | Upload video (`multipart/form-data`, field `video`) |
| `GET /api/download/{filename}` | GET | Download enhanced video |

```bash
# Enhance a video
curl -X POST http://localhost:5001/api/enhance -F "video=@input.mp4"

# Download result
curl http://localhost:5001/api/download/enhanced_input.mp4 -o output.mp4
```

## Requirements

- Docker Desktop
- For GPU path: NVIDIA GPU with CUDA, 4 GB+ VRAM
- For CPU path: 8 GB+ RAM, multi-core processor
- 10 GB+ disk space for models and working files
- Internet for initial model download only

## Limitations

Swin2SR processes frames individually — there is no temporal consistency model, so fine detail may flicker slightly between frames on high-motion footage. Processing time scales roughly linearly with video length (~1–2 min per minute of 1080p on a modern GPU). Audio enhancement is applied to the mixed track; stereo spatial information is preserved but not separately enhanced per channel.
