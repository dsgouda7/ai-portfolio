# Mini Projects - Local Model Conversions

## Conversation Analyzer

**File**: `conversation_analyzer.py`

### Overview
Audio transcription and conversation analysis tool using 100% local models - no credentials or API keys required.

### Features
- **Speech-to-Text**: Transcribes audio files using Whisper (tiny.en model)
- **Key Points Extraction**: Summarizes conversations using FLAN-T5
- **Web Interface**: Gradio UI for easy file upload
- **CPU-Optimized**: Runs on standard CPUs without GPU

### Models Used

| Task | Model | Size | Description |
|------|-------|------|-------------|
| Speech Recognition | `openai/whisper-tiny.en` | ~150MB | Fast, accurate English transcription |
| Summarization | `google/flan-t5-small` | ~250MB | Instruction-following text summarization |

### Alternative Models

For better quality (at the cost of speed), uncomment in the code:
```python
# FLAN-T5-base: ~900MB, better quality summaries
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-base",
    device=-1
)
```

### Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python conversation_analyzer.py
   ```

3. **Access the web UI**:
   - Open browser to `http://localhost:7860`
   - Upload an audio file (MP3, WAV, etc.)
   - View transcription and key points

### Changes from Original

**Removed**:
- IBM Watson Machine Learning dependencies
- IBM credentials and API keys
- Langchain LLM chains
- Remote API calls

**Added**:
- Local FLAN-T5 model for summarization
- CPU-optimized inference
- Chunking for long conversations
- Combined output (transcription + key points)

### Technical Details

**Processing Flow**:
1. Audio file → Whisper model → Transcription
2. Transcription → FLAN-T5 → Key points summary
3. Combined output displayed in UI

**Memory Requirements**:
- Minimum: 2GB RAM
- Recommended: 4GB+ RAM for smooth operation

**First Run**:
- Models download automatically (~400MB total)
- Cached locally for future use
- No internet required after initial download

### Troubleshooting

**Out of memory errors**:
- Close other applications
- Use `whisper-tiny.en` instead of larger Whisper models
- Reduce `batch_size` in transcription

**Slow processing**:
- Normal for CPU inference
- First run slower due to model loading
- Consider upgrading to `flan-t5-base` for better results if speed is acceptable

### Performance

| Audio Length | Processing Time (CPU) |
|--------------|----------------------|
| 1 minute | ~30 seconds |
| 5 minutes | ~2-3 minutes |
| 10 minutes | ~5-6 minutes |

*Times vary based on CPU speed and audio complexity*
