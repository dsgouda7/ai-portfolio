"""
Tests for Whisper model
"""

import pytest
import numpy as np
from src.models.whisper import WhisperModel


@pytest.fixture
def whisper_model(device):
    """Initialize Whisper model for testing."""
    return WhisperModel(device=device, cache_dir="./models/whisper")


def test_whisper_initialization(whisper_model):
    """Test Whisper model initialization."""
    assert whisper_model is not None
    assert whisper_model.model is not None
    assert whisper_model.processor is not None


def test_transcribe_audio_array(whisper_model, sample_audio):
    """Test transcription with audio array."""
    result = whisper_model.transcribe(sample_audio)
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "language" in result
    assert "duration" in result
    assert isinstance(result["text"], str)


def test_transcribe_audio_file(whisper_model, temp_audio_file):
    """Test transcription with audio file."""
    result = whisper_model.transcribe(temp_audio_file)
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "duration" in result
    assert result["duration"] > 0


def test_transcribe_with_confidence(whisper_model, sample_audio):
    """Test transcription with confidence scores."""
    result = whisper_model.transcribe_with_confidence(sample_audio)
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    
    # Confidence might be None if not available
    if result["confidence"] is not None:
        assert 0.0 <= result["confidence"] <= 1.0


def test_transcription_duration_matches(whisper_model, sample_audio):
    """Test that transcription duration matches audio duration."""
    expected_duration = len(sample_audio) / 16000  # 16kHz sample rate
    
    result = whisper_model.transcribe(sample_audio)
    
    assert abs(result["duration"] - expected_duration) < 0.1  # Within 0.1s


def test_language_setting(device):
    """Test Whisper with different language settings."""
    model_en = WhisperModel(device=device, language="en")
    model_es = WhisperModel(device=device, language="es")
    
    assert model_en.language == "en"
    assert model_es.language == "es"


def test_transcribe_empty_audio(whisper_model):
    """Test transcription with empty audio."""
    empty_audio = np.zeros(16000, dtype=np.float32)
    
    result = whisper_model.transcribe(empty_audio)
    
    assert isinstance(result, dict)
    assert "text" in result
    # Empty audio should produce empty or minimal transcription


def test_transcribe_short_audio(whisper_model):
    """Test transcription with very short audio."""
    short_audio = np.random.randn(1600).astype(np.float32)  # 0.1 seconds
    
    result = whisper_model.transcribe(short_audio)
    
    assert isinstance(result, dict)
    assert result["duration"] < 0.2
