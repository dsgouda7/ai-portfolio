"""
Pytest configuration and fixtures
"""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "A red car on a highway"


@pytest.fixture
def sample_image():
    """Create sample RGB image."""
    # Create 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_audio():
    """Create sample audio waveform."""
    # Create 1 second of audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    return audio


@pytest.fixture
def temp_image_file(tmp_path, sample_image):
    """Create temporary image file."""
    filepath = tmp_path / "test_image.png"
    sample_image.save(filepath)
    return str(filepath)


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio):
    """Create temporary audio file."""
    import soundfile as sf
    filepath = tmp_path / "test_audio.wav"
    sf.write(str(filepath), sample_audio, 16000)
    return str(filepath)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "models": {
            "clip": {
                "model_name": "openai/clip-vit-base-patch32",
                "device": "cpu",
                "cache_dir": "./models/clip"
            },
            "whisper": {
                "model_name": "openai/whisper-base",
                "device": "cpu",
                "cache_dir": "./models/whisper",
                "language": "en"
            },
            "stable_diffusion": {
                "model_name": "runwayml/stable-diffusion-v1-5",
                "device": "cpu",
                "cache_dir": "./models/stable-diffusion"
            },
            "blip": {
                "model_name": "Salesforce/blip-image-captioning-base",
                "device": "cpu",
                "cache_dir": "./models/blip"
            }
        },
        "image": {
            "max_width": 512,
            "max_height": 512,
            "max_file_size_mb": 10
        },
        "audio": {
            "sample_rate": 16000,
            "max_duration_sec": 30,
            "max_file_size_mb": 50
        }
    }


@pytest.fixture
def device():
    """Get device for testing (CPU)."""
    return "cpu"
