"""
PixelSmith - Multimodal AI System

A production-grade multimodal AI system for processing text, images, and audio.
"""

from .models.clip import CLIPModel
from .models.whisper import WhisperModel
from .models.image_gen import ImageGenerator
from .models.image_caption import ImageCaptioner
from .features import MultimodalFeatureExtractor
from .utils import setup_logger, load_config

__version__ = "1.0.0"

__all__ = [
    "CLIPModel",
    "WhisperModel",
    "ImageGenerator",
    "ImageCaptioner",
    "MultimodalFeatureExtractor",
    "setup_logger",
    "load_config",
]
