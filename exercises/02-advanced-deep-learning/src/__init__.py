"""ProductionCV - Edge-Deployable Computer Vision System

A production-grade object detection and instance segmentation system
with model compression and edge deployment capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Portfolio"

from src.data import COCODataLoader
from src.features import ImagePreprocessor
from src.models import ModelRegistry
from src.evaluate import CVEvaluator
from src.edge import EdgeDeployer

__all__ = [
    "COCODataLoader",
    "ImagePreprocessor",
    "ModelRegistry",
    "CVEvaluator",
    "EdgeDeployer",
]
