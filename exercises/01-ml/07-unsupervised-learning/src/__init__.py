"""SegmentAI - Production Clustering System

A production-grade customer segmentation API implementing unsupervised learning
with comprehensive error handling, logging, and monitoring.
"""

__version__ = "1.0.0"
__author__ = "AI Portfolio"

from src.data import load_and_prepare
from src.features import FeatureEngineer
from src.models import ModelRegistry
from src.evaluate import AutoEvaluator

__all__ = [
    "load_and_prepare",
    "FeatureEngineer",
    "ModelRegistry",
    "AutoEvaluator",
]
