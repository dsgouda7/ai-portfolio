"""UnifiedAI - Production Neural Network System

A production-grade neural network classification API implementing deep learning
with comprehensive error handling, logging, and monitoring.
"""

__version__ = "1.0.0"
__author__ = "AI Portfolio"

from src.data import load_and_split
from src.features import FeatureEngineer
from src.models import ModelRegistry
from src.evaluate import AutoEvaluator

__all__ = [
    "load_and_split",
    "FeatureEngineer",
    "ModelRegistry",
    "AutoEvaluator",
]
