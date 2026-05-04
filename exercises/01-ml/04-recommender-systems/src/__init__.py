"""FlixAI - Production Recommender System

A production-grade movie recommendation API implementing collaborative filtering
with comprehensive error handling, logging, and monitoring.
"""

__version__ = "1.0.0"
__author__ = "AI Portfolio"

from src.data import load_and_split_ratings
from src.features import EmbeddingGenerator
from src.models import ModelRegistry
from src.evaluate import RecommenderEvaluator

__all__ = [
    "load_and_split_ratings",
    "EmbeddingGenerator",
    "ModelRegistry",
    "RecommenderEvaluator",
]
