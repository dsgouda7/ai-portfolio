"""
PizzaBot - RAG-based Pizza Ordering Chatbot
"""

from .models import ChatbotEngine
from .rag import RAGPipeline
from .embeddings import EmbeddingManager
from .api import app

__version__ = "1.0.0"

__all__ = [
    "ChatbotEngine",
    "RAGPipeline",
    "EmbeddingManager",
    "app",
]
