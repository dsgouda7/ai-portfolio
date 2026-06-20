"""
Context Optimizer Gateway - LiteLLM Wrapper

A pip-installable package that adds semantic compression to LiteLLM.
"""

from .litellm_wrapper import CompressedLiteLLM, CompressedRouter
from .middleware import CompressionMiddleware
from .cache import SemanticCache

__version__ = "0.1.0"

__all__ = [
    "CompressedLiteLLM",
    "CompressedRouter",
    "CompressionMiddleware",
    "SemanticCache",
]
