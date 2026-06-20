"""
Compression Middleware for LiteLLM

Can be used as a callback or standalone middleware component.
"""

from typing import Dict, Any, List, Optional
import hashlib
import json


class CompressionMiddleware:
    """
    Middleware for automatic compression in LiteLLM pipelines.

    Usage:
        middleware = CompressionMiddleware()

        # Register as LiteLLM callback
        litellm.success_callback = [middleware.on_success]
        litellm.failure_callback = [middleware.on_failure]
    """

    def __init__(self, compression_threshold: int = 2000):
        self.compression_threshold = compression_threshold
        self.stats = {
            "total_requests": 0,
            "compressed_requests": 0,
            "total_savings_tokens": 0,
            "cache_hits": 0
        }

    def should_compress(self, messages: List[Dict[str, str]]) -> bool:
        """Determine if messages should be compressed."""
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        return total_length > self.compression_threshold

    def get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate cache key for messages."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def on_success(self, kwargs: Dict, response: Any, start_time: float, end_time: float):
        """Callback for successful LiteLLM requests."""
        self.stats["total_requests"] += 1

        # Log compression metrics if applicable
        if "compressed" in kwargs.get("model", ""):
            self.stats["compressed_requests"] += 1

    def on_failure(self, kwargs: Dict, error: Exception, start_time: float, end_time: float):
        """Callback for failed LiteLLM requests."""
        print(f"[Middleware] Request failed: {error}")

    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return self.stats.copy()
