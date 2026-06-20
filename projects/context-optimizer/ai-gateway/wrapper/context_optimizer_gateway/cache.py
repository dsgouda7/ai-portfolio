"""
Semantic Cache for Compressed Contexts

Redis-backed semantic caching with cross-user support.
"""

from typing import Optional, Dict, Any, List
import hashlib
import json
from datetime import datetime, timedelta


class SemanticCache:
    """
    Semantic cache for compressed contexts.

    Stores compressed chunks with semantic similarity matching.
    Supports multi-tenant caching (cross-user savings).

    Usage:
        cache = SemanticCache(redis_url="redis://localhost:6379")

        # Store compressed context
        cache.set("doc_key", compressed_chunks, ttl=3600)

        # Retrieve
        chunks = cache.get("doc_key")
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        enable_stats: bool = True
    ):
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats

        # Try to use Redis if available
        self.backend = self._init_backend(redis_url)

        # In-memory fallback
        self._memory_cache: Dict[str, Any] = {}

        # Stats tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }

    def _init_backend(self, redis_url: Optional[str]):
        """Initialize Redis backend if available."""
        if redis_url:
            try:
                import redis
                return redis.from_url(redis_url)
            except ImportError:
                print("[Cache] Redis not installed, using in-memory cache")
                return None
        return None

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value."""
        if self.backend:
            # Redis backend
            value = self.backend.get(key)
            if value:
                self.stats["hits"] += 1
                return json.loads(value)
            self.stats["misses"] += 1
            return None
        else:
            # In-memory backend
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if datetime.now() < entry["expires_at"]:
                    self.stats["hits"] += 1
                    return entry["value"]
                else:
                    # Expired
                    del self._memory_cache[key]
                    self.stats["evictions"] += 1

            self.stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache."""
        ttl = ttl or self.default_ttl
        self.stats["sets"] += 1

        if self.backend:
            # Redis backend
            self.backend.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        else:
            # In-memory backend
            self._memory_cache[key] = {
                "value": value,
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
            return True

    def semantic_key(
        self,
        content: str,
        query: Optional[str] = None
    ) -> str:
        """
        Generate semantic cache key.

        Uses content hash + optional query for similarity matching.
        """
        # Simple hash-based key (can be enhanced with embeddings)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if query:
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
            return f"sem:{content_hash}:{query_hash}"

        return f"sem:{content_hash}"

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.1%}",
            "backend": "redis" if self.backend else "memory"
        }

    def clear(self):
        """Clear all cached entries."""
        if self.backend:
            self.backend.flushdb()
        else:
            self._memory_cache.clear()

        self.stats["evictions"] += len(self._memory_cache)


# Example usage
if __name__ == "__main__":
    cache = SemanticCache()

    # Store compressed context
    cache.set("doc123", {"compressed": "summary", "tokens": 150})

    # Retrieve
    result = cache.get("doc123")
    print("Cached result:", result)

    # Stats
    print("Cache stats:", cache.get_stats())
