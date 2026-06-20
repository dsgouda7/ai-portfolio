"""
LiteLLM Wrapper with Semantic Compression

Drop-in replacement for LiteLLM with automatic context compression.
"""

from typing import List, Dict, Any, Optional, Union
import time
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

try:
    import litellm
    from litellm import Router, completion
except ImportError:
    raise ImportError("LiteLLM not installed. Run: pip install litellm")

from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.retriever import DualStorageRetriever


class CompressedLiteLLM:
    """
    LiteLLM wrapper with automatic compression.

    Usage:
        client = CompressedLiteLLM(
            compression_threshold=2000,  # Compress if > 2K tokens
            target_compression_ratio=0.02  # Target 2% of original
        )

        response = client.completion(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Large context..."},
                {"role": "user", "content": "Query"}
            ]
        )
    """

    def __init__(
        self,
        compression_threshold: int = 2000,
        target_compression_ratio: float = 0.02,
        cache_enabled: bool = True,
        track_costs: bool = True,
        **litellm_kwargs
    ):
        self.compression_threshold = compression_threshold
        self.target_compression_ratio = target_compression_ratio
        self.cache_enabled = cache_enabled
        self.track_costs = track_costs

        # Initialize LiteLLM with config
        litellm.drop_params = True  # Auto-drop unsupported params
        litellm.set_verbose = litellm_kwargs.get("verbose", False)

        # Cost tracking
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.total_cost_without_compression = 0.0
        self.total_cost_with_compression = 0.0

        # Cache for compressed contexts
        self._compression_cache: Dict[str, Any] = {}

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """
        Drop-in replacement for litellm.completion() with compression.

        Automatically detects large contexts and compresses them before
        sending to the LLM provider.
        """
        start_time = time.time()

        # Estimate tokens in messages
        total_content = self._extract_content(messages)
        original_tokens = self._estimate_tokens(total_content)

        # Decide whether to compress
        should_compress = original_tokens > self.compression_threshold

        if should_compress:
            # Compress large context
            compressed_messages = self._compress_messages(messages)
            compressed_tokens = self._estimate_tokens(
                self._extract_content(compressed_messages)
            )

            # Track savings
            self.total_original_tokens += original_tokens
            self.total_compressed_tokens += compressed_tokens

            if self.track_costs:
                self._update_cost_tracking(
                    model, original_tokens, compressed_tokens
                )

            # Log compression stats
            compression_ratio = compressed_tokens / original_tokens
            print(f"[Compression] {original_tokens} → {compressed_tokens} tokens "
                  f"({compression_ratio:.1%}, {time.time() - start_time:.2f}s)")

            # Use compressed messages
            messages_to_send = compressed_messages
        else:
            messages_to_send = messages

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=model,
                messages=messages_to_send,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"[LiteLLM Error] {e}")
            raise

    def _compress_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Compress large messages using rolling window strategy."""
        compressed_messages = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Only compress long system/user messages
            if role in ["system", "user"] and len(content) > 1000:
                # Split into lines for compression
                lines = content.split("\n")

                # Compress using rolling window
                compressed_chunks = compress_corpus_rolling(
                    corpus_lines=lines,
                    chunk_size_threshold=512,
                    chunk_overlap_tokens=128
                )

                # Reconstruct compressed content
                compressed_content = "\n".join(
                    chunk.compressed_summary for chunk in compressed_chunks
                )

                compressed_messages.append({
                    "role": role,
                    "content": compressed_content
                })
            else:
                # Keep short messages as-is
                compressed_messages.append(msg)

        return compressed_messages

    def _extract_content(self, messages: List[Dict[str, str]]) -> str:
        """Extract all content from messages."""
        return "\n".join(msg.get("content", "") for msg in messages)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return max(1, len(text) // 4)

    def _update_cost_tracking(
        self,
        model: str,
        original_tokens: int,
        compressed_tokens: int
    ):
        """Track cost savings from compression."""
        # Approximate cost per 1K tokens (can be enhanced with actual pricing)
        cost_map = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.0015,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
        }

        base_model = model.split("-compressed")[0]
        cost_per_1k = cost_map.get(base_model, 0.01)  # Default

        cost_without = (original_tokens / 1000) * cost_per_1k
        cost_with = (compressed_tokens / 1000) * cost_per_1k

        self.total_cost_without_compression += cost_without
        self.total_cost_with_compression += cost_with

    def get_stats(self) -> Dict[str, Any]:
        """Get compression and cost statistics."""
        if self.total_original_tokens == 0:
            return {"message": "No compressions performed yet"}

        savings = self.total_cost_without_compression - self.total_cost_with_compression
        roi = (
            self.total_cost_without_compression / self.total_cost_with_compression
            if self.total_cost_with_compression > 0 else 0
        )

        return {
            "total_compressions": self.total_original_tokens // self.compression_threshold,
            "original_tokens": self.total_original_tokens,
            "compressed_tokens": self.total_compressed_tokens,
            "compression_ratio": self.total_compressed_tokens / self.total_original_tokens,
            "cost_without_compression": f"${self.total_cost_without_compression:.2f}",
            "cost_with_compression": f"${self.total_cost_with_compression:.2f}",
            "total_savings": f"${savings:.2f}",
            "roi": f"{roi:.1f}x"
        }


class CompressedRouter(Router):
    """
    LiteLLM Router with compression middleware.

    Extends LiteLLM Router to add automatic compression for all models.
    """

    def __init__(
        self,
        model_list: List[Dict],
        compression_threshold: int = 2000,
        **kwargs
    ):
        super().__init__(model_list=model_list, **kwargs)
        self.compression_threshold = compression_threshold
        self.compressor = CompressedLiteLLM(
            compression_threshold=compression_threshold
        )

    def completion(self, model: str, messages: List[Dict], **kwargs):
        """Override completion to add compression."""
        return self.compressor.completion(
            model=model,
            messages=messages,
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    # Initialize compressed client
    client = CompressedLiteLLM(
        compression_threshold=1000,
        track_costs=True
    )

    # Test with large context
    large_context = "Documentation: " + ("AWS Lambda is a serverless compute service. " * 500)

    response = client.completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": large_context},
            {"role": "user", "content": "Explain Lambda pricing"}
        ]
    )

    print("\nResponse:", response.choices[0].message.content[:200])
    print("\nStats:", client.get_stats())
