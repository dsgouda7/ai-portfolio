"""
Shared protocols (interfaces) for context-optimizer components.

These are cross-cutting structural types that multiple modules depend on.
Keeping them here avoids circular imports and makes the dependency direction
clear: concrete implementations import from protocols, not from each other.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Minimal retriever interface accepted by ToTReasoner and other consumers.

    Any object with a ``.search()`` method that matches this signature satisfies
    the protocol — no inheritance required.  Concrete implementations include
    ``CachedChromaRetriever`` and ``ChromaRetriever``.
    """

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return a list of result dicts; each should contain 'compressed_summary'."""
        ...
