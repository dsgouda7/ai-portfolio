from __future__ import annotations

from typing import Protocol, Sequence

from .models import RetrievedChunk, SearchQuery


class SearchIndex(Protocol):
    async def search(self, query: SearchQuery) -> Sequence[RetrievedChunk]: ...


class TelemetryHooks(Protocol):
    async def retrieval_finished(
        self,
        *,
        tenant_tier: str,
        requested_top_k: int,
        result_count_bucket: str,
        outcome: str,
    ) -> None: ...

    async def generation_finished(
        self,
        *,
        tenant_tier: str,
        outcome: str,
        citation_count_bucket: str,
        finish_reason: str,
    ) -> None: ...

    async def failed(self, *, tenant_tier: str, category: str) -> None: ...


class NoopTelemetryHooks:
    async def retrieval_finished(
        self,
        *,
        tenant_tier: str,
        requested_top_k: int,
        result_count_bucket: str,
        outcome: str,
    ) -> None:
        return None

    async def generation_finished(
        self,
        *,
        tenant_tier: str,
        outcome: str,
        citation_count_bucket: str,
        finish_reason: str,
    ) -> None:
        return None

    async def failed(self, *, tenant_tier: str, category: str) -> None:
        return None
