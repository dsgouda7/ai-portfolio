"""Ports for the Delta-to-vector-index workflow."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import AuthorizationContext


class ParsedDocumentSource(Protocol):
    def read_documents(self, *, since: str | None = None) -> Sequence[Mapping[str, Any]]:
        """Return parsed-document v1 rows changed after the optional watermark."""


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text in the same order."""


class ContractRecordRepository(Protocol):
    def record_ids_for_document(self, tenant_id: str, document_id: str) -> set[str]:
        """Return active or pending vector record IDs for one document."""

    def merge_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Idempotently merge document-chunk contract rows."""

    def merge_vector_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Idempotently merge vector-index-record contract rows."""

    def mark_records_deleted(
        self,
        record_ids: Sequence[str],
        *,
        requested_at: str,
        deleted_at: str,
    ) -> None:
        """Persist deletion evidence for records removed from the serving index."""

    def write_quality_report(self, report: Mapping[str, Any]) -> str:
        """Persist a durable report and return its table or artifact URI."""


class VectorIndex(Protocol):
    def ensure_exists(self) -> None:
        """Create the configured immutable-schema direct-access index if absent."""

    def upsert(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Idempotently insert or replace records by primary key."""

    def delete(self, record_ids: Sequence[str]) -> None:
        """Idempotently delete records by primary key."""

    def query_authorized(
        self,
        query_vector: Sequence[float],
        context: AuthorizationContext,
        *,
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return only records authorized for the supplied server-derived context."""
