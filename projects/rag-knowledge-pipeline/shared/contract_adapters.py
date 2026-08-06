"""Adapters from local and Databricks retrieval records to one serving shape."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse

from shared.vector_contract_v1 import (
    CONTRACT_VERSION,
    VectorContractError,
    validate_vector_record_v1,
)

_IDENTIFIER_CHARACTER = re.compile(r"[^A-Za-z0-9._:-]")
_SYNTHETIC_LOCAL_INDEXED_AT = "1970-01-01T00:00:00Z"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_identifier(value: Any, fallback: str) -> str:
    normalized = _IDENTIFIER_CHARACTER.sub("-", str(value or fallback)).strip("-")
    return normalized[:128] or fallback


@dataclass(frozen=True)
class AuthorizationContext:
    tenant_id: str
    region: str
    classifications: tuple[str, ...]
    principal_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.region or not self.classifications:
            raise ValueError("Tenant, region, and classifications are required for retrieval")


@dataclass(frozen=True)
class RetrievedRecord:
    tenant_id: str
    chunk_id: str
    document_id: str
    source_uri: str
    source_version: str
    content_hash: str
    title: str
    content: str
    score: float
    region: str
    classification: str
    index_version: str
    indexed_at: str
    lineage_kind: str = "contract-v1"


def adapt_local_document(document: Any, score: float, *, ordinal: int) -> RetrievedRecord:
    """Adapt legacy Chroma metadata with explicit synthetic local lineage."""
    metadata = getattr(document, "metadata", {}) or {}
    content = str(getattr(document, "page_content", ""))
    document_id = safe_identifier(metadata.get("id"), f"local-document-{ordinal}")
    content_hash = sha256_text(content)
    chunk_id = safe_identifier(
        metadata.get("chunk_id"),
        f"chunk:{sha256_text(f'{document_id}:{content_hash}')[:32]}",
    )
    return RetrievedRecord(
        tenant_id="local",
        chunk_id=chunk_id,
        document_id=document_id,
        source_uri=str(metadata.get("source_uri") or f"urn:riverside:local:{document_id}"),
        source_version=str(metadata.get("source_version") or "legacy-local"),
        content_hash=content_hash,
        title=str(metadata.get("title") or document_id),
        content=content,
        score=float(score),
        region="local",
        classification="public",
        index_version=str(metadata.get("index_version") or "0.1.0-legacy-local"),
        indexed_at=str(metadata.get("indexed_at") or _SYNTHETIC_LOCAL_INDEXED_AT),
        lineage_kind="synthetic-local",
    )


def _title_from_source(source_uri: str, document_id: str) -> str:
    parsed = urlparse(source_uri)
    candidate = PurePosixPath(parsed.path).name if parsed.path else ""
    return candidate or document_id


def adapt_vector_index_record(record: Mapping[str, Any], score: float = 0.0) -> RetrievedRecord:
    """Validate and adapt a Databricks vector-index-record v1 result."""
    validate_vector_record_v1(record)
    deletion = record["deletion_state"]
    if deletion.get("status") != "active":
        raise VectorContractError("Serving rejects non-active vector index records")
    source_uri = str(record["source_uri"])
    document_id = str(record["document_id"])
    return RetrievedRecord(
        tenant_id=str(record["tenant_id"]),
        chunk_id=str(record["chunk_id"]),
        document_id=document_id,
        source_uri=source_uri,
        source_version=str(record["source_version"]),
        content_hash=str(record["content_hash"]),
        title=str(record.get("title") or _title_from_source(source_uri, document_id)),
        content=str(record["content"]),
        score=float(score),
        region=str(record["region"]),
        classification=str(record["classification"]),
        index_version=str(record["index_version"]),
        indexed_at=str(record["indexed_at"]),
    )
