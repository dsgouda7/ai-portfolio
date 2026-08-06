"""Configuration, contract builders, and runtime invariants for remote indexing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_VERSION = "1.0.0"
SEMANTIC_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    value = mapping.get(name)
    if value is None or value == "":
        raise ValueError(f"Remote vectorization requires {name!r}")
    return value


def _configured(
    mapping: Mapping[str, Any],
    name: str,
    environment_name: str,
    default: Any = None,
) -> Any:
    value = mapping.get(name)
    if value is not None and value != "":
        return value
    return os.getenv(environment_name, default)


def _semantic_version(value: str, field_name: str) -> str:
    if not SEMANTIC_VERSION.fullmatch(value):
        raise ValueError(f"{field_name} must be a semantic version, got {value!r}")
    return value


def utc_now() -> str:
    """Return a contract-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_identifier(prefix: str, parts: Iterable[Any]) -> str:
    canonical = json.dumps(list(parts), ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}:{sha256_text(canonical)[:32]}"


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    model_revision: str
    dimensions: int
    endpoint_name: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision or not self.endpoint_name:
            raise ValueError("Embedding model, revision, and endpoint must be pinned")
        if not 1 <= self.dimensions <= 65536:
            raise ValueError("Embedding dimensions must be between 1 and 65536")

    def descriptor(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True)
class ChunkingSpec:
    strategy: str = "recursive-character"
    version: str = "1.0.0"
    size: int = 1024
    overlap: int = 128
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

    def __post_init__(self) -> None:
        _semantic_version(self.version, "chunk_strategy_version")
        if self.strategy != "recursive-character":
            raise ValueError(f"Unsupported chunk strategy: {self.strategy}")
        if self.size < 32:
            raise ValueError("chunk_size must be at least 32 code points")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


@dataclass(frozen=True)
class RemoteSettings:
    catalog: str
    schema: str
    source_table: str
    chunk_table: str
    vector_record_table: str
    quality_table: str
    evaluation_table: str
    evaluation_output_table: str
    vector_search_endpoint: str
    index_name: str
    index_version: str
    pipeline_version: str
    embedding: EmbeddingSpec
    chunking: ChunkingSpec
    batch_size: int = 128
    query_endpoint_type: str = "standard"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RemoteSettings":
        catalog = str(_required(mapping, "catalog"))
        schema = str(_required(mapping, "schema"))
        endpoint = str(_required(mapping, "vector_search_endpoint"))
        model_id = str(_required(mapping, "embedding_model"))
        model_revision = str(
            _configured(
                mapping,
                "embedding_model_revision",
                "RIVERSIDE_EMBEDDING_MODEL_REVISION",
                "",
            )
        )
        if not model_revision:
            raise ValueError(
                "Remote vectorization requires embedding_model_revision or "
                "RIVERSIDE_EMBEDDING_MODEL_REVISION"
            )
        dimensions_value = _configured(
            mapping,
            "embedding_dimensions",
            "RIVERSIDE_EMBEDDING_DIMENSIONS",
        )
        if dimensions_value is None:
            raise ValueError(
                "Remote vectorization requires embedding_dimensions or "
                "RIVERSIDE_EMBEDDING_DIMENSIONS"
            )

        index_version = _semantic_version(
            str(_configured(mapping, "index_version", "RIVERSIDE_INDEX_VERSION", "1.0.0")),
            "index_version",
        )
        pipeline_version = _semantic_version(
            str(
                _configured(
                    mapping,
                    "pipeline_version",
                    "RIVERSIDE_VECTORIZATION_PIPELINE_VERSION",
                    "1.0.0",
                )
            ),
            "pipeline_version",
        )
        chunk_version = str(
            _configured(
                mapping,
                "chunk_strategy_version",
                "RIVERSIDE_CHUNK_STRATEGY_VERSION",
                "1.0.0",
            )
        )
        index_name = str(
            _configured(
                mapping,
                "index_name",
                "RIVERSIDE_VECTOR_INDEX_NAME",
                f"{catalog}.{schema}.riverside_manuscripts_v1",
            )
        )
        embedding_endpoint = str(
            _configured(
                mapping,
                "embedding_endpoint",
                "RIVERSIDE_EMBEDDING_ENDPOINT",
                model_id,
            )
        )
        batch_size = int(
            _configured(mapping, "embedding_batch_size", "RIVERSIDE_EMBEDDING_BATCH_SIZE", 128)
        )
        if batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")

        return cls(
            catalog=catalog,
            schema=schema,
            source_table=str(mapping.get("source_table", f"{catalog}.{schema}.parsed_documents")),
            chunk_table=str(mapping.get("chunk_table", f"{catalog}.{schema}.document_chunks")),
            vector_record_table=str(
                mapping.get("vector_record_table", f"{catalog}.{schema}.vector_index_records")
            ),
            quality_table=str(
                mapping.get("quality_table", f"{catalog}.{schema}.vectorization_quality_reports")
            ),
            evaluation_table=str(
                mapping.get("evaluation_table", f"{catalog}.{schema}.retrieval_evaluation_cases")
            ),
            evaluation_output_table=str(
                mapping.get(
                    "evaluation_output_table",
                    f"{catalog}.{schema}.retrieval_evaluation_reports",
                )
            ),
            vector_search_endpoint=endpoint,
            index_name=index_name,
            index_version=index_version,
            pipeline_version=pipeline_version,
            embedding=EmbeddingSpec(
                model_id=model_id,
                model_revision=model_revision,
                dimensions=int(dimensions_value),
                endpoint_name=embedding_endpoint,
            ),
            chunking=ChunkingSpec(
                version=chunk_version,
                size=int(mapping.get("chunk_size", 1024)),
                overlap=int(mapping.get("chunk_overlap", 128)),
            ),
            batch_size=batch_size,
            query_endpoint_type=str(mapping.get("vector_search_endpoint_type", "standard")),
        )


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


def validate_parsed_document(document: Mapping[str, Any]) -> None:
    required = {
        "contract_version",
        "kind",
        "tenant_id",
        "document_id",
        "source_uri",
        "source_version",
        "content_hash",
        "acl",
        "region",
        "classification",
        "ingested_at",
        "pipeline_version",
        "deletion_state",
        "text",
    }
    missing = sorted(name for name in required if name not in document)
    if missing:
        raise ValueError(f"Parsed document is missing required fields: {', '.join(missing)}")
    if document["contract_version"] != CONTRACT_VERSION or document["kind"] != "parsed_document":
        raise ValueError("Source row is not a parsed-document v1 record")
    if document["deletion_state"].get("status") not in {"active", "pending", "deleted"}:
        raise ValueError("Unsupported deletion state")
    if document["deletion_state"].get("status") == "active" and not document["text"]:
        raise ValueError("Active parsed documents require text")


def build_document_chunk(
    document: Mapping[str, Any],
    *,
    content: str,
    ordinal: int,
    start: int,
    end: int,
    settings: RemoteSettings,
) -> dict[str, Any]:
    content_hash = sha256_text(content)
    chunk_id = stable_identifier(
        "chunk",
        (
            CONTRACT_VERSION,
            document["tenant_id"],
            document["document_id"],
            document["source_version"],
            document["content_hash"],
            settings.chunking.strategy,
            settings.chunking.version,
            ordinal,
            start,
            end,
            content_hash,
        ),
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": "document_chunk",
        "tenant_id": document["tenant_id"],
        "chunk_id": chunk_id,
        "document_id": document["document_id"],
        "parent_document_id": document["document_id"],
        "source_uri": document["source_uri"],
        "source_version": document["source_version"],
        "content_hash": content_hash,
        "document_content_hash": document["content_hash"],
        "acl": document["acl"],
        "region": document["region"],
        "classification": document["classification"],
        "ingested_at": document["ingested_at"],
        "pipeline_version": settings.pipeline_version,
        "deletion_state": document["deletion_state"],
        "ordinal": ordinal,
        "offsets": {"unit": "unicode_code_points", "start": start, "end": end},
        "chunking": {
            "strategy": settings.chunking.strategy,
            "version": settings.chunking.version,
        },
        "embedding": settings.embedding.descriptor(),
        "index_version": settings.index_version,
        "content": content,
    }


def build_vector_record(
    chunk: Mapping[str, Any],
    vector: Sequence[float],
    *,
    indexed_at: str,
    settings: RemoteSettings,
) -> dict[str, Any]:
    normalized_vector = [float(value) for value in vector]
    if len(normalized_vector) != settings.embedding.dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.embedding.dimensions}, "
            f"received {len(normalized_vector)}"
        )
    if any(not math.isfinite(value) for value in normalized_vector):
        raise ValueError("Embedding vectors must contain only finite values")
    record_id = stable_identifier(
        "vector",
        (
            settings.index_name,
            settings.index_version,
            chunk["chunk_id"],
            settings.embedding.model_id,
            settings.embedding.model_revision,
        ),
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": "vector_index_record",
        "tenant_id": chunk["tenant_id"],
        "record_id": record_id,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "parent_document_id": chunk["parent_document_id"],
        "source_uri": chunk["source_uri"],
        "source_version": chunk["source_version"],
        "content_hash": chunk["content_hash"],
        "acl": chunk["acl"],
        "region": chunk["region"],
        "classification": chunk["classification"],
        "ingested_at": chunk["ingested_at"],
        "indexed_at": indexed_at,
        "pipeline_version": chunk["pipeline_version"],
        "deletion_state": chunk["deletion_state"],
        "embedding": settings.embedding.descriptor(),
        "index_name": settings.index_name,
        "index_version": settings.index_version,
        "content": chunk["content"],
        "vector": normalized_vector,
    }


def validate_chunk_invariants(chunk: Mapping[str, Any], source_text: str) -> None:
    if chunk["document_id"] != chunk["parent_document_id"]:
        raise ValueError("Chunk parent_document_id must equal document_id")
    start = chunk["offsets"]["start"]
    end = chunk["offsets"]["end"]
    if end <= start or source_text[start:end] != chunk["content"]:
        raise ValueError("Chunk offsets must exactly span chunk content")
    if sha256_text(chunk["content"]) != chunk["content_hash"]:
        raise ValueError("Chunk content hash does not match content")


def is_record_authorized(record: Mapping[str, Any], context: AuthorizationContext) -> bool:
    if record.get("tenant_id") != context.tenant_id or record.get("region") != context.region:
        return False
    if record.get("classification") not in context.classifications:
        return False
    if record.get("deletion_state", {}).get("status") != "active":
        return False
    acl = record.get("acl", {})
    if acl.get("visibility") == "tenant":
        return True
    return bool(
        set(acl.get("principals", ())).intersection(context.principal_ids)
        or set(acl.get("groups", ())).intersection(context.group_ids)
    )
