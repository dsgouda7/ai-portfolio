"""Vendored Riverside vector-index-record v1 validation and transport adapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


CONTRACT_VERSION = "1.0.0"
CONTRACT_PIN = "riverside-vector-index-record-v1@1.0.0"
_CONTRACT_ROOT = Path(__file__).parent / "contracts" / "riverside-v1"


class VectorContractError(ValueError):
    """Raised when a remote retrieval result violates the pinned v1 contract."""


def _load_schema(name: str) -> dict[str, Any]:
    with (_CONTRACT_ROOT / name).open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


_COMMON_SCHEMA = _load_schema("common.schema.json")
_VECTOR_SCHEMA = _load_schema("vector-index-record.schema.json")
_REGISTRY = Registry().with_resources(
    (
        (str(_COMMON_SCHEMA["$id"]), Resource.from_contents(_COMMON_SCHEMA)),
        (str(_VECTOR_SCHEMA["$id"]), Resource.from_contents(_VECTOR_SCHEMA)),
    )
)
_VALIDATOR = Draft202012Validator(
    _VECTOR_SCHEMA,
    registry=_REGISTRY,
    format_checker=FormatChecker(),
)

_FLAT_REQUIRED_FIELDS = (
    "contract_version",
    "kind",
    "tenant_id",
    "record_id",
    "chunk_id",
    "document_id",
    "parent_document_id",
    "source_uri",
    "source_version",
    "content_hash",
    "acl_visibility",
    "acl_principals_json",
    "acl_groups_json",
    "region",
    "classification",
    "ingested_at",
    "indexed_at",
    "pipeline_version",
    "deletion_status",
    "embedding_model_id",
    "embedding_model_revision",
    "embedding_dimensions",
    "index_name",
    "index_version",
    "content",
    "vector",
)


def _json_string_array(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, str):
        raise VectorContractError(f"{field_name} must be a JSON-encoded string array")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise VectorContractError(f"{field_name} is not valid JSON") from error
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise VectorContractError(f"{field_name} must encode an array of strings")
    return decoded


def normalize_flat_vector_record_v1(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the closed v1 object from Databricks index transport columns."""
    missing = [name for name in _FLAT_REQUIRED_FIELDS if name not in record]
    if missing:
        raise VectorContractError(
            f"Vector search result is missing contract columns: {', '.join(missing)}"
        )

    deletion_state = {"status": record["deletion_status"]}
    if record.get("deletion_requested_at"):
        deletion_state["requested_at"] = record["deletion_requested_at"]
    if record.get("deletion_deleted_at"):
        deletion_state["deleted_at"] = record["deletion_deleted_at"]

    normalized = {
        "contract_version": record["contract_version"],
        "kind": record["kind"],
        "tenant_id": record["tenant_id"],
        "record_id": record["record_id"],
        "chunk_id": record["chunk_id"],
        "document_id": record["document_id"],
        "parent_document_id": record["parent_document_id"],
        "source_uri": record["source_uri"],
        "source_version": record["source_version"],
        "content_hash": record["content_hash"],
        "acl": {
            "visibility": record["acl_visibility"],
            "principals": _json_string_array(
                record["acl_principals_json"], "acl_principals_json"
            ),
            "groups": _json_string_array(record["acl_groups_json"], "acl_groups_json"),
        },
        "region": record["region"],
        "classification": record["classification"],
        "ingested_at": record["ingested_at"],
        "indexed_at": record["indexed_at"],
        "pipeline_version": record["pipeline_version"],
        "deletion_state": deletion_state,
        "embedding": {
            "model_id": record["embedding_model_id"],
            "model_revision": record["embedding_model_revision"],
            "dimensions": record["embedding_dimensions"],
        },
        "index_name": record["index_name"],
        "index_version": record["index_version"],
        "content": record["content"],
        "vector": record["vector"],
    }
    validate_vector_record_v1(normalized)
    return normalized


def validate_vector_record_v1(record: Mapping[str, Any]) -> None:
    """Validate the pinned schema and runtime invariants required by serving."""
    errors = sorted(_VALIDATOR.iter_errors(dict(record)), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "record"
        raise VectorContractError(
            f"{CONTRACT_PIN} validation failed at {location}: {error.message}"
        )

    if record["document_id"] != record["parent_document_id"]:
        raise VectorContractError("parent_document_id must equal document_id")
    content_hash = hashlib.sha256(str(record["content"]).encode("utf-8")).hexdigest()
    if content_hash != record["content_hash"]:
        raise VectorContractError("content_hash does not match content")

    vector = record["vector"]
    dimensions = record["embedding"]["dimensions"]
    if len(vector) != dimensions:
        raise VectorContractError("vector length must equal embedding.dimensions")
    if any(not math.isfinite(float(value)) for value in vector):
        raise VectorContractError("vector values must be finite")
