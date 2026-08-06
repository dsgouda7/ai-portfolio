from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.contract_adapters import adapt_local_document, adapt_vector_index_record, sha256_text
from shared.vector_contract_v1 import (
    CONTRACT_PIN,
    VectorContractError,
    normalize_flat_vector_record_v1,
)


def vector_record() -> dict:
    content = "Remote content"
    return {
        "contract_version": "1.0.0",
        "kind": "vector_index_record",
        "tenant_id": "tenant-1",
        "record_id": "vector-1",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "parent_document_id": "doc-1",
        "source_uri": "https://content.example.invalid/doc-1.md",
        "source_version": "v1",
        "content_hash": sha256_text(content),
        "acl": {"visibility": "restricted", "principals": [], "groups": ["editors"]},
        "region": "eastus2",
        "classification": "internal",
        "ingested_at": "2026-08-05T00:00:00Z",
        "indexed_at": "2026-08-05T00:01:00Z",
        "pipeline_version": "1.0.0",
        "deletion_state": {"status": "active"},
        "embedding": {
            "model_id": "example-embedding-model",
            "model_revision": "revision-1",
            "dimensions": 3,
        },
        "index_name": "main.rag_demo.documents_v1",
        "index_version": "1.0.0",
        "content": content,
        "vector": [0.125, -0.25, 0.5],
    }


def flat_vector_record() -> dict:
    record = vector_record()
    return {
        **{name: value for name, value in record.items() if name not in {"acl", "deletion_state", "embedding"}},
        "acl_visibility": "restricted",
        "acl_principals_json": "[]",
        "acl_groups_json": '["editors"]',
        "deletion_status": "active",
        "deletion_requested_at": "",
        "deletion_deleted_at": "",
        "embedding_model_id": "example-embedding-model",
        "embedding_model_revision": "revision-1",
        "embedding_dimensions": 3,
    }


def test_legacy_local_document_gets_deterministic_contract_fields() -> None:
    document = SimpleNamespace(page_content="A local chunk.", metadata={"id": "doc-1", "title": "One"})
    first = adapt_local_document(document, 0.75, ordinal=0)
    second = adapt_local_document(document, 0.75, ordinal=0)

    assert first.chunk_id == second.chunk_id
    assert first.source_uri == "urn:riverside:local:doc-1"
    assert first.source_version == "legacy-local"
    assert first.lineage_kind == "synthetic-local"
    assert first.indexed_at == "1970-01-01T00:00:00Z"
    assert len(first.content_hash) == 64


def test_remote_adapter_rejects_deleted_records() -> None:
    record = vector_record()
    record["deletion_state"] = {
        "status": "deleted",
        "requested_at": "2026-08-05T00:02:00Z",
        "deleted_at": "2026-08-05T00:03:00Z",
    }
    with pytest.raises(VectorContractError, match="non-active"):
        adapt_vector_index_record(record)


@pytest.mark.parametrize("missing_field", ["acl", "parent_document_id", "ingested_at", "pipeline_version"])
def test_remote_adapter_fails_closed_on_missing_security_or_lineage(missing_field: str) -> None:
    record = vector_record()
    del record[missing_field]

    with pytest.raises(VectorContractError, match=CONTRACT_PIN):
        adapt_vector_index_record(record)


def test_flat_remote_record_normalizes_to_closed_v1_contract() -> None:
    normalized = normalize_flat_vector_record_v1(flat_vector_record())

    assert normalized["acl"] == {
        "visibility": "restricted",
        "principals": [],
        "groups": ["editors"],
    }
    assert normalized["embedding"]["dimensions"] == len(normalized["vector"]) == 3


def test_flat_remote_record_rejects_malformed_acl_transport() -> None:
    record = flat_vector_record()
    record["acl_groups_json"] = "not-json"

    with pytest.raises(VectorContractError, match="acl_groups_json is not valid JSON"):
        normalize_flat_vector_record_v1(record)
