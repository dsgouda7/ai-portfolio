"""Static, cloud-free tests for chunk and vector contract production."""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from remote.chunking import chunk_document
from remote.contracts import (
    AuthorizationContext,
    RemoteSettings,
    build_vector_record,
    is_record_authorized,
    validate_chunk_invariants,
)


FIXTURES = Path(__file__).parent / "fixtures"


def settings(**overrides):
    values = {
        "catalog": "main",
        "schema": "rag_demo",
        "vector_search_endpoint": "riverside-search",
        "index_name": "main.rag_demo.riverside_manuscripts_v1",
        "index_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "embedding_model": "databricks-gte-large-en",
        "embedding_model_revision": "revision-2026-08-05",
        "embedding_endpoint": "databricks-gte-large-en",
        "embedding_dimensions": 3,
        "chunk_strategy_version": "1.0.0",
        "chunk_size": 64,
        "chunk_overlap": 8,
    }
    values.update(overrides)
    return RemoteSettings.from_mapping(values)


def active_document():
    return json.loads((FIXTURES / "parsed-documents-v1.json").read_text())[0]


def test_chunking_is_deterministic_versioned_and_offset_preserving():
    document = active_document()
    first = chunk_document(document, settings())
    second = chunk_document(document, settings())

    assert first == second
    assert len(first) > 1
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)
    for ordinal, chunk in enumerate(first):
        assert chunk["ordinal"] == ordinal
        assert chunk["chunking"] == {"strategy": "recursive-character", "version": "1.0.0"}
        assert chunk["embedding"]["model_revision"] == "revision-2026-08-05"
        validate_chunk_invariants(chunk, document["text"])

    changed = chunk_document(document, settings(chunk_strategy_version="1.1.0"))
    assert [chunk["chunk_id"] for chunk in changed] != [chunk["chunk_id"] for chunk in first]


def test_vector_record_pins_model_revision_dimensions_and_index_version():
    chunk = chunk_document(active_document(), settings())[0]
    record = build_vector_record(
        chunk,
        [0.125, -0.25, 0.5],
        indexed_at="2026-08-05T10:07:00Z",
        settings=settings(),
    )

    assert record["parent_document_id"] == record["document_id"]
    assert record["embedding"] == {
        "model_id": "databricks-gte-large-en",
        "model_revision": "revision-2026-08-05",
        "dimensions": 3,
    }
    assert record["index_name"] == "main.rag_demo.riverside_manuscripts_v1"
    assert record["index_version"] == "1.0.0"

    with pytest.raises(ValueError, match="dimension mismatch"):
        build_vector_record(
            chunk,
            [0.125, -0.25],
            indexed_at="2026-08-05T10:07:00Z",
            settings=settings(),
        )


def test_authorization_fails_closed_for_tenant_acl_region_and_deletion():
    chunk = chunk_document(active_document(), settings())[0]
    allowed = AuthorizationContext(
        tenant_id="tenant-editorial-standard",
        region="eastus2",
        classifications=("confidential",),
        group_ids=("editors",),
    )
    assert is_record_authorized(chunk, allowed)

    assert not is_record_authorized(
        chunk,
        AuthorizationContext(
            tenant_id="tenant-other",
            region="eastus2",
            classifications=("confidential",),
            group_ids=("editors",),
        ),
    )
    assert not is_record_authorized(
        chunk,
        AuthorizationContext(
            tenant_id="tenant-editorial-standard",
            region="eastus2",
            classifications=("confidential",),
            group_ids=("unassigned",),
        ),
    )
    deleted = {**chunk, "deletion_state": {"status": "deleted"}}
    assert not is_record_authorized(deleted, allowed)


def test_remote_settings_require_pinned_embedding_lineage(monkeypatch):
    values = {
        "catalog": "main",
        "schema": "rag_demo",
        "vector_search_endpoint": "riverside-search",
        "embedding_model": "databricks-gte-large-en",
    }
    monkeypatch.delenv("RIVERSIDE_EMBEDDING_MODEL_REVISION", raising=False)
    monkeypatch.delenv("RIVERSIDE_EMBEDDING_DIMENSIONS", raising=False)
    with pytest.raises(ValueError, match="embedding_model_revision"):
        RemoteSettings.from_mapping(values)
