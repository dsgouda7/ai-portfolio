"""Cloud-free tests for Databricks index payloads and mandatory filters."""

import json
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from remote.chunking import chunk_document
from remote.contracts import AuthorizationContext, RemoteSettings, build_vector_record
from remote.databricks_adapters import DatabricksDirectVectorIndex


FIXTURES = Path(__file__).parent / "fixtures"


def settings(endpoint_type="standard"):
    return RemoteSettings.from_mapping(
        {
            "catalog": "main",
            "schema": "rag_demo",
            "vector_search_endpoint": "riverside-search",
            "vector_search_endpoint_type": endpoint_type,
            "index_name": "main.rag_demo.riverside_manuscripts_v1",
            "index_version": "1.0.0",
            "pipeline_version": "1.0.0",
            "embedding_model": "databricks-gte-large-en",
            "embedding_model_revision": "revision-2026-08-05",
            "embedding_endpoint": "databricks-gte-large-en",
            "embedding_dimensions": 3,
            "chunk_size": 64,
            "chunk_overlap": 8,
        }
    )


def record():
    document = json.loads((FIXTURES / "parsed-documents-v1.json").read_text())[0]
    chunk = chunk_document(document, settings())[0]
    return build_vector_record(
        chunk,
        [0.125, -0.25, 0.5],
        indexed_at="2026-08-05T10:07:00Z",
        settings=settings(),
    )


def context():
    return AuthorizationContext(
        tenant_id="tenant-editorial-standard",
        region="eastus2",
        classifications=("confidential",),
        principal_ids=("editor-17",),
        group_ids=("editors",),
    )


def test_direct_index_payload_flattens_security_fields_without_losing_contract_values():
    payload = DatabricksDirectVectorIndex._to_index_payload(record())

    assert payload["tenant_id"] == "tenant-editorial-standard"
    assert payload["deletion_status"] == "active"
    assert payload["embedding_model_revision"] == "revision-2026-08-05"
    assert payload["embedding_dimensions"] == len(payload["vector"]) == 3
    assert json.loads(payload["acl_groups_json"]) == ["editors"]
    assert " group:editors " in payload["acl_scope_tokens"]


def test_standard_endpoint_filter_pushes_all_security_partitions_to_ai_search():
    adapter = DatabricksDirectVectorIndex(settings())
    filters = adapter._backend_filters(context())

    assert filters["tenant_id"] == "tenant-editorial-standard"
    assert filters["region"] == "eastus2"
    assert filters["classification"] == ["confidential"]
    assert filters["deletion_status"] == "active"
    assert set(filters["acl_scope_tokens LIKE"]) == {
        "tenant",
        "principal:editor-17",
        "group:editors",
    }


def test_storage_optimized_filter_escapes_values_and_requires_acl_scope():
    adapter = DatabricksDirectVectorIndex(settings("storage-optimized"))
    filters = adapter._backend_filters(context())

    assert "tenant_id = 'tenant-editorial-standard'" in filters
    assert "region = 'eastus2'" in filters
    assert "deletion_status = 'active'" in filters
    assert "acl_scope_tokens LIKE '% group:editors %'" in filters
