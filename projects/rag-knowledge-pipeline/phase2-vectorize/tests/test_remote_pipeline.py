"""Static fixture tests for upsert, stale-version deletion, and tombstones."""

import json
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from remote.contracts import RemoteSettings
from remote.pipeline import RemoteVectorizationPipeline


FIXTURES = Path(__file__).parent / "fixtures"


class FakeSource:
    def __init__(self, documents):
        self.documents = documents

    def read_documents(self, *, since=None):
        return self.documents


class FakeEmbedder:
    def embed(self, texts):
        return [[len(text) / 1000.0, 0.25, -0.5] for text in texts]


class FakeRepository:
    def __init__(self):
        self.existing = {
            ("tenant-editorial-standard", "doc-active-001"): {"vector:stale-version"},
            ("tenant-editorial-standard", "doc-deleted-001"): {"vector:deleted-source"},
        }
        self.chunks = []
        self.records = []
        self.deleted = []
        self.reports = []
        self.events = []

    def record_ids_for_document(self, tenant_id, document_id):
        return set(self.existing.get((tenant_id, document_id), set()))

    def merge_chunks(self, chunks):
        self.events.append("merge_chunks")
        self.chunks.extend(chunks)

    def merge_vector_records(self, records):
        self.events.append("merge_records")
        self.records.extend(records)

    def mark_records_deleted(self, record_ids, *, requested_at, deleted_at):
        self.events.append("mark_deleted")
        self.deleted.extend(record_ids)

    def write_quality_report(self, report):
        self.reports.append(dict(report))
        return f"delta://quality/{report['run_id']}"


class FakeIndex:
    def __init__(self, events):
        self.events = events
        self.upserted = []
        self.deleted = []

    def ensure_exists(self):
        self.events.append("ensure_index")

    def upsert(self, records):
        self.events.append("index_upsert")
        self.upserted.extend(records)

    def delete(self, record_ids):
        self.events.append("index_delete")
        self.deleted.extend(record_ids)


def settings():
    return RemoteSettings.from_mapping(
        {
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
    )


def test_pipeline_upserts_current_version_and_propagates_all_deletions():
    documents = json.loads((FIXTURES / "parsed-documents-v1.json").read_text())
    repository = FakeRepository()
    index = FakeIndex(repository.events)
    timestamps = iter(
        [
            "2026-08-05T11:00:00Z",
            "2026-08-05T11:00:01Z",
            "2026-08-05T11:00:02Z",
        ]
    )
    pipeline = RemoteVectorizationPipeline(
        settings=settings(),
        source=FakeSource(documents),
        embedder=FakeEmbedder(),
        repository=repository,
        index=index,
        clock=lambda: next(timestamps),
    )

    report = pipeline.run(run_id="index-run-fixture-001")

    expected = json.loads((FIXTURES / "expected-quality-report.json").read_text())
    actual = report.to_dict()
    for key, value in expected.items():
        assert actual[key] == value
    assert report.vectors_upserted == len(repository.records) == len(index.upserted)
    assert set(index.deleted) == {"vector:stale-version", "vector:deleted-source"}
    assert set(repository.deleted) == set(index.deleted)
    assert repository.events.index("index_delete") < repository.events.index("mark_deleted")
    assert report.artifact_uri == "delta://quality/index-run-fixture-001"


def test_contract_invalid_batch_does_not_touch_serving_index():
    invalid = json.loads((FIXTURES / "parsed-documents-v1.json").read_text())
    invalid[0].pop("tenant_id")
    repository = FakeRepository()
    index = FakeIndex(repository.events)
    pipeline = RemoteVectorizationPipeline(
        settings=settings(),
        source=FakeSource(invalid),
        embedder=FakeEmbedder(),
        repository=repository,
        index=index,
        clock=lambda: "2026-08-05T11:00:00Z",
    )

    report = pipeline.run(run_id="index-run-invalid-001")

    assert report.decision == "fail"
    assert report.required_field_failures == 1
    assert "ensure_index" not in repository.events
    assert not index.upserted
    assert not index.deleted
