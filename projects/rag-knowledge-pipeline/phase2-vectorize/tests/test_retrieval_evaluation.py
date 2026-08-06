"""Static retrieval metric and authorization evidence tests."""

import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from remote.contracts import RemoteSettings
from remote.evaluation import RetrievalEvaluator


FIXTURE = Path(__file__).parent / "fixtures" / "retrieval-evaluation-v1.json"


class FakeEmbedder:
    def embed(self, texts):
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


class FakeAuthorizedIndex:
    def ensure_exists(self):
        pass

    def upsert(self, records):
        pass

    def delete(self, record_ids):
        pass

    def query_authorized(self, query_vector, context, *, top_k):
        base = {
            "tenant_id": context.tenant_id,
            "region": context.region,
            "classification": context.classifications[0],
            "acl": {"visibility": "restricted", "principals": [], "groups": ["editors"]},
            "deletion_state": {"status": "active"},
        }
        if query_vector[0] == 0.0:
            return [{**base, "chunk_id": "chunk-relevant-001"}]
        return [
            {**base, "chunk_id": "chunk-distractor-001"},
            {**base, "chunk_id": "chunk-relevant-002"},
        ]


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
        }
    )


def test_retrieval_report_emits_metrics_slices_and_no_query_or_tenant_content():
    fixture = json.loads(FIXTURE.read_text())
    evaluator = RetrievalEvaluator(
        settings=settings(),
        embedder=FakeEmbedder(),
        index=FakeAuthorizedIndex(),
        clock=lambda: "2026-08-05T12:00:00Z",
    )
    report = evaluator.evaluate(
        fixture["cases"],
        dataset_id=fixture["dataset_id"],
        dataset_version=fixture["dataset_version"],
        report_id="retrieval-eval-static-001",
    )

    assert report["aggregate"]["recall_at_k"] == 1.0
    assert report["aggregate"]["mrr"] == 0.75
    expected_ndcg = (1.0 + 1.0 / math.log2(3)) / 2
    assert math.isclose(report["aggregate"]["ndcg_at_k"], expected_ndcg)
    assert report["aggregate"]["authorization_leakage_count"] == 0
    assert set(report["slices"]) == {"editorial", "policy"}
    assert report["decision"] == "pass"
    serialized = json.dumps(report)
    assert "What did Aria" not in serialized
    assert "tenant-editorial-standard" not in serialized
