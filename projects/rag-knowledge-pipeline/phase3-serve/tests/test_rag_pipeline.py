from __future__ import annotations

from shared.contract_adapters import AuthorizationContext, RetrievedRecord

from src import rag_pipeline as pipeline_module
from src.providers import GenerationResult


class FakeRetriever:
    def __init__(self, records):
        self.records = records
        self.authorization = None

    def retrieve(self, query, *, top_k, search_type, authorization, filters=None):
        self.authorization = authorization
        return self.records[:top_k]

    def status(self):
        return "fake retrieval"


class FakeGenerator:
    def __init__(self, content):
        self.content = content

    def complete(self, messages, *, temperature, max_tokens, max_input_tokens):
        return GenerationResult(self.content, 10, 4, "stop")

    def status(self):
        return "fake generation"


def config() -> dict:
    return {
        "mode": "remote",
        "local": {},
        "remote": {"index_version": "1.0.0"},
        "retrieval": {"top_k": 6, "search_type": "similarity"},
        "serving": {
            "retrieval": {"provider": "auto"},
            "generation": {
                "provider": "azure_endpoint",
                "model_alias": "riverside-editor",
                "max_input_tokens": 1024,
                "max_tokens": 128,
            },
        },
    }


def record() -> RetrievedRecord:
    return RetrievedRecord(
        tenant_id="tenant-1",
        chunk_id="chunk-1",
        document_id="doc-1",
        source_uri="https://content.example.invalid/doc-1.md",
        source_version="v1",
        content_hash="a" * 64,
        title="Document one",
        content="Grounded content",
        score=0.9,
        region="eastus2",
        classification="internal",
        index_version="1.0.0",
        indexed_at="2026-08-05T00:00:00Z",
    )


def test_completion_normalizes_citation_markers(monkeypatch) -> None:
    retriever = FakeRetriever([record()])
    monkeypatch.setattr(pipeline_module, "build_retriever", lambda *_: retriever)
    monkeypatch.setattr(
        pipeline_module,
        "build_generation_provider",
        lambda *_: FakeGenerator("Grounded answer. <cite:chunk-1>"),
    )
    pipeline = pipeline_module.RAGPipeline(config())
    result = pipeline.complete(
        [{"role": "user", "content": "Question"}],
        authorization=AuthorizationContext("tenant-1", "eastus2", ("internal",)),
    )

    assert result.answer == "Grounded answer. "
    assert result.citations[0]["chunk_id"] == "chunk-1"
    assert result.sources_count == 1


def test_request_classifications_only_narrow_trusted_scope(monkeypatch) -> None:
    retriever = FakeRetriever([record()])
    monkeypatch.setattr(pipeline_module, "build_retriever", lambda *_: retriever)
    monkeypatch.setattr(
        pipeline_module,
        "build_generation_provider",
        lambda *_: FakeGenerator("No marker"),
    )
    pipeline = pipeline_module.RAGPipeline(config())
    pipeline.complete(
        [{"role": "user", "content": "Question"}],
        authorization=AuthorizationContext(
            "tenant-1", "eastus2", ("public", "internal", "confidential")
        ),
        retrieval_filters={"classification": ["internal"]},
    )

    assert retriever.authorization.classifications == ("internal",)


def test_legacy_query_shape_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "build_retriever", lambda *_: FakeRetriever([record()]))
    monkeypatch.setattr(
        pipeline_module,
        "build_generation_provider",
        lambda *_: FakeGenerator("Answer"),
    )
    pipeline = pipeline_module.RAGPipeline(config())
    result = pipeline.query(
        "Question",
        authorization=AuthorizationContext("tenant-1", "eastus2", ("internal",)),
    )
    assert result == {"answer": "Answer", "question": "Question", "sources_count": 1}
