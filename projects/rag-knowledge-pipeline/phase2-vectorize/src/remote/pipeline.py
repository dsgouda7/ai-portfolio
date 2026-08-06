"""Deletion-aware orchestration from parsed Delta rows to a direct vector index."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .chunking import chunk_document
from .contracts import (
    RemoteSettings,
    build_vector_record,
    stable_identifier,
    utc_now,
    validate_parsed_document,
)
from .interfaces import ContractRecordRepository, EmbeddingProvider, ParsedDocumentSource, VectorIndex
from .quality import QualityReport, evaluate_records


class RemoteVectorizationPipeline:
    """Coordinate idempotent contract writes and serving-index updates."""

    def __init__(
        self,
        *,
        settings: RemoteSettings,
        source: ParsedDocumentSource,
        embedder: EmbeddingProvider,
        repository: ContractRecordRepository,
        index: VectorIndex,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.settings = settings
        self.source = source
        self.embedder = embedder
        self.repository = repository
        self.index = index
        self.clock = clock

    def run(self, *, since: str | None = None, run_id: str | None = None) -> QualityReport:
        generated_at = self.clock()
        report = QualityReport(
            run_id=run_id
            or stable_identifier(
                "index-run",
                (self.settings.index_name, self.settings.index_version, generated_at),
            ),
            generated_at=generated_at,
            pipeline_version=self.settings.pipeline_version,
            index_name=self.settings.index_name,
            index_version=self.settings.index_version,
            source_table=self.settings.source_table,
        )
        documents = list(self.source.read_documents(since=since))
        report.documents_seen = len(documents)
        for document in documents:
            try:
                validate_parsed_document(document)
            except (KeyError, TypeError, ValueError) as exc:
                report.add_contract_error(
                    f"{document.get('tenant_id', 'unknown')}/"
                    f"{document.get('document_id', 'unknown')}: {exc}"
                )

        if report.errors:
            report.artifact_uri = self.repository.write_quality_report(report.to_dict())
            return report

        self.index.ensure_exists()
        for document in documents:
            try:
                self._process_document(document, report)
            except (KeyError, TypeError, ValueError) as exc:
                report.add_contract_error(
                    f"{document.get('tenant_id', 'unknown')}/"
                    f"{document.get('document_id', 'unknown')}: {exc}"
                )

        report.artifact_uri = self.repository.write_quality_report(report.to_dict())
        return report

    def _process_document(self, document: Mapping[str, object], report: QualityReport) -> None:
        tenant_id = str(document["tenant_id"])
        document_id = str(document["document_id"])
        existing_ids = self.repository.record_ids_for_document(tenant_id, document_id)
        deletion_state = document["deletion_state"]
        if not isinstance(deletion_state, Mapping):
            raise ValueError("deletion_state must be an object")
        if deletion_state["status"] != "active":
            report.deletion_documents += 1
            self._delete_records(existing_ids, deletion_state, report)
            return

        report.active_documents += 1
        chunks = chunk_document(document, self.settings)
        vectors = self._embed_chunks(chunks)
        indexed_at = self.clock()
        records = [
            build_vector_record(
                chunk,
                vector,
                indexed_at=indexed_at,
                settings=self.settings,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        errors_before_evaluation = len(report.errors)
        evaluate_records(chunks, records, report)
        if len(report.errors) > errors_before_evaluation:
            return

        self.repository.merge_chunks(chunks)
        self.repository.merge_vector_records(records)
        self.index.upsert(records)
        current_ids = {str(record["record_id"]) for record in records}
        stale_ids = sorted(existing_ids - current_ids)
        if stale_ids:
            deleted_at = self.clock()
            self.index.delete(stale_ids)
            self.repository.mark_records_deleted(
                stale_ids,
                requested_at=indexed_at,
                deleted_at=deleted_at,
            )
        report.chunks_written += len(chunks)
        report.vectors_upserted += len(records)
        report.vectors_deleted += len(stale_ids)

    def _embed_chunks(self, chunks: Sequence[Mapping[str, object]]) -> Sequence[Sequence[float]]:
        vectors: list[Sequence[float]] = []
        for start in range(0, len(chunks), self.settings.batch_size):
            batch = chunks[start : start + self.settings.batch_size]
            embedded = list(self.embedder.embed([str(chunk["content"]) for chunk in batch]))
            if len(embedded) != len(batch):
                raise ValueError("Embedding provider returned a different number of vectors than inputs")
            vectors.extend(embedded)
        return vectors

    def _delete_records(
        self,
        record_ids: set[str],
        deletion_state: Mapping[str, object],
        report: QualityReport,
    ) -> None:
        if not record_ids:
            return
        requested_at = str(deletion_state.get("requested_at") or self.clock())
        deleted_at = str(deletion_state.get("deleted_at") or self.clock())
        ordered_ids = sorted(record_ids)
        self.index.delete(ordered_ids)
        self.repository.mark_records_deleted(
            ordered_ids,
            requested_at=requested_at,
            deleted_at=deleted_at,
        )
        report.vectors_deleted += len(ordered_ids)
