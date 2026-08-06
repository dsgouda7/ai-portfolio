"""Machine-readable quality evidence for one remote vectorization run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class QualityReport:
    run_id: str
    generated_at: str
    pipeline_version: str
    index_name: str
    index_version: str
    source_table: str
    documents_seen: int = 0
    active_documents: int = 0
    deletion_documents: int = 0
    chunks_written: int = 0
    vectors_upserted: int = 0
    vectors_deleted: int = 0
    required_field_failures: int = 0
    embedding_dimension_failures: int = 0
    authorization_metadata_failures: int = 0
    lineage_failures: int = 0
    duplicate_record_ids: int = 0
    errors: list[str] = field(default_factory=list)
    artifact_uri: str | None = None

    @property
    def decision(self) -> str:
        return "pass" if not self.errors else "fail"

    def add_contract_error(self, message: str) -> None:
        self.required_field_failures += 1
        self.errors.append(message[:512])

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision
        return result


def evaluate_records(
    chunks: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], report: QualityReport
) -> None:
    record_ids = [str(record["record_id"]) for record in records]
    report.duplicate_record_ids += len(record_ids) - len(set(record_ids))
    for chunk in chunks:
        if chunk["document_id"] != chunk["parent_document_id"]:
            report.lineage_failures += 1
        acl = chunk.get("acl", {})
        if not {"visibility", "principals", "groups"}.issubset(acl):
            report.authorization_metadata_failures += 1
    for record in records:
        if len(record["vector"]) != record["embedding"]["dimensions"]:
            report.embedding_dimension_failures += 1

    failure_count = (
        report.duplicate_record_ids
        + report.lineage_failures
        + report.authorization_metadata_failures
        + report.embedding_dimension_failures
    )
    if failure_count:
        report.errors.append(f"Record quality gates found {failure_count} invariant failures")
