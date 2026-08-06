"""Distributed raw-to-parsed Databricks ingestion orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from typing import Any, Iterable, Mapping

from .config import IngestionSettings
from .contracts import (
    ContractValueError,
    SourceObject,
    build_parsed_record,
    build_raw_record,
    canonical_json,
    sha256_bytes,
)
from .delta import DeltaIngestionSink
from .parsers import ParserRegistry, ParsingError
from .quality import QualityCounts, build_quality_report, enforce_quality_report
from .sources import SourceAdapter, create_source_adapter


logger = logging.getLogger(__name__)
_RESULT_SCHEMA = """
  status STRING,
  raw_json STRING,
  parsed_json STRING,
  quarantine_json STRING,
  tenant_id STRING,
  document_id STRING,
  source_version STRING,
  content_hash STRING
"""


def _contract_uri(value: str) -> str:
    return f"dbfs:{value}" if value.startswith("/Volumes/") else value


def _quarantine_record(
    settings: IngestionSettings,
    source: Mapping[str, Any],
    ingested_at: datetime,
    error: Exception,
    raw_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    error_code = error.code if isinstance(error, ParsingError) else "contract_validation_failed"
    error_message = (
        error.safe_message
        if isinstance(error, ParsingError)
        else "source metadata could not produce a valid v1 record"
    )
    retryable = error.retryable if isinstance(error, ParsingError) else False
    source_uri = _contract_uri(str(source.get("source_uri") or "unknown:/source"))
    source_version = raw_record.get("source_version") if raw_record else None
    event_identity = (
        f"{settings.tenant_id}\n{source_uri}\n{source_version or ''}\n{error_code}".encode("utf-8")
    )
    return {
        "event_id": f"quarantine-{sha256(event_identity).hexdigest()[:40]}",
        "run_id": settings.run_id or f"manual-{ingested_at.strftime('%Y%m%dT%H%M%SZ')}",
        "tenant_id": settings.tenant_id,
        "document_id": raw_record.get("document_id") if raw_record else None,
        "source_uri": source_uri,
        "source_version": source_version,
        "raw_content_hash": raw_record.get("content_hash") if raw_record else None,
        "media_type": source.get("media_type"),
        "byte_size": source.get("byte_size"),
        "region": settings.region,
        "classification": settings.classification,
        "acl": settings.governance().acl(),
        "deletion_state": settings.governance().deletion_state(),
        "ingested_at": ingested_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pipeline_version": settings.pipeline_version,
        "error_code": error_code,
        "error_message": error_message,
        "retryable": retryable,
    }


def _transform_partition(
    rows: Iterable[Any],
    settings: IngestionSettings,
    ingested_at: datetime,
) -> Iterable[tuple[Any, ...]]:
    parsers = ParserRegistry.default()
    governance = settings.governance()
    for row in rows:
        values = row.asDict(recursive=True)
        raw_record: dict[str, Any] | None = None
        try:
            content = bytes(values["content"])
            source = SourceObject(
                source_uri=_contract_uri(values["source_uri"]),
                storage_uri=_contract_uri(values["storage_uri"]),
                source_name=values["source_name"],
                media_type=values["media_type"],
                content=content,
                metadata={
                    "source_modified_at": values["modified_at"].isoformat()
                    if values.get("modified_at")
                    else None
                },
            )
            raw_record = build_raw_record(source, governance, ingested_at)
            if len(content) > settings.max_file_bytes:
                raise ParsingError("file_too_large", "source exceeds the configured parser byte limit")
            parsed = parsers.parse(source)
            parsed_record = build_parsed_record(raw_record, parsed, ingested_at)
            yield (
                "parsed",
                canonical_json(raw_record),
                canonical_json(parsed_record),
                None,
                raw_record["tenant_id"],
                raw_record["document_id"],
                raw_record["source_version"],
                raw_record["content_hash"],
            )
        except (ContractValueError, ParsingError) as error:
            quarantine = _quarantine_record(settings, values, ingested_at, error, raw_record)
            yield (
                "quarantined",
                canonical_json(raw_record) if raw_record else None,
                None,
                canonical_json(quarantine),
                settings.tenant_id,
                raw_record.get("document_id") if raw_record else None,
                raw_record.get("source_version") if raw_record else None,
                raw_record.get("content_hash") if raw_record else sha256_bytes(bytes(values.get("content", b""))),
            )


class SparkIngestionPipeline:
    def __init__(
        self,
        spark: Any,
        settings: IngestionSettings,
        source: SourceAdapter,
        sink: DeltaIngestionSink,
    ) -> None:
        self.spark = spark
        self.settings = settings
        self.source = source
        self.sink = sink

    def run(self) -> dict[str, Any]:
        from pyspark.sql import functions as functions

        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        source_frame = self.source.read(self.spark)
        freshest_source_at = source_frame.select(
            functions.max("modified_at").alias("freshest_source_at")
        ).first()["freshest_source_at"]
        if freshest_source_at is not None and freshest_source_at.tzinfo is None:
            freshest_source_at = freshest_source_at.replace(tzinfo=timezone.utc)
        rows = source_frame.rdd.mapPartitions(
            lambda partition: _transform_partition(partition, self.settings, started_at)
        )
        results = self.spark.createDataFrame(rows, schema=_RESULT_SCHEMA).persist()
        try:
            source_count = results.count()
            raw = results.filter(functions.col("raw_json").isNotNull())
            parsed = results.filter(functions.col("status") == "parsed")
            quarantined = results.filter(functions.col("status") == "quarantined")
            raw_count = raw.count()
            parsed_count = parsed.count()
            quarantine_count = quarantined.count()
            unique_input_count = raw.select(
                "tenant_id", "document_id", "source_version"
            ).dropDuplicates().count()
            duplicate_input_count = raw_count - unique_input_count
            unique_content_count = raw.select("tenant_id", "content_hash").dropDuplicates().count()
            duplicate_content_count = raw_count - unique_content_count
            schema_drift_count = quarantined.filter(
                functions.get_json_object("quarantine_json", "$.error_code")
                == "contract_validation_failed"
            ).count()

            raw_contract = self.sink._raw_frame(raw)
            duplicate_existing_count = self.sink.count_existing(raw_contract, "raw_documents")
            self.sink.merge_raw(raw)
            if parsed_count:
                self.sink.merge_parsed(parsed)
            if quarantine_count:
                self.sink.merge_quarantine(quarantined)

            expected_required = source_count * 17 + parsed_count * 19
            observed_required = raw_count * 17 + parsed_count * 19
            report = build_quality_report(
                self.settings,
                QualityCounts(
                    source_count=source_count,
                    parsed_count=parsed_count,
                    quarantine_count=quarantine_count,
                    duplicate_input_count=duplicate_input_count,
                    duplicate_content_count=duplicate_content_count,
                    duplicate_existing_count=duplicate_existing_count,
                    schema_drift_count=schema_drift_count,
                    required_field_count=observed_required,
                    expected_required_field_count=expected_required,
                    acl_covered_count=raw_count,
                    classification_covered_count=raw_count,
                    region_covered_count=raw_count,
                    lineage_complete_count=raw_count,
                    deletion_lineage_complete_count=raw_count,
                    freshest_source_at=freshest_source_at,
                ),
                started_at,
            )
            self.sink.append_quality_report(report)
            enforce_quality_report(report)
            logger.info(
                "Remote ingestion completed: run_id=%s source_count=%s parsed_count=%s quarantine_count=%s",
                report["run_id"],
                source_count,
                parsed_count,
                quarantine_count,
            )
            return report
        finally:
            results.unpersist()


def run_remote_ingestion(
    remote_config: Mapping[str, Any] | None = None,
    *,
    spark: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    settings = IngestionSettings.from_mapping(remote_config, environ)
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    source = create_source_adapter(settings)
    sink = DeltaIngestionSink(spark, settings)
    sink.ensure_tables()
    return SparkIngestionPipeline(spark, settings, source, sink).run()
