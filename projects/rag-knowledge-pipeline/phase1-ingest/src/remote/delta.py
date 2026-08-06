"""Unity Catalog Delta tables, idempotent merges, and latest-version views."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .config import IngestionSettings


_ACL_SCHEMA = "STRUCT<visibility:STRING,principals:ARRAY<STRING>,groups:ARRAY<STRING>>"
_DELETION_SCHEMA = "STRUCT<status:STRING,requested_at:STRING,deleted_at:STRING>"
_PARSER_SCHEMA = "STRUCT<name:STRING,version:STRING>"


class DeltaIngestionSink:
    def __init__(self, spark: Any, settings: IngestionSettings) -> None:
        self.spark = spark
        self.settings = settings

    def ensure_tables(self) -> None:
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{self.settings.catalog}`.`{self.settings.schema}`")
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS {self.settings.table('raw_documents')} (
              contract_version STRING NOT NULL,
              kind STRING NOT NULL,
              tenant_id STRING NOT NULL,
              document_id STRING NOT NULL,
              source_uri STRING NOT NULL,
              source_version STRING NOT NULL,
              content_hash STRING NOT NULL,
              acl STRUCT<visibility:STRING,principals:ARRAY<STRING>,groups:ARRAY<STRING>> NOT NULL,
              region STRING NOT NULL,
              classification STRING NOT NULL,
              ingested_at STRING NOT NULL,
              pipeline_version STRING NOT NULL,
              deletion_state STRUCT<status:STRING,requested_at:STRING,deleted_at:STRING> NOT NULL,
              media_type STRING NOT NULL,
              byte_size BIGINT NOT NULL,
              storage_uri STRING NOT NULL,
              source_name STRING NOT NULL,
              metadata VARIANT
            ) USING DELTA
            TBLPROPERTIES (
              'delta.enableChangeDataFeed' = 'true',
              'riverside.contract' = 'raw-document',
              'riverside.contract.version' = '1.0.0'
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.settings.table('parsed_documents')} (
              contract_version STRING NOT NULL,
              kind STRING NOT NULL,
              tenant_id STRING NOT NULL,
              document_id STRING NOT NULL,
              source_uri STRING NOT NULL,
              source_version STRING NOT NULL,
              content_hash STRING NOT NULL,
              raw_content_hash STRING NOT NULL,
              acl STRUCT<visibility:STRING,principals:ARRAY<STRING>,groups:ARRAY<STRING>> NOT NULL,
              region STRING NOT NULL,
              classification STRING NOT NULL,
              ingested_at STRING NOT NULL,
              parsed_at STRING NOT NULL,
              pipeline_version STRING NOT NULL,
              deletion_state STRUCT<status:STRING,requested_at:STRING,deleted_at:STRING> NOT NULL,
              parser STRUCT<name:STRING,version:STRING> NOT NULL,
              language STRING NOT NULL,
              title STRING NOT NULL,
              text STRING NOT NULL,
              metadata VARIANT
            ) USING DELTA
            TBLPROPERTIES (
              'delta.enableChangeDataFeed' = 'true',
              'riverside.contract' = 'parsed-document',
              'riverside.contract.version' = '1.0.0'
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.settings.table('document_quarantine')} (
              event_id STRING NOT NULL,
              run_id STRING NOT NULL,
              tenant_id STRING NOT NULL,
              document_id STRING,
              source_uri STRING NOT NULL,
              source_version STRING,
              raw_content_hash STRING,
              media_type STRING,
              byte_size BIGINT,
              region STRING NOT NULL,
              classification STRING NOT NULL,
              acl STRUCT<visibility:STRING,principals:ARRAY<STRING>,groups:ARRAY<STRING>> NOT NULL,
              deletion_state STRUCT<status:STRING,requested_at:STRING,deleted_at:STRING> NOT NULL,
              ingested_at STRING NOT NULL,
              pipeline_version STRING NOT NULL,
              error_code STRING NOT NULL,
              error_message STRING NOT NULL,
              retryable BOOLEAN NOT NULL
            ) USING DELTA
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.settings.table('ingestion_quality_reports')} (
              run_id STRING NOT NULL,
              pipeline_version STRING NOT NULL,
              started_at STRING NOT NULL,
              completed_at STRING NOT NULL,
              status STRING NOT NULL,
              source_count BIGINT NOT NULL,
              parsed_count BIGINT NOT NULL,
              quarantine_count BIGINT NOT NULL,
              duplicate_input_count BIGINT NOT NULL,
              duplicate_content_count BIGINT NOT NULL,
              duplicate_existing_count BIGINT NOT NULL,
              schema_drift_count BIGINT NOT NULL,
              parse_success_rate DOUBLE NOT NULL,
              quarantine_rate DOUBLE NOT NULL,
              required_field_completeness DOUBLE NOT NULL,
              acl_coverage DOUBLE NOT NULL,
              classification_coverage DOUBLE NOT NULL,
              region_coverage DOUBLE NOT NULL,
              lineage_completeness DOUBLE NOT NULL,
              deletion_lineage_coverage DOUBLE NOT NULL,
              freshest_source_at STRING,
              freshness_lag_seconds DOUBLE,
              gates VARIANT NOT NULL
            ) USING DELTA
            """,
        )
        for statement in statements:
            self.spark.sql(statement)
        self._create_latest_views()

    def _create_latest_views(self) -> None:
        self.spark.sql(
            f"""
            CREATE OR REPLACE VIEW {self.settings.table('raw_documents_latest')} AS
            SELECT contract_version, kind, tenant_id, document_id, source_uri, source_version,
                   content_hash, acl, region, classification, ingested_at, pipeline_version,
                   deletion_state, media_type, byte_size, storage_uri, source_name, metadata
            FROM (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY tenant_id, document_id ORDER BY ingested_at DESC, source_version DESC
              ) AS row_number
              FROM {self.settings.table('raw_documents')}
            ) WHERE row_number = 1
            """
        )
        self.spark.sql(
            f"""
            CREATE OR REPLACE VIEW {self.settings.table('parsed_documents_latest')} AS
            SELECT contract_version, kind, tenant_id, document_id, source_uri, source_version,
                   content_hash, raw_content_hash, acl, region, classification, ingested_at,
                   parsed_at, pipeline_version, deletion_state, parser, language, title, text, metadata
            FROM (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY tenant_id, document_id ORDER BY ingested_at DESC, source_version DESC
              ) AS row_number
              FROM {self.settings.table('parsed_documents')}
            ) WHERE row_number = 1
            """
        )

    def _raw_frame(self, frame: Any) -> Any:
        return frame.selectExpr(
            "get_json_object(raw_json, '$.contract_version') AS contract_version",
            "get_json_object(raw_json, '$.kind') AS kind",
            "get_json_object(raw_json, '$.tenant_id') AS tenant_id",
            "get_json_object(raw_json, '$.document_id') AS document_id",
            "get_json_object(raw_json, '$.source_uri') AS source_uri",
            "get_json_object(raw_json, '$.source_version') AS source_version",
            "get_json_object(raw_json, '$.content_hash') AS content_hash",
            f"from_json(get_json_object(raw_json, '$.acl'), '{_ACL_SCHEMA}') AS acl",
            "get_json_object(raw_json, '$.region') AS region",
            "get_json_object(raw_json, '$.classification') AS classification",
            "get_json_object(raw_json, '$.ingested_at') AS ingested_at",
            "get_json_object(raw_json, '$.pipeline_version') AS pipeline_version",
            f"from_json(get_json_object(raw_json, '$.deletion_state'), '{_DELETION_SCHEMA}') AS deletion_state",
            "get_json_object(raw_json, '$.media_type') AS media_type",
            "CAST(get_json_object(raw_json, '$.byte_size') AS BIGINT) AS byte_size",
            "get_json_object(raw_json, '$.storage_uri') AS storage_uri",
            "get_json_object(raw_json, '$.source_name') AS source_name",
            "parse_json(get_json_object(raw_json, '$.metadata')) AS metadata",
        )

    def _parsed_frame(self, frame: Any) -> Any:
        return frame.selectExpr(
            "get_json_object(parsed_json, '$.contract_version') AS contract_version",
            "get_json_object(parsed_json, '$.kind') AS kind",
            "get_json_object(parsed_json, '$.tenant_id') AS tenant_id",
            "get_json_object(parsed_json, '$.document_id') AS document_id",
            "get_json_object(parsed_json, '$.source_uri') AS source_uri",
            "get_json_object(parsed_json, '$.source_version') AS source_version",
            "get_json_object(parsed_json, '$.content_hash') AS content_hash",
            "get_json_object(parsed_json, '$.raw_content_hash') AS raw_content_hash",
            f"from_json(get_json_object(parsed_json, '$.acl'), '{_ACL_SCHEMA}') AS acl",
            "get_json_object(parsed_json, '$.region') AS region",
            "get_json_object(parsed_json, '$.classification') AS classification",
            "get_json_object(parsed_json, '$.ingested_at') AS ingested_at",
            "get_json_object(parsed_json, '$.parsed_at') AS parsed_at",
            "get_json_object(parsed_json, '$.pipeline_version') AS pipeline_version",
            f"from_json(get_json_object(parsed_json, '$.deletion_state'), '{_DELETION_SCHEMA}') AS deletion_state",
            f"from_json(get_json_object(parsed_json, '$.parser'), '{_PARSER_SCHEMA}') AS parser",
            "get_json_object(parsed_json, '$.language') AS language",
            "get_json_object(parsed_json, '$.title') AS title",
            "get_json_object(parsed_json, '$.text') AS text",
            "parse_json(get_json_object(parsed_json, '$.metadata')) AS metadata",
        )

    def count_existing(self, frame: Any, table_name: str) -> int:
        keys = ["tenant_id", "document_id", "source_version"]
        target = self.spark.table(self.settings.table(table_name)).select(*keys)
        return frame.select(*keys).dropDuplicates(keys).join(target, keys, "inner").count()

    def merge_raw(self, frame: Any) -> None:
        self._merge_contract(self._raw_frame(frame), "raw_documents")

    def merge_parsed(self, frame: Any) -> None:
        self._merge_contract(self._parsed_frame(frame), "parsed_documents")

    def _merge_contract(self, source: Any, table_name: str) -> None:
        from delta.tables import DeltaTable

        keys = ["tenant_id", "document_id", "source_version"]
        source = source.dropDuplicates(keys)
        condition = " AND ".join(f"target.{key} = source.{key}" for key in keys)
        updates = {
            "acl": "source.acl",
            "region": "source.region",
            "classification": "source.classification",
            "deletion_state": "source.deletion_state",
            "metadata": "source.metadata",
        }
        (
            DeltaTable.forName(self.spark, self.settings.table(table_name))
            .alias("target")
            .merge(source.alias("source"), condition)
            .whenMatchedUpdate(set=updates)
            .whenNotMatchedInsertAll()
            .execute()
        )

    def merge_quarantine(self, frame: Any) -> None:
        from delta.tables import DeltaTable

        source = frame.selectExpr(
            "get_json_object(quarantine_json, '$.event_id') AS event_id",
            "get_json_object(quarantine_json, '$.run_id') AS run_id",
            "get_json_object(quarantine_json, '$.tenant_id') AS tenant_id",
            "get_json_object(quarantine_json, '$.document_id') AS document_id",
            "get_json_object(quarantine_json, '$.source_uri') AS source_uri",
            "get_json_object(quarantine_json, '$.source_version') AS source_version",
            "get_json_object(quarantine_json, '$.raw_content_hash') AS raw_content_hash",
            "get_json_object(quarantine_json, '$.media_type') AS media_type",
            "CAST(get_json_object(quarantine_json, '$.byte_size') AS BIGINT) AS byte_size",
            "get_json_object(quarantine_json, '$.region') AS region",
            "get_json_object(quarantine_json, '$.classification') AS classification",
            f"from_json(get_json_object(quarantine_json, '$.acl'), '{_ACL_SCHEMA}') AS acl",
            f"from_json(get_json_object(quarantine_json, '$.deletion_state'), '{_DELETION_SCHEMA}') AS deletion_state",
            "get_json_object(quarantine_json, '$.ingested_at') AS ingested_at",
            "get_json_object(quarantine_json, '$.pipeline_version') AS pipeline_version",
            "get_json_object(quarantine_json, '$.error_code') AS error_code",
            "get_json_object(quarantine_json, '$.error_message') AS error_message",
            "CAST(get_json_object(quarantine_json, '$.retryable') AS BOOLEAN) AS retryable",
        ).dropDuplicates(["event_id"])
        (
            DeltaTable.forName(self.spark, self.settings.table("document_quarantine"))
            .alias("target")
            .merge(source.alias("source"), "target.event_id = source.event_id")
            .whenNotMatchedInsertAll()
            .execute()
        )

    def append_quality_report(self, report: dict[str, Any]) -> None:
        from delta.tables import DeltaTable

        row = dict(report)
        row["gates_json"] = json.dumps(row.pop("gates"), separators=(",", ":"), sort_keys=True)
        frame = self.spark.createDataFrame([row]).selectExpr(
            *[name for name in row if name != "gates_json"],
            "parse_json(gates_json) AS gates",
        )
        (
            DeltaTable.forName(
                self.spark, self.settings.table("ingestion_quality_reports")
            )
            .alias("target")
            .merge(frame.alias("source"), "target.run_id = source.run_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
