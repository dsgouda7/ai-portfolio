"""Create fixed-schema Unity Catalog Delta tables for the indexing contracts."""

from __future__ import annotations

import argparse
import re
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _qualified_identifier(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 3 or any(not _IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"Expected a three-part Unity Catalog identifier, got {value!r}")
    return ".".join(f"`{part}`" for part in parts)


def create_contract_tables(
    spark: Any,
    *,
    chunk_table: str,
    vector_record_table: str,
    quality_table: str,
    evaluation_output_table: str,
) -> None:
    acl_type = "STRUCT<visibility: STRING, principals: ARRAY<STRING>, groups: ARRAY<STRING>>"
    deletion_type = "STRUCT<status: STRING, requested_at: STRING, deleted_at: STRING>"
    embedding_type = "STRUCT<model_id: STRING, model_revision: STRING, dimensions: INT>"
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_qualified_identifier(chunk_table)} (
          contract_version STRING NOT NULL,
          kind STRING NOT NULL,
          tenant_id STRING NOT NULL,
          chunk_id STRING NOT NULL,
          document_id STRING NOT NULL,
          parent_document_id STRING NOT NULL,
          source_uri STRING NOT NULL,
          source_version STRING NOT NULL,
          content_hash STRING NOT NULL,
          document_content_hash STRING NOT NULL,
          acl {acl_type} NOT NULL,
          region STRING NOT NULL,
          classification STRING NOT NULL,
          ingested_at STRING NOT NULL,
          pipeline_version STRING NOT NULL,
          deletion_state {deletion_type} NOT NULL,
          ordinal BIGINT NOT NULL,
          offsets STRUCT<unit: STRING, start: BIGINT, end: BIGINT> NOT NULL,
          chunking STRUCT<strategy: STRING, version: STRING> NOT NULL,
          embedding {embedding_type} NOT NULL,
          index_version STRING NOT NULL,
          content STRING NOT NULL
        ) USING DELTA
        TBLPROPERTIES (
          'delta.enableChangeDataFeed' = 'true',
          'riverside.contract' = 'document-chunk/1.0.0'
        )
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_qualified_identifier(evaluation_output_table)} (
          report_id STRING NOT NULL,
          generated_at STRING NOT NULL,
          dataset_id STRING NOT NULL,
          dataset_version STRING NOT NULL,
          index_name STRING NOT NULL,
          index_version STRING NOT NULL,
          evaluator_version STRING NOT NULL,
          case_count BIGINT NOT NULL,
          recall_at_k DOUBLE NOT NULL,
          mrr DOUBLE NOT NULL,
          ndcg_at_k DOUBLE NOT NULL,
          authorization_leakage_count BIGINT NOT NULL,
          decision STRING NOT NULL,
          report_json STRING NOT NULL
        ) USING DELTA
        TBLPROPERTIES ('riverside.contract' = 'retrieval-evaluation-report/1.0.0')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_qualified_identifier(vector_record_table)} (
          contract_version STRING NOT NULL,
          kind STRING NOT NULL,
          tenant_id STRING NOT NULL,
          record_id STRING NOT NULL,
          chunk_id STRING NOT NULL,
          document_id STRING NOT NULL,
          parent_document_id STRING NOT NULL,
          source_uri STRING NOT NULL,
          source_version STRING NOT NULL,
          content_hash STRING NOT NULL,
          acl {acl_type} NOT NULL,
          region STRING NOT NULL,
          classification STRING NOT NULL,
          ingested_at STRING NOT NULL,
          indexed_at STRING NOT NULL,
          pipeline_version STRING NOT NULL,
          deletion_state {deletion_type} NOT NULL,
          embedding {embedding_type} NOT NULL,
          index_name STRING NOT NULL,
          index_version STRING NOT NULL,
          content STRING NOT NULL,
          vector ARRAY<FLOAT> NOT NULL
        ) USING DELTA
        TBLPROPERTIES (
          'delta.enableChangeDataFeed' = 'true',
          'riverside.contract' = 'vector-index-record/1.0.0'
        )
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_qualified_identifier(quality_table)} (
          run_id STRING NOT NULL,
          generated_at STRING NOT NULL,
          pipeline_version STRING NOT NULL,
          index_name STRING NOT NULL,
          index_version STRING NOT NULL,
          source_table STRING NOT NULL,
          documents_seen BIGINT NOT NULL,
          active_documents BIGINT NOT NULL,
          deletion_documents BIGINT NOT NULL,
          chunks_written BIGINT NOT NULL,
          vectors_upserted BIGINT NOT NULL,
          vectors_deleted BIGINT NOT NULL,
          required_field_failures BIGINT NOT NULL,
          embedding_dimension_failures BIGINT NOT NULL,
          authorization_metadata_failures BIGINT NOT NULL,
          lineage_failures BIGINT NOT NULL,
          duplicate_record_ids BIGINT NOT NULL,
          errors ARRAY<STRING> NOT NULL,
          artifact_uri STRING,
          decision STRING NOT NULL
        ) USING DELTA
        TBLPROPERTIES ('riverside.contract' = 'vectorization-quality-report/1.0.0')
        """
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap Riverside indexing Delta tables")
    parser.add_argument("--chunk-table", required=True)
    parser.add_argument("--vector-record-table", required=True)
    parser.add_argument("--quality-table", required=True)
    parser.add_argument("--evaluation-output-table", required=True)
    return parser


def main(**overrides: Any) -> None:
    values = vars(_parser().parse_args()) if not overrides else overrides
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("The bootstrap task requires an active Spark session")
    create_contract_tables(
        spark,
        chunk_table=str(values["chunk_table"]),
        vector_record_table=str(values["vector_record_table"]),
        quality_table=str(values["quality_table"]),
        evaluation_output_table=str(values["evaluation_output_table"]),
    )


if __name__ == "__main__":
    main()
