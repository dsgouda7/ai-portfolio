"""Databricks Asset Bundle wheel entry points."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from remote.config import IngestionSettings
from remote.delta import DeltaIngestionSink
from remote.pipeline import run_remote_ingestion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", default="rag_data")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--source-adapter", default="auto")
    parser.add_argument("--file-glob")
    parser.add_argument("--acl-visibility", default="tenant")
    parser.add_argument("--acl-principals", default="")
    parser.add_argument("--acl-groups", default="")
    parser.add_argument("--deletion-status", default="active")
    parser.add_argument("--deletion-requested-at")
    parser.add_argument("--deleted-at")
    parser.add_argument("--max-quarantine-rate", default="0.02")
    parser.add_argument("--min-parse-success-rate", default="0.98")
    parser.add_argument("--run-id")
    return parser


def _environment(arguments: argparse.Namespace) -> dict[str, str]:
    values = {
        "RIVERSIDE_INGEST_SOURCE_URI": arguments.source_uri,
        "RIVERSIDE_INGEST_CATALOG": arguments.catalog,
        "RIVERSIDE_INGEST_SCHEMA": arguments.schema,
        "RIVERSIDE_INGEST_VOLUME": arguments.volume,
        "RIVERSIDE_INGEST_TENANT_ID": arguments.tenant_id,
        "RIVERSIDE_INGEST_REGION": arguments.region,
        "RIVERSIDE_INGEST_CLASSIFICATION": arguments.classification,
        "RIVERSIDE_INGEST_SOURCE_ADAPTER": arguments.source_adapter,
        "RIVERSIDE_INGEST_ACL_VISIBILITY": arguments.acl_visibility,
        "RIVERSIDE_INGEST_ACL_PRINCIPALS": arguments.acl_principals,
        "RIVERSIDE_INGEST_ACL_GROUPS": arguments.acl_groups,
        "RIVERSIDE_INGEST_DELETION_STATUS": arguments.deletion_status,
        "RIVERSIDE_INGEST_MAX_QUARANTINE_RATE": arguments.max_quarantine_rate,
        "RIVERSIDE_INGEST_MIN_PARSE_SUCCESS_RATE": arguments.min_parse_success_rate,
    }
    if arguments.file_glob:
        values["RIVERSIDE_INGEST_FILE_GLOB"] = arguments.file_glob
    if arguments.deletion_requested_at:
        values["RIVERSIDE_INGEST_DELETION_REQUESTED_AT"] = arguments.deletion_requested_at
    if arguments.deleted_at:
        values["RIVERSIDE_INGEST_DELETED_AT"] = arguments.deleted_at
    run_id = arguments.run_id or os.getenv("DATABRICKS_RUN_ID")
    if run_id:
        values["DATABRICKS_RUN_ID"] = run_id
    return values


def ingest_entrypoint(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    run_remote_ingestion(environ=_environment(args))


def setup_tables_entrypoint(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    settings = IngestionSettings.from_mapping(environ=_environment(args))
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    DeltaIngestionSink(spark, settings).ensure_tables()
