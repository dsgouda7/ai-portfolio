"""Composition root for the remote Azure Databricks vectorization mode."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .contracts import RemoteSettings
from .databricks_adapters import (
    DatabricksDirectVectorIndex,
    MlflowDatabricksEmbeddingProvider,
    SparkDeltaContractRepository,
    SparkDeltaParsedDocumentSource,
)
from .pipeline import RemoteVectorizationPipeline


def run_remote_vectorization(
    remote_config: Mapping[str, Any],
    *,
    logger: Any,
    spark: Any | None = None,
) -> dict[str, Any]:
    """Run the remote flow using Databricks runtime identity and managed services."""
    settings = RemoteSettings.from_mapping(remote_config)
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Remote vectorization must run in an active Spark session")

    logger.info(
        "Starting remote vectorization",
        extra={
            "source_table": settings.source_table,
            "index_name": settings.index_name,
            "index_version": settings.index_version,
            "embedding_model": settings.embedding.model_id,
            "embedding_revision": settings.embedding.model_revision,
            "embedding_dimensions": settings.embedding.dimensions,
        },
    )
    pipeline = RemoteVectorizationPipeline(
        settings=settings,
        source=SparkDeltaParsedDocumentSource(spark, settings.source_table),
        embedder=MlflowDatabricksEmbeddingProvider(settings),
        repository=SparkDeltaContractRepository(spark, settings),
        index=DatabricksDirectVectorIndex(settings),
    )
    report = pipeline.run(
        since=remote_config.get("since") or os.getenv("RIVERSIDE_INDEX_SINCE"),
        run_id=remote_config.get("run_id") or os.getenv("RIVERSIDE_INDEX_RUN_ID"),
    )
    result = report.to_dict()
    logger.info(
        "Remote vectorization completed",
        extra={
            "decision": report.decision,
            "documents_seen": report.documents_seen,
            "vectors_upserted": report.vectors_upserted,
            "vectors_deleted": report.vectors_deleted,
            "quality_artifact": report.artifact_uri,
        },
    )
    if report.decision != "pass":
        raise RuntimeError(
            f"Remote vectorization quality gates failed; evidence: {report.artifact_uri}"
        )
    return result
