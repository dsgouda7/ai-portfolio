"""Databricks job entry point for post-index retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from .contracts import RemoteSettings
from .databricks_adapters import DatabricksDirectVectorIndex, MlflowDatabricksEmbeddingProvider
from .evaluation import RetrievalEvaluator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Riverside's authorized retrieval index")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--vector-search-endpoint", required=True)
    parser.add_argument("--vector-search-endpoint-type", default="standard")
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-model-revision", required=True)
    parser.add_argument("--embedding-endpoint", required=True)
    parser.add_argument("--embedding-dimensions", type=int, required=True)
    parser.add_argument("--evaluation-table", required=True)
    parser.add_argument("--evaluation-output-table", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--report-id")
    return parser


def main(**overrides: Any) -> dict[str, Any]:
    values = vars(_parser().parse_args()) if not overrides else overrides
    from delta.tables import DeltaTable
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Retrieval evaluation requires an active Spark session")
    settings = RemoteSettings.from_mapping(values)
    cases = [row.asDict(recursive=True) for row in spark.table(values["evaluation_table"]).collect()]
    evaluator = RetrievalEvaluator(
        settings=settings,
        embedder=MlflowDatabricksEmbeddingProvider(settings),
        index=DatabricksDirectVectorIndex(settings),
    )
    report = evaluator.evaluate(
        cases,
        dataset_id=str(values["dataset_id"]),
        dataset_version=str(values["dataset_version"]),
        report_id=values.get("report_id") or None,
    )
    aggregate = report["aggregate"]
    row = {
        "report_id": report["report_id"],
        "generated_at": report["generated_at"],
        "dataset_id": report["dataset"]["id"],
        "dataset_version": report["dataset"]["version"],
        "index_name": report["index"]["name"],
        "index_version": report["index"]["version"],
        "evaluator_version": report["evaluator"]["version"],
        "case_count": aggregate["case_count"],
        "recall_at_k": aggregate["recall_at_k"],
        "mrr": aggregate["mrr"],
        "ndcg_at_k": aggregate["ndcg_at_k"],
        "authorization_leakage_count": aggregate["authorization_leakage_count"],
        "decision": report["decision"],
        "report_json": json.dumps(report, sort_keys=True, separators=(",", ":")),
    }
    output_table = str(values["evaluation_output_table"])
    source = spark.createDataFrame([row], schema=spark.table(output_table).schema)
    (
        DeltaTable.forName(spark, output_table)
        .alias("target")
        .merge(source.alias("source"), "target.report_id = source.report_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logging.getLogger("riverside.indexing.evaluation").info(
        "Retrieval evaluation completed",
        extra={
            "report_id": report["report_id"],
            "decision": report["decision"],
            "case_count": aggregate["case_count"],
        },
    )
    if report["decision"] != "pass":
        raise RuntimeError(f"Authorization leakage detected in retrieval report {report['report_id']}")
    return report


if __name__ == "__main__":
    main()
