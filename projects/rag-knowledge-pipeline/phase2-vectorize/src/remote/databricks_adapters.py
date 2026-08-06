"""Azure Databricks implementations of the remote vectorization ports."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Mapping, Sequence

from .contracts import AuthorizationContext, RemoteSettings, is_record_authorized


class SparkDeltaParsedDocumentSource:
    def __init__(self, spark: Any, table_name: str) -> None:
        self.spark = spark
        self.table_name = table_name

    def read_documents(self, *, since: str | None = None) -> Sequence[Mapping[str, Any]]:
        from pyspark.sql import functions as functions

        frame = self.spark.table(self.table_name)
        if since:
            frame = frame.where(
                (functions.col("parsed_at") > functions.lit(since))
                | (functions.col("deletion_state.requested_at") > functions.lit(since))
                | (functions.col("deletion_state.deleted_at") > functions.lit(since))
            )
        return [row.asDict(recursive=True) for row in frame.toLocalIterator()]


class MlflowDatabricksEmbeddingProvider:
    """Call a Databricks model-serving endpoint using its OpenAI-compatible shape."""

    def __init__(self, settings: RemoteSettings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def _deploy_client(self) -> Any:
        if self._client is None:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client("databricks")
        return self._client

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        response = self._deploy_client().predict(
            endpoint=self.settings.embedding.endpoint_name,
            inputs={"input": list(texts)},
        )
        payload = response.to_dict() if hasattr(response, "to_dict") else response
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Sequence):
            raise ValueError("Embedding endpoint response is missing the data array")
        vectors = [item["embedding"] for item in data]
        if any(len(vector) != self.settings.embedding.dimensions for vector in vectors):
            raise ValueError("Embedding endpoint returned an unexpected vector dimension")
        return vectors


class SparkDeltaContractRepository:
    def __init__(self, spark: Any, settings: RemoteSettings) -> None:
        self.spark = spark
        self.settings = settings

    def record_ids_for_document(self, tenant_id: str, document_id: str) -> set[str]:
        from pyspark.sql import functions as functions

        if not self.spark.catalog.tableExists(self.settings.vector_record_table):
            return set()
        rows = (
            self.spark.table(self.settings.vector_record_table)
            .where(
                (functions.col("tenant_id") == tenant_id)
                & (functions.col("document_id") == document_id)
                & (functions.col("deletion_state.status") != "deleted")
            )
            .select("record_id")
            .collect()
        )
        return {str(row["record_id"]) for row in rows}

    def merge_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self._merge(self.settings.chunk_table, chunks, "chunk_id")

    def merge_vector_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._merge(self.settings.vector_record_table, records, "record_id")

    def mark_records_deleted(
        self,
        record_ids: Sequence[str],
        *,
        requested_at: str,
        deleted_at: str,
    ) -> None:
        from pyspark.sql import functions as functions

        if not record_ids or not self.spark.catalog.tableExists(self.settings.vector_record_table):
            return
        rows = (
            self.spark.table(self.settings.vector_record_table)
            .where(functions.col("record_id").isin(list(record_ids)))
            .collect()
        )
        tombstones = []
        chunk_ids = []
        for row in rows:
            record = row.asDict(recursive=True)
            chunk_ids.append(str(record["chunk_id"]))
            record["deletion_state"] = {
                "status": "deleted",
                "requested_at": requested_at,
                "deleted_at": deleted_at,
            }
            tombstones.append(record)
        self._merge(self.settings.vector_record_table, tombstones, "record_id")
        if chunk_ids and self.spark.catalog.tableExists(self.settings.chunk_table):
            chunk_rows = (
                self.spark.table(self.settings.chunk_table)
                .where(functions.col("chunk_id").isin(chunk_ids))
                .collect()
            )
            chunk_tombstones = []
            for row in chunk_rows:
                chunk = row.asDict(recursive=True)
                chunk["deletion_state"] = {
                    "status": "deleted",
                    "requested_at": requested_at,
                    "deleted_at": deleted_at,
                }
                chunk_tombstones.append(chunk)
            self._merge(self.settings.chunk_table, chunk_tombstones, "chunk_id")

    def write_quality_report(self, report: Mapping[str, Any]) -> str:
        artifact_uri = f"delta://{self.settings.quality_table}/{report['run_id']}"
        stored_report = {**report, "artifact_uri": artifact_uri}
        self._merge(self.settings.quality_table, [stored_report], "run_id")
        return artifact_uri

    def _merge(
        self,
        table_name: str,
        rows: Sequence[Mapping[str, Any]],
        primary_key: str,
    ) -> None:
        if not rows:
            return
        if not self.spark.catalog.tableExists(table_name):
            raise RuntimeError(
                f"Required Delta table {table_name!r} does not exist; run the indexing bootstrap task"
            )
        target_schema = self.spark.table(table_name).schema
        source = self.spark.createDataFrame([dict(row) for row in rows], schema=target_schema)

        from delta.tables import DeltaTable

        target = DeltaTable.forName(self.spark, table_name)
        (
            target.alias("target")
            .merge(source.alias("source"), f"target.{primary_key} = source.{primary_key}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )


class DatabricksDirectVectorIndex:
    """Fixed-schema Direct Vector Access index with fail-closed retrieval filters."""

    _RETURN_COLUMNS = [
        "record_id",
        "tenant_id",
        "chunk_id",
        "document_id",
        "parent_document_id",
        "source_uri",
        "source_version",
        "content_hash",
        "acl_visibility",
        "acl_principals_json",
        "acl_groups_json",
        "region",
        "classification",
        "ingested_at",
        "indexed_at",
        "pipeline_version",
        "deletion_status",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dimensions",
        "index_name",
        "index_version",
        "content",
    ]

    def __init__(self, settings: RemoteSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._index: Any | None = None

    def _search_client(self) -> Any:
        if self._client is None:
            from databricks.ai_search.client import AISearchClient

            self._client = AISearchClient()
        return self._client

    def _get_index(self) -> Any:
        if self._index is None:
            self._index = self._search_client().get_index(
                endpoint_name=self.settings.vector_search_endpoint,
                index_name=self.settings.index_name,
            )
        return self._index

    def ensure_exists(self) -> None:
        client = self._search_client()
        if client.index_exists(
            endpoint_name=self.settings.vector_search_endpoint,
            index_name=self.settings.index_name,
        ):
            self._index = client.get_index(
                endpoint_name=self.settings.vector_search_endpoint,
                index_name=self.settings.index_name,
            )
        else:
            self._index = client.create_direct_access_index(
                endpoint_name=self.settings.vector_search_endpoint,
                index_name=self.settings.index_name,
                primary_key="record_id",
                embedding_dimension=self.settings.embedding.dimensions,
                embedding_vector_column="vector",
                schema=self._index_schema(),
            )
        self._index.wait_until_ready(timeout=timedelta(hours=1))

    def upsert(self, records: Sequence[Mapping[str, Any]]) -> None:
        if records:
            self._get_index().upsert([self._to_index_payload(record) for record in records])

    def delete(self, record_ids: Sequence[str]) -> None:
        if record_ids:
            self._get_index().delete(list(record_ids))

    def query_authorized(
        self,
        query_vector: Sequence[float],
        context: AuthorizationContext,
        *,
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]:
        if len(query_vector) != self.settings.embedding.dimensions:
            raise ValueError("Query vector dimensions do not match the index contract")
        candidate_count = min(200, max(top_k, top_k * 8))
        response = self._get_index().similarity_search(
            query_vector=[float(value) for value in query_vector],
            columns=self._RETURN_COLUMNS,
            filters=self._backend_filters(context),
            num_results=candidate_count,
        )
        candidates = self._parse_search_response(response)
        authorized = [record for record in candidates if is_record_authorized(record, context)]
        return authorized[:top_k]

    def _backend_filters(self, context: AuthorizationContext) -> Any:
        scope_tokens = ["tenant"]
        scope_tokens.extend(f"principal:{value}" for value in context.principal_ids)
        scope_tokens.extend(f"group:{value}" for value in context.group_ids)
        if self.settings.query_endpoint_type == "storage-optimized":
            classifications = ", ".join(
                f"'{self._sql_literal(value)}'" for value in context.classifications
            )
            scope_predicates = " OR ".join(
                "acl_scope_tokens LIKE "
                f"'% {self._sql_literal(token)} %'" for token in scope_tokens
            )
            return (
                f"tenant_id = '{self._sql_literal(context.tenant_id)}' AND "
                f"region = '{self._sql_literal(context.region)}' AND "
                f"classification IN ({classifications}) AND "
                "deletion_status = 'active' AND "
                f"({scope_predicates})"
            )
        return {
            "tenant_id": context.tenant_id,
            "region": context.region,
            "classification": list(context.classifications),
            "deletion_status": "active",
            "acl_scope_tokens LIKE": scope_tokens,
        }

    @staticmethod
    def _sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _index_schema() -> dict[str, str]:
        string_fields = {
            "contract_version",
            "kind",
            "record_id",
            "tenant_id",
            "chunk_id",
            "document_id",
            "parent_document_id",
            "source_uri",
            "source_version",
            "content_hash",
            "acl_visibility",
            "acl_principals_json",
            "acl_groups_json",
            "acl_scope_tokens",
            "region",
            "classification",
            "ingested_at",
            "indexed_at",
            "pipeline_version",
            "deletion_status",
            "deletion_requested_at",
            "deletion_deleted_at",
            "embedding_model_id",
            "embedding_model_revision",
            "index_name",
            "index_version",
            "content",
        }
        schema = {name: "string" for name in sorted(string_fields)}
        schema["embedding_dimensions"] = "int"
        schema["vector"] = "array<float>"
        return schema

    @staticmethod
    def _to_index_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        acl = record["acl"]
        deletion = record["deletion_state"]
        scope_tokens = (
            ["tenant"]
            if acl["visibility"] == "tenant"
            else [f"principal:{value}" for value in acl["principals"]]
            + [f"group:{value}" for value in acl["groups"]]
        )
        return {
            "contract_version": record["contract_version"],
            "kind": record["kind"],
            "record_id": record["record_id"],
            "tenant_id": record["tenant_id"],
            "chunk_id": record["chunk_id"],
            "document_id": record["document_id"],
            "parent_document_id": record["parent_document_id"],
            "source_uri": record["source_uri"],
            "source_version": record["source_version"],
            "content_hash": record["content_hash"],
            "acl_visibility": acl["visibility"],
            "acl_principals_json": json.dumps(acl["principals"], separators=(",", ":")),
            "acl_groups_json": json.dumps(acl["groups"], separators=(",", ":")),
            "acl_scope_tokens": f" {' '.join(scope_tokens)} ",
            "region": record["region"],
            "classification": record["classification"],
            "ingested_at": record["ingested_at"],
            "indexed_at": record["indexed_at"],
            "pipeline_version": record["pipeline_version"],
            "deletion_status": deletion["status"],
            "deletion_requested_at": deletion.get("requested_at", ""),
            "deletion_deleted_at": deletion.get("deleted_at", ""),
            "embedding_model_id": record["embedding"]["model_id"],
            "embedding_model_revision": record["embedding"]["model_revision"],
            "embedding_dimensions": record["embedding"]["dimensions"],
            "index_name": record["index_name"],
            "index_version": record["index_version"],
            "content": record["content"],
            "vector": record["vector"],
        }

    def _parse_search_response(self, response: Any) -> list[dict[str, Any]]:
        payload = response.to_dict() if hasattr(response, "to_dict") else response
        if not isinstance(payload, Mapping):
            raise ValueError("AI Search response must be an object")
        result = payload.get("result", payload)
        manifest = result.get("manifest", {})
        columns = [column["name"] for column in manifest.get("columns", [])]
        rows = result.get("data_array", [])
        parsed = []
        for row in rows:
            flat = dict(zip(columns, row)) if columns else dict(row)
            parsed.append(
                {
                    **flat,
                    "acl": {
                        "visibility": flat["acl_visibility"],
                        "principals": json.loads(flat["acl_principals_json"]),
                        "groups": json.loads(flat["acl_groups_json"]),
                    },
                    "deletion_state": {"status": flat["deletion_status"]},
                    "embedding": {
                        "model_id": flat["embedding_model_id"],
                        "model_revision": flat["embedding_model_revision"],
                        "dimensions": flat["embedding_dimensions"],
                    },
                }
            )
        return parsed
