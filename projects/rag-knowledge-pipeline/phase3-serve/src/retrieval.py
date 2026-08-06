"""Provider-neutral retrieval adapters for local and Databricks modes."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from shared.contract_adapters import (
    AuthorizationContext,
    RetrievedRecord,
    adapt_local_document,
    adapt_vector_index_record,
)
from shared.vector_contract_v1 import normalize_flat_vector_record_v1


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        search_type: str,
        authorization: AuthorizationContext,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedRecord]: ...

    def status(self) -> str: ...


class LocalChromaRetriever:
    def __init__(self, config: Mapping[str, Any], device: str) -> None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        self._path = str(config["vector_store_path"])
        self._embeddings = HuggingFaceEmbeddings(
            model_name=str(config["embedding_model"]),
            model_kwargs={"device": device},
        )
        self._store = Chroma(
            persist_directory=self._path,
            embedding_function=self._embeddings,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        search_type: str,
        authorization: AuthorizationContext,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedRecord]:
        if authorization.tenant_id != "local" or authorization.region != "local":
            return []
        if search_type == "mmr":
            documents = self._store.max_marginal_relevance_search(query, k=top_k)
            pairs = [(document, 0.0) for document in documents]
        elif search_type == "similarity":
            pairs = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        else:
            raise ValueError(f"Local retrieval does not implement search type {search_type!r}")
        records = [
            adapt_local_document(document, score, ordinal=index)
            for index, (document, score) in enumerate(pairs)
        ]
        if filters and filters.get("source_version"):
            records = [record for record in records if record.source_version == filters["source_version"]]
        return records

    def status(self) -> str:
        return f"{self._store._collection.count()} local vectors"


class DatabricksRetriever:
    _RETURN_COLUMNS = [
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
        "embedding_dimensions",
        "index_name",
        "index_version",
        "content",
        "vector",
    ]

    def __init__(self, config: Mapping[str, Any]) -> None:
        from databricks.ai_search.client import AISearchClient
        from mlflow.deployments import get_deploy_client

        self._config = config
        self._dimensions = int(config["embedding_dimensions"])
        self._embedding_endpoint = str(config.get("embedding_endpoint") or config["embedding_model"])
        self._endpoint_name = str(config["vector_search_endpoint"])
        self._index_name = str(config["index_name"])
        self._endpoint_type = str(config.get("vector_search_endpoint_type", "standard"))
        self._embedding_client = get_deploy_client("databricks")
        self._index = AISearchClient().get_index(
            endpoint_name=self._endpoint_name,
            index_name=self._index_name,
        )

    def _embed(self, query: str) -> Sequence[float]:
        response = self._embedding_client.predict(
            endpoint=self._embedding_endpoint,
            inputs={"input": [query]},
        )
        payload = response.to_dict() if hasattr(response, "to_dict") else response
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not data or len(data[0].get("embedding", [])) != self._dimensions:
            raise ValueError("Databricks embedding response does not match the configured dimensions")
        return data[0]["embedding"]

    def _filters(
        self,
        authorization: AuthorizationContext,
        requested: Mapping[str, Any] | None,
    ) -> Any:
        scope_tokens = ["tenant"]
        scope_tokens.extend(f"principal:{value}" for value in authorization.principal_ids)
        scope_tokens.extend(f"group:{value}" for value in authorization.group_ids)
        if self._endpoint_type == "storage-optimized":
            classifications = ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in authorization.classifications)
            scopes = " OR ".join(
                f"acl_scope_tokens LIKE '% {value.replace(chr(39), chr(39) * 2)} %'"
                for value in scope_tokens
            )
            tenant = authorization.tenant_id.replace("'", "''")
            region = authorization.region.replace("'", "''")
            filters = (
                f"tenant_id = '{tenant}' AND region = '{region}' AND "
                f"classification IN ({classifications}) AND deletion_status = 'active' AND ({scopes})"
            )
            if requested and requested.get("source_version"):
                source_version = str(requested["source_version"]).replace("'", "''")
                filters += f" AND source_version = '{source_version}'"
            return filters
        filters = {
            "tenant_id": authorization.tenant_id,
            "region": authorization.region,
            "classification": list(authorization.classifications),
            "deletion_status": "active",
            "acl_scope_tokens LIKE": scope_tokens,
        }
        if requested and requested.get("source_version"):
            filters["source_version"] = str(requested["source_version"])
        return filters

    def _parse(self, response: Any) -> list[Mapping[str, Any]]:
        payload = response.to_dict() if hasattr(response, "to_dict") else response
        if not isinstance(payload, Mapping):
            raise ValueError("Databricks AI Search response must be an object")
        result = payload.get("result", payload)
        manifest = result.get("manifest", {})
        columns = [column["name"] for column in manifest.get("columns", [])]
        records = []
        for row in result.get("data_array", []):
            flat = dict(zip(columns, row)) if columns else dict(row)
            records.append(normalize_flat_vector_record_v1(flat))
        return records

    @staticmethod
    def _authorized(record: Mapping[str, Any], authorization: AuthorizationContext) -> bool:
        if record.get("tenant_id") != authorization.tenant_id:
            return False
        if record.get("region") != authorization.region:
            return False
        if record.get("classification") not in authorization.classifications:
            return False
        if record.get("deletion_state", {}).get("status") != "active":
            return False
        acl = record.get("acl", {})
        if acl.get("visibility") == "tenant":
            return True
        return bool(
            set(acl.get("principals", ())).intersection(authorization.principal_ids)
            or set(acl.get("groups", ())).intersection(authorization.group_ids)
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        search_type: str,
        authorization: AuthorizationContext,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedRecord]:
        if search_type != "similarity":
            raise ValueError(
                f"Databricks Direct Vector Access retrieval does not implement {search_type!r}"
            )
        candidate_count = min(200, max(top_k, top_k * 8))
        response = self._index.similarity_search(
            query_vector=list(self._embed(query)),
            columns=self._RETURN_COLUMNS,
            filters=self._filters(authorization, filters),
            num_results=candidate_count,
        )
        authorized = [record for record in self._parse(response) if self._authorized(record, authorization)]
        return [adapt_vector_index_record(record) for record in authorized[:top_k]]

    def status(self) -> str:
        return f"Databricks index {self._index_name}"


def build_retriever(config: Mapping[str, Any], device: str) -> Retriever:
    mode = str(config["mode"])
    provider = str(config["serving"]["retrieval"].get("provider", "auto"))
    resolved = mode if provider == "auto" else provider
    if resolved == "local":
        return LocalChromaRetriever(config["local"], device)
    if resolved in {"remote", "databricks"}:
        return DatabricksRetriever(config["remote"])
    raise ValueError(f"Unsupported retrieval provider: {resolved}")
