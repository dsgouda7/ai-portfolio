from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from endpoint_client.auth import AsyncTokenCredential, create_default_credential
from rag_orchestrator import ACL, RetrievedChunk, SearchQuery

from .config import PlatformConfig


_RETURN_COLUMNS = (
    "tenant_id",
    "chunk_id",
    "document_id",
    "source_uri",
    "source_version",
    "content_hash",
    "acl_visibility",
    "acl_principals_json",
    "acl_groups_json",
    "region",
    "classification",
    "indexed_at",
    "index_version",
    "deletion_status",
    "content",
)


@dataclass(frozen=True, slots=True)
class DatabricksSearchConfig:
    host: str
    vector_search_endpoint: str
    index_name: str
    index_version: str
    embedding_endpoint: str
    embedding_dimensions: int
    region: str
    token_scope: str = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
    timeout_seconds: int = 30
    managed_identity_client_id: str | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.host)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("DATABRICKS_HOST must be an HTTPS workspace origin")
        if not self.vector_search_endpoint or not self.embedding_endpoint:
            raise ValueError("Databricks vector-search and embedding endpoints are required")
        if not self.token_scope.endswith("/.default"):
            raise ValueError("Databricks token scope must end in /.default")
        if not 1 <= self.embedding_dimensions <= 65536:
            raise ValueError("embedding dimensions must be between 1 and 65536")
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("Databricks timeout must be between 1 and 120 seconds")

    @classmethod
    def from_environment(
        cls,
        profile: PlatformConfig,
        environ: Mapping[str, str] | None = None,
    ) -> DatabricksSearchConfig:
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(name)
            if not value:
                raise ValueError(f"missing required environment variable: {name}")
            return value

        return cls(
            host=required("DATABRICKS_HOST"),
            vector_search_endpoint=required("RIVERSIDE_VECTOR_SEARCH_ENDPOINT"),
            index_name=profile.data.index_name,
            index_version=profile.data.index_version,
            embedding_endpoint=required("RIVERSIDE_EMBEDDING_ENDPOINT"),
            embedding_dimensions=int(required("RIVERSIDE_EMBEDDING_DIMENSIONS")),
            region=profile.region,
            token_scope=values.get(
                "RIVERSIDE_DATABRICKS_TOKEN_SCOPE",
                "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default",
            ),
            timeout_seconds=int(values.get("RIVERSIDE_DATABRICKS_TIMEOUT_SECONDS", "30")),
            managed_identity_client_id=values.get("AZURE_CLIENT_ID"),
        )


class DatabricksSearchIndex:
    def __init__(
        self,
        config: DatabricksSearchConfig,
        *,
        credential: AsyncTokenCredential | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._credential = credential or create_default_credential(config.managed_identity_client_id)
        self._http = http_client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_credential = credential is None
        self._owns_http_client = http_client is None

    async def check_ready(self) -> None:
        response = await self._http.get(
            self._index_url(),
            headers=await self._headers(),
        )
        response.raise_for_status()

    async def search(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        if query.filters is not None and query.filters.region not in {None, self._config.region}:
            return []
        vector = await self._embed(query.text)
        response = await self._http.post(
            f"{self._index_url()}/query",
            headers=await self._headers(),
            json={
                "query_vector": vector,
                "columns": list(_RETURN_COLUMNS),
                "filters_json": json.dumps(self._filters(query), separators=(",", ":")),
                "num_results": min(200, max(query.top_k, query.top_k * 8)),
            },
        )
        response.raise_for_status()
        return self._parse_results(response.json(), query)[: query.top_k]

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()
        if self._owns_credential:
            await self._credential.close()

    async def _headers(self) -> dict[str, str]:
        token = await self._credential.get_token(self._config.token_scope)
        return {
            "authorization": f"Bearer {token.token}",
            "content-type": "application/json",
        }

    def _index_url(self) -> str:
        index_name = quote(self._config.index_name, safe="")
        return f"{self._config.host.rstrip('/')}/api/2.0/vector-search/indexes/{index_name}"

    async def _embed(self, text: str) -> list[float]:
        endpoint_name = quote(self._config.embedding_endpoint, safe="")
        response = await self._http.post(
            f"{self._config.host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations",
            headers=await self._headers(),
            json={"input": [text]},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("embedding endpoint response is missing the data array")
        vector = data[0].get("embedding")
        if not isinstance(vector, list) or len(vector) != self._config.embedding_dimensions:
            raise ValueError("embedding endpoint returned an unexpected vector dimension")
        return [float(value) for value in vector]

    def _filters(self, query: SearchQuery) -> dict[str, Any]:
        classifications = (
            query.filters.classification
            if query.filters is not None and query.filters.classification is not None
            else ["public", "internal", "confidential", "restricted"]
        )
        scope_tokens = ["tenant", f"principal:{query.auth.principal_id}"]
        scope_tokens.extend(f"group:{group_id}" for group_id in sorted(query.auth.group_ids))
        filters: dict[str, Any] = {
            "tenant_id": query.auth.tenant_id,
            "region": self._config.region,
            "classification": classifications,
            "deletion_status": "active",
            "index_version": self._config.index_version,
            "acl_scope_tokens LIKE": scope_tokens,
        }
        if query.filters is not None and query.filters.source_version is not None:
            filters["source_version"] = query.filters.source_version
        return filters

    def _parse_results(self, payload: Any, query: SearchQuery) -> list[RetrievedChunk]:
        if not isinstance(payload, dict):
            raise ValueError("Databricks vector-search response must be an object")
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise ValueError("Databricks vector-search result must be an object")
        manifest = result.get("manifest", {})
        columns = manifest.get("columns", []) if isinstance(manifest, dict) else []
        column_names = [column["name"] for column in columns if isinstance(column, dict) and "name" in column]
        rows = result.get("data_array", [])
        if not isinstance(rows, list):
            raise ValueError("Databricks vector-search data_array must be an array")
        parsed: list[RetrievedChunk] = []
        for row in rows:
            flat = dict(zip(column_names, row)) if isinstance(row, list) else dict(row)
            acl = ACL(
                visibility=flat["acl_visibility"],
                principals=frozenset(json.loads(flat["acl_principals_json"])),
                groups=frozenset(json.loads(flat["acl_groups_json"])),
            )
            score = float(flat.get("score", flat.get("similarity_score", 0.0)))
            parsed.append(
                RetrievedChunk(
                    tenant_id=flat["tenant_id"],
                    chunk_id=flat["chunk_id"],
                    document_id=flat["document_id"],
                    source_uri=flat["source_uri"],
                    source_version=flat["source_version"],
                    content_hash=flat["content_hash"],
                    title=flat.get("title") or flat["document_id"],
                    content=flat["content"],
                    score=score,
                    acl=acl,
                    region=flat["region"],
                    classification=flat["classification"],
                    index_version=flat["index_version"],
                    deletion_status=flat["deletion_status"],
                    indexed_at=datetime.fromisoformat(str(flat["indexed_at"]).replace("Z", "+00:00")),
                )
            )
        parsed.sort(key=lambda item: item.score, reverse=True)
        return parsed
