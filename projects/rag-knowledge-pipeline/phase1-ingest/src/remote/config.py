"""Environment-backed configuration for Databricks ingestion jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import re
from typing import Mapping, Any

from .contracts import GovernanceContext, PIPELINE_VERSION


_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_SECRET_KEYS = {"token", "access_token", "api_key", "client_secret", "connection_string"}


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in (value or "").split(",") if part.strip()))


def _optional_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _env(environ: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    value = environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class IngestionSettings:
    source_uri: str
    catalog: str
    schema: str
    tenant_id: str
    region: str
    classification: str
    volume: str = "rag_data"
    source_adapter: str = "auto"
    file_glob: str | None = None
    acl_visibility: str = "tenant"
    acl_principals: tuple[str, ...] = ()
    acl_groups: tuple[str, ...] = ()
    deletion_status: str = "active"
    deletion_requested_at: datetime | None = None
    deleted_at: datetime | None = None
    max_file_bytes: int = 100 * 1024 * 1024
    max_quarantine_rate: float = 0.02
    min_parse_success_rate: float = 0.98
    pipeline_version: str = PIPELINE_VERSION
    run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("catalog", "schema", "volume"):
            if not _SQL_IDENTIFIER.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a valid Unity Catalog identifier")
        if self.source_adapter not in {"auto", "adls", "volume"}:
            raise ValueError("source_adapter must be auto, adls, or volume")
        if not 1 <= self.max_file_bytes <= 1_073_741_824:
            raise ValueError("max_file_bytes must be between one byte and one GiB")
        if not 0 <= self.max_quarantine_rate <= 1:
            raise ValueError("max_quarantine_rate must be between zero and one")
        if not 0 <= self.min_parse_success_rate <= 1:
            raise ValueError("min_parse_success_rate must be between zero and one")
        if self.pipeline_version != PIPELINE_VERSION:
            raise ValueError(f"this package emits only pipeline version {PIPELINE_VERSION}")
        self.governance()

    @classmethod
    def from_mapping(
        cls,
        remote_config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "IngestionSettings":
        config = dict(remote_config or {})
        environment = environ or os.environ
        supplied_secrets = _SECRET_KEYS.intersection(config)
        if supplied_secrets:
            raise ValueError("credentials must use Databricks managed or workload identity")

        source_uri = _env(environment, "RIVERSIDE_INGEST_SOURCE_URI", config.get("source_uri"))
        tenant_id = _env(environment, "RIVERSIDE_INGEST_TENANT_ID", config.get("tenant_id"))
        region = _env(environment, "RIVERSIDE_INGEST_REGION", config.get("region"))
        classification = _env(
            environment,
            "RIVERSIDE_INGEST_CLASSIFICATION",
            config.get("classification"),
        )
        missing = [
            name
            for name, value in (
                ("RIVERSIDE_INGEST_SOURCE_URI", source_uri),
                ("RIVERSIDE_INGEST_TENANT_ID", tenant_id),
                ("RIVERSIDE_INGEST_REGION", region),
                ("RIVERSIDE_INGEST_CLASSIFICATION", classification),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"missing required ingestion settings: {', '.join(missing)}")

        return cls(
            source_uri=str(source_uri),
            catalog=str(_env(environment, "RIVERSIDE_INGEST_CATALOG", config.get("catalog", "main"))),
            schema=str(_env(environment, "RIVERSIDE_INGEST_SCHEMA", config.get("schema", "rag_demo"))),
            volume=str(_env(environment, "RIVERSIDE_INGEST_VOLUME", config.get("volume", "rag_data"))),
            tenant_id=str(tenant_id),
            region=str(region),
            classification=str(classification),
            source_adapter=str(_env(environment, "RIVERSIDE_INGEST_SOURCE_ADAPTER", "auto")),
            file_glob=_env(environment, "RIVERSIDE_INGEST_FILE_GLOB"),
            acl_visibility=str(_env(environment, "RIVERSIDE_INGEST_ACL_VISIBILITY", "tenant")),
            acl_principals=_csv(_env(environment, "RIVERSIDE_INGEST_ACL_PRINCIPALS")),
            acl_groups=_csv(_env(environment, "RIVERSIDE_INGEST_ACL_GROUPS")),
            deletion_status=str(_env(environment, "RIVERSIDE_INGEST_DELETION_STATUS", "active")),
            deletion_requested_at=_optional_timestamp(
                _env(environment, "RIVERSIDE_INGEST_DELETION_REQUESTED_AT")
            ),
            deleted_at=_optional_timestamp(_env(environment, "RIVERSIDE_INGEST_DELETED_AT")),
            max_file_bytes=int(_env(environment, "RIVERSIDE_INGEST_MAX_FILE_BYTES", "104857600")),
            max_quarantine_rate=float(
                _env(environment, "RIVERSIDE_INGEST_MAX_QUARANTINE_RATE", "0.02")
            ),
            min_parse_success_rate=float(
                _env(environment, "RIVERSIDE_INGEST_MIN_PARSE_SUCCESS_RATE", "0.98")
            ),
            run_id=_env(environment, "DATABRICKS_RUN_ID"),
        )

    def governance(self) -> GovernanceContext:
        return GovernanceContext(
            tenant_id=self.tenant_id,
            region=self.region,
            classification=self.classification,
            acl_visibility=self.acl_visibility,
            acl_principals=self.acl_principals,
            acl_groups=self.acl_groups,
            deletion_status=self.deletion_status,
            deletion_requested_at=self.deletion_requested_at,
            deleted_at=self.deleted_at,
        )

    def table(self, name: str) -> str:
        if not _SQL_IDENTIFIER.fullmatch(name):
            raise ValueError("table name must be a valid Unity Catalog identifier")
        return f"`{self.catalog}`.`{self.schema}`.`{name}`"
