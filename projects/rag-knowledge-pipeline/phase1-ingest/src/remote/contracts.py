"""Construction of contract-exact Riverside raw and parsed document records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


CONTRACT_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REGION = re.compile(r"^[a-z0-9-]{2,64}$")
_METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted"}
_VISIBILITIES = {"tenant", "restricted"}
_DELETION_STATES = {"active", "pending", "deleted"}


class ContractValueError(ValueError):
    """A source or governance value cannot produce a valid v1 record."""


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ContractValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_identifier(name: str, value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ContractValueError(f"{name} is not a valid v1 identifier")
    return value


def _bounded_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool | None]:
    if len(metadata) > 32:
        raise ContractValueError("metadata exceeds the v1 limit of 32 properties")

    bounded: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if not _METADATA_KEY.fullmatch(key):
            raise ContractValueError(f"metadata key {key!r} is not allowed by the v1 contract")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ContractValueError(f"metadata value for {key!r} must be scalar")
        if isinstance(value, str) and len(value) > 512:
            raise ContractValueError(f"metadata value for {key!r} exceeds 512 characters")
        bounded[key] = value
    return bounded


def canonical_source_uri(uri: str) -> str:
    """Normalize URI casing and dot segments without changing source identity."""
    parts = urlsplit(uri)
    if not parts.scheme:
        raise ContractValueError("source and storage locations must be absolute URIs")

    path_parts: list[str] = []
    for part in parts.path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if path_parts:
                path_parts.pop()
            continue
        path_parts.append(part)

    normalized_path = "/" + "/".join(path_parts) if parts.path.startswith("/") else "/".join(path_parts)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, parts.query, ""))


def sha256_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def deterministic_document_id(tenant_id: str, source_uri: str) -> str:
    identity = f"{tenant_id}\n{canonical_source_uri(source_uri)}".encode("utf-8")
    return f"doc-{sha256(identity).hexdigest()[:40]}"


@dataclass(frozen=True)
class GovernanceContext:
    tenant_id: str
    region: str
    classification: str
    acl_visibility: str = "tenant"
    acl_principals: tuple[str, ...] = ()
    acl_groups: tuple[str, ...] = ()
    deletion_status: str = "active"
    deletion_requested_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_identifier("tenant_id", self.tenant_id)
        if not _REGION.fullmatch(self.region):
            raise ContractValueError("region is not valid for the v1 contract")
        if self.classification not in _CLASSIFICATIONS:
            raise ContractValueError("classification is not valid for the v1 contract")
        if self.acl_visibility not in _VISIBILITIES:
            raise ContractValueError("ACL visibility is not valid for the v1 contract")
        if len(self.acl_principals) > 64 or len(self.acl_groups) > 64:
            raise ContractValueError("ACL principal or group count exceeds the v1 limit")
        if len(set(self.acl_principals)) != len(self.acl_principals):
            raise ContractValueError("ACL principals must be unique")
        if len(set(self.acl_groups)) != len(self.acl_groups):
            raise ContractValueError("ACL groups must be unique")
        for principal in (*self.acl_principals, *self.acl_groups):
            _validate_identifier("ACL entry", principal)
        if self.deletion_status not in _DELETION_STATES:
            raise ContractValueError("deletion status is not valid for the v1 contract")
        if self.deletion_status in {"pending", "deleted"} and self.deletion_requested_at is None:
            raise ContractValueError("pending and deleted records require deletion_requested_at")
        if self.deletion_status == "deleted" and self.deleted_at is None:
            raise ContractValueError("deleted records require deleted_at")

    def acl(self) -> dict[str, Any]:
        return {
            "visibility": self.acl_visibility,
            "principals": list(self.acl_principals),
            "groups": list(self.acl_groups),
        }

    def deletion_state(self) -> dict[str, str]:
        state = {"status": self.deletion_status}
        if self.deletion_requested_at is not None:
            state["requested_at"] = _utc_timestamp(self.deletion_requested_at)
        if self.deleted_at is not None:
            state["deleted_at"] = _utc_timestamp(self.deleted_at)
        return state


@dataclass(frozen=True)
class SourceObject:
    source_uri: str
    storage_uri: str
    source_name: str
    media_type: str
    content: bytes
    source_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_source_uri(self.source_uri)
        canonical_source_uri(self.storage_uri)
        if not self.source_name or len(self.source_name) > 255:
            raise ContractValueError("source_name must contain 1 to 255 characters")
        if "/" not in self.media_type or len(self.media_type) > 128:
            raise ContractValueError("media_type must be a bounded MIME type")
        if len(self.content) > 1_073_741_824:
            raise ContractValueError("source content exceeds the v1 one-GiB limit")
        if self.source_version is not None and not (1 <= len(self.source_version) <= 128):
            raise ContractValueError("source_version must contain 1 to 128 characters")
        _bounded_metadata(self.metadata)


@dataclass(frozen=True)
class ParsedContent:
    title: str
    text: str
    language: str
    parser_name: str
    parser_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_raw_record(
    source: SourceObject,
    governance: GovernanceContext,
    ingested_at: datetime,
) -> dict[str, Any]:
    raw_hash = sha256_bytes(source.content)
    source_uri = canonical_source_uri(source.source_uri)
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": "raw_document",
        "tenant_id": governance.tenant_id,
        "document_id": deterministic_document_id(governance.tenant_id, source_uri),
        "source_uri": source_uri,
        "source_version": source.source_version or f"sha256:{raw_hash}",
        "content_hash": raw_hash,
        "acl": governance.acl(),
        "region": governance.region,
        "classification": governance.classification,
        "ingested_at": _utc_timestamp(ingested_at),
        "pipeline_version": PIPELINE_VERSION,
        "deletion_state": governance.deletion_state(),
        "media_type": source.media_type,
        "byte_size": len(source.content),
        "storage_uri": canonical_source_uri(source.storage_uri),
        "source_name": source.source_name,
        "metadata": _bounded_metadata(source.metadata),
    }


def build_parsed_record(
    raw_record: Mapping[str, Any],
    parsed: ParsedContent,
    parsed_at: datetime,
) -> dict[str, Any]:
    text = normalize_text(parsed.text)
    if not text:
        raise ContractValueError("parsed text must not be empty")
    title = normalize_text(parsed.title) or PurePosixPath(str(raw_record["source_name"])).stem
    if len(title) > 512:
        raise ContractValueError("parsed title exceeds the v1 limit")
    if len(text) > 10_000_000:
        raise ContractValueError("parsed text exceeds the v1 limit")

    metadata = dict(raw_record.get("metadata", {}))
    metadata.update(parsed.metadata)
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": "parsed_document",
        "tenant_id": raw_record["tenant_id"],
        "document_id": raw_record["document_id"],
        "source_uri": raw_record["source_uri"],
        "source_version": raw_record["source_version"],
        "content_hash": sha256_bytes(text.encode("utf-8")),
        "raw_content_hash": raw_record["content_hash"],
        "acl": raw_record["acl"],
        "region": raw_record["region"],
        "classification": raw_record["classification"],
        "ingested_at": raw_record["ingested_at"],
        "parsed_at": _utc_timestamp(parsed_at),
        "pipeline_version": PIPELINE_VERSION,
        "deletion_state": raw_record["deletion_state"],
        "parser": {"name": parsed.parser_name, "version": parsed.parser_version},
        "language": parsed.language,
        "title": title,
        "text": text,
        "metadata": _bounded_metadata(metadata),
    }


def canonical_json(record: Mapping[str, Any]) -> str:
    """Serialize a contract record deterministically for Delta staging."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
