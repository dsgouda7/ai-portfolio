"""Configuration and governance tests for identity-only remote jobs."""

from pathlib import Path
import sys

import pytest


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from remote.config import IngestionSettings  # noqa: E402


ENVIRONMENT = {
    "RIVERSIDE_INGEST_SOURCE_URI": "abfss://raw@riverside.dfs.core.windows.net/manuscripts",
    "RIVERSIDE_INGEST_TENANT_ID": "tenant-editorial-standard",
    "RIVERSIDE_INGEST_REGION": "eastus2",
    "RIVERSIDE_INGEST_CLASSIFICATION": "confidential",
    "RIVERSIDE_INGEST_ACL_VISIBILITY": "restricted",
    "RIVERSIDE_INGEST_ACL_GROUPS": "editors,legal-reviewers,editors",
}


def test_settings_use_environment_and_deduplicate_acl_entries() -> None:
    settings = IngestionSettings.from_mapping(environ=ENVIRONMENT)
    assert settings.source_adapter == "auto"
    assert settings.acl_groups == ("editors", "legal-reviewers")
    assert settings.governance().acl()["visibility"] == "restricted"


def test_embedded_credentials_are_rejected() -> None:
    with pytest.raises(ValueError, match="managed or workload identity"):
        IngestionSettings.from_mapping({"token": "not-allowed"}, ENVIRONMENT)


def test_deleted_state_requires_complete_lineage() -> None:
    environment = dict(ENVIRONMENT, RIVERSIDE_INGEST_DELETION_STATUS="deleted")
    with pytest.raises(ValueError, match="deletion_requested_at"):
        IngestionSettings.from_mapping(environ=environment)
