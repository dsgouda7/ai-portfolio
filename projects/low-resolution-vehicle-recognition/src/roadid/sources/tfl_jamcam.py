"""Disabled TfL JamCam catalog shell restricted to official API configuration."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from roadid.contracts import FramePacket
from roadid.sources.base import (
    BaseSourceAdapter,
    SourceAccessError,
    SourceDisabledError,
    SourceSecurityError,
    SourceTermsError,
    SourceUnavailableError,
)


class TfLJamCamSource(BaseSourceAdapter):
    OFFICIAL_API_HOST = "api.tfl.gov.uk"

    def _capture_once(self, run_id: str) -> FramePacket:
        self._validate_access()
        raise SourceUnavailableError(
            "TfL JamCam frame acquisition is not implemented; the provider remains a catalog shell",
            source_id=self.source.source_id,
        )

    def list_catalog(self) -> list[dict[str, Any]]:
        if not self.source.enabled:
            raise SourceDisabledError(
                f"source '{self.source.source_id}' is disabled",
                source_id=self.source.source_id,
            )
        self._validate_access()
        raise SourceUnavailableError(
            "TfL JamCam catalog access is not implemented; no page scraping fallback is permitted",
            source_id=self.source.source_id,
        )

    def catalog(self) -> list[dict[str, Any]]:
        return self.list_catalog()

    def _validate_access(self) -> None:
        options = self.source.options
        if options.get("terms_reviewed") is not True:
            raise SourceTermsError(
                "TfL JamCam terms must be reviewed and explicitly accepted before access",
                source_id=self.source.source_id,
            )
        api_base_url = options.get("official_api_base_url")
        if not isinstance(api_base_url, str) or not api_base_url:
            raise SourceAccessError(
                "TfL JamCam requires an official_api_base_url configuration",
                source_id=self.source.source_id,
            )
        parsed = urlsplit(api_base_url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() != self.OFFICIAL_API_HOST:
            raise SourceSecurityError(
                "TfL JamCam permits only the official HTTPS API host; scraping is forbidden",
                source_id=self.source.source_id,
            )
        if not options.get("app_key"):
            raise SourceAccessError(
                "TfL JamCam official API credentials are not configured",
                source_id=self.source.source_id,
            )
