"""Official TfL JamCam catalog and hardened snapshot acquisition."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from roadid.contracts import CameraSource, FramePacket
from roadid.sources.base import (
    BaseSourceAdapter,
    SourceAccessError,
    SourceDisabledError,
    SourceFetchError,
    SourceSecurityError,
    SourceTermsError,
    SourceUnavailableError,
    SourceValidationError,
)
from roadid.sources.snapshot_http import SnapshotHttpSource


class TfLJamCamSource(BaseSourceAdapter):
    OFFICIAL_API_HOST = "api.tfl.gov.uk"
    DEFAULT_IMAGE_HOSTS = ("s3-eu-west-1.amazonaws.com",)

    def __init__(
        self,
        source: CameraSource,
        *,
        session: Any | None = None,
        resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(source, clock=clock)
        self._session = session or requests.Session()
        if session is None:
            self._session.trust_env = False
        self._resolver = resolver
        self._snapshot: SnapshotHttpSource | None = None
        self._selected_camera: dict[str, Any] | None = None

    def _capture_once(self, run_id: str) -> FramePacket:
        self._validate_access()
        if self._snapshot is None:
            camera = self._select_camera(self._load_catalog())
            self._snapshot = self._snapshot_source(camera)
        packet = self._snapshot.capture(run_id)
        camera = self._selected_camera or {}
        return FramePacket(
            run_id=packet.run_id,
            source_id=packet.source_id,
            frame_id=packet.frame_id,
            captured_at=packet.captured_at,
            received_at=packet.received_at,
            image_bgr=packet.image_bgr,
            source_metadata={
                **packet.source_metadata,
                "provider": "Transport for London",
                "camera_id": str(camera.get("id", "")),
                "camera_name": str(camera.get("name", "")),
            },
        )

    def list_catalog(self) -> list[dict[str, Any]]:
        if not self.source.enabled:
            raise SourceDisabledError(
                f"source '{self.source.source_id}' is disabled",
                source_id=self.source.source_id,
            )
        self._validate_access()
        return self._load_catalog()

    def catalog(self) -> list[dict[str, Any]]:
        return self.list_catalog()

    def close(self) -> None:
        if self._snapshot is not None:
            self._snapshot.close()
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

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
        camera_id = options.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id.startswith("JamCams_"):
            raise SourceAccessError(
                "TfL JamCam requires an explicit JamCams_ camera_id",
                source_id=self.source.source_id,
            )

    def _load_catalog(self) -> list[dict[str, Any]]:
        base_url = str(self.source.options["official_api_base_url"]).rstrip("/") + "/"
        url = urljoin(base_url, "Place/Type/JamCam")
        params = {}
        app_key = self.source.options.get("app_key")
        if isinstance(app_key, str) and app_key:
            params["app_key"] = app_key
        try:
            response = self._session.get(url, params=params, timeout=(2.0, 10.0))
        except requests.RequestException as error:
            raise SourceUnavailableError(
                "TfL JamCam catalog request failed",
                source_id=self.source.source_id,
            ) from error
        if not 200 <= int(response.status_code) < 300:
            raise SourceFetchError(
                f"TfL JamCam catalog returned HTTP {response.status_code}",
                source_id=self.source.source_id,
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            raise SourceValidationError(
                "TfL JamCam catalog did not return JSON",
                source_id=self.source.source_id,
            ) from error
        if not isinstance(payload, list):
            raise SourceValidationError(
                "TfL JamCam catalog must be a list",
                source_id=self.source.source_id,
            )
        catalog = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            properties = _properties(item.get("additionalProperties"))
            image_url = properties.get("imageUrl")
            available = properties.get("available", "false").lower() == "true"
            if isinstance(image_url, str) and image_url:
                catalog.append(
                    {
                        "id": str(item.get("id", "")),
                        "name": str(item.get("commonName", "")),
                        "available": available,
                        "image_url": image_url,
                        "lat": item.get("lat"),
                        "lon": item.get("lon"),
                    }
                )
        if not catalog:
            raise SourceValidationError(
                "TfL JamCam catalog contained no image snapshots",
                source_id=self.source.source_id,
            )
        return catalog

    def _select_camera(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        camera_id = self.source.options["camera_id"]
        camera = next((item for item in catalog if item["id"] == camera_id), None)
        if camera is None:
            raise SourceAccessError(
                "configured TfL JamCam camera_id was not found",
                source_id=self.source.source_id,
            )
        if not camera["available"]:
            raise SourceUnavailableError(
                "configured TfL JamCam is currently unavailable",
                source_id=self.source.source_id,
            )
        self._selected_camera = camera
        return camera

    def _snapshot_source(self, camera: dict[str, Any]) -> SnapshotHttpSource:
        image_url = str(camera["image_url"])
        image_host = (urlsplit(image_url).hostname or "").lower()
        configured_hosts = self.source.options.get("allowed_image_hosts")
        hosts = (
            tuple(str(value).lower() for value in configured_hosts)
            if isinstance(configured_hosts, list)
            else self.DEFAULT_IMAGE_HOSTS
        )
        if image_host not in hosts:
            raise SourceSecurityError(
                "TfL JamCam image host is not allowlisted",
                source_id=self.source.source_id,
            )
        snapshot_source = CameraSource(
            source_id=self.source.source_id,
            name=self.source.name,
            adapter_type="snapshot_http",
            enabled=True,
            attribution=self.source.attribution,
            terms_url=self.source.terms_url,
            refresh_seconds=self.source.refresh_seconds,
            location_label=self.source.location_label,
            location_precision=self.source.location_precision,
            options={
                "url": image_url,
                "allowed_hosts": list(hosts),
                "allowed_schemes": ["https"],
                "allowed_ports": [443],
                "max_retries": 2,
                "retry_backoff_seconds": 0.2,
                "connect_timeout_seconds": 2.0,
                "read_timeout_seconds": 10.0,
                "max_bytes": 1_000_000,
                "max_width": 1_920,
                "max_height": 1_080,
                "max_pixels": 2_073_600,
                "max_age_seconds": max(300.0, self.source.refresh_seconds * 30),
                "require_freshness": True,
                "freshness_headers": ["Last-Modified"],
                "content_types": ["image/jpeg"],
                "max_redirects": 0,
            },
        )
        return SnapshotHttpSource(
            snapshot_source,
            session=self._session,
            resolver=self._resolver,
            clock=self._clock,
        )


def _properties(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    result = {}
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            result[item["key"]] = str(item.get("value", ""))
    return result
