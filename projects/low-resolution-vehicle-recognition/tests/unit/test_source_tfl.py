from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from roadid.contracts import CameraSource
from roadid.sources import SourceDisabledError, SourceTermsError, TfLJamCamSource


def _source(*, enabled: bool, options=None) -> CameraSource:
    return CameraSource(
        source_id="tfl-jamcam",
        name="TfL JamCam",
        adapter_type="tfl_jamcam",
        enabled=enabled,
        attribution="Transport for London",
        terms_url="https://tfl.gov.uk/terms",
        refresh_seconds=10,
        options=options or {},
    )


def test_tfl_provider_is_disabled_by_default() -> None:
    with pytest.raises(SourceDisabledError, match="disabled"):
        TfLJamCamSource(_source(enabled=False)).capture("run-a")


def test_tfl_provider_requires_terms_before_official_api_access() -> None:
    with pytest.raises(SourceTermsError, match="terms"):
        TfLJamCamSource(_source(enabled=True)).list_catalog()


class _CatalogResponse:
    status_code = 200

    def json(self):
        return [
            {
                "id": "JamCams_00002.00865",
                "commonName": "A406 Billet Underpass E",
                "lat": 51.60067,
                "lon": -0.01594,
                "additionalProperties": [
                    {"key": "available", "value": "true"},
                    {
                        "key": "imageUrl",
                        "value": "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00002.00865.jpg",
                    },
                ],
            }
        ]


class _SnapshotResponse:
    status_code = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {
            "Content-Type": "image/jpeg",
            "Content-Length": str(len(payload)),
            "Last-Modified": "Thu, 13 Aug 2026 12:00:00 GMT",
        }

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        pass


class _Session:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _CatalogResponse() if url.endswith("/Place/Type/JamCam") else _SnapshotResponse(
            self.payload
        )

    def close(self):
        pass


def _public_resolver(host, port, **kwargs):
    return [(2, 1, 6, "", ("52.216.0.1", port))]


def test_tfl_catalog_selects_and_decodes_official_snapshot() -> None:
    image = np.full((48, 72, 3), (40, 80, 120), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    session = _Session(buffer.tobytes())
    source = _source(
        enabled=True,
        options={
            "terms_reviewed": True,
            "official_api_base_url": "https://api.tfl.gov.uk",
            "camera_id": "JamCams_00002.00865",
            "allowed_image_hosts": ["s3-eu-west-1.amazonaws.com"],
        },
    )
    adapter = TfLJamCamSource(
        source,
        session=session,
        resolver=_public_resolver,
        clock=lambda: datetime(2026, 8, 13, 12, 0, 10, tzinfo=UTC),
    )

    packet = adapter.capture("run-live")

    assert packet.image_bgr.shape == (48, 72, 3)
    assert packet.source_metadata["camera_id"] == "JamCams_00002.00865"
    assert packet.source_metadata["provider"] == "Transport for London"
    assert session.urls == [
        "https://api.tfl.gov.uk/Place/Type/JamCam",
        "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00002.00865.jpg",
    ]
