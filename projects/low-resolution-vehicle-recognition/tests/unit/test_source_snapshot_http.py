from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from io import BytesIO
from typing import Any

import pytest
import requests
from PIL import Image

from roadid.contracts import CameraSource
from roadid.sources import (
    HealthStatus,
    SnapshotHttpSource,
    SourceOversizedError,
    SourceSecurityError,
    SourceStaleError,
    SourceTimeoutError,
)

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
PUBLIC_IP = "93.184.216.34"


class MockResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class MockSession:
    def __init__(self, responses: list[MockResponse | Exception]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> MockResponse:
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        return None


def _png(width: int = 16, height: int = 12) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), (180, 70, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _headers(*, captured_at: datetime = NOW, body: bytes | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "image/png",
        "Last-Modified": format_datetime(captured_at, usegmt=True),
    }
    if body is not None:
        headers["Content-Length"] = str(len(body))
    return headers


def _source(**option_overrides: Any) -> CameraSource:
    options: dict[str, Any] = {
        "url": "https://camera.example/snapshot.png",
        "allowed_hosts": ["camera.example"],
        "allowed_schemes": ["https"],
        "allowed_ports": [443],
        "max_retries": 0,
        "max_bytes": 4096,
        "max_width": 64,
        "max_height": 64,
        "max_pixels": 4096,
        "max_age_seconds": 20,
    }
    options.update(option_overrides)
    return CameraSource(
        source_id="snapshot-test",
        name="Snapshot",
        adapter_type="snapshot_http",
        enabled=True,
        attribution="fixture",
        refresh_seconds=5,
        options=options,
    )


def _resolver(host: str, port: int, **kwargs: Any):
    address = "127.0.0.1" if host == "internal.example" else PUBLIC_IP
    return [(2, 1, 6, "", (address, port))]


def test_rejects_private_dns_and_metadata_targets() -> None:
    with pytest.raises(SourceSecurityError, match="forbidden address"):
        SnapshotHttpSource(
            _source(url="https://internal.example/image.png", allowed_hosts=["internal.example"]),
            resolver=_resolver,
        )

    with pytest.raises(SourceSecurityError, match="metadata"):
        SnapshotHttpSource(
            _source(
                url="http://metadata.google.internal/latest",
                allowed_hosts=["metadata.google.internal"],
                allowed_schemes=["http"],
                allowed_ports=[80],
            ),
            resolver=_resolver,
        )


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "127.0.0.1",
        "169.254.10.20",
        "224.0.0.1",
        "0.0.0.0",
        "240.0.0.1",
        "100.64.0.1",
    ],
)
def test_rejects_every_non_global_dns_address(address: str) -> None:
    def resolver(host: str, port: int, **kwargs: Any):
        return [(2, 1, 6, "", (address, port))]

    with pytest.raises(SourceSecurityError, match="forbidden address"):
        SnapshotHttpSource(_source(), resolver=resolver)


def test_redirect_target_is_revalidated_before_second_request() -> None:
    redirect = MockResponse(302, headers={"Location": "https://internal.example/image.png"})
    session = MockSession([redirect])
    adapter = SnapshotHttpSource(
        _source(allowed_hosts=["camera.example", "internal.example"]),
        session=session,
        resolver=_resolver,
        clock=lambda: NOW,
    )

    with pytest.raises(SourceSecurityError, match="forbidden address"):
        adapter.capture("run-a")

    assert session.urls == ["https://camera.example/snapshot.png"]
    assert redirect.closed


def test_oversized_content_length_fails_before_body_or_decode() -> None:
    response = MockResponse(
        200,
        headers={
            "Content-Type": "image/png",
            "Content-Length": "2048",
            "Date": format_datetime(NOW, usegmt=True),
        },
        body=b"not-read",
    )
    adapter = SnapshotHttpSource(
        _source(max_bytes=1024),
        session=MockSession([response]),
        resolver=_resolver,
        clock=lambda: NOW,
    )

    with pytest.raises(SourceOversizedError):
        adapter.capture("run-a")


def test_stale_snapshot_is_rejected() -> None:
    body = _png()
    response = MockResponse(
        200,
        headers=_headers(captured_at=NOW - timedelta(seconds=21), body=body),
        body=body,
    )
    adapter = SnapshotHttpSource(
        _source(),
        session=MockSession([response]),
        resolver=_resolver,
        clock=lambda: NOW,
    )

    with pytest.raises(SourceStaleError):
        adapter.capture("run-a")


def test_dimensions_are_limited_before_opencv_decode() -> None:
    body = _png(width=65, height=12)
    response = MockResponse(200, headers=_headers(body=body), body=body)
    adapter = SnapshotHttpSource(
        _source(),
        session=MockSession([response]),
        resolver=_resolver,
        clock=lambda: NOW,
    )

    with pytest.raises(SourceOversizedError, match="dimension"):
        adapter.capture("run-a")


def test_bounded_retries_recover_and_update_health() -> None:
    body = _png()
    responses = [
        MockResponse(503),
        MockResponse(503),
        MockResponse(200, headers=_headers(body=body), body=body),
    ]
    sleeps: list[float] = []
    adapter = SnapshotHttpSource(
        _source(max_retries=2, retry_backoff_seconds=0.01),
        session=MockSession(responses),
        resolver=_resolver,
        clock=lambda: NOW,
        sleeper=sleeps.append,
    )

    packet = adapter.capture("run-a")

    assert packet.frame_id == 0
    assert packet.image_bgr.shape == (12, 16, 3)
    assert sleeps == [0.01, 0.02]
    assert adapter.health.status is HealthStatus.HEALTHY
    assert adapter.health.successful_captures == 1
    assert adapter.health.consecutive_failures == 0


def test_request_timeout_is_typed_without_real_network() -> None:
    adapter = SnapshotHttpSource(
        _source(),
        session=MockSession([requests.Timeout("fixture")]),
        resolver=_resolver,
        clock=lambda: NOW,
    )

    with pytest.raises(SourceTimeoutError) as caught:
        adapter.capture("run-a")
    assert caught.value.code == "source_timeout"
