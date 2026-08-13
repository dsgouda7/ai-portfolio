"""Secure still-image acquisition from administrator-configured HTTP endpoints."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from roadid.contracts import CameraSource, FramePacket, JSONValue
from roadid.sources.base import (
    BaseSourceAdapter,
    SourceConfigurationError,
    SourceFetchError,
    SourceOversizedError,
    SourceSecurityError,
    SourceStaleError,
    SourceTimeoutError,
    SourceUnavailableError,
    SourceValidationError,
)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_METADATA_HOSTS = {
    "instance-data",
    "metadata",
    "metadata.azure.com",
    "metadata.google.internal",
}
_METADATA_IPS = {
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("169.254.170.2"),
    ipaddress.ip_address("fd00:ec2::254"),
}


class SnapshotHttpSource(BaseSourceAdapter):
    def __init__(
        self,
        source: CameraSource,
        *,
        session: Any | None = None,
        resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        options = source.options
        max_retries = _option_int(options, "max_retries", 2, minimum=0, maximum=5)
        retry_backoff = _option_float(
            options, "retry_backoff_seconds", 0.1, minimum=0.0, maximum=30.0
        )
        base_kwargs: dict[str, Any] = {
            "max_retries": max_retries,
            "retry_backoff_seconds": retry_backoff,
            "clock": clock,
        }
        if sleeper is not None:
            base_kwargs["sleeper"] = sleeper
        super().__init__(source, **base_kwargs)
        self._url = _required_string(options, "url")
        self._allowed_hosts = frozenset(
            host.lower().rstrip(".") for host in _string_list(options, "allowed_hosts")
        )
        if not self._allowed_hosts:
            raise SourceConfigurationError("snapshot_http requires a non-empty allowed_hosts list")
        self._allowed_schemes = frozenset(
            value.lower() for value in _string_list(options, "allowed_schemes", ["https"])
        )
        if not self._allowed_schemes or not self._allowed_schemes <= {"http", "https"}:
            raise SourceConfigurationError("allowed_schemes may contain only http and https")
        self._allowed_ports = frozenset(
            _int_list(options, "allowed_ports", [80, 443], minimum=1, maximum=65_535)
        )
        self._connect_timeout = _option_float(
            options, "connect_timeout_seconds", 2.0, minimum=0.05, maximum=30.0
        )
        self._read_timeout = _option_float(
            options, "read_timeout_seconds", 4.0, minimum=0.05, maximum=60.0
        )
        self._max_bytes = _option_int(
            options, "max_bytes", 5_000_000, minimum=1_024, maximum=50_000_000
        )
        self._max_width = _option_int(options, "max_width", 4_096, minimum=1, maximum=16_384)
        self._max_height = _option_int(options, "max_height", 2_160, minimum=1, maximum=16_384)
        self._max_pixels = _option_int(
            options,
            "max_pixels",
            self._max_width * self._max_height,
            minimum=1,
            maximum=100_000_000,
        )
        self._max_age_seconds = _option_float(
            options, "max_age_seconds", 30.0, minimum=0.1, maximum=86_400.0
        )
        self._max_future_skew_seconds = _option_float(
            options, "max_future_skew_seconds", 5.0, minimum=0.0, maximum=300.0
        )
        self._require_freshness = bool(options.get("require_freshness", True))
        self._freshness_headers = tuple(
            name.lower()
            for name in _string_list(
                options,
                "freshness_headers",
                ["X-Captured-At", "Last-Modified"],
            )
        )
        self._content_types = frozenset(
            value.lower()
            for value in _string_list(
                options,
                "content_types",
                ["image/jpeg", "image/png", "image/webp"],
            )
        )
        self._max_redirects = _option_int(options, "max_redirects", 3, minimum=0, maximum=5)
        if session is None:
            self._session = requests.Session()
            self._session.trust_env = False
        else:
            self._session = session
        self._resolver = resolver
        self._next_frame_id = 0

        self._validate_remote_url(self._url)

    def _capture_once(self, run_id: str) -> FramePacket:
        received_at = self._clock().astimezone(UTC)
        current_url = self._url
        response: Any | None = None
        for redirect_count in range(self._max_redirects + 1):
            target = self._validate_remote_url(current_url)
            try:
                response = self._session.get(
                    current_url,
                    allow_redirects=False,
                    stream=True,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
            except requests.Timeout as error:
                raise SourceTimeoutError(
                    "snapshot endpoint timed out", source_id=self.source.source_id
                ) from error
            except requests.RequestException as error:
                raise SourceFetchError(
                    "snapshot endpoint request failed", source_id=self.source.source_id
                ) from error

            if response.status_code in _REDIRECT_STATUSES:
                location = _header(response.headers, "location")
                _close_response(response)
                response = None
                if redirect_count >= self._max_redirects:
                    raise SourceSecurityError(
                        "snapshot redirect limit exceeded", source_id=self.source.source_id
                    )
                if not location:
                    raise SourceFetchError(
                        "snapshot redirect omitted Location", source_id=self.source.source_id
                    )
                current_url = urljoin(current_url, location)
                self._validate_remote_url(current_url)
                continue

            try:
                payload, content_type, captured_at = self._validated_payload(response, received_at)
            finally:
                _close_response(response)
            image_bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise SourceValidationError(
                    "snapshot image could not be decoded", source_id=self.source.source_id
                )
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            return FramePacket(
                run_id=run_id,
                source_id=self.source.source_id,
                frame_id=frame_id,
                captured_at=captured_at,
                received_at=received_at,
                image_bgr=image_bgr,
                source_metadata={
                    "content_type": content_type,
                    "payload_bytes": len(payload),
                    "remote_host": target.hostname or "",
                    "redirect_count": redirect_count,
                },
            )
        raise AssertionError("bounded redirect loop terminated unexpectedly")

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _validate_remote_url(self, url: str) -> SplitResult:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in self._allowed_schemes:
            raise SourceSecurityError(
                "snapshot URL scheme is not allowlisted", source_id=self.source.source_id
            )
        if parsed.username or parsed.password or parsed.fragment:
            raise SourceSecurityError(
                "snapshot URL credentials and fragments are forbidden",
                source_id=self.source.source_id,
            )
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host or host not in self._allowed_hosts:
            raise SourceSecurityError(
                "snapshot host is not allowlisted", source_id=self.source.source_id
            )
        if host in _METADATA_HOSTS or host.endswith(".metadata.google.internal"):
            raise SourceSecurityError(
                "cloud metadata targets are forbidden", source_id=self.source.source_id
            )
        try:
            port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        except ValueError as error:
            raise SourceSecurityError(
                "snapshot URL port is invalid", source_id=self.source.source_id
            ) from error
        if port not in self._allowed_ports:
            raise SourceSecurityError(
                "snapshot port is not allowlisted", source_id=self.source.source_id
            )

        try:
            addresses = self._resolver(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            raise SourceUnavailableError(
                "snapshot host DNS resolution failed", source_id=self.source.source_id
            ) from error
        resolved = {_address_from_result(item) for item in addresses}
        if not resolved:
            raise SourceUnavailableError(
                "snapshot host DNS resolution returned no addresses",
                source_id=self.source.source_id,
            )
        # Every address must be globally routable because the HTTP client may select any DNS result.
        for address_text in resolved:
            self._validate_public_ip(address_text)
        return parsed

    def _validate_public_ip(self, address_text: str) -> None:
        try:
            address = ipaddress.ip_address(address_text)
        except ValueError as error:
            raise SourceSecurityError(
                "snapshot DNS returned an invalid address", source_id=self.source.source_id
            ) from error
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if (
            address in _METADATA_IPS
            or not address.is_global
            or any(
                (
                    address.is_private,
                    address.is_loopback,
                    address.is_link_local,
                    address.is_multicast,
                    address.is_unspecified,
                    address.is_reserved,
                )
            )
        ):
            raise SourceSecurityError(
                "snapshot DNS resolved to a forbidden address", source_id=self.source.source_id
            )

    def _validated_payload(
        self, response: Any, received_at: datetime
    ) -> tuple[bytes, str, datetime]:
        status_code = int(response.status_code)
        if status_code < 200 or status_code >= 300:
            error = SourceFetchError(
                f"snapshot endpoint returned HTTP {status_code}",
                source_id=self.source.source_id,
            )
            if status_code < 500 and status_code not in {408, 429}:
                error.retryable = False
            raise error

        content_type = (_header(response.headers, "content-type") or "").split(";", 1)[0].lower()
        if content_type not in self._content_types:
            raise SourceValidationError(
                "snapshot content type is not allowlisted", source_id=self.source.source_id
            )
        content_length = _header(response.headers, "content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise SourceValidationError(
                    "snapshot Content-Length is invalid", source_id=self.source.source_id
                ) from error
            if declared_size > self._max_bytes:
                raise SourceOversizedError(
                    "snapshot exceeds the configured byte limit",
                    source_id=self.source.source_id,
                )

        captured_at = self._captured_at(response.headers, received_at)
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=min(65_536, self._max_bytes)):
            if not chunk:
                continue
            total += len(chunk)
            if total > self._max_bytes:
                raise SourceOversizedError(
                    "snapshot exceeds the configured byte limit",
                    source_id=self.source.source_id,
                )
            chunks.append(chunk)
        if total == 0:
            raise SourceValidationError(
                "snapshot response was empty", source_id=self.source.source_id
            )
        payload = b"".join(chunks)
        self._validate_dimensions(payload)
        return payload, content_type, captured_at

    def _captured_at(self, headers: Mapping[str, Any], received_at: datetime) -> datetime:
        captured_at: datetime | None = None
        for header_name in self._freshness_headers:
            value = _header(headers, header_name)
            if value:
                captured_at = _parse_timestamp(value)
                if captured_at is None:
                    raise SourceValidationError(
                        f"snapshot freshness header '{header_name}' is invalid",
                        source_id=self.source.source_id,
                    )
                break
        if captured_at is None:
            if self._require_freshness:
                raise SourceStaleError(
                    "snapshot response has no configured freshness header",
                    source_id=self.source.source_id,
                )
            return received_at
        age_seconds = (received_at - captured_at).total_seconds()
        if age_seconds > self._max_age_seconds:
            raise SourceStaleError(
                "snapshot is older than the configured freshness limit",
                source_id=self.source.source_id,
            )
        if age_seconds < -self._max_future_skew_seconds:
            raise SourceStaleError(
                "snapshot timestamp is implausibly far in the future",
                source_id=self.source.source_id,
            )
        return captured_at

    def _validate_dimensions(self, payload: bytes) -> None:
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                exceeds_dimensions = width > self._max_width or height > self._max_height
                if exceeds_dimensions or width * height > self._max_pixels:
                    raise SourceOversizedError(
                        "snapshot exceeds the configured dimension limit",
                        source_id=self.source.source_id,
                    )
                image.verify()
        except SourceOversizedError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
            raise SourceValidationError(
                "snapshot payload is not a valid image", source_id=self.source.source_id
            ) from error


def _required_string(options: Mapping[str, JSONValue], name: str) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value.strip():
        raise SourceConfigurationError(f"snapshot_http requires string option '{name}'")
    return value.strip()


def _string_list(
    options: Mapping[str, JSONValue], name: str, default: list[str] | None = None
) -> list[str]:
    value = options.get(name, default or [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SourceConfigurationError(f"snapshot_http option '{name}' must be a string list")
    return value


def _int_list(
    options: Mapping[str, JSONValue],
    name: str,
    default: list[int],
    *,
    minimum: int,
    maximum: int,
) -> list[int]:
    value = options.get(name, default)
    if not isinstance(value, list):
        raise SourceConfigurationError(f"snapshot_http option '{name}' must be an integer list")
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(
            f"snapshot_http option '{name}' must be an integer list"
        ) from error
    if not all(minimum <= item <= maximum for item in result):
        raise SourceConfigurationError(
            f"snapshot_http option '{name}' values must be between {minimum} and {maximum}"
        )
    return result


def _option_int(
    options: Mapping[str, JSONValue],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(options.get(name, default))  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(
            f"snapshot_http option '{name}' must be an integer"
        ) from error
    if not minimum <= value <= maximum:
        raise SourceConfigurationError(
            f"snapshot_http option '{name}' must be between {minimum} and {maximum}"
        )
    return value


def _option_float(
    options: Mapping[str, JSONValue],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(options.get(name, default))  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(f"snapshot_http option '{name}' must be numeric") from error
    if not minimum <= value <= maximum:
        raise SourceConfigurationError(
            f"snapshot_http option '{name}' must be between {minimum} and {maximum}"
        )
    return value


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return None


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _address_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result[4][0])
    except (IndexError, TypeError) as error:
        raise SourceSecurityError("snapshot DNS returned a malformed result") from error


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()
