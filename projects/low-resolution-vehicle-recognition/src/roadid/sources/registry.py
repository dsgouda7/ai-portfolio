"""Server-owned camera source registry and lazy adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from roadid.contracts import CameraSource, JSONValue
from roadid.sources.base import (
    HealthStatus,
    SourceAdapter,
    SourceConfigurationError,
    SourceDisabledError,
    SourceHealth,
)
from roadid.sources.local_camera import LocalCameraSource
from roadid.sources.replay import ReplaySource
from roadid.sources.snapshot_http import SnapshotHttpSource
from roadid.sources.tfl_jamcam import TfLJamCamSource

AdapterFactory = Callable[[CameraSource], SourceAdapter]


class SourceRegistry:
    def __init__(
        self,
        sources: Iterable[CameraSource],
        *,
        factories: Mapping[str, AdapterFactory] | None = None,
    ) -> None:
        self._sources: dict[str, CameraSource] = {}
        for source in sources:
            if source.source_id in self._sources:
                raise SourceConfigurationError(f"duplicate source id: {source.source_id}")
            self._sources[source.source_id] = source
        self._factories: dict[str, AdapterFactory] = {
            "local_camera": LocalCameraSource,
            "replay": ReplaySource,
            "snapshot_http": SnapshotHttpSource,
            "tfl_jamcam": TfLJamCamSource,
        }
        if factories:
            self._factories.update(factories)
        unknown_types = sorted(
            {source.adapter_type for source in self._sources.values()} - self._factories.keys()
        )
        if unknown_types:
            raise SourceConfigurationError(
                f"unsupported source adapter type(s): {', '.join(unknown_types)}"
            )
        self._adapters: dict[str, SourceAdapter] = {}

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        factories: Mapping[str, AdapterFactory] | None = None,
    ) -> SourceRegistry:
        return cls(load_camera_sources(path), factories=factories)

    def list_sources(self, *, enabled_only: bool = False) -> tuple[CameraSource, ...]:
        values = self._sources.values()
        if enabled_only:
            values = (source for source in values if source.enabled)
        return tuple(values)

    def public_sources(self, *, enabled_only: bool = False) -> list[dict[str, JSONValue]]:
        return [source.public_dict() for source in self.list_sources(enabled_only=enabled_only)]

    def get_source(self, source_id: str) -> CameraSource:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise SourceConfigurationError(f"unknown source id: {source_id}") from error

    def create_adapter(self, source_id: str) -> SourceAdapter:
        source = self.get_source(source_id)
        if not source.enabled:
            raise SourceDisabledError(
                f"source '{source_id}' is disabled", source_id=source.source_id
            )
        if source_id not in self._adapters:
            self._adapters[source_id] = self._factories[source.adapter_type](source)
        return self._adapters[source_id]

    def get_adapter(self, source_id: str) -> SourceAdapter:
        return self.create_adapter(source_id)

    def capture(self, source_id: str, run_id: str):
        return self.create_adapter(source_id).capture(run_id)

    def health(self, source_id: str) -> SourceHealth:
        adapter = self._adapters.get(source_id)
        if adapter is None:
            self.get_source(source_id)
            return SourceHealth(status=HealthStatus.UNKNOWN)
        return adapter.health

    def close(self) -> None:
        for adapter in self._adapters.values():
            adapter.close()
        self._adapters.clear()


def load_camera_sources(path: str | Path) -> tuple[CameraSource, ...]:
    config_path = Path(path)
    if not config_path.exists():
        raise SourceConfigurationError(f"source configuration file not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise SourceConfigurationError(
            f"source configuration is invalid YAML: {config_path}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise SourceConfigurationError("source configuration must contain a 'sources' list")
    return tuple(_camera_source(item, index) for index, item in enumerate(payload["sources"]))


def load_source_registry(path: str | Path) -> SourceRegistry:
    return SourceRegistry.from_yaml(path)


def _camera_source(value: Any, index: int) -> CameraSource:
    if not isinstance(value, dict):
        raise SourceConfigurationError(f"source entry {index} must be a mapping")
    required = ("id", "name", "type", "enabled", "attribution", "refresh_seconds")
    missing = [name for name in required if name not in value]
    if missing:
        raise SourceConfigurationError(f"source entry {index} is missing: {', '.join(missing)}")
    options = value.get("options", {})
    if not isinstance(options, dict) or not all(isinstance(key, str) for key in options):
        raise SourceConfigurationError(f"source entry {index} options must be a mapping")
    try:
        return CameraSource(
            source_id=str(value["id"]),
            name=str(value["name"]),
            adapter_type=str(value["type"]),
            enabled=_strict_bool(value["enabled"], index),
            attribution=str(value["attribution"]),
            terms_url=_optional_string(value.get("terms_url"), "terms_url", index),
            refresh_seconds=float(value["refresh_seconds"]),
            location_label=_optional_string(value.get("location_label"), "location_label", index),
            location_precision=str(value.get("location_precision", "none")),
            options=options,
        )
    except (TypeError, ValueError) as error:
        raise SourceConfigurationError(f"source entry {index} is invalid: {error}") from error


def _strict_bool(value: Any, index: int) -> bool:
    if not isinstance(value, bool):
        raise SourceConfigurationError(f"source entry {index} enabled must be a boolean")
    return value


def _optional_string(value: Any, name: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceConfigurationError(f"source entry {index} {name} must be a string")
    return value
