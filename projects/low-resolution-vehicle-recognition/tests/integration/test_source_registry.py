from pathlib import Path

import pytest

from roadid.contracts import FramePacket
from roadid.sources import (
    BaseSourceAdapter,
    HealthStatus,
    SourceDisabledError,
    SourceRegistry,
    SourceUnavailableError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FailingSource(BaseSourceAdapter):
    def _capture_once(self, run_id: str) -> FramePacket:
        raise SourceUnavailableError("provider outage", source_id=self.source.source_id)


def test_example_registry_loads_camera_sources_without_exposing_options() -> None:
    registry = SourceRegistry.from_yaml(PROJECT_ROOT / "configs" / "camera_sources.example.yaml")

    source_ids = [source.source_id for source in registry.list_sources()]
    assert source_ids == ["replay-demo", "local-camera-0", "reviewed-snapshot", "tfl-jamcam"]
    assert registry.get_source("replay-demo").adapter_type == "replay"
    assert all("options" not in source for source in registry.public_sources())
    with pytest.raises(SourceDisabledError):
        registry.create_adapter("tfl-jamcam")


def test_failing_provider_does_not_break_replay(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
sources:
  - id: outage
    name: Unavailable provider
    type: failing
    enabled: true
    attribution: fixture
    refresh_seconds: 1
    options: {}
  - id: replay
    name: Replay
    type: replay
    enabled: true
    attribution: fixture
    refresh_seconds: 0.1
    options:
      frame_count: 30
      width: 96
      height: 64
""".lstrip(),
        encoding="utf-8",
    )
    registry = SourceRegistry.from_yaml(config, factories={"failing": FailingSource})

    with pytest.raises(SourceUnavailableError):
        registry.capture("outage", "run-a")

    replay = registry.capture("replay", "run-a")
    assert replay.frame_id == 0
    assert registry.health("outage").status is HealthStatus.UNAVAILABLE
    assert registry.health("replay").status is HealthStatus.HEALTHY
