"""RoadID camera source adapters."""

from roadid.sources.base import (
    BaseSourceAdapter,
    HealthStatus,
    SourceAccessError,
    SourceAdapter,
    SourceConfigurationError,
    SourceDisabledError,
    SourceError,
    SourceExhaustedError,
    SourceFetchError,
    SourceHealth,
    SourceOversizedError,
    SourceSecurityError,
    SourceStaleError,
    SourceTermsError,
    SourceTimeoutError,
    SourceUnavailableError,
    SourceValidationError,
)
from roadid.sources.local_camera import LocalCameraSource
from roadid.sources.registry import SourceRegistry, load_camera_sources, load_source_registry
from roadid.sources.replay import ReplaySource
from roadid.sources.snapshot_http import SnapshotHttpSource
from roadid.sources.tfl_jamcam import TfLJamCamSource

__all__ = [
    "BaseSourceAdapter",
    "HealthStatus",
    "LocalCameraSource",
    "ReplaySource",
    "SnapshotHttpSource",
    "SourceAccessError",
    "SourceAdapter",
    "SourceConfigurationError",
    "SourceDisabledError",
    "SourceError",
    "SourceExhaustedError",
    "SourceFetchError",
    "SourceHealth",
    "SourceOversizedError",
    "SourceRegistry",
    "SourceSecurityError",
    "SourceStaleError",
    "SourceTermsError",
    "SourceTimeoutError",
    "SourceUnavailableError",
    "SourceValidationError",
    "TfLJamCamSource",
    "load_camera_sources",
    "load_source_registry",
]
