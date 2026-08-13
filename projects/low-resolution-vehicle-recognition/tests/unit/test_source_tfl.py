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
