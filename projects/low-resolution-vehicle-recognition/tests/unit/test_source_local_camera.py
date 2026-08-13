from roadid.contracts import CameraSource
from roadid.sources import LocalCameraSource, SourceUnavailableError


class MissingCamera:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        self.released = True


def test_local_camera_opens_lazily_and_reports_unavailable_device() -> None:
    created: list[MissingCamera] = []

    def factory(device_index: int) -> MissingCamera:
        assert device_index == 7
        camera = MissingCamera()
        created.append(camera)
        return camera

    source = CameraSource(
        source_id="camera-7",
        name="Local camera",
        adapter_type="local_camera",
        enabled=True,
        attribution="user",
        refresh_seconds=0.2,
        options={"device_index": 7},
    )
    adapter = LocalCameraSource(source, capture_factory=factory)
    assert created == []

    try:
        adapter.capture("run-a")
    except SourceUnavailableError as error:
        assert error.code == "source_unavailable"
    else:
        raise AssertionError("missing camera did not raise SourceUnavailableError")

    assert len(created) == 1
    assert created[0].released
