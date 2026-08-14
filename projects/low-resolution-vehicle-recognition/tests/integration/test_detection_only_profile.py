from datetime import UTC, datetime

import numpy as np

from roadid.contracts import CameraSource, Decision, Detection, FramePacket
from roadid.inference.tracker import DeterministicTracker
from roadid.settings import Paths, Settings, WebSettings
from roadid.telemetry.recorder import EventRecorder
from roadid.web.dependencies import DetrDetectionPipelineFactory, RecorderHub


class _Detector:
    def detect(self, frame):
        return (Detection(frame.frame_id, (12, 12, 60, 44), "car", 0.94),)


def test_detection_only_factory_accepts_vehicle_and_abstains_make_model(tmp_path) -> None:
    source = CameraSource(
        source_id="tfl-live",
        name="TfL live",
        adapter_type="tfl_jamcam",
        enabled=True,
        attribution="Transport for London",
        refresh_seconds=10,
    )
    settings = Settings(
        web=WebSettings(),
        paths=Paths(tmp_path, tmp_path / "inference.yaml", tmp_path / "sources.yaml", None),
        inference={
            "runtime": {"profile": "detection-only", "max_evidence_per_track": 4},
            "detector": {"model_id": "facebook/detr-resnet-50"},
            "tracker": {"implementation": "deterministic"},
            "quality": {"minimum_height_px": 12, "maximum_blur": 0.95},
        },
    )
    recorder = RecorderHub(EventRecorder, capacity=100)
    recorder.ensure_run("run-live", source)
    factory = DetrDetectionPipelineFactory(
        settings,
        recorder,
        detector=_Detector(),
        tracker=DeterministicTracker(profile_label="test-real-source-tracker"),
    )
    image = np.indices((64, 96)).sum(axis=0) % 2 * 255
    image_bgr = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    now = datetime(2026, 8, 13, tzinfo=UTC)
    frame = FramePacket("run-live", source.source_id, 1, now, now, image_bgr)

    result = factory.create("run-live", source, {}).process_frame(frame)

    prediction = result.predictions[0]
    assert prediction.decision is Decision.ACCEPT_BODY_ONLY
    assert prediction.body_type.label == "vehicle"
    assert prediction.body_type.accepted
    assert not prediction.make.accepted
    assert not prediction.model_family.accepted
    assert not np.array_equal(result.display_frame_bgr, image_bgr)
    assert result.privacy.redactor_version.startswith("full-frame-pixelation-v1")
