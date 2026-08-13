from datetime import UTC, datetime, timedelta

import numpy as np

from roadid.contracts import CalibrationContract, FramePacket, StageStatus
from roadid.inference.calibration import HierarchicalDecisionEngine, TemperatureCalibrator
from roadid.inference.classifier import DeterministicDemoClassifier
from roadid.inference.detector import DeterministicVehicleDetector
from roadid.inference.fusion import TrackFuser
from roadid.inference.pipeline import SynchronousInferencePipeline
from roadid.inference.privacy import DeterministicNoPIIRedactor, PrivacyGuard
from roadid.inference.quality import CropQualityScorer, QualityConfig
from roadid.inference.tracker import DeterministicTracker


def _frames() -> tuple[FramePacket, ...]:
    started = datetime(2026, 8, 13, tzinfo=UTC)
    frames = []
    checker = (np.indices((18, 28)).sum(axis=0) % 2).astype(bool)
    for frame_id, x in enumerate((10, 13, 16), start=1):
        image = np.zeros((50, 90, 3), dtype=np.uint8)
        crop = image[12:30, x : x + 28]
        crop[checker, 2] = 255
        frames.append(
            FramePacket(
                run_id="run-replay",
                source_id="replay-demo",
                frame_id=frame_id,
                captured_at=started + timedelta(milliseconds=100 * frame_id),
                received_at=started + timedelta(milliseconds=100 * frame_id),
                image_bgr=image,
            )
        )
    return tuple(frames)


def test_replay_pipeline_accumulates_observations_and_emits_checkpoints() -> None:
    classifier = DeterministicDemoClassifier()
    contract = CalibrationContract(
        method="temperature_scaling",
        dataset_manifest_sha256="1" * 64,
        validation_split_sha256="2" * 64,
        test_split_sha256="3" * 64,
        body_threshold=0.55,
        make_threshold=0.55,
        model_threshold=0.55,
    )
    checkpoints = []
    pipeline = SynchronousInferencePipeline(
        detector=DeterministicVehicleDetector(minimum_area=40),
        tracker=DeterministicTracker(lost_track_buffer=2, maximum_centroid_distance=20),
        quality_scorer=CropQualityScorer(QualityConfig(minimum_height_px=12, maximum_blur=0.95)),
        classifier=classifier,
        fuser=TrackFuser(classifier.label_space, max_evidence_per_track=8),
        calibrator=TemperatureCalibrator(
            contract,
            dataset_manifest_sha256="1" * 64,
            validation_split_sha256="2" * 64,
        ),
        decision_engine=HierarchicalDecisionEngine(
            label_space=classifier.label_space,
            calibration=contract,
            model_version="deterministic-demo",
            model_to_make={"blue-line": "blue-motors", "red-line": "red-motors"},
        ),
        privacy_guard=PrivacyGuard(DeterministicNoPIIRedactor()),
        checkpoint_callback=checkpoints.append,
    )

    results = pipeline.run(_frames(), public_source=False)

    track_ids = [result.observations[0].track_id for result in results]
    assert len(set(track_ids)) == 1
    assert results[-1].predictions[0].usable_frames == 3
    assert all(result.privacy.safe_for_display for result in results)
    assert any(
        item.stage == "track_fusion" and item.status is StageStatus.COMPLETED
        for item in checkpoints
    )
    assert all(not hasattr(item, "sequence_id") for item in checkpoints)
    assert "ABC123" not in repr(results) + repr(checkpoints)
