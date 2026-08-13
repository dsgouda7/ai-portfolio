from types import SimpleNamespace

import numpy as np

from roadid.contracts import Detection
from roadid.inference.detector import (
    DeterministicVehicleDetector,
    DetrConfig,
    HuggingFaceDetrDetector,
)
from roadid.inference.quality import CropQualityScorer, QualityConfig
from roadid.inference.tracker import DeterministicTracker


class _ProcessorDouble:
    def __call__(self, *, images: np.ndarray, return_tensors: str) -> dict[str, np.ndarray]:
        assert images.shape == (60, 100, 3)
        return {"pixel_values": np.zeros((1, 3, 20, 20), dtype=np.float32)}

    def post_process_object_detection(self, outputs, *, threshold, target_sizes):
        assert target_sizes.tolist() == [[60, 100]]
        return [
            {
                "scores": np.asarray([0.95, 0.99]),
                "labels": np.asarray([3, 1]),
                "boxes": np.asarray([[10.2, 5.1, 80.7, 50.3], [0, 0, 10, 10]]),
            }
        ]


class _ModelDouble:
    config = SimpleNamespace(id2label={1: "person", 3: "car"})

    def __call__(self, **inputs):
        return SimpleNamespace()


def test_detr_filters_vehicles_and_returns_original_coordinates() -> None:
    detector = HuggingFaceDetrDetector(
        DetrConfig(score_threshold=0.7),
        processor=_ProcessorDouble(),
        model=_ModelDouble(),
    )
    image = np.zeros((60, 100, 3), dtype=np.uint8)

    detections = detector.detect(image, frame_id=7)

    assert detections == (Detection(7, (10, 5, 81, 51), "car", 0.95),)


def test_synthetic_detector_and_tracker_accumulate_then_expire() -> None:
    detector = DeterministicVehicleDetector(minimum_area=20)
    tracker = DeterministicTracker(lost_track_buffer=1, maximum_centroid_distance=20)
    frame_a = np.zeros((50, 80, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_a[10:25, 15:35, 2] = 255
    frame_b[10:25, 18:38, 2] = 255

    first = tracker.update(detector.detect(frame_a, frame_id=1))
    second = tracker.update(detector.detect(frame_b, frame_id=2))
    tracker.update((), frame_id=4)

    assert first[0].track_id == second[0].track_id
    assert tracker.active_track_ids == ()
    assert "test-demo-only" in tracker.profile_label


def test_quality_scores_are_bounded_and_low_quality_is_excluded() -> None:
    scorer = CropQualityScorer(QualityConfig(minimum_height_px=12, maximum_blur=0.6))
    sharp = np.indices((24, 24)).sum(axis=0) % 2 * 255
    sharp_bgr = np.repeat(sharp[:, :, None], 3, axis=2).astype(np.uint8)
    blurred = np.full((8, 24, 3), 127, dtype=np.uint8)

    good = scorer.score(sharp_bgr)
    bad = scorer.score(blurred, occlusion_score=0.9)

    assert good.usable and 0.0 < good.fusion_weight <= 1.0
    assert not bad.usable and bad.fusion_weight == 0.0
    assert {"insufficient_height", "blur", "occlusion"} <= set(bad.rejection_reasons)
    bounded_scores = (bad.blur_score, bad.exposure_score, bad.occlusion_score)
    assert all(0.0 <= value <= 1.0 for value in bounded_scores)
