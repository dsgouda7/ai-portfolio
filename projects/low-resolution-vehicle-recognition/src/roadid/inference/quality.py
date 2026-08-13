"""Bounded crop-quality scoring for resolution, blur, exposure, and occlusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadid.contracts import CropQuality


@dataclass(frozen=True, slots=True)
class QualityConfig:
    minimum_height_px: int = 12
    maximum_blur: float = 0.8
    maximum_exposure: float = 0.85
    maximum_occlusion: float = 0.75
    sharp_variance: float = 400.0
    resolution_weight: float = 0.35
    sharpness_weight: float = 0.30
    exposure_weight: float = 0.20
    visibility_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.minimum_height_px <= 0 or self.sharp_variance <= 0:
            raise ValueError("quality size and variance references must be positive")
        bounded = (self.maximum_blur, self.maximum_exposure, self.maximum_occlusion)
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("quality thresholds must be in [0, 1]")
        weights = (
            self.resolution_weight,
            self.sharpness_weight,
            self.exposure_weight,
            self.visibility_weight,
        )
        if any(value < 0 for value in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("quality fusion weights must be non-negative and sum to one")


class CropQualityScorer:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def score(
        self,
        crop_bgr: np.ndarray,
        *,
        bbox_xyxy: tuple[int, int, int, int] | None = None,
        frame_shape: tuple[int, ...] | None = None,
        occlusion_score: float | None = None,
    ) -> CropQuality:
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
            raise ValueError("crop_bgr must be a non-empty BGR image")
        apparent_height = int(crop_bgr.shape[0])
        gray = np.tensordot(crop_bgr.astype(float), np.asarray([0.114, 0.587, 0.299]), axes=1)
        laplacian_variance = _laplacian_variance(gray)
        blur = _bounded(1.0 - laplacian_variance / self.config.sharp_variance)
        dark_fraction = float(np.mean(gray <= 12))
        bright_fraction = float(np.mean(gray >= 243))
        exposure = _bounded(max(dark_fraction, bright_fraction))
        occlusion = (
            _bounded(occlusion_score)
            if occlusion_score is not None
            else _edge_occlusion(bbox_xyxy, frame_shape)
        )

        reasons = []
        if apparent_height < self.config.minimum_height_px:
            reasons.append("insufficient_height")
        if blur > self.config.maximum_blur:
            reasons.append("blur")
        if exposure > self.config.maximum_exposure:
            reasons.append("exposure")
        if occlusion > self.config.maximum_occlusion:
            reasons.append("occlusion")
        usable = not reasons
        resolution = min(1.0, apparent_height / (2.0 * self.config.minimum_height_px))
        weight = (
            self.config.resolution_weight * resolution
            + self.config.sharpness_weight * (1.0 - blur)
            + self.config.exposure_weight * (1.0 - exposure)
            + self.config.visibility_weight * (1.0 - occlusion)
        )
        return CropQuality(
            apparent_height_px=apparent_height,
            blur_score=blur,
            exposure_score=exposure,
            occlusion_score=occlusion,
            usable=usable,
            rejection_reasons=tuple(reasons),
            fusion_weight=_bounded(weight) if usable else 0.0,
        )

    def score_detection(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        occlusion_score: float | None = None,
    ) -> tuple[np.ndarray, CropQuality]:
        crop = crop_original(image_bgr, bbox_xyxy)
        quality = self.score(
            crop,
            bbox_xyxy=bbox_xyxy,
            frame_shape=image_bgr.shape,
            occlusion_score=occlusion_score,
        )
        return crop, quality


def crop_original(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box has no area inside the original frame")
    return image_bgr[y1:y2, x1:x2].copy()


def _edge_occlusion(
    bbox: tuple[int, int, int, int] | None, frame_shape: tuple[int, ...] | None
) -> float:
    if bbox is None or frame_shape is None:
        return 0.0
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    touched = sum((x1 <= 0, y1 <= 0, x2 >= width, y2 >= height))
    return touched / 4.0


def _bounded(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("quality scores must be finite")
    return float(max(0.0, min(1.0, value)))


def _laplacian_variance(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(laplacian.var())
