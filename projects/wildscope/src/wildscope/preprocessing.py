"""Full-frame normalization and conservative low-resolution enhancement."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class ImagePipelineResult:
    source_path: Path
    normalized_path: Path
    enhanced_path: Path
    model_input_path: Path
    source_size: tuple[int, int]
    model_input_size: tuple[int, int]
    enhancement_method: str
    enhancement_applied: bool


def prepare_image(
    source_path: Path,
    output_root: Path,
    *,
    minimum_width: int = 1280,
    minimum_height: int = 720,
) -> ImagePipelineResult:
    """Normalize a complete frame and enhance only when its resolution is low."""
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output_root / "normalized.jpg"
    enhanced_path = output_root / "enhanced.jpg"

    with Image.open(source_path) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        source_size = normalized.size
        normalized.save(normalized_path, format="JPEG", quality=95, optimize=True)

    needs_enhancement = (
        source_size[0] < minimum_width or source_size[1] < minimum_height
    )
    if needs_enhancement:
        rgb = np.asarray(normalized)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        restored_luminance = cv2.createCLAHE(
            clipLimit=1.8, tileGridSize=(8, 8)
        ).apply(luminance)
        restored_rgb = cv2.cvtColor(
            cv2.merge((restored_luminance, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )
        enhanced_rgb = cv2.resize(
            restored_rgb,
            (source_size[0] * 2, source_size[1] * 2),
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 5, 24, 24)
        Image.fromarray(enhanced_rgb).save(
            enhanced_path, format="JPEG", quality=95, optimize=True
        )
        method = "opencv-clahe-bicubic-x2"
        model_input_size = (source_size[0] * 2, source_size[1] * 2)
    else:
        shutil.copyfile(normalized_path, enhanced_path)
        method = "original-resolution-passthrough"
        model_input_size = source_size

    return ImagePipelineResult(
        source_path=source_path,
        normalized_path=normalized_path,
        enhanced_path=enhanced_path,
        model_input_path=enhanced_path,
        source_size=source_size,
        model_input_size=model_input_size,
        enhancement_method=method,
        enhancement_applied=needs_enhancement,
    )
