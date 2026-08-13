"""Deterministic low-resolution curriculum transforms with replayable recipes."""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True, slots=True)
class DegradationRecipe:
    schema_version: int
    seed: int
    apparent_height_px: int
    horizontal_shear: float
    subpixel_shift_x: float
    blur_radius: float
    jpeg_quality: int
    noise_sigma: float
    brightness: float
    contrast: float
    occlusion_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sample_recipe(
    seed: int, *, apparent_height_range: tuple[int, int] = (12, 64)
) -> DegradationRecipe:
    minimum, maximum = apparent_height_range
    if minimum < 4 or maximum < minimum:
        raise ValueError("apparent height range must be ordered and at least four pixels")
    random = np.random.default_rng(seed)
    return DegradationRecipe(
        schema_version=1,
        seed=seed,
        apparent_height_px=int(random.integers(minimum, maximum + 1)),
        horizontal_shear=float(random.uniform(-0.16, 0.16)),
        subpixel_shift_x=float(random.uniform(-0.75, 0.75)),
        blur_radius=float(random.uniform(0.4, 2.4)),
        jpeg_quality=int(random.integers(18, 56)),
        noise_sigma=float(random.uniform(2.0, 13.0)),
        brightness=float(random.uniform(0.55, 1.2)),
        contrast=float(random.uniform(0.65, 1.25)),
        occlusion_fraction=float(random.uniform(0.0, 0.22)),
    )


def degrade_image(
    image: Image.Image,
    recipe: DegradationRecipe,
    *,
    output_size: int = 224,
) -> Image.Image:
    if output_size < 32:
        raise ValueError("output_size must be at least 32")
    source = ImageOps.exif_transpose(image).convert("RGB")
    width = max(4, round(source.width * recipe.apparent_height_px / source.height))
    tiny = source.resize((width, recipe.apparent_height_px), Image.Resampling.LANCZOS)
    shear_pixels = abs(recipe.horizontal_shear) * recipe.apparent_height_px
    warped_width = max(4, round(width + shear_pixels))
    tiny = tiny.transform(
        (warped_width, recipe.apparent_height_px),
        Image.Transform.AFFINE,
        (1.0, recipe.horizontal_shear, -recipe.subpixel_shift_x, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
    )
    tiny = tiny.filter(ImageFilter.GaussianBlur(recipe.blur_radius))
    tiny = ImageEnhance.Brightness(tiny).enhance(recipe.brightness)
    tiny = ImageEnhance.Contrast(tiny).enhance(recipe.contrast)

    encoded = io.BytesIO()
    tiny.save(encoded, format="JPEG", quality=recipe.jpeg_quality, optimize=False)
    encoded.seek(0)
    compressed = Image.open(encoded).convert("RGB")

    pixels = np.asarray(compressed, dtype=np.float32)
    random = np.random.default_rng(recipe.seed ^ 0xA51CE)
    pixels += random.normal(0.0, recipe.noise_sigma, pixels.shape)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    degraded = Image.fromarray(pixels, mode="RGB")

    if recipe.occlusion_fraction > 0:
        occlusion_width = max(1, round(degraded.width * recipe.occlusion_fraction))
        array = np.asarray(degraded).copy()
        start = int(random.integers(0, max(1, degraded.width - occlusion_width + 1)))
        array[:, start : start + occlusion_width] = np.median(array, axis=(0, 1)).astype(np.uint8)
        degraded = Image.fromarray(array, mode="RGB")

    canvas = Image.new("RGB", (output_size, output_size), (114, 114, 114))
    left = (output_size - degraded.width) // 2
    top = (output_size - degraded.height) // 2
    canvas.paste(degraded, (left, top))
    return canvas


def degrade_file(
    source: Path,
    destination: Path,
    *,
    seed: int,
    apparent_height_range: tuple[int, int] = (12, 64),
    output_size: int = 224,
) -> tuple[DegradationRecipe, str]:
    recipe = sample_recipe(seed, apparent_height_range=apparent_height_range)
    with Image.open(source) as image:
        degraded = degrade_image(image, recipe, output_size=output_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    degraded.save(destination, format="PNG", compress_level=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return recipe, digest
