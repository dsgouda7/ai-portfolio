import hashlib
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from roadid.training.datasets import DatasetItem
from roadid.training.degradations import degrade_image, sample_recipe
from roadid.training.hierarchy import build_hierarchy, masked_hierarchical_loss


def source_item(
    item_id: str,
    body: str | None,
    make: str | None,
    model: str | None,
) -> DatasetItem:
    return DatasetItem(
        item_id=item_id,
        image_path=f"{item_id}.png",
        source_id="fixture",
        source_version="1",
        source_terms_url="https://example.test/terms",
        source_license="fixture-only",
        source_sha256=hashlib.sha256(item_id.encode()).hexdigest(),
        body_type=body,
        make=make,
        model_family=model,
    )


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue()


def test_same_seed_reproduces_recipe_and_pixels() -> None:
    array = np.arange(96 * 64 * 3, dtype=np.uint8).reshape(64, 96, 3)
    image = Image.fromarray(array)
    first_recipe = sample_recipe(2608, apparent_height_range=(12, 18))
    second_recipe = sample_recipe(2608, apparent_height_range=(12, 18))
    first = degrade_image(image, first_recipe, output_size=64)
    second = degrade_image(image, second_recipe, output_size=64)

    assert first_recipe == second_recipe
    assert png_bytes(first) == png_bytes(second)
    assert first_recipe.apparent_height_px <= 18


def test_hierarchy_rejects_conflicting_parents() -> None:
    with pytest.raises(ValueError, match="conflicting parents"):
        build_hierarchy(
            [
                source_item("one", "suv", "toyota", "rav4"),
                source_item("two", "sedan", "toyota", "camry"),
            ]
        )


def test_missing_model_label_contributes_no_model_loss_or_gradient() -> None:
    logits = {
        "body_type": torch.tensor([[1.0, -1.0]], requires_grad=True),
        "make": torch.tensor([[0.5, -0.5]], requires_grad=True),
        "model_family": torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True),
    }
    targets = {
        "body_type": torch.tensor([0]),
        "make": torch.tensor([0]),
        "model_family": torch.tensor([-1]),
    }
    total, levels = masked_hierarchical_loss(logits, targets)
    total.backward()

    assert levels["model_family"].item() == 0.0
    assert torch.count_nonzero(logits["model_family"].grad).item() == 0
    assert torch.count_nonzero(logits["body_type"].grad).item() > 0
