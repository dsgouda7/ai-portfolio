import sys

import torch

from roadid.training.model import (
    build_tiny_classifier,
    configure_frozen_head,
    configure_partial_unfreeze,
    one_step_gradient_check,
)

CLASS_COUNTS = {"body_type": 2, "make": 3, "model_family": 4}


def targets() -> dict[str, torch.Tensor]:
    return {
        "body_type": torch.tensor([0, 1]),
        "make": torch.tensor([1, 2]),
        "model_family": torch.tensor([2, -1]),
    }


def test_training_model_module_has_no_hf_import_side_effect() -> None:
    assert "transformers" not in sys.modules


def test_frozen_head_one_step_has_no_backbone_gradients() -> None:
    model = build_tiny_classifier(CLASS_COUNTS, seed=7)
    configure_frozen_head(model)
    report = one_step_gradient_check(model, torch.rand(2, 3, 32, 32), targets())

    assert report["backbone"].trainable_parameters == 0
    assert report["backbone"].finite_nonzero_gradients == 0
    assert report["heads"].finite_nonzero_gradients > 0


def test_partial_unfreeze_updates_only_selected_late_stage() -> None:
    model = build_tiny_classifier(CLASS_COUNTS, seed=7)
    selected = ("backbone.stage3",)
    matched = configure_partial_unfreeze(model, backbone_prefixes=selected)
    report = one_step_gradient_check(
        model,
        torch.rand(2, 3, 32, 32),
        targets(),
        selected_backbone_prefixes=selected,
    )

    assert matched
    assert report["backbone"].finite_nonzero_gradients > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and not name.startswith(selected)
    )
