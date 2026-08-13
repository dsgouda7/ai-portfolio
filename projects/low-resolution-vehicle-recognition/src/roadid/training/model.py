"""Lazy hierarchical classifier builders and transfer-learning phase controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roadid.training.hierarchy import LEVELS, masked_hierarchical_loss


@dataclass(frozen=True, slots=True)
class GradientCheck:
    trainable_parameters: int
    finite_nonzero_gradients: int
    frozen_parameters_with_gradients: int


def build_tiny_classifier(
    class_counts: Mapping[str, int], *, feature_dim: int = 32, seed: int = 2608
) -> Any:
    """Build a small local CNN with the production three-head interface."""
    torch, neural = _torch_modules()
    torch.manual_seed(seed)

    class TinyBackbone(neural.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = neural.Sequential(
                neural.Conv2d(3, 8, 3, stride=2, padding=1), neural.ReLU()
            )
            self.stage2 = neural.Sequential(
                neural.Conv2d(8, 16, 3, stride=2, padding=1), neural.ReLU()
            )
            self.stage3 = neural.Sequential(
                neural.Conv2d(16, feature_dim, 3, stride=2, padding=1), neural.ReLU()
            )
            self.pool = neural.AdaptiveAvgPool2d((1, 1))

        def forward(self, pixel_values: Any) -> Any:
            features = self.stage1(pixel_values)
            features = self.stage2(features)
            features = self.stage3(features)
            return self.pool(features).flatten(1)

    return _wrap_backbone(TinyBackbone(), feature_dim, class_counts)


def build_hf_resnet50_classifier(
    class_counts: Mapping[str, int],
    *,
    model_id: str = "microsoft/resnet-50",
    revision: str = "main",
    offline_only: bool = True,
) -> Any:
    """Load HF ResNet only when explicitly called; imports never trigger a download."""
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face training requires the optional `ml` dependencies; "
            "install the project with `pip install -e .[ml]`."
        ) from error

    backbone = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=offline_only,
    )
    hidden_sizes = getattr(backbone.config, "hidden_sizes", None)
    feature_dim = (
        hidden_sizes[-1] if hidden_sizes else getattr(backbone.config, "hidden_size", None)
    )
    if not isinstance(feature_dim, int):
        raise ValueError("unable to determine Hugging Face backbone feature width")
    return _wrap_backbone(backbone, feature_dim, class_counts)


def configure_frozen_head(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def configure_partial_unfreeze(
    model: Any, *, backbone_prefixes: Sequence[str] = ("backbone.stage3",)
) -> tuple[str, ...]:
    configure_frozen_head(model)
    matched: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in backbone_prefixes):
            parameter.requires_grad = True
            matched.append(name)
    if not matched:
        raise ValueError(f"no backbone parameters matched prefixes: {tuple(backbone_prefixes)}")
    return tuple(matched)


def one_step_gradient_check(
    model: Any,
    pixel_values: Any,
    targets: Mapping[str, Any],
    *,
    selected_backbone_prefixes: Sequence[str] = (),
) -> dict[str, GradientCheck]:
    """Run one backward pass and assert the configured transfer boundary."""
    torch, _ = _torch_modules()
    model.zero_grad(set_to_none=True)
    logits = model(pixel_values)
    loss, _ = masked_hierarchical_loss(logits, targets)
    loss.backward()

    groups: dict[str, list[tuple[str, Any]]] = {"backbone": [], "heads": []}
    for name, parameter in model.named_parameters():
        group = "backbone" if name.startswith("backbone.") else "heads"
        groups[group].append((name, parameter))

    report: dict[str, GradientCheck] = {}
    for group, parameters in groups.items():
        report[group] = GradientCheck(
            trainable_parameters=sum(
                parameter.numel() for _, parameter in parameters if parameter.requires_grad
            ),
            finite_nonzero_gradients=sum(
                parameter.numel()
                for _, parameter in parameters
                if parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.count_nonzero(parameter.grad))
            ),
            frozen_parameters_with_gradients=sum(
                parameter.numel()
                for _, parameter in parameters
                if not parameter.requires_grad and parameter.grad is not None
            ),
        )

    if report["heads"].finite_nonzero_gradients == 0:
        raise AssertionError("hierarchical heads received no finite non-zero gradients")
    if report["backbone"].frozen_parameters_with_gradients:
        raise AssertionError("frozen backbone parameters received gradients")
    if selected_backbone_prefixes:
        selected = [
            parameter
            for name, parameter in model.named_parameters()
            if any(name.startswith(prefix) for prefix in selected_backbone_prefixes)
        ]
        if not selected or not any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in selected
        ):
            raise AssertionError("selected backbone stages received no finite non-zero gradients")
    elif report["backbone"].finite_nonzero_gradients:
        raise AssertionError("frozen-head phase produced backbone gradients")
    return report


def _wrap_backbone(backbone: Any, feature_dim: int, class_counts: Mapping[str, int]) -> Any:
    _, neural = _torch_modules()
    invalid = {
        level: class_counts.get(level, 0) for level in LEVELS if class_counts.get(level, 0) < 1
    }
    if invalid:
        raise ValueError(f"each hierarchy head requires classes: {invalid}")

    class HierarchicalClassifier(neural.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.heads = neural.ModuleDict(
                {level: neural.Linear(feature_dim, int(class_counts[level])) for level in LEVELS}
            )

        def forward(self, pixel_values: Any) -> dict[str, Any]:
            output = self.backbone(pixel_values)
            features = _pooled_features(output)
            return {level: head(features) for level, head in self.heads.items()}

    return HierarchicalClassifier()


def _pooled_features(output: Any) -> Any:
    if hasattr(output, "pooler_output"):
        features = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state
    else:
        features = output
    if features.ndim == 4:
        features = features.mean(dim=(-2, -1))
    elif features.ndim == 3:
        features = features.mean(dim=1)
    if features.ndim != 2:
        raise ValueError(f"backbone must produce batch features, got shape {tuple(features.shape)}")
    return features


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as neural
    except ImportError as error:
        raise RuntimeError("CarFace training requires PyTorch; install `.[ml]`.") from error
    return torch, neural
