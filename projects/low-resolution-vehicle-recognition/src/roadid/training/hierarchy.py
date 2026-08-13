"""Label hierarchy validation and missing-label-aware losses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from roadid.training.datasets import DatasetItem

LEVELS = ("body_type", "make", "model_family")


@dataclass(frozen=True, slots=True)
class LabelHierarchy:
    body_types: tuple[str, ...]
    makes: tuple[str, ...]
    model_families: tuple[str, ...]
    make_to_body: Mapping[str, str]
    model_to_make: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.body_types or not self.makes or not self.model_families:
            raise ValueError("all hierarchy levels require at least one label")
        if set(self.make_to_body) != set(self.makes):
            raise ValueError("every make must name exactly one body-type parent")
        if set(self.model_to_make) != set(self.model_families):
            raise ValueError("every model family must name exactly one make parent")
        if not set(self.make_to_body.values()) <= set(self.body_types):
            raise ValueError("make references an unknown body type")
        if not set(self.model_to_make.values()) <= set(self.makes):
            raise ValueError("model family references an unknown make")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def label_to_index(self) -> dict[str, dict[str, int]]:
        return {
            "body_type": {label: index for index, label in enumerate(self.body_types)},
            "make": {label: index for index, label in enumerate(self.makes)},
            "model_family": {label: index for index, label in enumerate(self.model_families)},
        }


def build_hierarchy(items: Iterable[DatasetItem]) -> LabelHierarchy:
    body_types: set[str] = set()
    makes: set[str] = set()
    models: set[str] = set()
    make_to_body: dict[str, str] = {}
    model_to_make: dict[str, str] = {}
    for item in items:
        if item.body_type:
            body_types.add(item.body_type)
        if item.make:
            if not item.body_type:
                raise ValueError(f"make has no body-type parent: {item.item_id}")
            makes.add(item.make)
            _set_parent(make_to_body, item.make, item.body_type)
        if item.model_family:
            if not item.make:
                raise ValueError(f"model family has no make parent: {item.item_id}")
            models.add(item.model_family)
            _set_parent(model_to_make, item.model_family, item.make)
    return LabelHierarchy(
        body_types=tuple(sorted(body_types)),
        makes=tuple(sorted(makes)),
        model_families=tuple(sorted(models)),
        make_to_body=dict(sorted(make_to_body.items())),
        model_to_make=dict(sorted(model_to_make.items())),
    )


def masked_hierarchical_loss(
    logits: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    level_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute cross entropy only where the target index is non-negative."""
    import torch.nn.functional as functional

    weights = level_weights or {level: 1.0 for level in LEVELS}
    missing = set(LEVELS) - set(logits) | (set(LEVELS) - set(targets))
    if missing:
        raise ValueError(f"missing hierarchy tensors: {sorted(missing)}")

    losses: dict[str, Any] = {}
    total = logits["body_type"].sum() * 0.0
    for level in LEVELS:
        valid = targets[level] >= 0
        if valid.any():
            level_loss = functional.cross_entropy(logits[level][valid], targets[level][valid])
        else:
            level_loss = logits[level].sum() * 0.0
        losses[level] = level_loss
        total = total + float(weights.get(level, 1.0)) * level_loss
    return total, losses


def _set_parent(mapping: dict[str, str], child: str, parent: str) -> None:
    previous = mapping.setdefault(child, parent)
    if previous != parent:
        raise ValueError(f"label {child!r} has conflicting parents: {previous!r}, {parent!r}")
