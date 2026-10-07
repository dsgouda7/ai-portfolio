from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_norm: float | None = None,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_norm if clip_norm is not None else float("inf"),
        )
    )
    post_clip_norm = min(pre_clip_norm, clip_norm) if clip_norm is not None else pre_clip_norm
    optimizer.step()
    return pre_clip_norm, post_clip_norm
