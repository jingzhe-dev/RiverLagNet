"""Mask-aware forecasting losses."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def masked_huber_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Return mean Huber loss over observed target positions only."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    losses = functional.huber_loss(prediction, target, reduction="none", delta=delta)
    weights = mask.to(losses.dtype)
    count = weights.sum()
    if count.item() == 0:
        return prediction.sum() * 0.0
    return (losses * weights).sum() / count
