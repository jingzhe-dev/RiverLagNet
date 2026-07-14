"""Mask-aware forecasting losses."""

from __future__ import annotations

import torch
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


def masked_nse_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return target-balanced `1 - NSE` over observed positions.

    This is a selection-alignment auxiliary term; masked Huber remains the
    primary training objective.
    """
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    reduce_dims = tuple(range(target.ndim - 1))
    weights = mask.to(target.dtype)
    counts = weights.sum(dim=reduce_dims)
    means = (target * weights).sum(dim=reduce_dims) / counts.clamp_min(1.0)
    denominator = ((target - means) * weights).square().sum(dim=reduce_dims)
    numerator = ((prediction - target).square() * weights).sum(dim=reduce_dims)
    epsilon = torch.finfo(target.dtype).eps
    valid = (counts > 0) & (denominator > epsilon)
    ratios = numerator / denominator.clamp_min(epsilon)
    if not valid.any():
        return prediction.sum() * 0.0
    return ratios[valid].mean()
