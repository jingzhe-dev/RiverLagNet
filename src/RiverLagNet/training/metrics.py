"""Shared mask-aware evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from RiverLagNet.data.schema import TARGET_NAMES


def masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute error over observed positions."""
    return _masked_mean((prediction - target).abs(), mask)


def masked_rmse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Root mean squared error over observed positions."""
    return _masked_mean((prediction - target).square(), mask).sqrt()


def masked_nse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return one Nash-Sutcliffe efficiency value per target channel."""
    _validate_shapes(prediction, target, mask)
    reduce_dims = tuple(range(target.ndim - 1))
    weights = mask.to(target.dtype)
    counts = weights.sum(dim=reduce_dims)
    safe_counts = counts.clamp_min(1.0)
    means = (target * weights).sum(dim=reduce_dims) / safe_counts
    centered = (target - means) * weights
    denominator = centered.square().sum(dim=reduce_dims)
    numerator = ((prediction - target).square() * weights).sum(dim=reduce_dims)
    score = 1.0 - numerator / denominator.clamp_min(torch.finfo(target.dtype).eps)
    valid = (counts > 0) & (denominator > torch.finfo(target.dtype).eps)
    return torch.where(valid, score, torch.full_like(score, torch.nan))


def masked_metric_dict(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, Tensor]:
    """Compute scalar macro metrics and target-specific NSE values."""
    nse = masked_nse(prediction, target, mask)
    mae_by_target = _masked_channel_mean((prediction - target).abs(), mask)
    rmse_by_target = _masked_channel_mean(
        (prediction - target).square(), mask
    ).sqrt()
    valid = torch.isfinite(nse)
    macro_nse = nse[valid].mean() if valid.any() else prediction.sum() * 0.0
    metrics: dict[str, Tensor] = {
        "macro_nse": macro_nse,
        "macro_mae": masked_mae(prediction, target, mask),
        "macro_rmse": masked_rmse(prediction, target, mask),
    }
    metrics.update({f"nse_{name}": nse[index] for index, name in enumerate(TARGET_NAMES)})
    metrics.update(
        {f"mae_{name}": mae_by_target[index] for index, name in enumerate(TARGET_NAMES)}
    )
    metrics.update(
        {f"rmse_{name}": rmse_by_target[index] for index, name in enumerate(TARGET_NAMES)}
    )
    return metrics


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    weights = mask.to(values.dtype)
    count = weights.sum()
    if count.item() == 0:
        return values.sum() * 0.0
    return (values * weights).sum() / count


def _masked_channel_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    reduce_dims = tuple(range(values.ndim - 1))
    weights = mask.to(values.dtype)
    counts = weights.sum(dim=reduce_dims)
    totals = (values * weights).sum(dim=reduce_dims)
    means = totals / counts.clamp_min(1.0)
    return torch.where(counts > 0, means, torch.full_like(means, torch.nan))


def _validate_shapes(prediction: Tensor, target: Tensor, mask: Tensor) -> None:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
