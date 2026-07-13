"""Mask-aware feature standardization."""

from __future__ import annotations

import torch
from torch import Tensor


class MaskedStandardScaler:
    """Standardize the last tensor dimension using observed values only."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None

    def fit(self, values: Tensor, observed: Tensor) -> "MaskedStandardScaler":
        """Fit per-feature statistics across all leading dimensions."""
        if values.shape != observed.shape:
            raise ValueError("values and observed must have identical shapes")
        reduce_dims = tuple(range(values.ndim - 1))
        weights = observed.to(values.dtype)
        counts = weights.sum(dim=reduce_dims).clamp_min(1.0)
        self.mean = (values * weights).sum(dim=reduce_dims) / counts
        centered = (values - self.mean) * weights
        variance = centered.square().sum(dim=reduce_dims) / counts
        self.scale = variance.sqrt().clamp_min(self.eps)
        return self

    def transform(self, values: Tensor) -> Tensor:
        """Apply fitted standardization without changing tensor rank."""
        self._check_fitted()
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: Tensor) -> Tensor:
        """Return standardized values to the original feature scale."""
        self._check_fitted()
        return values * self.scale + self.mean

    def _check_fitted(self) -> None:
        if self.mean is None or self.scale is None:
            raise RuntimeError("MaskedStandardScaler must be fitted before use")
