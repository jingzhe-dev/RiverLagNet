"""Mask-aware input feature encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class InputMaskEncoder(nn.Module):
    """Encode values together with observation, quality, static, and time features."""

    def __init__(self, value_dim: int, static_dim: int, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = value_dim * 3 + static_dim + time_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_quality: Tensor | None,
        static: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        """Return encoded features with shape `[B, T, N, D]`."""
        if x.shape != x_mask.shape:
            raise ValueError("x and x_mask must have identical shapes")
        batch, history, nodes, _ = x.shape
        quality = torch.zeros_like(x) if x_quality is None else x_quality
        if quality.shape != x.shape:
            raise ValueError("x_quality must have the same shape as x")
        if static.ndim == 2:
            static_features = static[None, None].expand(batch, history, -1, -1)
        elif static.ndim == 3:
            static_features = static[:, None].expand(-1, history, -1, -1)
        else:
            raise ValueError("static must have shape [N,S] or [B,N,S]")
        if static_features.shape[2] != nodes:
            raise ValueError("static node dimension must match x")
        if time_features.ndim == 3:
            temporal = time_features[:, :, None].expand(-1, -1, nodes, -1)
        elif time_features.ndim == 4:
            temporal = time_features
        else:
            raise ValueError("time_features must have shape [B,T,F] or [B,T,N,F]")
        features = torch.cat(
            (x * x_mask, x_mask.to(x.dtype), quality, static_features, temporal), dim=-1
        )
        return self.projection(features)
