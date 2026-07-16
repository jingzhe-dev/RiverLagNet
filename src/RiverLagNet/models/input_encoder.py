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
        observed_values = torch.where(x_mask, x, torch.zeros_like(x))
        observed_quality = torch.where(x_mask, quality, torch.zeros_like(quality))
        features = torch.cat(
            (
                observed_values,
                x_mask.to(x.dtype),
                observed_quality,
                static_features,
                temporal,
            ),
            dim=-1,
        )
        return self.projection(features)


class TargetExogenousInputEncoder(nn.Module):
    """Encode target history separately from named exogenous context."""

    def __init__(
        self,
        value_dim: int,
        static_dim: int,
        time_dim: int,
        hidden_dim: int,
        target_dim: int = 3,
    ) -> None:
        super().__init__()
        if target_dim <= 0 or value_dim < target_dim:
            raise ValueError("value_dim must contain every target channel")
        self.value_dim = value_dim
        self.target_dim = target_dim
        self.exogenous_dim = value_dim - target_dim
        self.static_dim = static_dim
        self.time_dim = time_dim
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        external_input_dim = self.exogenous_dim * 3 + static_dim + time_dim
        self.external_encoder: nn.Module
        if external_input_dim > 0:
            self.external_encoder = nn.Sequential(
                nn.Linear(external_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        else:
            self.external_encoder = nn.Identity()
        self.no_exogenous_token = (
            nn.Parameter(torch.zeros(hidden_dim)) if self.exogenous_dim == 0 else None
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        """Return dual-stream encodings with shape ``[B,T,N,D]``."""
        if x.ndim != 4 or x.shape[-1] != self.value_dim:
            raise ValueError(f"x must have shape [B,T,N,{self.value_dim}]")
        if x_mask.shape != x.shape:
            raise ValueError("x and x_mask must have identical shapes")
        quality = torch.zeros_like(x) if x_quality is None else x_quality
        if quality.shape != x.shape:
            raise ValueError("x_quality must have the same shape as x")

        values = torch.where(x_mask, x, torch.zeros_like(x))
        quality = torch.where(x_mask, quality, torch.zeros_like(quality))
        target_slice = slice(0, self.target_dim)
        target_features = torch.cat(
            (
                values[..., target_slice],
                x_mask[..., target_slice].to(x.dtype),
                quality[..., target_slice],
            ),
            dim=-1,
        )
        target_state = self.target_encoder(target_features)

        static_features, temporal = _expand_static_and_time(
            x,
            static,
            time_features,
            expected_static_dim=self.static_dim,
            expected_time_dim=self.time_dim,
        )
        external_parts: list[Tensor] = []
        if self.exogenous_dim > 0:
            external_slice = slice(self.target_dim, self.value_dim)
            external_parts.extend(
                (
                    values[..., external_slice],
                    x_mask[..., external_slice].to(x.dtype),
                    quality[..., external_slice],
                )
            )
        if self.static_dim > 0:
            external_parts.append(static_features)
        if self.time_dim > 0:
            external_parts.append(temporal)
        if external_parts:
            external_state = self.external_encoder(torch.cat(external_parts, dim=-1))
        else:
            external_state = torch.zeros_like(target_state)
        if self.no_exogenous_token is not None:
            external_state = external_state + self.no_exogenous_token
        return self.fusion(torch.cat((target_state, external_state), dim=-1))


def _expand_static_and_time(
    x: Tensor,
    static: Tensor,
    time_features: Tensor,
    *,
    expected_static_dim: int,
    expected_time_dim: int,
) -> tuple[Tensor, Tensor]:
    """Broadcast static and time features without mixing station identities."""
    batch, history, nodes, _ = x.shape
    if static.ndim == 2:
        static_features = static[None, None].expand(batch, history, -1, -1)
    elif static.ndim == 3:
        static_features = static[:, None].expand(-1, history, -1, -1)
    else:
        raise ValueError("static must have shape [N,S] or [B,N,S]")
    if static_features.shape != (batch, history, nodes, expected_static_dim):
        raise ValueError("static dimensions do not match the configured input contract")
    if time_features.ndim == 3:
        temporal = time_features[:, :, None].expand(-1, -1, nodes, -1)
    elif time_features.ndim == 4:
        temporal = time_features
    else:
        raise ValueError("time_features must have shape [B,T,F] or [B,T,N,F]")
    if temporal.shape != (batch, history, nodes, expected_time_dim):
        raise ValueError("time feature dimensions do not match the configured input contract")
    return static_features, temporal
