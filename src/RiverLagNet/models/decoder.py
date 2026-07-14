"""Direct multi-horizon, multi-target decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiHorizonMultiTargetDecoder(nn.Module):
    """Decode shared node states through target-specific output heads."""

    def __init__(self, hidden_dim: int, output_window: int = 30, target_dim: int = 3) -> None:
        super().__init__()
        self.horizon_embedding = nn.Parameter(torch.empty(output_window, hidden_dim))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(target_dim))

    def contextualize(self, node_state: Tensor) -> Tensor:
        """Add lead embeddings and return ``[B,T_out,N,D]`` contexts."""
        if node_state.ndim == 3:
            context = node_state[:, None] + self.horizon_embedding[None, :, None]
        elif node_state.ndim == 4:
            if node_state.shape[1] != self.horizon_embedding.shape[0]:
                raise ValueError("horizon-specific state length must equal output_window")
            context = node_state + self.horizon_embedding[None, :, None]
        else:
            raise ValueError("node_state must have shape [B,N,D] or [B,T_out,N,D]")
        return context

    def decode_context(self, context: Tensor) -> Tensor:
        """Decode an already contextualized ``[B,T_out,N,D]`` trajectory."""
        if context.ndim != 4 or context.shape[1] != self.horizon_embedding.shape[0]:
            raise ValueError("context must have shape [B,T_out,N,D]")
        decoded = self.shared(context)
        return torch.cat([head(decoded) for head in self.heads], dim=-1)

    def forward(self, node_state: Tensor) -> Tensor:
        """Decode shared `[B,N,D]` or horizon-specific `[B,T_out,N,D]` states."""
        return self.decode_context(self.contextualize(node_state))


class UpstreamResidualDecoder(nn.Module):
    """Decode a horizon-specific upstream correction with exact zero fallback."""

    def __init__(self, hidden_dim: int, target_dim: int = 3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1, bias=False) for _ in range(target_dim)
        )

    def forward(self, upstream_state: Tensor) -> Tensor:
        """Return additive corrections `[B,T_out,N,target_dim]`."""
        if upstream_state.ndim != 4:
            raise ValueError("upstream_state must have shape [B,T_out,N,D]")
        decoded = self.shared(upstream_state)
        return torch.cat([head(decoded) for head in self.heads], dim=-1)
