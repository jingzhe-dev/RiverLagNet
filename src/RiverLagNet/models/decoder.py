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

    def forward(self, node_state: Tensor) -> Tensor:
        """Return forecasts with shape `[B, T_out, N, target_dim]`."""
        context = node_state[:, None] + self.horizon_embedding[None, :, None]
        decoded = self.shared(context)
        return torch.cat([head(decoded) for head in self.heads], dim=-1)
