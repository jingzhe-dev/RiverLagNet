"""Gated local and upstream state fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalUpstreamGatedFusion(nn.Module):
    """Add a gated upstream residual without weakening the local state."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.upstream_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, h_local: Tensor, h_upstream: Tensor) -> Tensor:
        if h_local.shape != h_upstream.shape:
            raise ValueError("local and upstream states must have identical shapes")
        gate = torch.sigmoid(self.gate(torch.cat((h_local, h_upstream), dim=-1)))
        upstream_residual = self.upstream_projection(h_upstream)
        return h_local + gate * upstream_residual
