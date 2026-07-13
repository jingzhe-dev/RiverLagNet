"""Gated local and upstream state fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalUpstreamGatedFusion(nn.Module):
    """Blend local and upstream states using a learned element-wise gate."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h_local: Tensor, h_upstream: Tensor) -> Tensor:
        if h_local.shape != h_upstream.shape:
            raise ValueError("local and upstream states must have identical shapes")
        gate = torch.sigmoid(self.gate(torch.cat((h_local, h_upstream), dim=-1)))
        return gate * h_local + (1.0 - gate) * h_upstream
