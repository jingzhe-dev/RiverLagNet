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


class BoundedLinearHorizonGate(nn.Module):
    """Scale upstream output corrections with two lead-dependent parameters.

    The scale is ``2 * sigmoid(offset + slope * normalized_lead)`` and is
    therefore bounded between zero and two. Both parameters start at zero, so
    the gate is exactly one and strictly nests an existing graph checkpoint.
    """

    def __init__(self, output_window: int) -> None:
        super().__init__()
        if output_window <= 0:
            raise ValueError("output_window must be positive")
        self.offset = nn.Parameter(torch.zeros(()))
        self.slope = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "normalized_lead", torch.linspace(-1.0, 1.0, output_window)
        )

    def scales(self) -> Tensor:
        """Return one bounded residual scale per forecast lead."""
        return 2.0 * torch.sigmoid(
            self.offset + self.slope * self.normalized_lead
        )

    def forward(self, upstream_correction: Tensor) -> Tensor:
        """Scale corrections shaped `[B,T_out,N,V]` without mixing axes."""
        if upstream_correction.ndim != 4:
            raise ValueError("upstream correction must have shape [B,T_out,N,V]")
        if upstream_correction.shape[1] != self.normalized_lead.numel():
            raise ValueError("upstream correction horizon does not match the gate")
        scale = self.scales().to(upstream_correction.dtype)
        return upstream_correction * scale[None, :, None, None]
