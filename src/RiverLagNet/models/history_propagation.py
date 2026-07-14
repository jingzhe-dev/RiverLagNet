"""Directed travel-time propagation over encoded observation histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectedLaggedHistoryPropagation(nn.Module):
    """Inject multi-hop upstream histories before the temporal GRU.

    Each edge reads only source states at ``t - round(travel_time)``. Values
    before the start of the 90-day input window are masked rather than clamped,
    so no future value or artificial repeated boundary value is introduced.
    The residual update is zero-initialized to reproduce the local forecaster
    exactly at warm start.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        max_lag: int,
        steps: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or edge_dim <= 0:
            raise ValueError("hidden_dim and edge_dim must be positive")
        if max_lag < 0 or steps <= 0:
            raise ValueError("max_lag must be non-negative and steps must be positive")
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.steps = steps
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.update[-1].weight)

    def aligned_source_states(
        self,
        states: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return lag-aligned sources and validity with shapes ``[B,T,E,D]``."""
        self._validate(states, edge_index, edge_attr)
        _, history, _, _ = states.shape
        source = edge_index[0]
        lag = edge_attr[:, -1].round().long().clamp(0, self.max_lag)
        time = torch.arange(history, device=states.device)
        source_time = time[:, None] - lag[None, :]
        valid = source_time >= 0
        source_time = source_time.clamp(0, history - 1)
        aligned = states[:, source_time, source]
        return aligned, valid

    def forward(
        self,
        local_history: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return graph-conditioned histories and normalized edge strengths."""
        self._validate(local_history, edge_index, edge_attr)
        batch, history, nodes, _ = local_history.shape
        edges = edge_index.shape[1]
        if edges == 0:
            return local_history, local_history.new_zeros(batch, history, 0, 1)

        destination = edge_index[1]
        edge_gate = torch.sigmoid(self.edge_gate(edge_attr)).to(local_history.dtype)
        states = local_history
        routing: Tensor | None = None
        for _ in range(self.steps):
            source_states, valid = self.aligned_source_states(
                states, edge_index, edge_attr
            )
            valid_float = valid.to(local_history.dtype)
            messages = self.message_projection(source_states)
            messages = messages * edge_gate[None, None]
            messages = messages * valid_float[None, :, :, None]

            upstream = local_history.new_zeros(
                batch, history, nodes, self.hidden_dim
            )
            upstream.index_add_(2, destination, messages)
            denominator = local_history.new_zeros(history, nodes)
            denominator.index_add_(1, destination, valid_float)
            upstream = upstream / denominator.clamp_min(1.0)[None, :, :, None]
            has_upstream = denominator > 0
            combined = torch.cat((local_history, states, upstream), dim=-1)
            candidate = states + self.update(combined)
            states = torch.where(
                has_upstream[None, :, :, None], candidate, states
            )

            strength = edge_gate.mean(dim=-1)
            weighted = strength[None, :] * valid_float
            strength_denominator = local_history.new_zeros(history, nodes)
            strength_denominator.index_add_(1, destination, weighted)
            normalized = weighted / strength_denominator[:, destination].clamp_min(1e-8)
            routing = normalized[None, :, :, None].expand(batch, -1, -1, -1)
        assert routing is not None
        return states, routing

    def _validate(
        self, states: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> None:
        if states.ndim != 4 or states.shape[-1] != self.hidden_dim:
            raise ValueError("states must have shape [B,T,N,hidden_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")
        if edge_attr.shape[1] == 0:
            raise ValueError("edge_attr must include travel time in its final channel")
        if edge_index.numel() and (
            edge_index.min() < 0 or edge_index.max() >= states.shape[2]
        ):
            raise ValueError("edge_index contains an invalid node")
