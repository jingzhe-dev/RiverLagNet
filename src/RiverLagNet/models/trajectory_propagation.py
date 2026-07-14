"""Leakage-safe multi-hop propagation over predicted future node trajectories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectedTrajectoryPropagation(nn.Module):
    """Propagate horizon-specific source states along directed travel-time paths.

    For destination lead ``h`` and rounded travel time ``tau``, the source state
    is taken from predicted lead ``h - tau`` when that lead is in the future and
    from the observed history otherwise. Repeating the shared update passes
    information over multiple upstream hops without reading future targets.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        max_lag: int,
        steps: int = 4,
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def aligned_source_states(
        self,
        future_states: Tensor,
        history_states: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Return travel-time aligned source states ``[B,H,E,D]``."""
        self._validate(future_states, history_states, edge_index, edge_attr)
        _, horizons, _, _ = future_states.shape
        history = history_states.shape[1]
        source = edge_index[0]
        lag = edge_attr[:, -1].round().long().clamp(0, self.max_lag)
        lead = torch.arange(1, horizons + 1, device=future_states.device)
        relative_lead = lead[:, None] - lag[None, :]

        future_index = (relative_lead - 1).clamp(0, horizons - 1)
        predicted = future_states[:, future_index, source]
        history_index = (history - 1 + relative_lead).clamp(0, history - 1)
        observed = history_states[:, history_index, source]
        use_prediction = relative_lead > 0
        return torch.where(use_prediction[None, :, :, None], predicted, observed)

    def routing_weights(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        *,
        batch_size: int,
        horizons: int,
        num_nodes: int,
    ) -> Tensor:
        """Return normalized diagnostic edge strengths ``[B,H,E,1]``."""
        if edge_index.shape[1] == 0:
            return edge_attr.new_zeros(batch_size, horizons, 0, 1)
        destination = edge_index[1]
        strength = torch.sigmoid(self.edge_gate(edge_attr)).mean(dim=-1)
        denominator = strength.new_zeros(num_nodes)
        denominator.index_add_(0, destination, strength)
        normalized = strength / denominator[destination].clamp_min(1e-8)
        return normalized[None, None, :, None].expand(
            batch_size, horizons, -1, -1
        )

    def forward(
        self,
        local_future: Tensor,
        history_states: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return propagated future states and normalized routing strengths."""
        self._validate(local_future, history_states, edge_index, edge_attr)
        batch, horizons, nodes, _ = local_future.shape
        if edge_index.shape[1] == 0:
            return local_future, local_future.new_zeros(batch, horizons, 0, 1)

        destination = edge_index[1]
        incoming_count = local_future.new_zeros(nodes)
        incoming_count.index_add_(
            0, destination, torch.ones_like(destination, dtype=local_future.dtype)
        )
        has_upstream = incoming_count > 0
        edge_gate = torch.sigmoid(self.edge_gate(edge_attr)).to(local_future.dtype)
        states = local_future
        for _ in range(self.steps):
            source_states = self.aligned_source_states(
                states, history_states, edge_index, edge_attr
            )
            messages = self.message_projection(source_states) * edge_gate[None, None]
            upstream = local_future.new_zeros(batch, horizons, nodes, self.hidden_dim)
            upstream.index_add_(2, destination, messages)
            upstream = upstream / incoming_count.clamp_min(1.0)[None, None, :, None]
            combined = torch.cat((states, upstream), dim=-1)
            gate = torch.sigmoid(self.update_gate(combined))
            candidate = self.norm(states + gate * self.update(combined))
            states = torch.where(
                has_upstream[None, None, :, None], candidate, states
            )
        routing = self.routing_weights(
            edge_index,
            edge_attr,
            batch_size=batch,
            horizons=horizons,
            num_nodes=nodes,
        )
        return states, routing

    def _validate(
        self,
        future_states: Tensor,
        history_states: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        if future_states.ndim != 4 or history_states.ndim != 4:
            raise ValueError("future and history states must have four dimensions")
        if future_states.shape[0] != history_states.shape[0]:
            raise ValueError("future and history batch sizes must match")
        if future_states.shape[2:] != history_states.shape[2:]:
            raise ValueError("future and history node/hidden dimensions must match")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")
        if edge_attr.shape[1] == 0:
            raise ValueError("edge_attr must include travel time in its final channel")
