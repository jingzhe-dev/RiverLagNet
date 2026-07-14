"""Directed travel-time transport over forecast water-quality trajectories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectedLaggedOutputTransport(nn.Module):
    """Correct downstream forecasts from time-aligned upstream trajectories.

    For a destination lead before the edge travel time, the message reads the
    corresponding observed source history. For later leads it reads a source
    forecast, never a future target. Repeated shared updates permit multi-hop
    transport while preserving the upstream-to-downstream direction.
    """

    def __init__(
        self,
        target_dim: int,
        edge_dim: int,
        max_lag: int,
        steps: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if target_dim <= 0 or edge_dim <= 0 or hidden_dim <= 0:
            raise ValueError("target_dim, edge_dim, and hidden_dim must be positive")
        if max_lag < 0 or steps <= 0:
            raise ValueError("max_lag must be non-negative and steps must be positive")
        self.target_dim = target_dim
        self.max_lag = max_lag
        self.steps = steps
        feature_dim = target_dim * 4 + edge_dim + 1
        self.message = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim, bias=False),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )
        nn.init.zeros_(self.message[-1].weight)

    def aligned_source_values(
        self,
        future_values: Tensor,
        history_values: Tensor,
        history_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return aligned source values and availability ``[B,H,E,V]``."""
        self._validate(
            future_values, history_values, history_mask, edge_index, edge_attr
        )
        _, horizons, _, _ = future_values.shape
        history = history_values.shape[1]
        source = edge_index[0]
        lag = edge_attr[:, -1].round().long().clamp(0, self.max_lag)
        lead = torch.arange(1, horizons + 1, device=future_values.device)
        relative_lead = lead[:, None] - lag[None, :]

        future_index = (relative_lead - 1).clamp(0, horizons - 1)
        predicted = future_values[:, future_index, source]
        history_index = (history - 1 + relative_lead).clamp(0, history - 1)
        observed = history_values[:, history_index, source]
        observed_mask = history_mask[:, history_index, source]
        use_prediction = relative_lead > 0
        values = torch.where(
            use_prediction[None, :, :, None], predicted, observed
        )
        availability = torch.where(
            use_prediction[None, :, :, None],
            torch.ones_like(predicted, dtype=torch.bool),
            observed_mask,
        )
        return values, availability

    def forward(
        self,
        local_prediction: Tensor,
        history_values: Tensor,
        history_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return transported predictions and normalized edge strengths."""
        self._validate(
            local_prediction, history_values, history_mask, edge_index, edge_attr
        )
        batch, horizons, nodes, _ = local_prediction.shape
        edges = edge_index.shape[1]
        if edges == 0:
            return local_prediction, local_prediction.new_zeros(batch, horizons, 0, 1)
        destination = edge_index[1]
        incoming_count = local_prediction.new_zeros(nodes)
        incoming_count.index_add_(
            0,
            destination,
            torch.ones(
                edges,
                device=destination.device,
                dtype=local_prediction.dtype,
            ),
        )
        has_upstream = incoming_count > 0
        normalized_lead = torch.linspace(
            0.0, 1.0, horizons, device=local_prediction.device,
            dtype=local_prediction.dtype,
        )
        edge_features = edge_attr[None, None].expand(batch, horizons, -1, -1)
        lead_features = normalized_lead[None, :, None, None].expand(
            batch, -1, edges, -1
        )
        gate_features = torch.cat((edge_features, lead_features), dim=-1)
        gate = torch.sigmoid(self.edge_gate(gate_features))

        states = local_prediction
        for _ in range(self.steps):
            source, available = self.aligned_source_values(
                states, history_values, history_mask, edge_index, edge_attr
            )
            destination_values = states[:, :, destination]
            delta = source - destination_values
            features = torch.cat(
                (
                    source,
                    destination_values,
                    delta,
                    available.to(source.dtype),
                    edge_features,
                    lead_features,
                ),
                dim=-1,
            )
            messages = self.message(features) * gate
            messages = messages * available.to(messages.dtype)
            correction = local_prediction.new_zeros(
                batch, horizons, nodes, self.target_dim
            )
            correction.index_add_(2, destination, messages)
            correction = correction / incoming_count.clamp_min(1.0)[
                None, None, :, None
            ]
            states = torch.where(
                has_upstream[None, None, :, None],
                states + correction,
                states,
            )

        strength = gate.mean(dim=-1, keepdim=True)
        denominator = strength.new_zeros(batch, horizons, nodes, 1)
        denominator.index_add_(2, destination, strength)
        routing = strength / denominator[:, :, destination].clamp_min(1e-8)
        return states, routing

    def _validate(
        self,
        future_values: Tensor,
        history_values: Tensor,
        history_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        if future_values.ndim != 4 or history_values.ndim != 4:
            raise ValueError("future and history values must have four dimensions")
        if history_values.shape != history_mask.shape:
            raise ValueError("history values and mask must have identical shape")
        if future_values.shape[0] != history_values.shape[0]:
            raise ValueError("future and history batch sizes must match")
        if future_values.shape[2:] != history_values.shape[2:]:
            raise ValueError("future and history node/target dimensions must match")
        if future_values.shape[-1] != self.target_dim:
            raise ValueError("future target dimension does not match the module")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")
        if edge_attr.shape[1] == 0:
            raise ValueError("edge_attr must include travel time in its final channel")
