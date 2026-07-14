"""Directed discrete-lag upstream message passing."""

from __future__ import annotations

import torch
from torch import Tensor, nn


LAG_MODES = {"no_lag", "fixed_lag", "learned_lag"}


class DirectedLagAwareMessagePassing(nn.Module):
    """Aggregate upstream states with joint incoming-edge and lag attention.

    Attention weights are normalized over all `(upstream edge, lag)` candidates
    for each destination node independently. They are learned routing weights,
    not estimates of causal contribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        max_lag: int = 14,
        lag_mode: str = "learned_lag",
        dropout: float = 0.0,
        prior_scale_days: float = 1.0,
        prior_strength: float = 8.0,
    ) -> None:
        super().__init__()
        if lag_mode not in LAG_MODES:
            raise ValueError(f"lag_mode must be one of {sorted(LAG_MODES)}")
        if prior_scale_days <= 0:
            raise ValueError("prior_scale_days must be positive")
        if prior_strength < 0:
            raise ValueError("prior_strength cannot be negative")
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.lag_mode = lag_mode
        self.prior_scale_days = prior_scale_days
        self.prior_strength = prior_strength
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.lag_embedding = nn.Embedding(max_lag + 1, hidden_dim)
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.lag_residual_scale = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_seq: Tensor,
        h_local: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return upstream states `[B,N,D]` and weights `[B,E,max_lag+1]`."""
        if h_seq.ndim != 4 or h_local.shape != (h_seq.shape[0], h_seq.shape[2], h_seq.shape[3]):
            raise ValueError("h_seq and h_local shapes are incompatible")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr and edge_index must have the same edge count")
        batch, history, nodes, hidden = h_seq.shape
        source, destination = edge_index
        lag_count = self.max_lag + 1
        lag_states = torch.stack(
            [h_seq[:, max(0, history - 1 - lag)] for lag in range(lag_count)], dim=2
        )
        source_states = lag_states[:, source]
        destination_states = h_local[:, destination, None].expand(-1, -1, lag_count, -1)
        edge_context = self.edge_encoder(edge_attr)[None, :, None].expand(
            batch, -1, lag_count, -1
        )
        lag_ids = torch.arange(lag_count, device=h_seq.device)
        lag_context = self.lag_embedding(lag_ids)[None, None].expand(
            batch, edge_index.shape[1], -1, -1
        )
        logits = self.score_network(
            torch.cat((destination_states, source_states, edge_context, lag_context), dim=-1)
        ).squeeze(-1)
        if self.lag_mode == "learned_lag" and self.prior_strength:
            logits = logits + self._travel_time_prior(edge_attr, lag_ids)[None]
        available = lag_ids[None, :] < history
        available = available.expand(edge_index.shape[1], -1).clone()
        if self.lag_mode == "no_lag":
            available.zero_()
            available[:, 0] = True
        elif self.lag_mode == "fixed_lag":
            fixed = edge_attr[:, -1].round().long().clamp(0, min(self.max_lag, history - 1))
            available.zero_()
            available.scatter_(1, fixed[:, None], True)
        logits = logits.masked_fill(~available[None], -torch.inf)
        attention = torch.zeros(logits.shape, device=logits.device, dtype=torch.float32)
        for node in destination.unique(sorted=True):
            incoming = destination == node
            normalized = torch.softmax(logits[:, incoming].reshape(batch, -1).float(), dim=-1)
            attention[:, incoming] = normalized.reshape(batch, int(incoming.sum()), lag_count)
        messages = self.message_projection(source_states)
        if self.lag_mode == "learned_lag":
            lag_mix = torch.tanh(self.lag_residual_scale).to(messages.dtype)
            current_messages = messages[:, :, :1]
            messages = current_messages + lag_mix * (messages - current_messages)
        message_weights = self.dropout(attention).to(messages.dtype)
        edge_messages = (message_weights[..., None] * messages).sum(dim=2).to(h_seq.dtype)
        upstream = h_seq.new_zeros(batch, nodes, hidden)
        upstream.index_add_(1, destination, edge_messages)
        return upstream, attention

    def forward_horizons(
        self,
        h_seq: Tensor,
        h_local: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        output_window: int,
    ) -> tuple[Tensor, Tensor]:
        """Route observable source states to each forecast horizon.

        Returns upstream states `[B,T_out,N,D]` and routing weights
        `[B,T_out,E,max_lag+1]`. For lead day `h` and lag `tau`, the source
        index is `t + h - tau`. When that index lies beyond the forecast
        origin, the last observed source hidden state is used as a leakage-free
        persistence proxy; the method never reads future observations.
        """
        if output_window <= 0:
            raise ValueError("output_window must be positive")
        if h_seq.ndim != 4 or h_local.shape != (
            h_seq.shape[0],
            h_seq.shape[2],
            h_seq.shape[3],
        ):
            raise ValueError("h_seq and h_local shapes are incompatible")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr and edge_index must have the same edge count")

        batch, history, nodes, hidden = h_seq.shape
        source, destination = edge_index
        edge_count = edge_index.shape[1]
        lag_count = self.max_lag + 1
        lag_ids = torch.arange(lag_count, device=h_seq.device)
        lead_days = torch.arange(1, output_window + 1, device=h_seq.device)
        source_indices = history - 1 + lead_days[:, None] - lag_ids[None, :]
        has_history = source_indices >= 0
        source_indices = source_indices.clamp(0, history - 1)
        aligned = h_seq[:, source_indices].permute(0, 1, 3, 2, 4)
        source_states = aligned[:, :, source]

        available = has_history[:, None, :].expand(-1, edge_count, -1).clone()
        if self.lag_mode == "no_lag":
            source_states = h_seq[:, -1, source][:, None, :, None].expand(
                -1, output_window, -1, lag_count, -1
            )
            available.zero_()
            available[:, :, 0] = True
        elif self.lag_mode == "fixed_lag":
            fixed = edge_attr[:, -1].round().long().clamp(0, self.max_lag)
            available &= lag_ids[None, None, :] == fixed[None, :, None]

        destination_states = h_local[:, None, destination, None].expand(
            -1, output_window, -1, lag_count, -1
        )
        edge_context = self.edge_encoder(edge_attr)[None, None, :, None].expand(
            batch, output_window, -1, lag_count, -1
        )
        lag_context = self.lag_embedding(lag_ids)[None, None, None].expand(
            batch, output_window, edge_count, -1, -1
        )
        logits = self.score_network(
            torch.cat((destination_states, source_states, edge_context, lag_context), dim=-1)
        ).squeeze(-1)
        if self.lag_mode == "learned_lag" and self.prior_strength:
            logits = logits + self._travel_time_prior(edge_attr, lag_ids)[None, None]

        attention = torch.zeros(logits.shape, device=logits.device, dtype=torch.float32)
        for node in destination.unique(sorted=True):
            incoming = destination == node
            incoming_count = int(incoming.sum())
            candidates = available[:, incoming].reshape(output_window, -1)
            node_logits = logits[:, :, incoming].reshape(batch, output_window, -1)
            masked = node_logits.masked_fill(~candidates[None], -torch.inf)
            has_candidate = candidates.any(dim=-1)
            safe_logits = torch.where(
                has_candidate[None, :, None], masked, torch.zeros_like(masked)
            )
            normalized = torch.softmax(safe_logits.float(), dim=-1)
            normalized = normalized * candidates[None].to(normalized.dtype)
            attention[:, :, incoming] = normalized.reshape(
                batch, output_window, incoming_count, lag_count
            )

        messages = self.message_projection(source_states)
        if self.lag_mode == "learned_lag":
            lag_mix = torch.tanh(self.lag_residual_scale).to(messages.dtype)
            current_messages = messages[:, :, :, :1]
            messages = current_messages + lag_mix * (messages - current_messages)
        message_weights = self.dropout(attention).to(messages.dtype)
        edge_messages = (message_weights[..., None] * messages).sum(dim=3).to(h_seq.dtype)
        upstream = h_seq.new_zeros(batch, output_window, nodes, hidden)
        upstream.index_add_(2, destination, edge_messages)
        return upstream, attention

    def _travel_time_prior(self, edge_attr: Tensor, lag_ids: Tensor) -> Tensor:
        """Return Gaussian log-prior scores `[E,max_lag+1]` from edge travel time."""
        prior_days = edge_attr[:, -1].to(torch.float32)
        offset = (lag_ids[None].to(torch.float32) - prior_days[:, None]) / self.prior_scale_days
        return (-0.5 * self.prior_strength * offset.square()).to(edge_attr.device)
