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
    ) -> None:
        super().__init__()
        if lag_mode not in LAG_MODES:
            raise ValueError(f"lag_mode must be one of {sorted(LAG_MODES)}")
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.lag_mode = lag_mode
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.lag_embedding = nn.Embedding(max_lag + 1, hidden_dim)
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
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
        attention = torch.zeros_like(logits)
        for node in destination.unique(sorted=True):
            incoming = destination == node
            normalized = torch.softmax(logits[:, incoming].reshape(batch, -1), dim=-1)
            attention[:, incoming] = normalized.reshape(batch, int(incoming.sum()), lag_count)
        messages = self.message_projection(source_states)
        message_weights = self.dropout(attention)
        edge_messages = (message_weights[..., None] * messages).sum(dim=2)
        upstream = h_seq.new_zeros(batch, nodes, hidden)
        upstream.index_add_(1, destination, edge_messages)
        return upstream, attention
