"""Graph-constrained edge-lag-horizon sparse cross-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sparsemax(values: Tensor, dim: int = -1) -> Tensor:
    """Project scores onto the probability simplex with exact zeros."""
    if values.shape[dim] == 0:
        raise ValueError("sparsemax dimension must be non-empty")
    shifted = values - values.amax(dim=dim, keepdim=True)
    ordered, _ = shifted.sort(dim=dim, descending=True)
    cumulative = ordered.cumsum(dim)
    count = torch.arange(
        1,
        ordered.shape[dim] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    shape = [1] * values.ndim
    shape[dim] = -1
    count = count.view(shape)
    support = 1.0 + count * ordered > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(dim, support_size - 1) - 1.0
    ) / support_size.to(values.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


class EdgeLagHorizonSparseAttention(nn.Module):
    """Attend only to causally observable upstream edge-lag candidates.

    A destination query at lead ``h`` can use an upstream history state at lag
    ``tau`` only when ``tau >= h``. Scores combine Transformer query-key
    similarity, edge attributes, a learned lag embedding, and a soft prior
    centered on the edge travel time. Sparsemax normalizes all incoming
    edge-lag candidates jointly for every destination, horizon, and head.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        max_lag: int = 30,
        prior_scale_days: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if edge_dim <= 0 or max_lag < 1 or prior_scale_days <= 0:
            raise ValueError("edge_dim, max_lag, and prior scale must be positive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_lag = max_lag
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_key = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.lag_embedding = nn.Parameter(
            torch.empty(max_lag + 1, num_heads, self.head_dim)
        )
        nn.init.normal_(self.lag_embedding, std=0.02)
        inverse_softplus = math.log(math.exp(prior_scale_days) - 1.0)
        self.prior_raw_scale = nn.Parameter(
            torch.full((num_heads,), inverse_softplus)
        )

    def forward(
        self,
        history_states: Tensor,
        destination_queries: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return head contexts and weights.

        Shapes are ``contexts [B,H,N,R,d]`` and
        ``weights [B,H,E,max_lag+1,R]``.
        """
        self._validate(history_states, destination_queries, edge_index, edge_attr)
        batch, history, nodes, _ = history_states.shape
        horizons = destination_queries.shape[1]
        edges = edge_index.shape[1]
        lags = self.max_lag + 1
        if horizons > self.max_lag:
            raise ValueError("forecast horizons cannot exceed max_lag")
        if edges == 0:
            context = history_states.new_zeros(
                batch, horizons, nodes, self.num_heads, self.head_dim
            )
            weights = history_states.new_zeros(
                batch, horizons, 0, lags, self.num_heads
            )
            return context, weights

        source, destination = edge_index
        queries = self.query_projection(destination_queries).view(
            batch, horizons, nodes, self.num_heads, self.head_dim
        )
        keys = self.key_projection(history_states).view(
            batch, history, nodes, self.num_heads, self.head_dim
        )
        values = self.value_projection(history_states).view(
            batch, history, nodes, self.num_heads, self.head_dim
        )

        lead = torch.arange(1, horizons + 1, device=history_states.device)
        lag = torch.arange(lags, device=history_states.device)
        source_time = history - 1 + lead[:, None] - lag[None, :]
        feasible = (lag[None, :] >= lead[:, None]) & (source_time >= 0)
        source_time = source_time.clamp(0, history - 1)
        time_index = source_time[:, None, :]
        source_index = source[None, :, None]
        candidate_keys = keys[:, time_index, source_index]
        candidate_values = values[:, time_index, source_index]

        edge_key = self.edge_key(edge_attr).view(
            edges, self.num_heads, self.head_dim
        )
        candidate_keys = (
            candidate_keys
            + edge_key[None, None, :, None]
            + self.lag_embedding[None, None]
        )
        edge_queries = queries[:, :, destination]
        scores = (
            edge_queries[:, :, :, None] * candidate_keys
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_attr)[None, None, :, None]

        travel_time = edge_attr[:, -1].to(history_states.dtype)
        prior_scale = F.softplus(self.prior_raw_scale).to(history_states.dtype) + 0.25
        prior = -0.5 * (
            (lag[None, :, None].to(history_states.dtype) - travel_time[:, None, None])
            / prior_scale[None, None]
        ).square()
        scores = scores + prior[None, None]
        scores = scores.masked_fill(
            ~feasible[None, :, None, :, None], -1e4
        )

        weights = torch.zeros_like(scores)
        contexts = history_states.new_zeros(
            batch, horizons, nodes, self.num_heads, self.head_dim
        )
        for node in destination.unique(sorted=True).tolist():
            edge_mask = destination == node
            node_scores = scores[:, :, edge_mask]
            flattened = node_scores.permute(0, 1, 4, 2, 3).flatten(3)
            node_weights = sparsemax(flattened, dim=-1)
            node_weights = node_weights.reshape(
                batch,
                horizons,
                self.num_heads,
                int(edge_mask.sum()),
                lags,
            ).permute(0, 1, 3, 4, 2)
            weights[:, :, edge_mask] = node_weights
            node_values = candidate_values[:, :, edge_mask]
            contexts[:, :, node] = (
                node_weights[..., None] * node_values
            ).sum(dim=(2, 3))
        return contexts, weights

    def _validate(
        self,
        history_states: Tensor,
        destination_queries: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        if history_states.ndim != 4 or history_states.shape[-1] != self.hidden_dim:
            raise ValueError("history_states must have shape [B,T,N,hidden_dim]")
        if (
            destination_queries.ndim != 4
            or destination_queries.shape[0] != history_states.shape[0]
            or destination_queries.shape[2:] != history_states.shape[2:]
        ):
            raise ValueError("destination_queries must have shape [B,H,N,hidden_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")
        if edge_attr.shape[1] == 0:
            raise ValueError("edge_attr must include travel time in its final channel")
        if edge_index.numel() and (
            edge_index.min() < 0 or edge_index.max() >= history_states.shape[2]
        ):
            raise ValueError("edge_index contains an invalid node")


class TransformerGraphCrossFusion(nn.Module):
    """Fuse Transformer queries with GNN routing-head tokens by cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.head_projection = nn.Parameter(
            torch.empty(num_heads, self.head_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.head_projection)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self, local_queries: Tensor, graph_head_contexts: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return fused graph residual states and head-fusion weights."""
        if local_queries.ndim != 4 or local_queries.shape[-1] != self.hidden_dim:
            raise ValueError("local_queries must have shape [B,H,N,hidden_dim]")
        expected = (*local_queries.shape[:-1], self.num_heads, self.head_dim)
        if graph_head_contexts.shape != expected:
            raise ValueError(f"graph_head_contexts must have shape {expected}")
        tokens = torch.einsum(
            "bhnrd,rdf->bhnrf", graph_head_contexts, self.head_projection
        )
        query = self.query_projection(local_queries)[..., None, :]
        scores = (query * tokens).sum(dim=-1) / math.sqrt(self.hidden_dim)
        valid = graph_head_contexts.square().sum(dim=-1) > 0
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        graph = (weights[..., None] * tokens).sum(dim=-2)
        graph = self.graph_projection(graph)
        gate = torch.sigmoid(self.gate(torch.cat((local_queries, graph), dim=-1)))
        return gate * graph, weights
