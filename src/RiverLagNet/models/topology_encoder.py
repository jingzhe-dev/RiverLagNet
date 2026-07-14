"""Deterministic directed-river topology features and trainable encoding."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn


TOPOLOGY_FEATURE_DIM = 10


class DirectedTopologyEncoder(nn.Module):
    """Encode graph position for every node, including headwaters.

    Features are computed from the directed forest only: in/out degree,
    headwater/outlet flags, upstream/downstream hop depth, ancestor/descendant
    counts, and cumulative travel time in both directions. The final projection
    starts at exact zero so a warm-started graph model initially reproduces its
    no-graph checkpoint.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(TOPOLOGY_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self._cache_signature: tuple[int, int, int, int, float] | None = None
        self._cached_features: Tensor | None = None

    def forward(
        self, edge_index: Tensor, edge_attr: Tensor, num_nodes: int
    ) -> Tensor:
        """Return encoded topology ``[N,D]`` on the graph device."""
        features = self.features(edge_index, edge_attr, num_nodes)
        return self.network(features)

    def features(
        self, edge_index: Tensor, edge_attr: Tensor, num_nodes: int
    ) -> Tensor:
        """Return normalized deterministic topology features ``[N,10]``."""
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")
        if edge_attr.shape[1] == 0:
            raise ValueError("edge_attr must include travel time")
        signature = (
            num_nodes,
            int(edge_index.shape[1]),
            int(edge_index[0].sum().detach().cpu()),
            int(edge_index[1].sum().detach().cpu()),
            round(float(edge_attr[:, -1].sum().detach().cpu()), 6),
        )
        if self._cached_features is None or signature != self._cache_signature:
            self._cached_features = self._compute_features(
                edge_index.detach().cpu(), edge_attr.detach().cpu(), num_nodes
            )
            self._cache_signature = signature
        return self._cached_features.to(device=edge_attr.device, dtype=edge_attr.dtype)

    @staticmethod
    def _compute_features(
        edge_index: Tensor, edge_attr: Tensor, num_nodes: int
    ) -> Tensor:
        parents: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
        children: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
        for edge, (source, destination) in enumerate(edge_index.t().tolist()):
            travel = max(0.0, float(edge_attr[edge, -1]))
            parents[destination].append((source, travel))
            children[source].append((destination, travel))

        indegree = [len(values) for values in parents]
        outdegree = [len(values) for values in children]
        remaining = indegree.copy()
        queue = deque(index for index, degree in enumerate(remaining) if degree == 0)
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child, _ in children[node]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)
        if len(order) != num_nodes:
            raise ValueError("topology encoding requires a directed acyclic graph")

        upstream_depth = [0.0] * num_nodes
        ancestor_count = [0.0] * num_nodes
        upstream_travel = [0.0] * num_nodes
        for node in order:
            if parents[node]:
                upstream_depth[node] = max(
                    upstream_depth[parent] + 1.0 for parent, _ in parents[node]
                )
                ancestor_count[node] = sum(
                    ancestor_count[parent] + 1.0 for parent, _ in parents[node]
                )
                upstream_travel[node] = max(
                    upstream_travel[parent] + travel
                    for parent, travel in parents[node]
                )

        downstream_depth = [0.0] * num_nodes
        descendant_count = [0.0] * num_nodes
        downstream_travel = [0.0] * num_nodes
        for node in reversed(order):
            if children[node]:
                downstream_depth[node] = max(
                    downstream_depth[child] + 1.0 for child, _ in children[node]
                )
                descendant_count[node] = sum(
                    descendant_count[child] + 1.0 for child, _ in children[node]
                )
                downstream_travel[node] = max(
                    downstream_travel[child] + travel
                    for child, travel in children[node]
                )

        raw = torch.tensor(
            [
                [
                    float(indegree[node]),
                    float(outdegree[node]),
                    float(indegree[node] == 0),
                    float(outdegree[node] == 0),
                    upstream_depth[node],
                    downstream_depth[node],
                    torch.log1p(torch.tensor(ancestor_count[node])).item(),
                    torch.log1p(torch.tensor(descendant_count[node])).item(),
                    torch.log1p(torch.tensor(upstream_travel[node])).item(),
                    torch.log1p(torch.tensor(downstream_travel[node])).item(),
                ]
                for node in range(num_nodes)
            ],
            dtype=torch.float32,
        )
        scale = raw.abs().amax(dim=0).clamp_min(1.0)
        return raw / scale
