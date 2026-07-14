"""Graph transformations used by structural ablations."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor


def build_graph_variant(
    edge_index: Tensor,
    edge_attr: Tensor,
    variant: str = "directed",
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Build directed, undirected, or destination-shuffled graph tensors."""
    if variant == "directed":
        return edge_index.clone(), edge_attr.clone()
    if variant == "undirected":
        reverse = edge_index.flip(0)
        return torch.cat((edge_index, reverse), dim=1), torch.cat((edge_attr, edge_attr), dim=0)
    if variant == "shuffled":
        count = edge_index.shape[1]
        if count == 0:
            return edge_index.clone(), edge_attr.clone()
        nodes = int(edge_index.max().item()) + 1
        true_edges = set(map(tuple, edge_index.detach().cpu().t().tolist()))
        candidates = [
            (source, destination)
            for source in range(nodes)
            for destination in range(nodes)
            if source != destination and (source, destination) not in true_edges
        ]
        if len(candidates) < count:
            raise ValueError("graph is too dense to build a disjoint shuffled variant")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selection = torch.randperm(len(candidates), generator=generator)[:count].tolist()
        shuffled = torch.tensor(
            [candidates[index] for index in selection],
            dtype=edge_index.dtype,
            device=edge_index.device,
        ).t()
        return shuffled, edge_attr.clone()
    raise ValueError(f"unknown graph variant: {variant}")


def expand_directed_paths(
    edge_index: Tensor,
    edge_attr: Tensor,
    *,
    max_hops: int,
    max_travel_time: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Expand a directed acyclic graph into deterministic multi-hop path edges.

    Each original edge is retained. A path attribute is the mean of its edge
    attributes except for the final travel-time channel, which is summed.
    ``path_hops`` records the number of original directed edges represented by
    every expanded edge.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have shape [E,A]")
    if edge_attr.shape[1] == 0:
        raise ValueError("edge_attr must include travel time in its final channel")
    if max_hops <= 0:
        raise ValueError("max_hops must be positive")
    if max_travel_time is not None and max_travel_time <= 0:
        raise ValueError("max_travel_time must be positive")
    if edge_index.shape[1] == 0:
        return (
            edge_index.clone(),
            edge_attr.clone(),
            torch.empty(0, dtype=torch.long, device=edge_index.device),
        )

    index_cpu = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    attr_cpu = edge_attr.detach().cpu().to(torch.float32).numpy()
    outgoing: dict[int, list[tuple[int, int]]] = {}
    for edge_id, (source, destination) in enumerate(index_cpu.T.tolist()):
        outgoing.setdefault(source, []).append((destination, edge_id))
    for children in outgoing.values():
        children.sort()

    paths: list[tuple[int, int, tuple[int, ...]]] = []
    for source, children in sorted(outgoing.items()):
        frontier = deque(
            (destination, (edge_id,), frozenset((source, destination)))
            for destination, edge_id in children
            if destination != source
        )
        while frontier:
            destination, edge_ids, visited = frontier.popleft()
            travel_time = float(attr_cpu[list(edge_ids), -1].sum())
            if max_travel_time is None or travel_time <= max_travel_time:
                paths.append((source, destination, edge_ids))
            if len(edge_ids) >= max_hops:
                continue
            for child, child_edge in outgoing.get(destination, []):
                if child not in visited:
                    frontier.append(
                        (child, (*edge_ids, child_edge), visited | {child})
                    )
    paths.sort(key=lambda row: (row[1], row[0], len(row[2]), row[2]))
    if not paths:
        return (
            edge_index.new_empty((2, 0)),
            edge_attr.new_empty((0, edge_attr.shape[1])),
            edge_index.new_empty((0,)),
        )
    expanded_index = torch.tensor(
        [[row[0] for row in paths], [row[1] for row in paths]],
        dtype=edge_index.dtype,
        device=edge_index.device,
    )
    attributes: list[np.ndarray] = []
    hops: list[int] = []
    for _, _, edge_ids in paths:
        path_attr = attr_cpu[list(edge_ids)].mean(axis=0)
        path_attr[-1] = attr_cpu[list(edge_ids), -1].sum()
        attributes.append(path_attr)
        hops.append(len(edge_ids))
    expanded_attr = torch.as_tensor(
        np.stack(attributes), dtype=edge_attr.dtype, device=edge_attr.device
    )
    path_hops = torch.tensor(hops, dtype=torch.long, device=edge_index.device)
    return expanded_index, expanded_attr, path_hops
