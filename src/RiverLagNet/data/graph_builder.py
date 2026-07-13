"""Graph transformations used by structural ablations."""

from __future__ import annotations

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
