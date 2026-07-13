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
        if count < 2:
            return edge_index.clone(), edge_attr.clone()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shift = int(torch.randint(1, count, (1,), generator=generator).item())
        shuffled = edge_index.clone()
        shuffled[1] = edge_index[1].roll(shift)
        return shuffled, edge_attr.clone()
    raise ValueError(f"unknown graph variant: {variant}")
