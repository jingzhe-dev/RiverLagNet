"""Typed tensor schemas for river-network observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


TARGET_NAMES = ("NH3N", "CODMn", "TP")


@dataclass(frozen=True)
class RiverGraph:
    """A directed graph with upstream-to-downstream edges."""

    edge_index: Tensor
    edge_attr: Tensor
    static: Tensor

    def validate(self) -> None:
        """Validate graph tensor ranks and compatible sizes."""
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long indices")
        if self.edge_attr.ndim != 2 or self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E, A]")
        if self.static.ndim != 2:
            raise ValueError("static must have shape [N, S]")
        if self.edge_index.numel() and (
            self.edge_index.min() < 0 or self.edge_index.max() >= self.static.shape[0]
        ):
            raise ValueError("edge_index contains an invalid node index")


@dataclass(frozen=True)
class TimeSeriesData:
    """Daily station observations with explicit masks and graph metadata."""

    values: Tensor
    observed: Tensor
    quality: Tensor | None
    graph: RiverGraph

    def validate(self) -> None:
        """Validate `[time, node, variable]` tensor contracts."""
        self.graph.validate()
        if self.values.ndim != 3:
            raise ValueError("values must have shape [T, N, V]")
        if self.observed.shape != self.values.shape:
            raise ValueError("observed must have the same shape as values")
        if self.observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean tensor")
        if self.quality is not None and self.quality.shape != self.values.shape:
            raise ValueError("quality must have the same shape as values")
        if self.values.shape[1] != self.graph.static.shape[0]:
            raise ValueError("values node count must match graph.static")
        if self.values.shape[2] < len(TARGET_NAMES):
            raise ValueError("values must contain NH3N, CODMn, and TP in the first channels")
