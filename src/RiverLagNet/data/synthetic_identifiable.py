"""Auditable synthetic signals with directed multi-hop lagged transport."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .schema import RiverGraph, TimeSeriesData


EVENT_TARGET_MULTIPLIERS = torch.tensor([0.8, 1.0, 0.6])
ROUTING_TARGET_MULTIPLIERS = torch.tensor([0.85, 1.0, 0.75])


@dataclass(frozen=True)
class SyntheticScenario:
    """Observed data plus truth components that never enter model batches."""

    data: TimeSeriesData
    true_lag_days: Tensor
    local_background: Tensor
    local_events: Tensor
    routed_load: Tensor


def route_pollutant_events(
    local_events: Tensor, edge_index: Tensor, edge_attr: Tensor
) -> tuple[Tensor, Tensor]:
    """Route local events over directed edges using causal integer travel lags."""
    if local_events.ndim != 3 or local_events.shape[-1] != 3:
        raise ValueError("local_events must have shape [T, N, 3]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 3):
        raise ValueError("edge_attr must have shape [E, 3]")
    days, nodes, _ = local_events.shape
    source, destination = edge_index
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= nodes):
        raise ValueError("edge_index contains an invalid node")
    lag_days = edge_attr[:, -1].round().long()
    if bool(((lag_days < 1) | (lag_days > 7)).any()):
        raise ValueError("identifiable v1 lags must be in 1..7 days")
    distance = edge_attr[:, 0]
    slope = edge_attr[:, 1]
    base = 2.0 * torch.exp(-distance / 100.0) * torch.exp(-20.0 * slope)
    attenuation = base[:, None] * ROUTING_TARGET_MULTIPLIERS.to(edge_attr)
    total = local_events.clone()
    routed = torch.zeros_like(local_events)
    for day in range(days):
        for node in range(nodes):
            incoming = torch.nonzero(destination == node, as_tuple=False).flatten()
            for edge in incoming.tolist():
                lag = int(lag_days[edge].item())
                if day >= lag:
                    contribution = attenuation[edge] * total[day - lag, source[edge]]
                    total[day, node] += contribution
                    routed[day, node] += contribution
    return total, routed


def generate_identifiable_synthetic_scenario(
    num_days: int = 520,
    num_nodes: int = 8,
    num_variables: int = 3,
    missing_rate: float = 0.08,
    seed: int = 42,
) -> SyntheticScenario:
    """Generate an identifiable event-routing scenario and hidden truth components."""
    if num_days < 2 or num_nodes < 2:
        raise ValueError("identifiable v1 requires at least two days and two nodes")
    if num_variables != 3:
        raise ValueError("identifiable v1 requires exactly three target variables")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    graph = _generate_graph(num_nodes, generator)
    background = _generate_background(num_days, num_nodes, generator)
    events = _generate_events(num_days, num_nodes, generator)
    transported, routed = route_pollutant_events(events, graph.edge_index, graph.edge_attr)
    values = (background + transported).clamp_min(0.0)
    observed = torch.rand(values.shape, generator=generator) >= missing_rate
    quality = observed.to(values.dtype) * (
        0.8 + 0.2 * torch.rand(values.shape, generator=generator)
    )
    data = TimeSeriesData(values, observed, quality, graph)
    data.validate()
    return SyntheticScenario(
        data=data,
        true_lag_days=graph.edge_attr[:, -1].round().long(),
        local_background=background,
        local_events=events,
        routed_load=routed,
    )


def _generate_graph(num_nodes: int, generator: torch.Generator) -> RiverGraph:
    destination = torch.arange(1, num_nodes, dtype=torch.long)
    source = torch.tensor(
        [
            int(torch.randint(0, node, (1,), generator=generator).item())
            for node in range(1, num_nodes)
        ],
        dtype=torch.long,
    )
    edge_index = torch.stack((source, destination))
    distance = 10.0 + 40.0 * torch.rand(num_nodes - 1, generator=generator)
    slope = 0.001 + 0.02 * torch.rand(num_nodes - 1, generator=generator)
    lag = torch.randint(1, 8, (num_nodes - 1,), generator=generator).float()
    edge_attr = torch.stack((distance, slope, lag), dim=-1)
    position = torch.linspace(0.0, 1.0, num_nodes)
    static = torch.stack(
        (position, torch.linspace(0.2, 1.0, num_nodes), 1.0 - position), dim=-1
    )
    return RiverGraph(edge_index=edge_index, edge_attr=edge_attr, static=static)


def _generate_background(
    num_days: int, num_nodes: int, generator: torch.Generator
) -> Tensor:
    background = torch.zeros(num_days, num_nodes, 3)
    phases = 2.0 * math.pi * torch.rand(num_nodes, 3, generator=generator)
    offsets = 0.5 + 1.5 * torch.rand(num_nodes, 3, generator=generator)
    for day in range(num_days):
        seasonal = torch.sin(torch.tensor(2.0 * math.pi * day / 30.0) + phases)
        innovation = 0.05 * torch.randn(num_nodes, 3, generator=generator)
        previous = background[day - 1] if day else offsets
        background[day] = (
            0.45 * previous + 0.45 * offsets + 0.05 * seasonal + innovation
        )
    return background


def _generate_events(
    num_days: int, num_nodes: int, generator: torch.Generator
) -> Tensor:
    events = torch.zeros(num_days, num_nodes, 3)
    target_scale = EVENT_TARGET_MULTIPLIERS.to(events)
    for node in range(num_nodes):
        for day in range(num_days):
            if float(torch.rand((), generator=generator).item()) >= 0.025:
                continue
            duration = int(torch.randint(12, 25, (1,), generator=generator).item())
            amplitude = 0.5 + 0.5 * float(torch.rand((), generator=generator).item())
            length = min(duration, num_days - day)
            age = torch.arange(length, dtype=events.dtype)
            shape = amplitude * torch.exp(-3.0 * age / duration)
            events[day : day + length, node] += shape[:, None] * target_scale
    return events
