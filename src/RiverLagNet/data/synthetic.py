"""Deterministic synthetic river-network observations for engineering tests."""

from __future__ import annotations

import math

import torch

from .schema import RiverGraph, TimeSeriesData


def generate_synthetic_river_data(
    num_days: int = 260,
    num_nodes: int = 8,
    num_variables: int = 3,
    missing_rate: float = 0.08,
    seed: int = 42,
) -> TimeSeriesData:
    """Generate a directed tree whose downstream signals contain lagged upstream input."""
    if num_days < 2 or num_nodes < 2 or num_variables < 3:
        raise ValueError("synthetic data needs >=2 days, >=2 nodes, and >=3 variables")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    destinations = torch.arange(1, num_nodes, dtype=torch.long)
    sources = torch.tensor(
        [int(torch.randint(0, node, (1,), generator=generator)) for node in range(1, num_nodes)],
        dtype=torch.long,
    )
    edge_index = torch.stack((sources, destinations))
    distance = 10.0 + 40.0 * torch.rand(num_nodes - 1, generator=generator)
    slope = 0.001 + 0.02 * torch.rand(num_nodes - 1, generator=generator)
    travel_lag = torch.randint(1, 8, (num_nodes - 1,), generator=generator).float()
    edge_attr = torch.stack((distance, slope, travel_lag), dim=-1)
    position = torch.linspace(0.0, 1.0, num_nodes)
    drainage = torch.linspace(0.2, 1.0, num_nodes)
    elevation = 1.0 - position
    static = torch.stack((position, drainage, elevation), dim=-1)

    values = torch.zeros(num_days, num_nodes, num_variables)
    phases = 2.0 * math.pi * torch.rand(num_nodes, num_variables, generator=generator)
    offsets = 0.5 + 1.5 * torch.rand(num_nodes, num_variables, generator=generator)
    for day in range(num_days):
        seasonal = torch.sin(torch.tensor(2.0 * math.pi * day / 30.0) + phases)
        innovations = 0.05 * torch.randn(num_nodes, num_variables, generator=generator)
        previous = values[day - 1] if day else offsets
        values[day] = 0.65 * previous + 0.25 * offsets + 0.1 * seasonal + innovations
        for edge, (source, destination) in enumerate(edge_index.t().tolist()):
            lag = int(travel_lag[edge].item())
            if day >= lag:
                values[day, destination] += 0.15 * values[day - lag, source]
    values = values.clamp_min(0.0)
    observed = torch.rand(values.shape, generator=generator) >= missing_rate
    quality = observed.to(values.dtype) * (
        0.8 + 0.2 * torch.rand(values.shape, generator=generator)
    )
    data = TimeSeriesData(values, observed, quality, RiverGraph(edge_index, edge_attr, static))
    data.validate()
    return data
