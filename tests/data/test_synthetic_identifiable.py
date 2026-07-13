import torch

from RiverLagNet.data.synthetic import generate_synthetic_river_data
from RiverLagNet.data.synthetic_identifiable import (
    generate_identifiable_synthetic_scenario,
    route_pollutant_events,
)


def test_identifiable_scenario_is_deterministic_for_one_seed() -> None:
    first = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=11)
    second = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=11)
    different = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=12)

    assert torch.equal(first.data.graph.edge_index, second.data.graph.edge_index)
    assert torch.allclose(first.data.values, second.data.values)
    assert not torch.allclose(first.data.values, different.data.values)


def test_identifiable_scenario_reconstructs_values_and_truth_contract() -> None:
    scenario = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=11)

    scenario.data.validate()
    reconstructed = (
        scenario.local_background + scenario.local_events + scenario.routed_load
    ).clamp_min(0.0)
    assert torch.allclose(scenario.data.values, reconstructed)
    assert torch.equal(
        scenario.true_lag_days,
        scenario.data.graph.edge_attr[:, -1].round().long(),
    )
    assert torch.all(scenario.data.graph.edge_index[0] < scenario.data.graph.edge_index[1])
    assert torch.isfinite(scenario.data.values).all()
    assert (scenario.data.values >= 0.0).all()


def test_route_pollutant_events_applies_exact_lags_and_multihop_attenuation() -> None:
    events = torch.zeros(7, 3, 3)
    events[0, 0] = 1.0
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    attrs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

    total, routed = route_pollutant_events(events, edges, attrs)

    first = 2.0 * torch.tensor([0.85, 1.0, 0.75])
    assert torch.allclose(routed[1, 1], first)
    assert torch.allclose(routed[3, 2], first.square())
    assert torch.count_nonzero(routed[:1, 1]) == 0
    assert torch.count_nonzero(routed[:3, 2]) == 0
    assert torch.allclose(total, events + routed)


def test_legacy_generator_remains_deterministic() -> None:
    first = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)
    second = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)

    assert torch.equal(first.graph.edge_index, second.graph.edge_index)
    assert torch.allclose(first.values, second.values)
