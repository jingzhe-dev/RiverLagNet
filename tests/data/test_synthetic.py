import torch

from RiverLagNet.data.synthetic import generate_synthetic_river_data


def test_synthetic_data_is_deterministic_directed_and_masked() -> None:
    first = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)
    second = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)
    first.validate()
    assert first.values.shape == (160, 6, 3)
    assert first.graph.static.shape == (6, 3)
    assert first.graph.edge_attr.shape[1] == 3
    assert torch.equal(first.graph.edge_index, second.graph.edge_index)
    assert torch.allclose(first.values, second.values)
    assert torch.all(first.graph.edge_index[0] < first.graph.edge_index[1])
    assert (~first.observed).any()
