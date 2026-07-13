import torch

from RiverLagNet.models.baselines import PersistenceModel, StationGRU, StaticDirectedGAT


def _inputs() -> dict[str, torch.Tensor]:
    batch, history, nodes, variables = 2, 10, 4, 3
    return {
        "x": torch.randn(batch, history, nodes, variables),
        "x_mask": torch.ones(batch, history, nodes, variables, dtype=torch.bool),
        "x_quality": torch.ones(batch, history, nodes, variables),
        "static": torch.randn(nodes, 2),
        "edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
        "edge_attr": torch.randn(3, 3),
        "time_features": torch.randn(batch, history, 4),
    }


def test_persistence_repeats_each_nodes_last_observed_targets() -> None:
    inputs = _inputs()
    inputs["x_mask"][:, -1, 0, 0] = False
    inputs["x"][:, -2, 0, 0] = 7.0
    model = PersistenceModel(output_window=5, target_dim=3)
    output = model(**inputs)
    assert output.shape == (2, 5, 4, 3)
    assert torch.all(output[:, :, 0, 0] == 7.0)


def test_trainable_baselines_return_common_output_shape() -> None:
    inputs = _inputs()
    kwargs = dict(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=12,
        output_window=5,
        target_dim=3,
    )
    station = StationGRU(**kwargs)
    gat = StaticDirectedGAT(**kwargs)
    assert station(**inputs).shape == (2, 5, 4, 3)
    assert gat(**inputs).shape == (2, 5, 4, 3)
