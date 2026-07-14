import pytest
import torch

from RiverLagNet.models.riverlag_net import RiverLagNet


@pytest.mark.parametrize("graph_variant", ["directed", "undirected", "shuffled", "no_graph"])
@pytest.mark.parametrize("lag_mode", ["no_lag", "fixed_lag", "learned_lag"])
def test_riverlagnet_preserves_output_axes_for_graph_and_lag_ablations(
    graph_variant: str, lag_mode: str
) -> None:
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        graph_variant=graph_variant,
        lag_mode=lag_mode,
    )
    output = model(
        x=torch.randn(2, 8, 4, 3),
        x_mask=torch.ones(2, 8, 4, 3, dtype=torch.bool),
        x_quality=torch.ones(2, 8, 4, 3),
        static=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]]),
        edge_attr=torch.tensor([[4.0, 0.1, 1.0], [3.0, 0.2, 2.0], [2.0, 0.3, 1.0]]),
        time_features=torch.randn(2, 8, 4),
    )
    assert output.shape == (2, 6, 4, 3)
    if graph_variant == "no_graph":
        assert model.attention_weights is None
    else:
        assert model.attention_weights is not None
        assert model.attention_weights.shape[:2] == (2, 6)


def test_riverlagnet_graph_path_equals_local_path_when_graph_is_empty() -> None:
    torch.manual_seed(5)
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        graph_variant="directed",
        lag_mode="learned_lag",
    ).eval()
    inputs = {
        "x": torch.randn(2, 8, 4, 3),
        "x_mask": torch.ones(2, 8, 4, 3, dtype=torch.bool),
        "x_quality": torch.ones(2, 8, 4, 3),
        "static": torch.randn(4, 2),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "edge_attr": torch.empty(0, 3),
        "time_features": torch.randn(2, 8, 4),
    }

    graph_output = model(**inputs)
    model.graph_variant = "no_graph"
    local_output = model(**inputs)

    assert torch.equal(graph_output, local_output)


def test_upstream_residual_training_zero_starts_and_freezes_local_backbone() -> None:
    torch.manual_seed(7)
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        graph_variant="directed",
        lag_mode="learned_lag",
    ).eval()
    inputs = {
        "x": torch.randn(2, 8, 4, 3),
        "x_mask": torch.ones(2, 8, 4, 3, dtype=torch.bool),
        "x_quality": torch.ones(2, 8, 4, 3),
        "static": torch.randn(4, 2),
        "edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
        "edge_attr": torch.tensor(
            [[4.0, 0.1, 1.0], [3.0, 0.2, 2.0], [2.0, 0.3, 1.0]]
        ),
        "time_features": torch.randn(2, 8, 4),
    }

    model.configure_upstream_residual_training(gate_bias=-1.0)
    directed_output = model(**inputs)
    model.graph_variant = "no_graph"
    local_output = model(**inputs)

    assert torch.equal(directed_output, local_output)
    assert all(not parameter.requires_grad for parameter in model.input_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.temporal_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert any(parameter.requires_grad for parameter in model.message_passing.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.parameters())
    assert any(parameter.requires_grad for parameter in model.upstream_decoder.parameters())


def test_linear_horizon_gate_strictly_nests_existing_graph_model() -> None:
    torch.manual_seed(19)
    base = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        graph_variant="directed",
        lag_mode="no_lag",
    ).eval()
    calibrated = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        graph_variant="directed",
        lag_mode="no_lag",
        horizon_gate_mode="linear",
    ).eval()
    incompatible = calibrated.load_state_dict(base.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "horizon_gate.offset",
        "horizon_gate.slope",
        "horizon_gate.normalized_lead",
    }
    inputs = {
        "x": torch.randn(2, 8, 4, 3),
        "x_mask": torch.ones(2, 8, 4, 3, dtype=torch.bool),
        "x_quality": torch.ones(2, 8, 4, 3),
        "static": torch.randn(4, 2),
        "edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
        "edge_attr": torch.tensor(
            [[4.0, 0.1, 1.0], [3.0, 0.2, 2.0], [2.0, 0.3, 1.0]]
        ),
        "time_features": torch.randn(2, 8, 4),
    }

    assert torch.equal(calibrated(**inputs), base(**inputs))


def test_horizon_calibration_trains_only_two_gate_parameters() -> None:
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        horizon_gate_mode="linear",
    )

    model.configure_horizon_calibration_training()
    model.train()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable == ["horizon_gate.offset", "horizon_gate.slope"]
    assert not model.message_passing.training
    assert model.horizon_gate is not None and model.horizon_gate.training


def test_lag_refinement_trains_only_lag_scale_and_global_bias() -> None:
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=3,
        lag_mode="learned_lag",
        lag_bias_mode="global",
        horizon_gate_mode="linear",
    )

    model.configure_lag_refinement_training()
    model.train()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable == [
        "message_passing.lag_residual_scale",
        "message_passing.lag_offset_bias",
    ]
    assert not model.message_passing.training
    assert model.horizon_gate is not None and not model.horizon_gate.training


def test_trajectory_mode_preserves_output_shape_and_headwater_prediction() -> None:
    torch.manual_seed(29)
    model = RiverLagNet(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        edge_dim=3,
        hidden_dim=8,
        output_window=6,
        max_lag=4,
        graph_variant="directed",
        lag_mode="no_lag",
        propagation_mode="trajectory",
        trajectory_steps=3,
    ).eval()
    inputs = {
        "x": torch.randn(2, 8, 4, 3),
        "x_mask": torch.ones(2, 8, 4, 3, dtype=torch.bool),
        "x_quality": torch.ones(2, 8, 4, 3),
        "static": torch.randn(4, 2),
        "edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
        "edge_attr": torch.tensor(
            [[4.0, 0.1, 1.0], [3.0, 0.2, 2.0], [2.0, 0.3, 1.0]]
        ),
        "time_features": torch.randn(2, 8, 4),
    }

    graph_output = model(**inputs)
    model.graph_variant = "no_graph"
    local_output = model(**inputs)

    assert graph_output.shape == (2, 6, 4, 3)
    assert torch.equal(graph_output[:, :, 0], local_output[:, :, 0])
