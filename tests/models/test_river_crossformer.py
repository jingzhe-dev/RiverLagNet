from __future__ import annotations

import torch

from RiverLagNet.models.river_crossformer import (
    RiverGraphCrossFormer,
    latest_observed_targets,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, 10, 4, 5),
        "x_mask": torch.ones(2, 10, 4, 5, dtype=torch.bool),
        "x_quality": torch.ones(2, 10, 4, 5),
        "static": torch.randn(4, 2),
        "edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
        "edge_attr": torch.tensor(
            [[0.1, 1.0], [0.2, 2.0], [0.3, 1.0]], dtype=torch.float32
        ),
        "time_features": torch.randn(2, 10, 4),
    }


def _model(
    graph_variant: str = "directed", max_path_hops: int = 1
) -> RiverGraphCrossFormer:
    return RiverGraphCrossFormer(
        value_dim=5,
        static_dim=2,
        time_dim=4,
        edge_dim=2,
        hidden_dim=8,
        output_window=4,
        target_dim=3,
        max_lag=4,
        graph_variant=graph_variant,
        transformer_heads=2,
        transformer_layers=1,
        graph_heads=2,
        history_steps=2,
        max_path_hops=max_path_hops,
        dropout=0.0,
    ).eval()


def test_crossformer_preserves_output_axes_and_attention_contract() -> None:
    model = _model()
    output = model(**_inputs())

    assert output.shape == (2, 4, 4, 3)
    assert model.attention_weights is not None
    assert model.attention_weights.shape == (2, 4, 3, 5, 2)
    assert model.fusion_weights is not None
    assert model.fusion_weights.shape == (2, 4, 4, 2)


def test_crossformer_zero_start_equals_same_local_transformer_without_graph() -> None:
    torch.manual_seed(17)
    graph = _model("directed")
    local = _model("no_graph")
    local.load_state_dict(graph.state_dict(), strict=True)
    inputs = _inputs()

    assert torch.equal(graph(**inputs), local(**inputs))


def test_crossformer_headwater_stays_local_after_graph_branch_is_activated() -> None:
    torch.manual_seed(23)
    graph = _model("directed")
    local = _model("no_graph")
    local.load_state_dict(graph.state_dict(), strict=True)
    with torch.no_grad():
        graph.history_diffusion.update[-1].weight.fill_(0.02)
        for head in graph.upstream_decoder.heads:
            head.weight.fill_(0.05)
    inputs = _inputs()
    graph_output = graph(**inputs)
    local_output = local(**inputs)

    assert torch.equal(graph_output[:, :, 0], local_output[:, :, 0])
    assert not torch.equal(graph_output[:, :, 1:], local_output[:, :, 1:])


def test_crossformer_residual_training_freezes_local_transformer() -> None:
    model = _model("directed")

    model.configure_upstream_residual_training()
    model.train()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable
    assert all(
        name.startswith(
            (
                "history_diffusion.",
                "graph_attention.",
                "cross_fusion.",
                "forecast_transport.",
                "upstream_decoder.",
            )
        )
        for name in trainable
    )
    assert not model.input_encoder.training
    assert not model.temporal_transformer.training
    assert not model.local_decoder.training


def test_crossformer_expands_directed_ancestor_paths_for_attention_only() -> None:
    model = _model("directed", max_path_hops=2)
    inputs = _inputs()

    expanded_index, expanded_attr, path_hops = model._expanded_graph(
        inputs["edge_index"], inputs["edge_attr"]
    )

    assert expanded_index.shape == (2, 5)
    assert expanded_attr.shape == (5, 2)
    assert path_hops.tolist().count(1) == 3
    assert path_hops.tolist().count(2) == 2
    path_lookup = {
        (int(source), int(destination), int(hops)): float(attributes[-1])
        for source, destination, hops, attributes in zip(
            expanded_index[0],
            expanded_index[1],
            path_hops,
            expanded_attr,
            strict=True,
        )
    }
    assert path_lookup[(0, 2, 2)] == 3.0
    assert path_lookup[(0, 3, 2)] == 2.0

    output = model(**inputs)
    assert output.shape == (2, 4, 4, 3)
    assert model.attention_weights is not None
    assert model.attention_weights.shape == (2, 4, 5, 5, 2)


def _recurrent_model(
    graph_variant: str = "directed",
    *,
    routing_mode: str = "attention",
    history_graph: bool = False,
) -> RiverGraphCrossFormer:
    return RiverGraphCrossFormer(
        value_dim=5,
        static_dim=2,
        time_dim=4,
        edge_dim=2,
        hidden_dim=8,
        output_window=4,
        target_dim=3,
        max_lag=4,
        graph_variant=graph_variant,
        transformer_heads=2,
        transformer_layers=1,
        graph_heads=2,
        history_steps=2,
        max_path_hops=1,
        fusion_mode="recurrent",
        recurrent_routing_mode=routing_mode,
        recurrent_history_graph=history_graph,
        dropout=0.0,
    ).eval()


def test_latest_observed_targets_respects_per_channel_missingness() -> None:
    x = torch.tensor(
        [
            [
                [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
                [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
                [[5.0, 50.0, 500.0], [6.0, 60.0, 600.0]],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [
                [[True, False, False], [False, False, False]],
                [[False, True, False], [True, False, False]],
                [[True, False, False], [False, False, True]],
            ]
        ]
    )

    values, available = latest_observed_targets(x, mask, target_dim=3)

    assert values.tolist() == [[[5.0, 30.0, 0.0], [4.0, 0.0, 600.0]]]
    assert available.tolist() == [
        [[True, True, False], [True, False, True]]
    ]


def test_recurrent_crossformer_zero_starts_at_same_no_graph_model() -> None:
    torch.manual_seed(31)
    graph = _recurrent_model("directed")
    local = _recurrent_model("no_graph")
    local.load_state_dict(graph.state_dict(), strict=True)
    inputs = _inputs()

    graph_output = graph(**inputs)
    local_output = local(**inputs)

    assert graph_output.shape == (2, 4, 4, 3)
    assert torch.equal(graph_output, local_output)
    assert graph.attention_weights is not None
    assert graph.attention_weights.shape == (2, 4, 3, 4, 2)
    assert graph.fusion_weights is not None
    assert graph.fusion_weights.shape == (2, 4, 4, 2)


def test_dual_stage_recurrent_model_uses_direct_lags_and_history_graph() -> None:
    torch.manual_seed(37)
    graph = _recurrent_model(
        "directed", routing_mode="fixed_direct", history_graph=True
    )
    local = _recurrent_model(
        "no_graph", routing_mode="fixed_direct", history_graph=True
    )
    local.load_state_dict(graph.state_dict(), strict=True)
    inputs = _inputs()

    initial_graph = graph(**inputs)
    initial_local = local(**inputs)

    assert torch.equal(initial_graph, initial_local)
    assert graph.attention_weights is not None
    assert graph.attention_weights.shape == (2, 4, 3, 1, 2)
    assert graph.history_routing is not None

    with torch.no_grad():
        graph.history_diffusion.update[-1].weight.fill_(0.03)
    activated_graph = graph(**inputs)
    activated_local = local(**inputs)
    assert torch.equal(activated_graph[:, :, 0], activated_local[:, :, 0])
    assert not torch.equal(activated_graph[:, :, 1:], activated_local[:, :, 1:])


def test_recurrent_residual_training_only_unfreezes_attention_and_fusion() -> None:
    model = _recurrent_model("directed")

    model.configure_upstream_residual_training()
    model.train()
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert trainable
    assert all(
        name.startswith(
            (
                "recurrent_decoder.attention.",
                "recurrent_decoder.fusion.",
            )
        )
        for name in trainable
    )
    assert model.recurrent_decoder is not None
    assert not model.recurrent_decoder.local_transition.training
    assert model.recurrent_decoder.attention.training
    assert model.recurrent_decoder.fusion.training
