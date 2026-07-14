from __future__ import annotations

import torch

from RiverLagNet.models.graph_cross_attention import (
    EdgeLagHorizonSparseAttention,
    TransformerGraphCrossFusion,
    sparsemax,
)


def test_sparsemax_normalizes_and_produces_exact_zeros() -> None:
    weights = sparsemax(torch.tensor([[3.0, 1.0, -2.0]]))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    assert weights[0, 0] == 1.0
    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 0.0


def test_edge_lag_horizon_attention_masks_unobservable_lags_and_normalizes() -> None:
    module = EdgeLagHorizonSparseAttention(
        hidden_dim=8, edge_dim=2, num_heads=2, max_lag=4
    )
    history = torch.randn(2, 7, 3, 8)
    queries = torch.randn(2, 4, 3, 8)
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_attr = torch.tensor([[0.0, 2.0], [0.0, 3.0]])

    contexts, weights = module(history, queries, queries, edge_index, edge_attr)

    assert contexts.shape == (2, 4, 3, 2, 4)
    assert weights.shape == (2, 4, 2, 5, 2)
    for horizon in range(4):
        normalized = weights[:, horizon].sum(dim=(1, 2))
        assert torch.allclose(normalized, torch.ones_like(normalized), atol=1e-5)
    assert torch.equal(contexts[:, :, :2], torch.zeros_like(contexts[:, :, :2]))


def test_transformer_graph_cross_fusion_keeps_headwaters_exactly_zero() -> None:
    fusion = TransformerGraphCrossFusion(hidden_dim=8, num_heads=2)
    local = torch.randn(2, 3, 4, 8)
    graph = torch.randn(2, 3, 4, 2, 4)
    graph[:, :, 0] = 0.0

    fused, weights = fusion(local, graph)

    assert fused.shape == local.shape
    assert weights.shape == (2, 3, 4, 2)
    assert torch.equal(fused[:, :, 0], torch.zeros_like(fused[:, :, 0]))
    assert torch.equal(weights[:, :, 0], torch.zeros_like(weights[:, :, 0]))


def test_edge_lag_horizon_attention_fast_path_normalizes_directed_chain() -> None:
    module = EdgeLagHorizonSparseAttention(
        hidden_dim=8, edge_dim=2, num_heads=2, max_lag=3
    )
    history = torch.randn(1, 6, 4, 8)
    queries = torch.randn(1, 3, 4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 1.0]])

    contexts, weights = module(history, queries, queries, edge_index, edge_attr)

    assert contexts.shape == (1, 3, 4, 2, 4)
    assert torch.allclose(
        weights.sum(dim=3), torch.ones(1, 3, 3, 2), atol=1e-5
    )
    assert torch.equal(contexts[:, :, 0], torch.zeros_like(contexts[:, :, 0]))


def test_edge_lag_attention_bridges_observed_history_and_predicted_future() -> None:
    module = EdgeLagHorizonSparseAttention(
        hidden_dim=2, edge_dim=1, num_heads=1, max_lag=2
    )
    history = torch.zeros(1, 3, 2, 2)
    future = torch.zeros(1, 2, 2, 2)
    history[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    future[0, :, 0, 0] = torch.tensor([10.0, 11.0])
    with torch.no_grad():
        module.value_projection.weight.copy_(torch.eye(2))
    source = torch.tensor([[0], [1]])
    edge_attr = torch.tensor([[1.0]])

    _, candidate_values, use_future, _ = module._candidate_states(
        history, future, source
    )

    assert candidate_values[0, 0, 0, :, 0, 0].tolist() == [10.0, 3.0, 2.0]
    assert candidate_values[0, 1, 0, :, 0, 0].tolist() == [11.0, 10.0, 3.0]
    assert use_future.tolist() == [[True, False, False], [True, True, False]]
