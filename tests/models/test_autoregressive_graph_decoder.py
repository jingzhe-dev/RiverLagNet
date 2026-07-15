from __future__ import annotations

import pytest
import torch

from RiverLagNet.models.autoregressive_graph_decoder import (
    DirectedAutoregressiveGraphDecoder,
    RecursiveCausalEdgeLagAttention,
)


def test_recursive_attention_uses_only_history_and_previous_predictions() -> None:
    attention = RecursiveCausalEdgeLagAttention(
        hidden_dim=4,
        edge_dim=2,
        num_heads=2,
        max_lag=4,
        max_path_hops=2,
    )
    history = torch.zeros(1, 4, 3, 4)
    future = torch.zeros(1, 2, 3, 4)
    for time in range(4):
        for node in range(3):
            history[:, time, node] = 100 * time + 10 * node
    for lead in range(2):
        for node in range(3):
            future[:, lead, node] = 1000 + 100 * lead + 10 * node

    candidates, feasible = attention._candidate_states(
        history, future, torch.tensor([0])
    )

    assert feasible.tolist() == [True, True, True, True]
    assert torch.equal(candidates[0, 0, 0], future[0, 1, 0])
    assert torch.equal(candidates[0, 0, 1], future[0, 0, 0])
    assert torch.equal(candidates[0, 0, 2], history[0, -1, 0])
    assert torch.equal(candidates[0, 0, 3], history[0, -2, 0])


def test_recursive_attention_jointly_normalizes_incoming_edges_and_lags() -> None:
    torch.manual_seed(3)
    attention = RecursiveCausalEdgeLagAttention(
        hidden_dim=8,
        edge_dim=2,
        num_heads=2,
        max_lag=4,
        max_path_hops=2,
    )
    history = torch.randn(2, 5, 4, 8)
    future = torch.empty(2, 0, 4, 8)
    query = torch.randn(2, 4, 8)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    edge_attr = torch.tensor([[0.2, 1.0], [0.3, 2.0], [0.4, 1.0]])

    contexts, weights = attention(
        history,
        future,
        query,
        edge_index,
        edge_attr,
        torch.tensor([1, 1, 1]),
    )

    assert contexts.shape == (2, 4, 2, 4)
    assert weights.shape == (2, 3, 4, 2)
    assert torch.equal(contexts[:, 0], torch.zeros_like(contexts[:, 0]))
    for destination in (1, 2):
        edge_ids = torch.nonzero(
            edge_index[1] == destination, as_tuple=False
        ).flatten()
        total = weights[:, edge_ids].sum(dim=(1, 2))
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_innovation_values_remove_destination_background_state() -> None:
    attention = RecursiveCausalEdgeLagAttention(
        hidden_dim=4,
        edge_dim=2,
        num_heads=2,
        max_lag=3,
        max_path_hops=2,
        value_mode="innovation",
    )
    candidates = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    destination_query = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
    )
    destination = torch.tensor([1, 2])

    messages = attention._message_states(
        candidates, destination_query, destination
    )

    expected = candidates - destination_query[:, destination, None]
    assert torch.equal(messages, expected)


def _decoder() -> DirectedAutoregressiveGraphDecoder:
    return DirectedAutoregressiveGraphDecoder(
        hidden_dim=8,
        edge_dim=2,
        output_window=4,
        target_dim=3,
        num_heads=2,
        max_lag=4,
        max_path_hops=2,
        dropout=0.0,
    ).eval()


def _decoder_inputs() -> dict[str, torch.Tensor]:
    return {
        "history_states": torch.randn(2, 6, 4, 8),
        "initial_state": torch.randn(2, 4, 8),
        "previous_values": torch.randn(2, 4, 3),
        "previous_mask": torch.ones(2, 4, 3, dtype=torch.bool),
    }


def test_recurrent_graph_decoder_zero_starts_at_exact_no_graph_forecast() -> None:
    torch.manual_seed(5)
    decoder = _decoder()
    inputs = _decoder_inputs()
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]])
    edge_attr = torch.tensor([[0.1, 1.0], [0.2, 2.0], [0.3, 1.0]])

    graph, routing, fusion = decoder(
        **inputs,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_hops=torch.ones(3, dtype=torch.long),
    )
    local, no_routing, no_fusion = decoder(**inputs)

    assert graph.shape == (2, 4, 4, 3)
    assert torch.equal(graph, local)
    assert routing is not None and routing.shape == (2, 4, 3, 4, 2)
    assert fusion is not None and fusion.shape == (2, 4, 4, 2)
    assert no_routing is None and no_fusion is None


def test_recurrent_fusion_changes_only_nodes_with_upstream_paths() -> None:
    torch.manual_seed(7)
    decoder = _decoder()
    inputs = _decoder_inputs()
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.1, 1.0], [0.2, 1.0]])
    with torch.no_grad():
        decoder.fusion.message_projection.weight.copy_(torch.eye(8))
        decoder.fusion.modulation.weight.zero_()
        decoder.fusion.modulation.weight[8:].copy_(torch.eye(8))
        decoder.fusion.output_projection.weight.copy_(torch.eye(8))

    graph, _, _ = decoder(
        **inputs,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_hops=torch.ones(2, dtype=torch.long),
    )
    local, _, _ = decoder(**inputs)

    assert torch.equal(graph[:, :, 0], local[:, :, 0])
    assert not torch.equal(graph[:, :, 1:3], local[:, :, 1:3])
    assert torch.equal(graph[:, :, 3], local[:, :, 3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_recurrent_decoder_supports_cuda_bfloat16_autocast() -> None:
    decoder = _decoder().cuda()
    inputs = {key: value.cuda() for key, value in _decoder_inputs().items()}
    edge_index = torch.tensor([[0, 1], [1, 2]], device="cuda")
    edge_attr = torch.tensor(
        [[0.1, 1.0], [0.2, 1.0]], device="cuda", dtype=torch.float32
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, routing, fusion = decoder(
            **inputs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_hops=torch.ones(2, dtype=torch.long, device="cuda"),
        )

    assert output.shape == (2, 4, 4, 3)
    assert routing is not None and torch.isfinite(routing).all()
    assert fusion is not None and torch.isfinite(fusion).all()
