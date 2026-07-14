from __future__ import annotations

import torch

from RiverLagNet.models.history_propagation import DirectedLaggedHistoryPropagation


def _module(steps: int = 1) -> DirectedLaggedHistoryPropagation:
    return DirectedLaggedHistoryPropagation(
        hidden_dim=1, edge_dim=1, max_lag=4, steps=steps
    )


def test_history_propagation_aligns_lags_without_repeating_window_boundary() -> None:
    module = _module()
    states = torch.tensor([[[[1.0], [9.0]], [[2.0], [9.0]], [[3.0], [9.0]]]])
    edge_index = torch.tensor([[0], [1]])
    edge_attr = torch.tensor([[2.0]])

    aligned, valid = module.aligned_source_states(states, edge_index, edge_attr)

    assert aligned[0, :, 0, 0].tolist() == [1.0, 1.0, 1.0]
    assert valid[:, 0].tolist() == [False, False, True]


def test_history_propagation_zero_starts_and_normalizes_incoming_edges() -> None:
    module = DirectedLaggedHistoryPropagation(
        hidden_dim=2, edge_dim=1, max_lag=2, steps=3
    )
    states = torch.randn(2, 5, 3, 2)
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_attr = torch.tensor([[1.0], [1.0]])

    output, routing = module(states, edge_index, edge_attr)

    assert torch.equal(output, states)
    assert routing.shape == (2, 5, 2, 1)
    assert torch.allclose(routing[:, 1:].sum(dim=2), torch.ones(2, 4, 1))
    assert torch.equal(routing[:, 0], torch.zeros_like(routing[:, 0]))


def test_history_propagation_is_directional_and_reaches_two_hops() -> None:
    module = _module(steps=2)
    with torch.no_grad():
        module.message_projection.weight.fill_(1.0)
        module.edge_gate[0].weight.zero_()
        module.edge_gate[0].bias.zero_()
        module.edge_gate[2].weight.zero_()
        module.edge_gate[2].bias.fill_(10.0)
        module.update[0].weight.zero_()
        module.update[0].bias.zero_()
        module.update[0].weight[0, 2] = 1.0
        module.update[-1].weight.fill_(1.0)
    states = torch.zeros(1, 4, 3, 1)
    states[:, :, 0] = 2.0
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.0], [0.0]])

    output, _ = module(states, edge_index, edge_attr)

    assert torch.equal(output[:, :, 0], states[:, :, 0])
    assert torch.all(output[:, :, 1] > 0)
    assert torch.all(output[:, :, 2] > 0)


def test_history_propagation_preserves_mixed_precision_dtype() -> None:
    module = _module().half()
    states = torch.zeros(1, 3, 2, 1, dtype=torch.float16)

    output, routing = module(
        states,
        torch.tensor([[0], [1]]),
        torch.tensor([[1.0]], dtype=torch.float16),
    )

    assert output.dtype == torch.float16
    assert routing.dtype == torch.float16
