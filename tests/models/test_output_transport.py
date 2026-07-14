from __future__ import annotations

import torch

from RiverLagNet.models.output_transport import DirectedLaggedOutputTransport


def _module(steps: int = 1) -> DirectedLaggedOutputTransport:
    return DirectedLaggedOutputTransport(
        target_dim=1, edge_dim=1, max_lag=4, steps=steps, hidden_dim=8
    )


def test_output_transport_aligns_observed_history_and_predicted_future() -> None:
    module = _module()
    future = torch.tensor([[[[10.0], [20.0]], [[11.0], [21.0]], [[12.0], [22.0]]]])
    history = torch.tensor([[[[1.0], [5.0]], [[2.0], [6.0]], [[3.0], [7.0]]]])
    mask = torch.ones_like(history, dtype=torch.bool)
    edge_index = torch.tensor([[0], [1]])
    edge_attr = torch.tensor([[2.0]])

    aligned, available = module.aligned_source_values(
        future, history, mask, edge_index, edge_attr
    )

    assert aligned[0, :, 0, 0].tolist() == [2.0, 3.0, 10.0]
    assert available.all()


def test_output_transport_zero_starts_and_preserves_headwater() -> None:
    module = _module(steps=3)
    local = torch.randn(2, 3, 3, 1)
    history = torch.randn(2, 5, 3, 1)
    mask = torch.ones_like(history, dtype=torch.bool)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[1.0], [1.0]])

    output, routing = module(local, history, mask, edge_index, edge_attr)

    assert torch.equal(output, local)
    assert torch.equal(output[:, :, 0], local[:, :, 0])
    assert routing.shape == (2, 3, 2, 1)
    assert torch.allclose(routing, torch.ones_like(routing))


def test_output_transport_propagates_only_downstream_after_training_signal() -> None:
    module = _module(steps=2)
    with torch.no_grad():
        module.message[-1].weight.fill_(0.1)
    local = torch.zeros(1, 3, 3, 1)
    history = torch.zeros(1, 5, 3, 1)
    history[:, :, 0] = 2.0
    mask = torch.ones_like(history, dtype=torch.bool)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[3.0], [1.0]])

    output, _ = module(local, history, mask, edge_index, edge_attr)

    assert torch.equal(output[:, :, 0], local[:, :, 0])
    assert not torch.equal(output[:, :, 1], local[:, :, 1])
    assert not torch.equal(output[:, :, 2], local[:, :, 2])


def test_output_transport_preserves_mixed_precision_dtype() -> None:
    module = _module().half()
    local = torch.zeros(1, 2, 2, 1, dtype=torch.float16)
    history = torch.zeros(1, 3, 2, 1, dtype=torch.float16)
    mask = torch.ones_like(history, dtype=torch.bool)

    output, routing = module(
        local,
        history,
        mask,
        torch.tensor([[0], [1]]),
        torch.tensor([[1.0]], dtype=torch.float16),
    )

    assert output.dtype == torch.float16
    assert routing.dtype == torch.float16
