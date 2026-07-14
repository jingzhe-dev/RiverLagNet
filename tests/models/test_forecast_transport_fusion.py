from __future__ import annotations

import torch

from RiverLagNet.models.forecast_transport_fusion import (
    TargetConditionedForecastTransport,
)


def test_target_transport_bridges_observed_history_and_local_forecast() -> None:
    module = TargetConditionedForecastTransport(
        hidden_dim=4, target_dim=3, num_heads=2
    )
    history = torch.zeros(1, 3, 2, 3)
    history_mask = torch.ones_like(history, dtype=torch.bool)
    local = torch.zeros(1, 2, 2, 3)
    history[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    local[0, :, 0, 0] = torch.tensor([10.0, 11.0])
    edges = torch.tensor([[0], [1]])

    candidates, valid = module._candidate_targets(
        history, history_mask, local, edges, lag_count=3
    )

    assert candidates[0, 0, 0, :, 0].tolist() == [10.0, 3.0, 2.0]
    assert candidates[0, 1, 0, :, 0].tolist() == [11.0, 10.0, 3.0]
    assert valid.all()


def test_target_transport_is_zero_started_and_keeps_headwater_zero() -> None:
    module = TargetConditionedForecastTransport(
        hidden_dim=4, target_dim=3, num_heads=2
    )
    history = torch.randn(1, 5, 2, 3)
    history_mask = torch.ones_like(history, dtype=torch.bool)
    local = torch.randn(1, 2, 2, 3)
    local_context = torch.randn(1, 2, 2, 4)
    graph_context = torch.randn(1, 2, 2, 4)
    graph_context[:, :, 0] = 0.0
    edges = torch.tensor([[0], [1]])
    attention = torch.zeros(1, 2, 1, 3, 2)
    attention[:, :, 0, 0] = 0.5

    correction = module(
        history,
        history_mask,
        local,
        local_context,
        graph_context,
        edges,
        attention,
    )

    assert torch.equal(correction, torch.zeros_like(correction))
    assert module.target_head_weights is not None
    assert torch.allclose(
        module.target_head_weights.sum(dim=-1), torch.ones(3)
    )

    with torch.no_grad():
        module.residual[-1].weight.fill_(0.1)
    correction = module(
        history,
        history_mask,
        local,
        local_context,
        graph_context,
        edges,
        attention,
    )

    assert torch.equal(correction[:, :, 0], torch.zeros_like(correction[:, :, 0]))
    assert torch.count_nonzero(correction[:, :, 1]) > 0


def test_target_transport_inherits_bfloat16_compute_dtype() -> None:
    module = TargetConditionedForecastTransport(
        hidden_dim=4, target_dim=3, num_heads=2
    )
    history = torch.randn(1, 5, 2, 3)
    history_mask = torch.ones_like(history, dtype=torch.bool)
    local = torch.randn(1, 2, 2, 3, dtype=torch.bfloat16)
    local_context = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    graph_context = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    edges = torch.tensor([[0], [1]])
    # Sparse attention may remain float32 under AMP while decoder outputs are bf16.
    attention = torch.zeros(1, 2, 1, 3, 2, dtype=torch.float32)
    attention[:, :, 0, 0] = 0.5

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        correction = module(
            history,
            history_mask,
            local,
            local_context,
            graph_context,
            edges,
            attention,
        )

    assert correction.dtype == torch.bfloat16
    assert module.upstream_forecast is not None
    assert module.upstream_forecast.dtype == torch.bfloat16
