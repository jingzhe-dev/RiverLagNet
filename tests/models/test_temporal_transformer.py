from __future__ import annotations

import pytest
import torch

from RiverLagNet.models.temporal_transformer import NodeTemporalTransformer


def test_temporal_transformer_preserves_time_and_node_axes() -> None:
    module = NodeTemporalTransformer(
        hidden_dim=8, num_heads=2, num_layers=1, dropout=0.0, max_history=12
    ).eval()
    encoded = torch.randn(3, 10, 4, 8)

    sequence, local = module(encoded)

    assert sequence.shape == (3, 10, 4, 8)
    assert local.shape == (3, 4, 8)
    assert torch.equal(local, sequence[:, -1])


def test_temporal_transformer_rejects_history_beyond_position_table() -> None:
    module = NodeTemporalTransformer(
        hidden_dim=8, num_heads=2, num_layers=1, max_history=4
    )
    with pytest.raises(ValueError, match="history exceeds"):
        module(torch.randn(1, 5, 2, 8))


def test_temporal_transformer_shares_weights_without_mixing_nodes() -> None:
    torch.manual_seed(5)
    module = NodeTemporalTransformer(
        hidden_dim=8, num_heads=2, num_layers=1, dropout=0.0
    ).eval()
    encoded = torch.randn(1, 6, 3, 8)
    changed = encoded.clone()
    changed[:, :, 0] += 10.0

    baseline, _ = module(encoded)
    perturbed, _ = module(changed)

    assert not torch.equal(baseline[:, :, 0], perturbed[:, :, 0])
    assert torch.equal(baseline[:, :, 1:], perturbed[:, :, 1:])
