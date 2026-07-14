from __future__ import annotations

import math

import torch

from RiverLagNet.models.topology_encoder import DirectedTopologyEncoder


def test_directed_topology_features_follow_chain_position() -> None:
    encoder = DirectedTopologyEncoder(hidden_dim=8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.0, 1.0], [0.0, 2.0]])

    features = encoder.features(edge_index, edge_attr, num_nodes=3)

    assert features.shape == (3, 10)
    assert torch.equal(features[:, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(features[:, 3], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(features[:, 4], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(features[:, 5], torch.tensor([1.0, 0.5, 0.0]))
    assert math.isclose(float(features[2, 8]), 1.0)
    assert math.isclose(float(features[0, 9]), 1.0)


def test_topology_encoder_starts_as_exact_zero() -> None:
    encoder = DirectedTopologyEncoder(hidden_dim=8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.0, 1.0], [0.0, 2.0]])

    output = encoder(edge_index, edge_attr, num_nodes=3)

    assert output.shape == (3, 8)
    assert torch.equal(output, torch.zeros_like(output))


def test_topology_encoder_rejects_cycles() -> None:
    encoder = DirectedTopologyEncoder(hidden_dim=8)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    try:
        encoder.features(edge_index, edge_attr, num_nodes=2)
    except ValueError as error:
        assert "acyclic" in str(error)
    else:
        raise AssertionError("cyclic graph should be rejected")
