from __future__ import annotations

import torch

from RiverLagNet.models.trajectory_propagation import DirectedTrajectoryPropagation


def test_trajectory_alignment_uses_history_then_predicted_future() -> None:
    module = DirectedTrajectoryPropagation(
        hidden_dim=1, edge_dim=2, max_lag=4, steps=1
    )
    history = torch.zeros(1, 4, 2, 1)
    history[0, :, 0, 0] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    future = torch.zeros(1, 4, 2, 1)
    future[0, :, 0, 0] = torch.tensor([20.0, 21.0, 22.0, 23.0])
    edge_index = torch.tensor([[0], [1]])
    edge_attr = torch.tensor([[0.0, 2.0]])

    aligned = module.aligned_source_states(
        future, history, edge_index, edge_attr
    )

    assert aligned.shape == (1, 4, 1, 1)
    assert torch.equal(aligned[0, :, 0, 0], torch.tensor([12.0, 13.0, 20.0, 21.0]))


def test_trajectory_routing_is_normalized_over_incoming_edges() -> None:
    module = DirectedTrajectoryPropagation(
        hidden_dim=4, edge_dim=2, max_lag=3, steps=1
    )
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]])
    edge_attr = torch.randn(3, 2)

    weights = module.routing_weights(
        edge_index,
        edge_attr,
        batch_size=2,
        horizons=5,
        num_nodes=4,
    )

    assert weights.shape == (2, 5, 3, 1)
    assert torch.allclose(weights[:, :, :2].sum(dim=2), torch.ones(2, 5, 1))
    assert torch.allclose(weights[:, :, 2:].sum(dim=2), torch.ones(2, 5, 1))


def test_repeated_trajectory_steps_reach_distant_downstream_nodes() -> None:
    torch.manual_seed(17)
    one_step = DirectedTrajectoryPropagation(
        hidden_dim=4, edge_dim=2, max_lag=2, steps=1
    ).eval()
    two_steps = DirectedTrajectoryPropagation(
        hidden_dim=4, edge_dim=2, max_lag=2, steps=2
    ).eval()
    two_steps.load_state_dict(one_step.state_dict())
    local = torch.randn(1, 3, 3, 4)
    history = torch.randn(1, 5, 3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.2, 0.0], [0.4, 0.0]])
    perturbed = local.clone()
    perturbed[:, :, 0] += 2.0

    base_one, _ = one_step(local, history, edge_index, edge_attr)
    changed_one, _ = one_step(perturbed, history, edge_index, edge_attr)
    base_two, _ = two_steps(local, history, edge_index, edge_attr)
    changed_two, _ = two_steps(perturbed, history, edge_index, edge_attr)

    assert torch.equal(base_one[:, :, 0], local[:, :, 0])
    assert torch.equal(changed_one[:, :, 0], perturbed[:, :, 0])
    assert torch.allclose(base_one[:, :, 2], changed_one[:, :, 2])
    assert not torch.allclose(base_two[:, :, 2], changed_two[:, :, 2])


def test_empty_trajectory_graph_is_exact_identity() -> None:
    module = DirectedTrajectoryPropagation(
        hidden_dim=4, edge_dim=2, max_lag=3, steps=4
    )
    local = torch.randn(2, 5, 3, 4)
    history = torch.randn(2, 8, 3, 4)

    output, weights = module(
        local,
        history,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 2),
    )

    assert torch.equal(output, local)
    assert weights.shape == (2, 5, 0, 1)
