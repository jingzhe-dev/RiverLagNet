import torch

from RiverLagNet.models.lag_message_passing import DirectedLagAwareMessagePassing


def _make_uniform(module: DirectedLagAwareMessagePassing) -> None:
    with torch.no_grad():
        module.message_projection.weight.fill_(1.0)
        for parameter in module.score_network.parameters():
            parameter.zero_()


def test_no_lag_flows_only_from_sources_to_destinations() -> None:
    module = DirectedLagAwareMessagePassing(1, edge_dim=1, max_lag=2, lag_mode="no_lag")
    _make_uniform(module)
    h_seq = torch.tensor([[[[2.0], [10.0], [6.0]], [[4.0], [20.0], [8.0]]]])
    local = h_seq[:, -1]
    edges = torch.tensor([[0, 2], [1, 1]])
    upstream, attention = module(h_seq, local, edges, torch.ones(2, 1))
    assert torch.allclose(upstream[0, 1], torch.tensor([6.0]))
    assert torch.allclose(upstream[0, 0], torch.tensor([0.0]))
    assert torch.allclose(upstream[0, 2], torch.tensor([0.0]))
    assert torch.allclose(attention[:, :, 0].sum(dim=1), torch.ones(1))
    assert torch.all(attention[:, :, 1:] == 0)


def test_fixed_lag_uses_t_minus_tau_and_attention_is_jointly_normalized() -> None:
    module = DirectedLagAwareMessagePassing(1, edge_dim=1, max_lag=3, lag_mode="fixed_lag")
    _make_uniform(module)
    h_seq = torch.tensor([[[[1.0], [0.0]], [[2.0], [0.0]], [[3.0], [0.0]], [[4.0], [0.0]]]])
    local = h_seq[:, -1]
    upstream, attention = module(
        h_seq,
        local,
        torch.tensor([[0], [1]]),
        torch.tensor([[2.0]]),
    )
    assert torch.allclose(upstream[0, 1], torch.tensor([2.0]))
    assert torch.allclose(attention[0, 0], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_learned_attention_sums_to_one_per_destination_over_edges_and_lags() -> None:
    module = DirectedLagAwareMessagePassing(4, edge_dim=2, max_lag=2, lag_mode="learned_lag")
    h_seq = torch.randn(2, 5, 4, 4)
    edges = torch.tensor([[0, 2, 1], [1, 1, 3]])
    upstream, attention = module(h_seq, h_seq[:, -1], edges, torch.randn(3, 2))
    assert upstream.shape == (2, 4, 4)
    for destination in (1, 3):
        incoming = edges[1] == destination
        assert torch.allclose(attention[:, incoming].sum(dim=(1, 2)), torch.ones(2), atol=1e-6)
    assert torch.all(upstream[:, 0] == 0)
    assert torch.all(upstream[:, 2] == 0)


def test_training_dropout_does_not_change_reported_attention_normalization() -> None:
    torch.manual_seed(0)
    module = DirectedLagAwareMessagePassing(
        4, edge_dim=2, max_lag=2, lag_mode="learned_lag", dropout=0.5
    )
    h_seq = torch.randn(2, 5, 3, 4)
    edges = torch.tensor([[0, 2], [1, 1]])
    _, attention = module(h_seq, h_seq[:, -1], edges, torch.randn(2, 2))
    assert torch.allclose(attention.sum(dim=(1, 2)), torch.ones(2), atol=1e-6)


@torch.no_grad()
def test_cuda_mixed_precision_keeps_attention_stable() -> None:
    if not torch.cuda.is_available():
        return
    module = DirectedLagAwareMessagePassing(8, edge_dim=3, max_lag=3).cuda()
    h_seq = torch.randn(2, 8, 4, 8, device="cuda")
    edges = torch.tensor([[0, 1, 1], [1, 2, 3]], device="cuda")
    edge_attr = torch.randn(3, 3, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        upstream, attention = module(h_seq, h_seq[:, -1], edges, edge_attr)
    assert torch.isfinite(upstream).all()
    assert attention.dtype == torch.float32
    assert torch.allclose(attention.sum(dim=(1, 2)), torch.full((2,), 3.0, device="cuda"))
