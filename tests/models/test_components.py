import torch

from RiverLagNet.models.decoder import MultiHorizonMultiTargetDecoder, UpstreamResidualDecoder
from RiverLagNet.models.fusion import BoundedLinearHorizonGate, LocalUpstreamGatedFusion
from RiverLagNet.models.input_encoder import InputMaskEncoder, TargetExogenousInputEncoder
from RiverLagNet.models.temporal_gru import NodeTemporalGRU


def test_input_encoder_and_temporal_gru_preserve_time_and_node_axes() -> None:
    batch, history, nodes, variables = 2, 12, 4, 3
    encoder = InputMaskEncoder(variables, static_dim=2, time_dim=4, hidden_dim=8)
    x = torch.randn(batch, history, nodes, variables)
    mask = torch.rand_like(x) > 0.2
    encoded = encoder(x, mask, mask.float(), torch.randn(nodes, 2), torch.randn(batch, history, 4))
    assert encoded.shape == (batch, history, nodes, 8)
    h_seq, h_local = NodeTemporalGRU(8, hidden_dim=10)(encoded)
    assert h_seq.shape == (batch, history, nodes, 10)
    assert h_local.shape == (batch, nodes, 10)


def test_target_exogenous_encoder_uses_learned_context_without_exogenous_channels() -> None:
    encoder = TargetExogenousInputEncoder(
        value_dim=3,
        static_dim=2,
        time_dim=4,
        hidden_dim=8,
        target_dim=3,
    )
    x = torch.randn(2, 6, 4, 3)
    mask = torch.ones_like(x, dtype=torch.bool)

    output = encoder(x, mask, None, torch.randn(4, 2), torch.randn(2, 6, 4))

    assert output.shape == (2, 6, 4, 8)
    assert encoder.no_exogenous_token is not None
    assert torch.isfinite(output).all()


def test_decoder_keeps_horizon_node_and_target_dimensions() -> None:
    decoder = MultiHorizonMultiTargetDecoder(hidden_dim=10, output_window=7, target_dim=3)
    output = decoder(torch.randn(2, 5, 10))
    assert output.shape == (2, 7, 5, 3)


def test_decoder_accepts_horizon_specific_node_states() -> None:
    decoder = MultiHorizonMultiTargetDecoder(hidden_dim=10, output_window=7, target_dim=3)

    output = decoder(torch.randn(2, 7, 5, 10))

    assert output.shape == (2, 7, 5, 3)


def test_decoder_context_methods_reconstruct_regular_forward() -> None:
    decoder = MultiHorizonMultiTargetDecoder(hidden_dim=10, output_window=7, target_dim=3)
    state = torch.randn(2, 5, 10)

    context = decoder.contextualize(state)

    assert context.shape == (2, 7, 5, 10)
    assert torch.equal(decoder.decode_context(context), decoder(state))


def test_upstream_residual_decoder_is_exactly_zero_without_upstream_state() -> None:
    decoder = UpstreamResidualDecoder(hidden_dim=10, target_dim=3)

    output = decoder(torch.zeros(2, 7, 5, 10))

    assert output.shape == (2, 7, 5, 3)
    assert torch.equal(output, torch.zeros_like(output))


def test_fusion_is_identity_when_no_upstream_message_exists() -> None:
    fusion = LocalUpstreamGatedFusion(hidden_dim=8)
    local = torch.randn(2, 5, 8)

    fused = fusion(local, torch.zeros_like(local))

    assert torch.equal(fused, local)


def test_fusion_starts_as_a_small_upstream_residual_with_gradient_flow() -> None:
    torch.manual_seed(7)
    fusion = LocalUpstreamGatedFusion(hidden_dim=8)
    local = torch.randn(2, 5, 8, requires_grad=True)
    upstream = torch.randn(2, 5, 8, requires_grad=True)

    fused = fusion(local, upstream)
    residual_ratio = (fused - local).norm() / upstream.norm()
    fused.square().mean().backward()

    assert residual_ratio < 0.1
    assert upstream.grad is not None
    assert upstream.grad.abs().sum() > 0
    assert fusion.gate.weight.grad is not None
    assert fusion.gate.weight.grad.abs().sum() > 0


def test_bounded_horizon_gate_strictly_starts_as_identity() -> None:
    gate = BoundedLinearHorizonGate(output_window=6)
    correction = torch.randn(2, 6, 4, 3)

    output = gate(correction)

    assert torch.equal(output, correction)
    assert torch.equal(gate.scales(), torch.ones(6))


def test_bounded_horizon_gate_can_increase_weight_with_lead() -> None:
    gate = BoundedLinearHorizonGate(output_window=6)
    with torch.no_grad():
        gate.slope.fill_(2.0)

    scales = gate.scales()

    assert torch.all(scales[1:] > scales[:-1])
    assert torch.all((scales > 0.0) & (scales < 2.0))
