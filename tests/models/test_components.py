import torch

from RiverLagNet.models.decoder import MultiHorizonMultiTargetDecoder
from RiverLagNet.models.input_encoder import InputMaskEncoder
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


def test_decoder_keeps_horizon_node_and_target_dimensions() -> None:
    decoder = MultiHorizonMultiTargetDecoder(hidden_dim=10, output_window=7, target_dim=3)
    output = decoder(torch.randn(2, 5, 10))
    assert output.shape == (2, 7, 5, 3)
