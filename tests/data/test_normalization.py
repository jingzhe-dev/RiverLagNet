import torch

from RiverLagNet.data.normalization import MaskedStandardScaler


def test_scaler_uses_only_observed_values_and_round_trips() -> None:
    values = torch.tensor([[[1.0, 10.0]], [[3.0, 99.0]], [[1000.0, 30.0]]])
    mask = torch.tensor([[[1, 1]], [[1, 0]], [[0, 1]]], dtype=torch.bool)
    scaler = MaskedStandardScaler().fit(values, mask)
    assert torch.allclose(scaler.mean, torch.tensor([2.0, 20.0]))
    transformed = scaler.transform(values)
    assert torch.allclose(scaler.inverse_transform(transformed), values)
