import torch

from RiverLagNet.data.dataset import RiverWindowDataset
from RiverLagNet.data.normalization import MaskedStandardScaler
from RiverLagNet.data.synthetic import generate_synthetic_river_data


def test_windows_use_past_inputs_and_targets_wholly_inside_split() -> None:
    data = generate_synthetic_river_data(num_days=180, num_nodes=4, seed=3)
    scaler = MaskedStandardScaler().fit(data.values[:126], data.observed[:126])
    dataset = RiverWindowDataset(data, scaler, 126, 153, input_window=20, output_window=10)
    assert dataset.forecast_starts[0] == 126
    assert dataset.forecast_starts[-1] == 143
    sample = dataset[0]
    assert sample["input_indices"].max() < sample["target_indices"].min()
    assert sample["target_indices"].min() >= 126
    assert sample["target_indices"].max() < 153
    assert sample["x"].shape == (20, 4, 3)
    assert sample["y"].shape == (10, 4, 3)
    assert sample["time_features"].shape == (20, 4)


def test_targets_remain_three_channels_when_inputs_have_extra_variables() -> None:
    data = generate_synthetic_river_data(
        num_days=180, num_nodes=4, num_variables=5, seed=4
    )
    scaler = MaskedStandardScaler().fit(data.values[:126], data.observed[:126])
    dataset = RiverWindowDataset(data, scaler, 126, 153, input_window=20, output_window=10)
    sample = dataset[0]
    assert sample["x"].shape[-1] == 5
    assert sample["y"].shape[-1] == 3
