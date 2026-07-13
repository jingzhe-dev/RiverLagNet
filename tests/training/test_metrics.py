import torch

from RiverLagNet.training.metrics import masked_mae, masked_metric_dict, masked_nse, masked_rmse


def test_masked_metrics_ignore_unobserved_values() -> None:
    prediction = torch.tensor([[[[2.0, 10.0, 3.0]], [[4.0, 20.0, 5.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 3.0]], [[3.0, 0.0, 7.0]]]])
    mask = torch.tensor([[[[1, 0, 1]], [[1, 0, 1]]]], dtype=torch.bool)
    assert torch.allclose(masked_mae(prediction, target, mask), torch.tensor(1.0))
    assert torch.allclose(masked_rmse(prediction, target, mask), torch.tensor(1.5).sqrt())
    changed = prediction.clone()
    changed[..., 1] = -10000.0
    assert torch.allclose(masked_mae(changed, target, mask), masked_mae(prediction, target, mask))


def test_nse_is_per_target_and_macro_excludes_zero_variance_channels() -> None:
    target = torch.tensor([[[[1.0, 5.0, 2.0]], [[3.0, 5.0, 4.0]]]])
    prediction = target.clone()
    mask = torch.ones_like(target, dtype=torch.bool)
    nse = masked_nse(prediction, target, mask)
    assert torch.allclose(nse[[0, 2]], torch.ones(2))
    assert torch.isnan(nse[1])
    metrics = masked_metric_dict(prediction, target, mask)
    assert torch.allclose(metrics["macro_nse"], torch.tensor(1.0))
    assert set(("nse_NH3N", "nse_CODMn", "nse_TP")).issubset(metrics)
