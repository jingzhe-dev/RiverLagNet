import torch

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


def test_lightning_module_trains_one_real_batch() -> None:
    data = RiverDataModule(num_days=180, num_nodes=5, input_window=20, output_window=10, batch_size=2)
    data.setup("fit")
    model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    module = RiverForecastModule(model, learning_rate=1e-3)
    loss = module.training_step(next(iter(data.train_dataloader())), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_logged_metrics_are_restored_to_physical_target_units() -> None:
    data = RiverDataModule(num_days=180, num_nodes=4, input_window=20, output_window=10)
    data.setup("fit")
    model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    module = RiverForecastModule(
        model,
        target_mean=[10.0, 20.0, 30.0],
        target_scale=[2.0, 3.0, 4.0],
    )
    prediction = torch.zeros(1, 1, 1, 3)
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    module._epoch_outputs["val"] = [(prediction, target, mask)]
    logged: dict[str, torch.Tensor] = {}
    module.log = lambda name, value, **_: logged.__setitem__(name, value)  # type: ignore[method-assign]

    module._log_epoch_metrics("val")

    assert torch.allclose(logged["val_macro_mae"], torch.tensor(3.0))
    assert torch.allclose(logged["val_mae_NH3N"], torch.tensor(2.0))
    assert torch.allclose(logged["val_mae_CODMn"], torch.tensor(3.0))
    assert torch.allclose(logged["val_mae_TP"], torch.tensor(4.0))


def test_fused_adamw_request_falls_back_for_cpu_parameters() -> None:
    data = RiverDataModule(num_days=180, num_nodes=4, input_window=20, output_window=10)
    data.setup("fit")
    model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    module = RiverForecastModule(
        model,
        learning_rate=6e-4,
        weight_decay=1e-3,
        fused_adamw=True,
    )

    optimizer = module.configure_optimizers()["optimizer"]

    assert optimizer.defaults["lr"] == 6e-4
    assert optimizer.defaults["weight_decay"] == 1e-3
    assert optimizer.defaults.get("fused") is not True
