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
