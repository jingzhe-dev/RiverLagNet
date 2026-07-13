import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


def test_checkpoint_saves_and_loads_forecaster(tmp_path) -> None:
    data = RiverDataModule(
        num_days=180, num_nodes=4, input_window=20, output_window=10, batch_size=2, seed=17
    )
    data.setup("fit")
    model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    module = RiverForecastModule(model)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_macro_nse", mode="max")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=data)
    assert checkpoint.best_model_path
    restored_model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    restored = RiverForecastModule.load_from_checkpoint(
        checkpoint.best_model_path, model=restored_model
    )
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        assert restored(batch).shape == batch["y"].shape
