import pytest
from lightning.pytorch import Trainer

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


@pytest.mark.parametrize("model_name", ["station_gru", "riverlagnet"])
def test_synthetic_fast_dev_run_completes_for_trainable_models(model_name: str) -> None:
    data = RiverDataModule(
        num_days=180,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=2,
        seed=13,
    )
    data.setup("fit")
    model = build_model(
        model_name,
        data.data_spec,
        output_window=10,
        hidden_dim=8,
        max_lag=3,
        dropout=0.0,
    )
    module = RiverForecastModule(model)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=data)
    assert trainer.state.finished
    assert "val_macro_nse" in trainer.callback_metrics
