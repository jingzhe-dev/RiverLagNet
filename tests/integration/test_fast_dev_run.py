from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer

from RiverLagNet.cli import train as train_cli
from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.callbacks import RuntimeStatsCallback
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "china-real-daily-contracted-1068-extended-v0.4"
    / "dataset.npz"
)


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
    runtime = RuntimeStatsCallback()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[runtime],
    )
    trainer.fit(module, datamodule=data)
    assert trainer.state.finished
    assert "val_macro_nse" in trainer.callback_metrics
    assert runtime.duration_s > 0.0


@pytest.mark.parametrize("model_name", ["station_gru", "riverlagnet"])
def test_identifiable_fast_dev_run_completes_for_trainable_models(
    model_name: str,
) -> None:
    data = RiverDataModule(
        scenario="identifiable_v1",
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


def test_local_multiscale_cli_fast_dev_writes_complexity_and_throughput(
    tmp_path: Path,
) -> None:
    config_dir = ROOT / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic",
                "model=local_multiscale",
                "data.num_days=180",
                "data.num_nodes=5",
                "data.num_variables=5",
                "data.input_window=20",
                "data.output_window=10",
                "data.batch_size=2",
                "model.hidden_dim=8",
                "model.scales=[1,3]",
                "model.num_layers=1",
                "model.dropout=0.0",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.precision=32-true",
                "trainer.effective_batch_size=2",
                "trainer.fast_dev_run=true",
                "trainer.use_gpu_lock=false",
                "experiment.record_result=false",
            ],
        )
    cfg.run_dir = str(tmp_path / "local-fast-dev")

    result = train_cli.run(cfg)
    payload = json.loads((Path(cfg.run_dir) / "hardware.json").read_text(encoding="utf-8"))

    expected_parameters = sum(
        parameter.numel()
        for parameter in result["module"].model.parameters()
        if parameter.requires_grad
    )
    assert result["trainer"].state.finished
    assert payload["trainable_parameters"] == expected_parameters
    assert payload["forward_flops"] > 0
    assert payload["flop_input_shape"] == [2, 20, 5, 5]
    assert payload["flop_estimation_method"] == "torch.utils.flop_counter"
    assert payload["samples_per_second"] > 0.0
    assert payload["optimizer_steps_per_second"] > 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not REAL_DATASET.is_file(),
    reason="formal CUDA dataset is required",
)
def test_local_multiscale_real_fold_a_fast_dev_run(tmp_path: Path) -> None:
    config_dir = ROOT / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=china_real_daily_contracted_1068_v02",
                "model=local_multiscale",
                "trainer=blackwell_96gb",
                "data.split_name=v02_fold_a",
                "model.hidden_dim=8",
                "model.scales=[1,3]",
                "model.num_layers=1",
                "model.dropout=0.0",
                "trainer.fast_dev_run=true",
                "experiment.record_result=false",
            ],
        )
    cfg.run_dir = str(tmp_path / "real-fold-a-fast-dev")

    result = train_cli.run(cfg)

    assert result["trainer"].state.finished
    assert not Path(str(cfg.trainer.gpu_lock_path)).exists()
