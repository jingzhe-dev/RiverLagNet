from __future__ import annotations

import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer

from RiverLagNet.cli import train as train_cli
from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.callbacks import RuntimeStatsCallback
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


def test_runtime_stats_write_throughput_and_memory_json(tmp_path: Path) -> None:
    data = RiverDataModule(
        num_days=180,
        num_nodes=4,
        input_window=20,
        output_window=10,
        batch_size=2,
    )
    data.setup("fit")
    model = build_model("station_gru", data.data_spec, output_window=10, hidden_dim=8)
    module = RiverForecastModule(model)
    output_path = tmp_path / "hardware.json"
    runtime = RuntimeStatsCallback(output_path=output_path)
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
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert runtime.duration_s > 0.0
    assert runtime.samples_per_second > 0.0
    assert runtime.optimizer_steps_per_second > 0.0
    assert runtime.peak_vram_gb == runtime.peak_allocated_vram_gb
    assert runtime.peak_reserved_vram_gb >= runtime.peak_allocated_vram_gb
    assert payload["samples_per_second"] == runtime.samples_per_second
    assert payload["optimizer_steps_per_second"] == runtime.optimizer_steps_per_second
    assert payload["peak_allocated_vram_gb"] == runtime.peak_allocated_vram_gb
    assert payload["peak_reserved_vram_gb"] == runtime.peak_reserved_vram_gb


def test_blackwell_config_and_tf32_high_contract() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "trainer=blackwell_96gb",
                "data=china_real_daily_contracted_1068_v02",
            ],
        )

    previous = torch.get_float32_matmul_precision()
    try:
        train_cli._set_matmul_precision(str(cfg.trainer.matmul_precision))
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
    assert cfg.trainer.accelerator == "gpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.precision == "bf16-mixed"
    assert cfg.trainer.fused_adamw is True
    assert cfg.trainer.effective_batch_size == 96
    assert cfg.trainer.compile_model is False
    assert cfg.data.batch_size == 8
    assert cfg.data.num_workers == 0
    assert cfg.data.persistent_workers is False
