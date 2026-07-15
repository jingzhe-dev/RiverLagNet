from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from RiverLagNet.cli import train as train_cli
from RiverLagNet.training.budget import resolve_accumulation, resolve_training_budget


def test_equal_budget_configs_keep_effective_batch_and_optimizer_updates() -> None:
    batch_four = resolve_training_budget(4, 32, 100)
    batch_eight = resolve_training_budget(8, 32, 100)

    assert batch_four.effective_batch_size == batch_eight.effective_batch_size == 32
    assert batch_four.max_optimizer_steps == batch_eight.max_optimizer_steps == 100
    assert batch_four.gradient_accumulation == 8
    assert batch_eight.gradient_accumulation == 4


def test_accumulation_rejects_nondivisible_or_nonpositive_batches() -> None:
    with pytest.raises(ValueError, match="divisible"):
        resolve_accumulation(6, 32)
    with pytest.raises(ValueError, match="positive"):
        resolve_accumulation(0, 32)


def test_train_cli_propagates_fixed_update_budget_on_cpu(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=station_gru",
                "model.hidden_dim=8",
                "data.num_days=180",
                "data.num_nodes=4",
                "data.input_window=20",
                "data.output_window=10",
                "data.batch_size=4",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.effective_batch_size=8",
                "trainer.max_steps=12",
                "trainer.fast_dev_run=true",
                "trainer.enable_progress_bar=false",
                "experiment.record_result=false",
            ],
        )
    cfg.run_dir = str(tmp_path / "run")

    result = train_cli.run(cfg)

    assert result["trainer"].accumulate_grad_batches == 2
    assert result["training_budget"].effective_batch_size == 8
    assert result["training_budget"].max_optimizer_steps == 12
    assert result["training_budget"].gradient_accumulation == 2
    assert not (tmp_path / "gpu.lock").exists()
