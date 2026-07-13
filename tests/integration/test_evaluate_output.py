from __future__ import annotations

import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from RiverLagNet.cli import evaluate as evaluate_cli
from RiverLagNet.cli import train as train_cli


def test_evaluate_run_returns_and_writes_finite_test_metrics(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=station_gru",
                "data.num_days=180",
                "data.num_nodes=4",
                "data.input_window=20",
                "data.output_window=10",
                "data.batch_size=4",
                "model.hidden_dim=8",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_epochs=1",
                "trainer.enable_progress_bar=false",
                "experiment.record_result=false",
            ],
        )
    cfg.run_dir = str(tmp_path / "run")
    trained = train_cli.run(cfg)
    cfg.checkpoint_path = trained["checkpoint_path"]
    output_path = tmp_path / "test_metrics.json"
    OmegaConf.update(cfg, "evaluation_output", str(output_path), force_add=True)

    metrics = evaluate_cli.run(cfg)

    expected = {
        "test_macro_nse",
        "test_macro_mae",
        "test_macro_rmse",
        "test_nse_NH3N",
        "test_nse_CODMn",
        "test_nse_TP",
    }
    assert set(metrics) >= expected
    assert all(math.isfinite(metrics[name]) for name in expected)
    assert json.loads(output_path.read_text(encoding="utf-8")) == metrics
