from __future__ import annotations

import csv
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from RiverLagNet.cli import train as train_cli


def _config(tmp_path: Path, *, fast_dev_run: bool = False) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    overrides = [
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
        "trainer.early_stopping_patience=1",
        "trainer.enable_progress_bar=false",
        f"trainer.fast_dev_run={str(fast_dev_run).lower()}",
        "experiment.name=test_recording",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.run_dir = str(tmp_path / "run")
    OmegaConf.update(cfg, "experiment.record_result", True, force_add=True)
    OmegaConf.update(
        cfg, "experiment.results_path", str(tmp_path / "results.tsv"), force_add=True
    )
    OmegaConf.update(cfg, "experiment.status", "baseline", force_add=True)
    return cfg


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_full_run_records_best_checkpoint_validation_metrics(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    result = train_cli.run(cfg)

    assert Path(result["checkpoint_path"]).is_file()
    assert set(result["validation_metrics"]) >= {
        "val_macro_nse",
        "val_macro_mae",
        "val_macro_rmse",
    }
    rows = _rows(tmp_path / "results.tsv")
    assert len(rows) == 1
    assert rows[0]["experiment"] == "test_recording"
    assert rows[0]["status"] == "baseline"
    assert rows[0]["val_macro_nse"]
    assert result["experiment_record"] is not None


def test_fast_dev_run_does_not_write_experiment_ledger(tmp_path: Path) -> None:
    cfg = _config(tmp_path, fast_dev_run=True)

    result = train_cli.run(cfg)

    assert result["experiment_record"] is None
    assert not (tmp_path / "results.tsv").exists()


def test_synthetic_smoke_experiment_disables_ledger(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=synthetic_smoke",
                "model=station_gru",
                "data.num_days=180",
                "data.num_nodes=4",
                "data.input_window=20",
                "data.output_window=10",
                "data.batch_size=4",
                "model.hidden_dim=8",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.fast_dev_run=true",
                "trainer.enable_progress_bar=false",
            ],
        )
    cfg.run_dir = str(tmp_path / "run")

    result = train_cli.run(cfg)

    assert result["experiment_record"] is None
    assert not (tmp_path / "results.tsv").exists()


def test_failed_full_run_records_crash_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("intentional training failure")

    monkeypatch.setattr(train_cli.Trainer, "fit", fail_fit)

    with pytest.raises(RuntimeError, match="intentional training failure"):
        train_cli.run(cfg)

    rows = _rows(tmp_path / "results.tsv")
    assert len(rows) == 1
    assert rows[0]["status"] == "crash"
    assert rows[0]["val_macro_nse"] == ""
    assert "intentional training failure" in rows[0]["description"]
