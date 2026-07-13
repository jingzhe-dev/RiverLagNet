"""Train RiverLagNet models through Lightning and Hydra."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.callbacks import RuntimeStatsCallback
from RiverLagNet.training.experiment_log import ExperimentRecord, append_experiment_record
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _metric_float(metrics: dict[str, Any], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"validation did not produce required metric: {name}")
    value = metrics[name]
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _experiment_record(
    cfg: DictConfig,
    runtime: RuntimeStatsCallback,
    metrics: dict[str, Any] | None,
    *,
    status: str | None = None,
    description: str | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit=_git_value("rev-parse", "HEAD"),
        branch=_git_value("branch", "--show-current"),
        experiment=str(cfg.experiment.name),
        seed=int(cfg.seed),
        val_macro_nse=(
            _metric_float(metrics, "val_macro_nse") if metrics is not None else None
        ),
        val_macro_mae=(
            _metric_float(metrics, "val_macro_mae") if metrics is not None else None
        ),
        val_macro_rmse=(
            _metric_float(metrics, "val_macro_rmse") if metrics is not None else None
        ),
        duration_s=runtime.duration_s or None,
        peak_vram_gb=runtime.peak_vram_gb,
        status=status or str(cfg.experiment.status),
        description=description or str(cfg.experiment.description),
    )


def run(cfg: DictConfig) -> dict[str, Any]:
    """Execute one configured Lightning training run."""
    seed_everything(int(cfg.seed), workers=True)
    data_kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_kwargs, dict)
    datamodule = RiverDataModule(**data_kwargs)
    datamodule.setup("fit")
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_kwargs, dict)
    model_name = str(model_kwargs.pop("name"))
    model = build_model(
        model_name,
        datamodule.data_spec,
        output_window=int(cfg.data.output_window),
        **model_kwargs,
    )
    module = RiverForecastModule(
        model,
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    run_dir = Path(str(cfg.run_dir))
    runtime = RuntimeStatsCallback()
    callbacks = [runtime, LearningRateMonitor(logging_interval="epoch")]
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch={epoch:03d}-val_nse={val_macro_nse:.4f}",
        monitor="val_macro_nse",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    if not cfg.trainer.fast_dev_run:
        callbacks.extend(
            [
                checkpoint,
                EarlyStopping(
                    monitor="val_macro_nse",
                    mode="max",
                    patience=int(cfg.trainer.early_stopping_patience),
                ),
            ]
        )
    loggers = [
        CSVLogger(save_dir=run_dir, name="csv"),
        TensorBoardLogger(save_dir=run_dir, name="tensorboard"),
    ]
    accelerator = str(cfg.trainer.accelerator)
    use_cuda = torch.cuda.is_available() and accelerator != "cpu"
    precision = str(cfg.trainer.precision) if use_cuda else "32-true"
    trainer = Trainer(
        accelerator=accelerator,
        devices=cfg.trainer.devices,
        precision=precision,
        max_epochs=int(cfg.trainer.max_epochs),
        deterministic=bool(cfg.trainer.deterministic),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        default_root_dir=run_dir,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
        enable_checkpointing=not bool(cfg.trainer.fast_dev_run),
    )
    should_record = bool(cfg.experiment.record_result) and not bool(cfg.trainer.fast_dev_run)
    try:
        trainer.fit(module, datamodule=datamodule)
    except Exception as error:
        if should_record:
            crash_record = _experiment_record(
                cfg,
                runtime,
                None,
                status="crash",
                description=f"{type(error).__name__}: {error}",
            )
            append_experiment_record(Path(str(cfg.experiment.results_path)), crash_record)
        raise

    validation_metrics: dict[str, Any] = {}
    record: ExperimentRecord | None = None
    if should_record:
        validation_results = trainer.validate(
            module, datamodule=datamodule, ckpt_path="best", verbose=False
        )
        if len(validation_results) != 1:
            raise RuntimeError("expected one validation metric dictionary")
        validation_metrics = validation_results[0]
        record = _experiment_record(cfg, runtime, validation_metrics)
        append_experiment_record(Path(str(cfg.experiment.results_path)), record)
    return {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
        "checkpoint_path": checkpoint.best_model_path if not cfg.trainer.fast_dev_run else "",
        "validation_metrics": validation_metrics,
        "experiment_record": record,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI wrapper."""
    result = run(cfg)
    print(f"best_checkpoint={result['checkpoint_path']}")


if __name__ == "__main__":
    main()
