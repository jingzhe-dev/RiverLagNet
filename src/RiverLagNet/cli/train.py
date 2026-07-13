"""Train RiverLagNet models through Lightning and Hydra."""

from __future__ import annotations

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
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


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
    callbacks = [RuntimeStatsCallback(), LearningRateMonitor(logging_interval="epoch")]
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
    trainer.fit(module, datamodule=datamodule)
    return {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
        "checkpoint_path": checkpoint.best_model_path if not cfg.trainer.fast_dev_run else "",
    }


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI wrapper."""
    result = run(cfg)
    print(f"best_checkpoint={result['checkpoint_path']}")


if __name__ == "__main__":
    main()
