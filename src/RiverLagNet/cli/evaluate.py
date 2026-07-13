"""Evaluate a selected checkpoint on the held-out chronological test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


def _metric_float(value: Any) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def run(cfg: DictConfig) -> dict[str, float]:
    """Evaluate one validation-selected checkpoint on the held-out test split."""
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be provided for evaluation")
    seed_everything(int(cfg.seed), workers=True)
    data_kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(data_kwargs, dict) and isinstance(model_kwargs, dict)
    datamodule = RiverDataModule(**data_kwargs)
    datamodule.setup("test")
    model_name = str(model_kwargs.pop("name"))
    model = build_model(
        model_name,
        datamodule.data_spec,
        output_window=int(cfg.data.output_window),
        **model_kwargs,
    )
    module = RiverForecastModule.load_from_checkpoint(str(cfg.checkpoint_path), model=model)
    trainer = Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=(
            str(cfg.trainer.precision)
            if torch.cuda.is_available() and str(cfg.trainer.accelerator) != "cpu"
            else "32-true"
        ),
        deterministic=bool(cfg.trainer.deterministic),
        logger=False,
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
    )
    results = trainer.test(module, datamodule=datamodule, verbose=False)
    if len(results) != 1:
        raise RuntimeError("expected one test metric dictionary")
    metrics = {name: _metric_float(value) for name, value in results[0].items()}
    if cfg.evaluation_output:
        output_path = Path(str(cfg.evaluation_output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI wrapper for held-out test evaluation."""
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
