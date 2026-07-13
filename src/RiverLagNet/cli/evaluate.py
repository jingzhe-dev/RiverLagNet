"""Evaluate a selected checkpoint on the held-out chronological test split."""

from __future__ import annotations

import hydra
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load one checkpoint and evaluate it without model selection on test data."""
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
        deterministic=bool(cfg.trainer.deterministic),
        logger=False,
    )
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
