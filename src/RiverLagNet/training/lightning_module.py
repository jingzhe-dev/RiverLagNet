"""Common LightningModule for all RiverLagNet forecasting models."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from RiverLagNet.data.datamodule import DataSpec
from RiverLagNet.models.baselines import PersistenceModel, StationGRU, StaticDirectedGAT
from RiverLagNet.models.riverlag_net import RiverLagNet

from .losses import masked_huber_loss
from .metrics import masked_metric_dict


MODEL_TYPES = {
    "persistence": PersistenceModel,
    "station_gru": StationGRU,
    "static_gat": StaticDirectedGAT,
    "riverlagnet": RiverLagNet,
}
MODEL_INPUT_KEYS = (
    "x",
    "x_mask",
    "x_quality",
    "static",
    "edge_index",
    "edge_attr",
    "time_features",
)


def build_model(
    name: str,
    data_spec: DataSpec,
    output_window: int = 30,
    **options: Any,
) -> nn.Module:
    """Build any required model against one shared data specification."""
    try:
        model_type = MODEL_TYPES[name]
    except KeyError as error:
        raise ValueError(f"unknown model: {name}") from error
    kwargs = {
        "value_dim": data_spec.num_variables,
        "static_dim": data_spec.static_dim,
        "time_dim": data_spec.time_dim,
        "edge_dim": data_spec.edge_dim,
        "output_window": output_window,
        **options,
    }
    return model_type(**kwargs)


class RiverForecastModule(LightningModule):
    """Optimize and evaluate a forecasting model with shared masked metrics."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._dummy_parameter = (
            nn.Parameter(torch.zeros(()))
            if not any(parameter.requires_grad for parameter in model.parameters())
            else None
        )
        self._epoch_outputs: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {
            "val": [],
            "test": [],
        }

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward only the tensors accepted by the common model API."""
        prediction = self.model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
        if self._dummy_parameter is not None:
            prediction = prediction + self._dummy_parameter * 0.0
        return prediction

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        prediction = self(batch)
        loss = masked_huber_loss(prediction, batch["y"], batch["y_mask"], self.hparams.huber_delta)
        if self._trainer is not None:
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch["x"].shape[0],
            )
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._evaluation_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._evaluation_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._epoch_outputs["val"].clear()

    def on_test_epoch_start(self) -> None:
        self._epoch_outputs["test"].clear()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_macro_nse",
                "interval": "epoch",
            },
        }

    def _evaluation_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        prediction = self(batch)
        loss = masked_huber_loss(prediction, batch["y"], batch["y_mask"], self.hparams.huber_delta)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self._epoch_outputs[stage].append(
            (prediction.detach(), batch["y"].detach(), batch["y_mask"].detach())
        )
        return loss

    def _log_epoch_metrics(self, stage: str) -> None:
        outputs = self._epoch_outputs[stage]
        if not outputs:
            return
        prediction = torch.cat([item[0] for item in outputs], dim=0)
        target = torch.cat([item[1] for item in outputs], dim=0)
        mask = torch.cat([item[2] for item in outputs], dim=0)
        metrics = masked_metric_dict(prediction, target, mask)
        for name, value in metrics.items():
            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=name == "macro_nse",
                sync_dist=True,
                batch_size=prediction.shape[0],
            )
