"""Common LightningModule for all RiverLagNet forecasting models."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import GradClipAlgorithmType
from torch import Tensor, nn
from torch.optim import Optimizer

from RiverLagNet.data.datamodule import DataSpec
from RiverLagNet.data.schema import TARGET_NAMES
from RiverLagNet.models.baselines import PersistenceModel, StationGRU, StaticDirectedGAT
from RiverLagNet.models.river_crossformer import RiverGraphCrossFormer
from RiverLagNet.models.riverlag_net import RiverLagNet

from .losses import masked_huber_loss, masked_nse_loss
from .metrics import masked_metric_dict


MODEL_TYPES = {
    "persistence": PersistenceModel,
    "station_gru": StationGRU,
    "static_gat": StaticDirectedGAT,
    "riverlagnet": RiverLagNet,
    "river_crossformer": RiverGraphCrossFormer,
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
        nse_aux_weight: float = 0.0,
        target_mean: Sequence[float] | None = None,
        target_scale: Sequence[float] | None = None,
        fused_adamw: bool = False,
        use_lr_scheduler: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        if (target_mean is None) != (target_scale is None):
            raise ValueError("target_mean and target_scale must be provided together")
        mean = torch.zeros(len(TARGET_NAMES)) if target_mean is None else torch.tensor(target_mean)
        scale = torch.ones(len(TARGET_NAMES)) if target_scale is None else torch.tensor(target_scale)
        if mean.shape != (len(TARGET_NAMES),) or scale.shape != (len(TARGET_NAMES),):
            raise ValueError("target normalization must contain exactly three values")
        if torch.any(scale <= 0):
            raise ValueError("target_scale must be positive")
        if nse_aux_weight < 0:
            raise ValueError("nse_aux_weight cannot be negative")
        self.register_buffer("target_mean", mean.to(torch.float32))
        self.register_buffer("target_scale", scale.to(torch.float32))
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
        huber = masked_huber_loss(
            prediction, batch["y"], batch["y_mask"], self.hparams.huber_delta
        )
        loss = huber
        if self.hparams.nse_aux_weight:
            nse_aux = masked_nse_loss(prediction, batch["y"], batch["y_mask"])
            loss = loss + self.hparams.nse_aux_weight * nse_aux
        if self._trainer is not None:
            self.log(
                "train_huber_loss",
                huber,
                on_step=False,
                on_epoch=True,
                batch_size=batch["x"].shape[0],
            )
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
        parameter = next(self.parameters())
        use_fused = (
            bool(self.hparams.fused_adamw)
            and parameter.device.type == "cuda"
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        optimizer_options: dict[str, Any] = {}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            optimizer_options["fused"] = use_fused
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            **optimizer_options,
        )
        if not bool(self.hparams.use_lr_scheduler):
            return {"optimizer": optimizer}
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

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: GradClipAlgorithmType | None = None,
    ) -> None:
        """Clip unscaled BF16 fused-optimizer gradients before the optimizer step."""
        fused = bool(optimizer.defaults.get("fused"))
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if fused and scaler is None:
            clip_value = float(gradient_clip_val or 0.0)
            if clip_value <= 0.0:
                return
            if gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip_value)
            else:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
            return
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

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
        metric_prediction = prediction * self.target_scale + self.target_mean
        metric_target = target * self.target_scale + self.target_mean
        metrics = masked_metric_dict(metric_prediction, metric_target, mask)
        for name, value in metrics.items():
            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=name == "macro_nse",
                sync_dist=True,
                batch_size=prediction.shape[0],
            )
