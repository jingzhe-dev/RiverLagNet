"""Runtime and GPU memory measurement callbacks."""

from __future__ import annotations

import time

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class RuntimeStatsCallback(Callback):
    """Record fit duration and peak allocated CUDA memory."""

    def __init__(self) -> None:
        self.started_at = 0.0
        self.duration_s = 0.0
        self.peak_vram_gb = 0.0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.duration_s = time.perf_counter() - self.started_at
        if torch.cuda.is_available():
            self.peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        metrics = {"duration_s": self.duration_s, "peak_vram_gb": self.peak_vram_gb}
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)
