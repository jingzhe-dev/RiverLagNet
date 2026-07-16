"""Runtime and GPU memory measurement callbacks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class RuntimeStatsCallback(Callback):
    """Record fit throughput and allocated/reserved CUDA memory."""

    def __init__(self, output_path: Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.started_at = 0.0
        self.duration_s = 0.0
        self.peak_vram_gb = 0.0
        self.peak_allocated_vram_gb = 0.0
        self.peak_reserved_vram_gb = 0.0
        self.samples_per_second = 0.0
        self.optimizer_steps_per_second = 0.0
        self.samples_processed = 0
        self.optimizer_steps = 0
        self._uses_cuda = False
        self._cuda_device: torch.device | None = None
        self._fit_device = "unknown"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.duration_s = 0.0
        self.peak_vram_gb = 0.0
        self.peak_allocated_vram_gb = 0.0
        self.peak_reserved_vram_gb = 0.0
        self.samples_per_second = 0.0
        self.optimizer_steps_per_second = 0.0
        self.samples_processed = 0
        self.optimizer_steps = 0
        self._uses_cuda = pl_module.device.type == "cuda"
        self._cuda_device = pl_module.device if self._uses_cuda else None
        self._fit_device = str(pl_module.device)
        self.started_at = time.perf_counter()
        if self._cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(self._cuda_device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Count physical samples presented to the model."""
        if isinstance(batch, dict) and isinstance(batch.get("x"), torch.Tensor):
            self.samples_processed += int(batch["x"].shape[0])

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._cuda_device is not None:
            torch.cuda.synchronize(self._cuda_device)
        self.duration_s = time.perf_counter() - self.started_at
        self.optimizer_steps = int(trainer.global_step)
        if self._cuda_device is not None:
            self.peak_allocated_vram_gb = (
                torch.cuda.max_memory_allocated(self._cuda_device) / 1024**3
            )
            self.peak_reserved_vram_gb = (
                torch.cuda.max_memory_reserved(self._cuda_device) / 1024**3
            )
        self.peak_vram_gb = self.peak_allocated_vram_gb
        if self.duration_s > 0.0:
            self.samples_per_second = self.samples_processed / self.duration_s
            self.optimizer_steps_per_second = self.optimizer_steps / self.duration_s
        metrics = self.as_dict(pl_module)
        scalar_metrics = {
            name: value
            for name, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for logger in trainer.loggers:
            logger.log_metrics(scalar_metrics, step=trainer.global_step)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False)
            with self.output_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")

    def as_dict(self, pl_module: LightningModule | None = None) -> dict[str, Any]:
        """Return JSON-safe runtime and hardware statistics."""
        device = self._fit_device
        if device == "unknown" and pl_module is not None:
            device = str(pl_module.device)
        metrics: dict[str, Any] = {
            "duration_s": self.duration_s,
            "samples_processed": self.samples_processed,
            "optimizer_steps": self.optimizer_steps,
            "samples_per_second": self.samples_per_second,
            "optimizer_steps_per_second": self.optimizer_steps_per_second,
            "peak_vram_gb": self.peak_vram_gb,
            "peak_allocated_vram_gb": self.peak_allocated_vram_gb,
            "peak_reserved_vram_gb": self.peak_reserved_vram_gb,
            "device": device,
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        complexity_metrics = getattr(pl_module, "complexity_metrics", None)
        if callable(complexity_metrics):
            metrics.update(complexity_metrics())
        return metrics
