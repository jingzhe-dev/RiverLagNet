"""Selection rules and measurement callback for Blackwell hardware profiles."""

from __future__ import annotations

import math
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


@dataclass(frozen=True)
class HardwareBenchmarkResult:
    """One measured training-throughput candidate or failed attempt."""

    physical_batch_size: int
    effective_batch_size: int
    gradient_accumulation: int
    num_workers: int
    prefetch_factor: int | None
    repetition: int
    compiled: bool
    warmup_steps: int
    measured_steps: int
    samples_per_second: float
    optimizer_steps_per_second: float
    peak_allocated_vram_gb: float
    peak_reserved_vram_gb: float
    median_sm_utilization_percent: float
    median_power_w: float
    duration_s: float
    status: str
    outputs_finite: bool
    validation_max_abs_diff: float | None
    error: str | None
    graph_break_count: int | None = None
    free_headroom_gb: float = 0.0


@dataclass(frozen=True)
class BenchmarkJournalEntry:
    """One stage-labelled measurement persisted before the matrix continues."""

    stage: str
    result: HardwareBenchmarkResult


def append_benchmark_journal(
    path: Path,
    *,
    signature: str,
    entry: BenchmarkJournalEntry,
) -> None:
    """Append and fsync one completed measurement for interruption recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "signature": signature,
        "stage": entry.stage,
        "result": asdict(entry.result),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_benchmark_journal(
    path: Path,
    *,
    signature: str,
) -> tuple[BenchmarkJournalEntry, ...]:
    """Load a compatible journal, rejecting measurements from another protocol."""
    if not path.is_file():
        return ()
    entries: list[BenchmarkJournalEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("signature") != signature:
                raise ValueError(
                    f"benchmark journal signature mismatch at line {line_number}"
                )
            entries.append(
                BenchmarkJournalEntry(
                    stage=str(record["stage"]),
                    result=HardwareBenchmarkResult(**record["result"]),
                )
            )
    return tuple(entries)


def rank_hardware_profiles(
    results: Sequence[HardwareBenchmarkResult],
    *,
    total_vram_gb: float,
    minimum_repetitions: int = 1,
) -> tuple[HardwareBenchmarkResult, ...]:
    """Aggregate valid repetitions and rank profiles by median optimizer throughput."""
    if minimum_repetitions <= 0:
        raise ValueError("minimum_repetitions must be positive")
    groups: dict[tuple[int, int, int | None, bool], list[HardwareBenchmarkResult]] = {}
    for result in results:
        if not _is_acceptable(result, total_vram_gb):
            continue
        key = (
            result.physical_batch_size,
            result.effective_batch_size,
            result.gradient_accumulation,
            result.num_workers,
            result.prefetch_factor,
            result.compiled,
        )
        groups.setdefault(key, []).append(result)

    summaries: list[HardwareBenchmarkResult] = []
    for repetitions in groups.values():
        if len(repetitions) < minimum_repetitions:
            continue
        first = repetitions[0]
        reserved = statistics.median(
            item.peak_reserved_vram_gb for item in repetitions
        )
        validation_diffs = [
            item.validation_max_abs_diff
            for item in repetitions
            if item.validation_max_abs_diff is not None
        ]
        summaries.append(
            replace(
                first,
                repetition=-1,
                measured_steps=sum(item.measured_steps for item in repetitions),
                samples_per_second=statistics.median(
                    item.samples_per_second for item in repetitions
                ),
                optimizer_steps_per_second=statistics.median(
                    item.optimizer_steps_per_second for item in repetitions
                ),
                peak_allocated_vram_gb=statistics.median(
                    item.peak_allocated_vram_gb for item in repetitions
                ),
                peak_reserved_vram_gb=reserved,
                median_sm_utilization_percent=statistics.median(
                    item.median_sm_utilization_percent for item in repetitions
                ),
                median_power_w=statistics.median(
                    item.median_power_w for item in repetitions
                ),
                duration_s=statistics.median(item.duration_s for item in repetitions),
                validation_max_abs_diff=(
                    max(validation_diffs) if validation_diffs else None
                ),
                free_headroom_gb=total_vram_gb - reserved,
            )
        )

    summaries.sort(
        key=lambda item: (
            item.optimizer_steps_per_second,
            item.samples_per_second,
            70.0 <= item.peak_reserved_vram_gb <= 82.0,
            item.median_sm_utilization_percent >= 70.0,
        ),
        reverse=True,
    )
    return tuple(summaries)


def select_hardware_profile(
    results: Sequence[HardwareBenchmarkResult],
    *,
    total_vram_gb: float,
    minimum_repetitions: int = 1,
) -> HardwareBenchmarkResult:
    """Return the fastest valid median profile with at least 10 GiB headroom."""
    ranked = rank_hardware_profiles(
        results,
        total_vram_gb=total_vram_gb,
        minimum_repetitions=minimum_repetitions,
    )
    if not ranked:
        raise ValueError("no valid hardware profile has at least 10 GiB headroom")
    return ranked[0]


def profile_is_faster_than_baseline(
    profile: HardwareBenchmarkResult,
    baseline: HardwareBenchmarkResult,
) -> bool:
    """Return whether two successful measurements clear the strict speed gate."""
    return (
        _is_finite_success(profile)
        and _is_finite_success(baseline)
        and profile.optimizer_steps_per_second
        > baseline.optimizer_steps_per_second
    )


def compile_is_acceptable(
    eager_results: Sequence[HardwareBenchmarkResult],
    compiled_results: Sequence[HardwareBenchmarkResult],
) -> bool:
    """Keep compile only for finite, equivalent runs with >=5% median speedup."""
    eager = [item for item in eager_results if _is_finite_success(item)]
    compiled = [
        item
        for item in compiled_results
        if _is_finite_success(item)
        and item.validation_max_abs_diff is not None
        and item.validation_max_abs_diff < 1e-5
        and item.graph_break_count in (None, 0)
    ]
    if not eager or not compiled:
        return False
    eager_speed = statistics.median(item.optimizer_steps_per_second for item in eager)
    compiled_speed = statistics.median(
        item.optimizer_steps_per_second for item in compiled
    )
    return compiled_speed >= eager_speed * 1.05


class StepBenchmarkCallback(Callback):
    """Measure only optimizer steps after a fixed warm-up interval."""

    def __init__(self, warmup_steps: int) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        self.warmup_steps = warmup_steps
        self.started_at: float | None = None
        self.duration_s = 0.0
        self.samples_processed = 0
        self.measured_steps = 0
        self.outputs_finite = True
        self.peak_allocated_vram_gb = 0.0
        self.peak_reserved_vram_gb = 0.0
        self.telemetry = GpuTelemetrySampler()
        self.cuda_device: torch.device | None = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.started_at is None and int(trainer.global_step) >= self.warmup_steps:
            self.cuda_device = pl_module.device
            torch.cuda.synchronize(self.cuda_device)
            torch.cuda.reset_peak_memory_stats(self.cuda_device)
            self.telemetry.start()
            self.started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.started_at is None:
            return
        if isinstance(batch, dict) and isinstance(batch.get("x"), torch.Tensor):
            self.samples_processed += int(batch["x"].shape[0])
        if isinstance(outputs, torch.Tensor) and not bool(torch.isfinite(outputs).all()):
            self.outputs_finite = False

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._finish(trainer, pl_module)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self._finish(trainer, pl_module)

    def _finish(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.started_at is None or self.duration_s > 0.0:
            self.telemetry.stop()
            return
        if self.cuda_device is None:
            return
        torch.cuda.synchronize(self.cuda_device)
        self.duration_s = time.perf_counter() - self.started_at
        self.measured_steps = max(0, int(trainer.global_step) - self.warmup_steps)
        self.peak_allocated_vram_gb = (
            torch.cuda.max_memory_allocated(self.cuda_device) / 1024**3
        )
        self.peak_reserved_vram_gb = (
            torch.cuda.max_memory_reserved(self.cuda_device) / 1024**3
        )
        self.telemetry.stop()


class GpuTelemetrySampler:
    """Sample global NVIDIA SM utilization and power during measured steps."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.sm_samples: list[float] = []
        self.power_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def median_sm_utilization_percent(self) -> float:
        return statistics.median(self.sm_samples) if self.sm_samples else 0.0

    @property
    def median_power_w(self) -> float:
        return statistics.median(self.power_samples) if self.power_samples else 0.0

    def _run(self) -> None:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                creationflags=creationflags,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    sm, power = result.stdout.splitlines()[0].split(",")[:2]
                    self.sm_samples.append(float(sm.strip()))
                    self.power_samples.append(float(power.strip()))
                except ValueError:
                    pass
            self._stop.wait(self.interval_s)


def measurement_result(
    callback: StepBenchmarkCallback,
    *,
    physical_batch_size: int,
    effective_batch_size: int,
    gradient_accumulation: int,
    num_workers: int,
    prefetch_factor: int | None,
    repetition: int,
    compiled: bool,
    warmup_steps: int,
    status: str,
    validation_max_abs_diff: float | None,
    error: str | None,
    graph_break_count: int | None = None,
) -> HardwareBenchmarkResult:
    """Convert a completed callback into the durable result contract."""
    duration = callback.duration_s
    return HardwareBenchmarkResult(
        physical_batch_size=physical_batch_size,
        effective_batch_size=effective_batch_size,
        gradient_accumulation=gradient_accumulation,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        repetition=repetition,
        compiled=compiled,
        warmup_steps=warmup_steps,
        measured_steps=callback.measured_steps,
        samples_per_second=(
            callback.samples_processed / duration if duration > 0.0 else 0.0
        ),
        optimizer_steps_per_second=(
            callback.measured_steps / duration if duration > 0.0 else 0.0
        ),
        peak_allocated_vram_gb=callback.peak_allocated_vram_gb,
        peak_reserved_vram_gb=callback.peak_reserved_vram_gb,
        median_sm_utilization_percent=(
            callback.telemetry.median_sm_utilization_percent
        ),
        median_power_w=callback.telemetry.median_power_w,
        duration_s=duration,
        status=status,
        outputs_finite=callback.outputs_finite,
        validation_max_abs_diff=validation_max_abs_diff,
        error=error,
        graph_break_count=graph_break_count,
    )


def _is_acceptable(result: HardwareBenchmarkResult, total_vram_gb: float) -> bool:
    return (
        _is_finite_success(result)
        and total_vram_gb - result.peak_reserved_vram_gb >= 10.0
    )


def _is_finite_success(result: HardwareBenchmarkResult) -> bool:
    numeric = (
        result.samples_per_second,
        result.optimizer_steps_per_second,
        result.peak_allocated_vram_gb,
        result.peak_reserved_vram_gb,
        result.median_sm_utilization_percent,
        result.median_power_w,
        result.duration_s,
    )
    return (
        result.status == "ok"
        and result.outputs_finite
        and result.measured_steps > 0
        and all(math.isfinite(value) for value in numeric)
        and result.optimizer_steps_per_second > 0.0
    )
