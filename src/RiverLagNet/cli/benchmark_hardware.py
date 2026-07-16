"""Benchmark real-data StationGRU throughput on one reserved NVIDIA GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning.pytorch import Trainer, seed_everything

from RiverLagNet.analysis.hardware_benchmark import (
    BenchmarkJournalEntry,
    HardwareBenchmarkResult,
    StepBenchmarkCallback,
    append_benchmark_journal,
    compile_is_acceptable,
    measurement_result,
    load_benchmark_journal,
    profile_is_faster_than_baseline,
    rank_hardware_profiles,
    select_hardware_profile,
)
from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.data.dataset import river_collate
from RiverLagNet.data.manifest import inspect_dataset
from RiverLagNet.training.gpu_lock import SingleGpuLock
from RiverLagNet.training.budget import resolve_accumulation
from RiverLagNet.training.lightning_module import (
    MODEL_INPUT_KEYS,
    RiverForecastModule,
    build_model,
)


DEFAULT_DATASET = Path(
    "data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz"
)
DEFAULT_BATCHES = (4, 8, 16, 24, 32)
DEFAULT_WORKERS = (0, 4, 8, 12)
EFFECTIVE_BATCH_SIZE = 96


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _prepare_data(dataset_path: Path, prefetch_factor: int) -> RiverDataModule:
    datamodule = RiverDataModule(
        scenario="real_daily",
        dataset_path=str(dataset_path),
        split_name="v02_fold_a",
        input_window=180,
        output_window=30,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        seed=42,
    )
    datamodule.setup("fit")
    return datamodule


def _set_loader_profile(
    datamodule: RiverDataModule,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> None:
    datamodule.hparams["batch_size"] = batch_size
    datamodule.hparams["num_workers"] = num_workers
    datamodule.hparams["pin_memory"] = True
    datamodule.hparams["persistent_workers"] = num_workers > 0
    datamodule.hparams["prefetch_factor"] = prefetch_factor
    datamodule.setup("fit")


def _build_station_module(
    datamodule: RiverDataModule,
    *,
    compiled: bool,
    fused_adamw: bool,
) -> tuple[RiverForecastModule, float | None]:
    seed_everything(42, workers=True)
    model = build_model(
        "station_gru",
        datamodule.data_spec,
        output_window=30,
        hidden_dim=64,
        target_dim=3,
    )
    validation_diff: float | None = None
    if compiled:
        reference = build_model(
            "station_gru",
            datamodule.data_spec,
            output_window=30,
            hidden_dim=64,
            target_dim=3,
        )
        reference.load_state_dict(model.state_dict())
        reference = reference.cuda().eval()
        model = model.cuda().eval()
        compiled_model = torch.compile(model)
        batch = river_collate([datamodule.val_dataset[0]])
        batch = {name: value.cuda(non_blocking=False) for name, value in batch.items()}
        inputs = {name: batch[name] for name in MODEL_INPUT_KEYS}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager_output = reference(**inputs)
            compiled_output = compiled_model(**inputs)
        validation_diff = float(
            (eager_output.float() - compiled_output.float()).abs().max().item()
        )
        if not bool(torch.isfinite(compiled_output).all()):
            raise RuntimeError("compiled validation output is nonfinite")
        del reference, eager_output, compiled_output, batch, inputs
        model = compiled_model.train()
        torch.cuda.empty_cache()
    return (
        RiverForecastModule(
            model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            nse_aux_weight=0.0,
            target_mean=datamodule.scaler.mean[:3].tolist(),
            target_scale=datamodule.scaler.scale[:3].tolist(),
            fused_adamw=fused_adamw,
            use_lr_scheduler=False,
        ),
        validation_diff,
    )


def _graph_break_count() -> int | None:
    try:
        counters = torch._dynamo.utils.counters["graph_break"]  # type: ignore[attr-defined]
        return int(sum(counters.values()))
    except (AttributeError, TypeError):
        return None


def _reset_graph_breaks() -> None:
    try:
        torch._dynamo.reset()  # type: ignore[attr-defined]
        torch._dynamo.utils.counters.clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass


def run_candidate(
    datamodule: RiverDataModule,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    repetition: int,
    compiled: bool,
    warmup_steps: int,
    measured_steps: int,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    precision: str = "bf16-mixed",
    fused_adamw: bool = True,
) -> HardwareBenchmarkResult:
    """Run one Lightning-only candidate and convert failures into result rows."""
    print(
        "BENCHMARK_START "
        f"batch={batch_size} workers={num_workers} repetition={repetition} "
        f"compiled={compiled}",
        flush=True,
    )
    gradient_accumulation = resolve_accumulation(
        batch_size,
        effective_batch_size,
    )
    _set_loader_profile(
        datamodule,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    callback = StepBenchmarkCallback(warmup_steps)
    validation_diff: float | None = None
    status = "ok"
    error_text: str | None = None
    graph_breaks: int | None = None
    trainer: Trainer | None = None
    module: RiverForecastModule | None = None
    try:
        if compiled:
            _reset_graph_breaks()
        module, validation_diff = _build_station_module(
            datamodule,
            compiled=compiled,
            fused_adamw=fused_adamw,
        )
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            precision=precision,
            max_epochs=-1,
            max_steps=warmup_steps + measured_steps,
            accumulate_grad_batches=gradient_accumulation,
            deterministic=True,
            gradient_clip_val=1.0,
            logger=False,
            callbacks=[callback],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            log_every_n_steps=max(1, measured_steps),
        )
        trainer.fit(module, datamodule=datamodule)
        if callback.measured_steps != measured_steps:
            raise RuntimeError(
                f"measured {callback.measured_steps} steps, expected {measured_steps}"
            )
        if not callback.outputs_finite:
            status = "nonfinite"
        if compiled:
            graph_breaks = _graph_break_count()
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
            status = "oom"
        else:
            status = "error"
    result = measurement_result(
        callback,
        physical_batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        gradient_accumulation=gradient_accumulation,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        repetition=repetition,
        compiled=compiled,
        warmup_steps=warmup_steps,
        status=status,
        validation_max_abs_diff=validation_diff,
        error=error_text,
        graph_break_count=graph_breaks,
    )
    print("BENCHMARK_RESULT " + json.dumps(asdict(result), sort_keys=True), flush=True)
    del trainer, module
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _profile_key(
    result: HardwareBenchmarkResult,
) -> tuple[int, int, int, int, int | None]:
    return (
        result.physical_batch_size,
        result.effective_batch_size,
        result.gradient_accumulation,
        result.num_workers,
        result.prefetch_factor,
    )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    temporary.replace(path)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    winner = payload["winner"]
    baseline = payload.get("old_batch4_worker0")
    compile_decision = payload["compile_decision"]
    lines = [
        "# RTX PRO 6000 Blackwell 96 GB hardware profile",
        "",
        "All rows are measured on the frozen 1,068-node fold-A training data with "
        "`T_in=180`, StationGRU hidden size 64, BF16, TF32 high, fused AdamW, "
        "and no validation metrics. The table records warm-up-excluded training throughput.",
        "",
        "| Batch | Effective | Accum | Workers | Rep | Compile | Status | Steps/s | Samples/s | Allocated GiB | Reserved GiB | SM % | Power W | Reason |",
        "|---:|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in payload["results"]:
        reason = result["error"] or ("accepted" if result["status"] == "ok" else result["status"])
        reason = str(reason).replace("|", "\\|").replace("\n", "<br>")
        lines.append(
            f"| {result['physical_batch_size']} | {result['effective_batch_size']} | "
            f"{result['gradient_accumulation']} | {result['num_workers']} | "
            f"{result['repetition']} | {str(result['compiled']).lower()} | "
            f"{result['status']} | {result['optimizer_steps_per_second']:.4f} | "
            f"{result['samples_per_second']:.2f} | "
            f"{result['peak_allocated_vram_gb']:.2f} | "
            f"{result['peak_reserved_vram_gb']:.2f} | "
            f"{result['median_sm_utilization_percent']:.1f} | "
            f"{result['median_power_w']:.1f} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Selected profile",
            "",
            f"The median-throughput winner is batch {winner['physical_batch_size']} with "
            f"{winner['num_workers']} workers: "
            f"effective batch {winner['effective_batch_size']} with "
            f"accumulation {winner['gradient_accumulation']}, "
            f"{winner['optimizer_steps_per_second']:.4f} optimizer steps/s, "
            f"{winner['samples_per_second']:.2f} samples/s, "
            f"{winner['peak_reserved_vram_gb']:.2f} GiB peak reserved VRAM, "
            f"{winner['free_headroom_gb']:.2f} GiB headroom, and "
            f"{winner['median_sm_utilization_percent']:.1f}% median SM utilization.",
            "",
            "Profiles with OOM, nonfinite output, errors, or less than 10 GiB free "
            "headroom were rejected. Remaining profiles lost on median optimizer "
            "throughput; the 70–82 GiB and ≥70% SM targets are reported as utilization "
            "goals rather than substituted for measured speed.",
            "",
            "## Compile decision",
            "",
            f"`torch.compile` retained: **{str(compile_decision['retained']).lower()}**. "
            f"Reason: {compile_decision['reason']}",
            "",
            "## Previous-profile comparison",
            "",
            (
                "The independently measured previous batch-4/worker-0 training "
                "stack (FP16, non-fused AdamW) reached "
                f"{baseline['optimizer_steps_per_second']:.4f} optimizer steps/s. "
                f"The selected profile is faster by "
                f"{((winner['optimizer_steps_per_second'] / baseline['optimizer_steps_per_second']) - 1.0) * 100.0:.2f}%: **"
                f"{str(payload.get('winner_faster_than_old_batch4_worker0', False)).lower()}**."
                if baseline is not None
                else "The previous profile has not been measured."
            ),
            "",
            "These are hardware measurements only and were not written to the experiment ledger.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def run_full_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the grid, repeat the top two, and test compile on the winner."""
    manifest = inspect_dataset(args.dataset)
    datamodule = _prepare_data(args.dataset, args.prefetch_factor)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    signature = json.dumps(
        {
            "dataset_sha256": manifest.sha256,
            "batches": list(args.batches),
            "workers": list(args.workers),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "prefetch_factor": args.prefetch_factor,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "input_window": 180,
            "hidden_dim": 64,
        },
        sort_keys=True,
    )
    if args.restart and args.journal.exists():
        args.journal.unlink()
    journal_entries = list(
        load_benchmark_journal(args.journal, signature=signature)
    )
    cached = {
        (
            entry.stage,
            entry.result.physical_batch_size,
            entry.result.num_workers,
            entry.result.repetition,
            entry.result.compiled,
        ): entry.result
        for entry in journal_entries
    }

    def measure(
        stage: str,
        *,
        batch_size: int,
        num_workers: int,
        repetition: int,
        compiled: bool = False,
        precision: str = "bf16-mixed",
        fused_adamw: bool = True,
    ) -> HardwareBenchmarkResult:
        key = (stage, batch_size, num_workers, repetition, compiled)
        if key in cached:
            result = cached[key]
            print(
                "BENCHMARK_REUSE "
                f"stage={stage} batch={batch_size} workers={num_workers} "
                f"repetition={repetition} compiled={compiled}",
                flush=True,
            )
            return result
        result = run_candidate(
            datamodule,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            repetition=repetition,
            compiled=compiled,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measure_steps,
            precision=precision,
            fused_adamw=fused_adamw,
        )
        entry = BenchmarkJournalEntry(stage=stage, result=result)
        append_benchmark_journal(
            args.journal,
            signature=signature,
            entry=entry,
        )
        cached[key] = result
        return result

    results: list[HardwareBenchmarkResult] = []
    for batch_size in args.batches:
        for num_workers in args.workers:
            results.append(
                measure(
                    "grid",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    repetition=0,
                )
            )

    initial_ranked = rank_hardware_profiles(results, total_vram_gb=total_vram_gb)
    if not initial_ranked:
        raise RuntimeError("hardware matrix produced no valid candidate")
    for profile in initial_ranked[:2]:
        for repetition in (1, 2):
            results.append(
                measure(
                    "repeat",
                    batch_size=profile.physical_batch_size,
                    num_workers=profile.num_workers,
                    repetition=repetition,
                )
            )

    eager_results = [item for item in results if not item.compiled]
    ranked = rank_hardware_profiles(
        eager_results,
        total_vram_gb=total_vram_gb,
        minimum_repetitions=3,
    )
    winner = select_hardware_profile(
        eager_results,
        total_vram_gb=total_vram_gb,
        minimum_repetitions=3,
    )
    winner_key = _profile_key(winner)
    winner_repetitions = [
        item for item in eager_results if _profile_key(item) == winner_key
    ]
    compiled_results: list[HardwareBenchmarkResult] = []
    for repetition in range(3):
        compiled_result = measure(
            "compile",
            batch_size=winner.physical_batch_size,
            num_workers=winner.num_workers,
            repetition=repetition,
            compiled=True,
        )
        compiled_results.append(compiled_result)
        results.append(compiled_result)
        if compiled_result.status != "ok":
            break
    retain_compile = compile_is_acceptable(winner_repetitions, compiled_results)
    if retain_compile:
        reason = "median speed improved by at least 5%, outputs were finite, graph breaks were zero, and max difference was <1e-5"
    elif compiled_results and compiled_results[0].error:
        reason = f"compile candidate failed: {compiled_results[0].error.splitlines()[0]}"
    else:
        reason = "speed, finiteness, graph-break, or <1e-5 numerical-equivalence gate was not met"

    legacy_measurements = [
        measure(
            "legacy",
            batch_size=4,
            num_workers=0,
            repetition=repetition,
            precision="16-mixed",
            fused_adamw=False,
        )
        for repetition in range(3)
    ]
    baseline = select_hardware_profile(
        legacy_measurements,
        total_vram_gb=total_vram_gb,
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "dataset_path": manifest.path,
        "dataset_sha256": manifest.sha256,
        "num_nodes": manifest.num_nodes,
        "input_window": 180,
        "output_window": 30,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "hidden_dim": 64,
        "precision": "bf16-mixed",
        "matmul_precision": "high",
        "fused_adamw": True,
        "warmup_steps": args.warmup_steps,
        "measured_optimizer_steps": args.measure_steps,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_vram_gb": total_vram_gb,
        "results": [asdict(item) for item in results],
        "ranked_profiles": [asdict(item) for item in ranked],
        "winner": asdict(winner),
        "compile_decision": {
            "retained": retain_compile,
            "reason": reason,
            "measurements": [asdict(item) for item in compiled_results],
        },
        "selected_training_config": {
            "physical_batch_size": winner.physical_batch_size,
            "effective_batch_size": winner.effective_batch_size,
            "gradient_accumulation": winner.gradient_accumulation,
            "num_workers": winner.num_workers,
            "prefetch_factor": winner.prefetch_factor,
            "compile_model": retain_compile,
        },
        "legacy_baseline_config": {
            "physical_batch_size": 4,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "gradient_accumulation": EFFECTIVE_BATCH_SIZE // 4,
            "num_workers": 0,
            "precision": "16-mixed",
            "fused_adamw": False,
            "input_window": 180,
            "hidden_dim": 64,
        },
        "legacy_baseline_measurements": [
            asdict(item) for item in legacy_measurements
        ],
        "old_batch4_worker0": asdict(baseline),
        "winner_faster_than_old_batch4_worker0": (
            profile_is_faster_than_baseline(winner, baseline)
        ),
        "uses_validation_metrics": False,
        "ledger_updated": False,
    }
    _write_json(payload, args.output)
    _write_report(payload, args.report)
    if args.journal.exists():
        args.journal.unlink()
    return payload


def measure_legacy_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the pre-optimization stack and update an existing matrix artifact."""
    if not args.output.is_file():
        raise FileNotFoundError(f"benchmark matrix not found: {args.output}")
    payload = json.loads(args.output.read_text(encoding="utf-8"))
    datamodule = _prepare_data(args.dataset, args.prefetch_factor)
    total_vram_gb = float(payload["total_vram_gb"])
    measurements = [
        run_candidate(
            datamodule,
            batch_size=4,
            num_workers=0,
            prefetch_factor=args.prefetch_factor,
            repetition=repetition,
            compiled=False,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measure_steps,
            precision="16-mixed",
            fused_adamw=False,
        )
        for repetition in range(3)
    ]
    baseline = select_hardware_profile(
        measurements,
        total_vram_gb=total_vram_gb,
    )
    winner = HardwareBenchmarkResult(**payload["winner"])
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["commit"] = _git_commit()
    payload["legacy_baseline_config"] = {
        "physical_batch_size": 4,
        "num_workers": 0,
        "precision": "16-mixed",
        "fused_adamw": False,
        "input_window": 180,
        "hidden_dim": 64,
    }
    payload["legacy_baseline_measurements"] = [
        asdict(item) for item in measurements
    ]
    payload["old_batch4_worker0"] = asdict(baseline)
    payload["winner_faster_than_old_batch4_worker0"] = (
        profile_is_faster_than_baseline(winner, baseline)
    )
    _write_json(payload, args.output)
    _write_report(payload, args.report)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--measure-legacy-baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_WORKERS))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/hardware/blackwell-96gb-benchmark.json"),
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("experiments/hardware/blackwell-96gb-benchmark.jsonl"),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard a compatible partial journal and rerun every candidate",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/hardware/blackwell-96gb-profile.md"),
    )
    parser.add_argument("--lock-path", type=Path, default=Path("runs/.gpu0.lock"))
    return parser


def main() -> None:
    """CLI entry point with one lock covering all GPU candidates."""
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Blackwell hardware benchmark")
    torch.set_float32_matmul_precision("high")
    with SingleGpuLock(args.lock_path, run_name="blackwell_hardware_benchmark"):
        if args.measure_legacy_baseline:
            payload = measure_legacy_baseline(args)
            print(json.dumps(payload["old_batch4_worker0"], indent=2, sort_keys=True))
            return
        if args.smoke:
            datamodule = _prepare_data(args.dataset, args.prefetch_factor)
            result = run_candidate(
                datamodule,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                repetition=0,
                compiled=False,
                warmup_steps=2,
                measured_steps=3,
            )
            if result.status != "ok":
                raise RuntimeError(result.error or result.status)
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return
        payload = run_full_matrix(args)
        print(json.dumps(payload["winner"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
