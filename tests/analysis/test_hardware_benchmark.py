from __future__ import annotations

from pathlib import Path

import pytest

from RiverLagNet.analysis.hardware_benchmark import (
    BenchmarkJournalEntry,
    HardwareBenchmarkResult,
    append_benchmark_journal,
    compile_is_acceptable,
    load_benchmark_journal,
    profile_is_faster_than_baseline,
    rank_hardware_profiles,
    select_hardware_profile,
)


def _result(
    batch: int,
    workers: int,
    *,
    steps_per_second: float,
    reserved_gb: float,
    sm_percent: float = 75.0,
    status: str = "ok",
    outputs_finite: bool = True,
    compiled: bool = False,
    validation_max_abs_diff: float | None = None,
    repetition: int = 0,
) -> HardwareBenchmarkResult:
    return HardwareBenchmarkResult(
        physical_batch_size=batch,
        effective_batch_size=96,
        gradient_accumulation=96 // batch,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
        repetition=repetition,
        compiled=compiled,
        warmup_steps=10,
        measured_steps=100,
        samples_per_second=steps_per_second * batch,
        optimizer_steps_per_second=steps_per_second,
        peak_allocated_vram_gb=reserved_gb - 2.0,
        peak_reserved_vram_gb=reserved_gb,
        median_sm_utilization_percent=sm_percent,
        median_power_w=300.0,
        duration_s=100.0 / steps_per_second,
        status=status,
        outputs_finite=outputs_finite,
        validation_max_abs_diff=validation_max_abs_diff,
        error=None,
    )


def test_selector_rejects_oom_nonfinite_and_low_headroom_results() -> None:
    results = (
        _result(4, 0, steps_per_second=5.0, reserved_gb=20.0),
        _result(8, 4, steps_per_second=9.0, reserved_gb=80.0),
        _result(16, 4, steps_per_second=20.0, reserved_gb=90.0),
        _result(24, 8, steps_per_second=30.0, reserved_gb=60.0, status="oom"),
        _result(32, 8, steps_per_second=40.0, reserved_gb=70.0, outputs_finite=False),
    )

    winner = select_hardware_profile(results, total_vram_gb=95.59)

    assert winner.physical_batch_size == 8
    assert winner.free_headroom_gb == pytest.approx(15.59)
    assert winner.status == "ok"


def test_profiles_are_ranked_by_median_optimizer_throughput() -> None:
    results = (
        _result(4, 0, steps_per_second=6.0, reserved_gb=20.0, repetition=0),
        _result(4, 0, steps_per_second=8.0, reserved_gb=22.0, repetition=1),
        _result(4, 0, steps_per_second=7.0, reserved_gb=21.0, repetition=2),
        _result(16, 8, steps_per_second=9.0, reserved_gb=60.0, repetition=0),
        _result(16, 8, steps_per_second=3.0, reserved_gb=62.0, repetition=1),
        _result(16, 8, steps_per_second=5.0, reserved_gb=61.0, repetition=2),
    )

    ranked = rank_hardware_profiles(results, total_vram_gb=95.59)

    assert [item.physical_batch_size for item in ranked] == [4, 16]
    assert ranked[0].optimizer_steps_per_second == 7.0
    assert ranked[0].effective_batch_size == 96
    assert ranked[0].gradient_accumulation == 24


def test_final_ranking_rejects_profiles_without_three_repetitions() -> None:
    results = (
        _result(4, 0, steps_per_second=6.0, reserved_gb=20.0, repetition=0),
        _result(4, 0, steps_per_second=8.0, reserved_gb=20.0, repetition=1),
        _result(4, 0, steps_per_second=7.0, reserved_gb=20.0, repetition=2),
        _result(24, 0, steps_per_second=9.0, reserved_gb=30.0, repetition=0),
    )

    ranked = rank_hardware_profiles(
        results,
        total_vram_gb=95.59,
        minimum_repetitions=3,
    )

    assert [item.physical_batch_size for item in ranked] == [4]


def test_compile_requires_speed_finiteness_and_numerical_equivalence() -> None:
    eager = (
        _result(8, 4, steps_per_second=10.0, reserved_gb=50.0, repetition=0),
        _result(8, 4, steps_per_second=11.0, reserved_gb=50.0, repetition=1),
        _result(8, 4, steps_per_second=9.0, reserved_gb=50.0, repetition=2),
    )
    compiled = (
        _result(
            8,
            4,
            steps_per_second=10.6,
            reserved_gb=50.0,
            compiled=True,
            validation_max_abs_diff=5e-6,
        ),
    )

    assert compile_is_acceptable(eager, compiled)
    assert not compile_is_acceptable(
        eager,
        (
            _result(
                8,
                4,
                steps_per_second=12.0,
                reserved_gb=50.0,
                compiled=True,
                validation_max_abs_diff=2e-5,
            ),
        ),
    )


def test_profile_speed_gate_uses_an_independent_measured_baseline() -> None:
    baseline = _result(4, 0, steps_per_second=9.0, reserved_gb=4.0)
    faster = _result(4, 0, steps_per_second=10.0, reserved_gb=4.0)
    tied = _result(4, 0, steps_per_second=9.0, reserved_gb=4.0)
    failed = _result(
        4,
        0,
        steps_per_second=20.0,
        reserved_gb=4.0,
        status="error",
    )

    assert profile_is_faster_than_baseline(faster, baseline)
    assert not profile_is_faster_than_baseline(tied, baseline)
    assert not profile_is_faster_than_baseline(failed, baseline)


def test_benchmark_journal_is_durable_and_signature_scoped(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    result = _result(16, 0, steps_per_second=0.5, reserved_gb=16.0)
    entry = BenchmarkJournalEntry(stage="grid", result=result)

    append_benchmark_journal(path, signature="dataset-sha:effective-96", entry=entry)

    loaded = load_benchmark_journal(
        path,
        signature="dataset-sha:effective-96",
    )
    assert loaded == (entry,)
    with pytest.raises(ValueError, match="signature"):
        load_benchmark_journal(path, signature="different-run")
