"""Immutable, resumable experiment matrix for robustness studies."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteCondition:
    """One model or RiverLagNet ablation in the paired experiment matrix."""

    name: str
    model: str
    model_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuitePreset:
    """Immutable naming, data, reporting, and decision contract for one suite."""

    name: str
    experiment_prefix: str
    data_override: str | None
    summary_json: Path
    summary_markdown: Path
    report_title: str
    primary_comparisons: tuple[str, ...]
    condition_names: tuple[str, ...] | None = None
    trainer_override: str | None = None
    experiment_override: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """One condition evaluated with one deterministic seed."""

    seed: int
    condition: SuiteCondition
    experiment_name: str
    run_dir: Path
    preset: SuitePreset


CONDITIONS = (
    SuiteCondition("persistence", "persistence"),
    SuiteCondition("station_gru", "station_gru"),
    SuiteCondition("static_gat", "static_gat"),
    SuiteCondition(
        "no_graph",
        "riverlagnet",
        ("model.graph_variant=no_graph", "model.lag_mode=learned_lag"),
    ),
    SuiteCondition(
        "undirected_graph",
        "riverlagnet",
        ("model.graph_variant=undirected", "model.lag_mode=learned_lag"),
    ),
    SuiteCondition(
        "shuffled_graph",
        "riverlagnet",
        ("model.graph_variant=shuffled", "model.lag_mode=learned_lag"),
    ),
    SuiteCondition(
        "no_lag",
        "riverlagnet",
        ("model.graph_variant=directed", "model.lag_mode=no_lag"),
    ),
    SuiteCondition(
        "fixed_lag",
        "riverlagnet",
        ("model.graph_variant=directed", "model.lag_mode=fixed_lag"),
    ),
    SuiteCondition(
        "learned_lag",
        "riverlagnet",
        ("model.graph_variant=directed", "model.lag_mode=learned_lag"),
    ),
)
CONDITION_NAMES = tuple(condition.name for condition in CONDITIONS)
SUCCESS_STATUSES = frozenset({"baseline", "keep", "discard"})
ROBUSTNESS_V1 = SuitePreset(
    name="robustness_v1",
    experiment_prefix="robust",
    data_override=None,
    summary_json=Path("experiments/robustness_summary.json"),
    summary_markdown=Path("docs/robustness_report_2026-07-13.md"),
    report_title="RiverLagNet multi-seed robustness and ablation report",
    primary_comparisons=(
        "no_graph",
        "undirected_graph",
        "shuffled_graph",
        "no_lag",
        "fixed_lag",
    ),
)
IDENTIFIABLE_V1 = SuitePreset(
    name="identifiable_v1",
    experiment_prefix="ident_v1",
    data_override="synthetic_identifiable_v1",
    summary_json=Path("experiments/identifiable_v1_summary.json"),
    summary_markdown=Path("docs/identifiable_v1_report_2026-07-13.md"),
    report_title="RiverLagNet identifiable synthetic benchmark report",
    primary_comparisons=("no_graph", "shuffled_graph", "no_lag"),
)
IDENTIFIABLE_FUSION_V2 = SuitePreset(
    name="identifiable_fusion_v2",
    experiment_prefix="ident_fusion_v2",
    data_override="synthetic_identifiable_v1",
    summary_json=Path("experiments/identifiable_fusion_v2_summary.json"),
    summary_markdown=Path("docs/identifiable_fusion_v2_report_2026-07-13.md"),
    report_title="RiverLagNet identity-safe fusion benchmark report",
    primary_comparisons=("no_graph", "shuffled_graph", "no_lag"),
)
IDENTIFIABLE_HORIZON_V5 = SuitePreset(
    name="identifiable_horizon_v5",
    experiment_prefix="ident_horizon_v5",
    data_override="synthetic_identifiable_v1",
    summary_json=Path("experiments/identifiable_horizon_v5_summary.json"),
    summary_markdown=Path("docs/identifiable_horizon_v5_report_2026-07-13.md"),
    report_title="RiverLagNet horizon-aligned routing benchmark report",
    primary_comparisons=("no_graph", "shuffled_graph", "no_lag"),
)
REAL_LAG_V1 = SuitePreset(
    name="real_lag_v1",
    experiment_prefix="real_lag_v1",
    data_override="china_real_daily",
    summary_json=Path("experiments/china_real_daily_lag_ablation_summary.json"),
    summary_markdown=Path("docs/china_real_daily_lag_ablation_report_2026-07-14.md"),
    report_title="RiverLagNet real daily multi-seed lag ablation report",
    primary_comparisons=("no_lag", "fixed_lag"),
    condition_names=("no_lag", "fixed_lag", "learned_lag"),
    trainer_override="formal_gpu",
    experiment_override="china_real_daily",
    description=(
        "Real-source daily China observations with imputed values masked in a "
        "paired five-seed lag ablation."
    ),
)
SUITE_PRESETS = {
    preset.name: preset
    for preset in (
        ROBUSTNESS_V1,
        IDENTIFIABLE_V1,
        IDENTIFIABLE_FUSION_V2,
        IDENTIFIABLE_HORIZON_V5,
        REAL_LAG_V1,
    )
}


def build_experiment_specs(
    seeds: Sequence[int], preset: SuitePreset = ROBUSTNESS_V1
) -> tuple[ExperimentSpec, ...]:
    """Return condition-major specs while pairing every condition within a seed."""
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("suite seeds must be unique")
    condition_names = preset.condition_names or CONDITION_NAMES
    unknown = sorted(set(condition_names) - set(CONDITION_NAMES))
    if unknown:
        raise ValueError(f"unknown suite conditions: {', '.join(unknown)}")
    selected_conditions = tuple(
        condition for condition in CONDITIONS if condition.name in condition_names
    )
    specs: list[ExperimentSpec] = []
    for seed in normalized:
        for condition in selected_conditions:
            name = f"{preset.experiment_prefix}_s{seed}_{condition.name}"
            specs.append(
                ExperimentSpec(
                    seed=seed,
                    condition=condition,
                    experiment_name=name,
                    run_dir=Path("runs") / name,
                    preset=preset,
                )
            )
    return tuple(specs)


def successful_experiment_names(ledger_path: Path) -> set[str]:
    """Read experiment names with a successful terminal status from a TSV ledger."""
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file() or ledger_path.stat().st_size == 0:
        return set()
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if rows.fieldnames is None or not {"experiment", "status"}.issubset(rows.fieldnames):
            raise ValueError("experiment ledger is missing experiment/status columns")
        return {
            row["experiment"]
            for row in rows
            if row["status"] in SUCCESS_STATUSES and row["experiment"]
        }


def pending_experiment_specs(
    specs: Iterable[ExperimentSpec], successful_names: set[str]
) -> tuple[ExperimentSpec, ...]:
    """Return specs that have no successful ledger row."""
    return tuple(spec for spec in specs if spec.experiment_name not in successful_names)


def training_command(spec: ExperimentSpec, python_executable: str) -> list[str]:
    """Build one exact Hydra training command without shell interpolation."""
    command = [
        python_executable,
        "-m",
        "RiverLagNet.cli.train",
        f"seed={spec.seed}",
    ]
    if spec.preset.data_override is not None:
        command.append(f"data={spec.preset.data_override}")
    if spec.preset.trainer_override is not None:
        command.append(f"trainer={spec.preset.trainer_override}")
    if spec.preset.experiment_override is not None:
        command.append(f"experiment={spec.preset.experiment_override}")
    command.extend(
        [
        f"model={spec.condition.model}",
        f"experiment.name={spec.experiment_name}",
        f"run_dir={spec.run_dir.as_posix()}",
        "trainer.enable_progress_bar=false",
        "experiment.status=baseline",
        *spec.condition.model_overrides,
        ]
    )
    if spec.preset.description is not None:
        command.append(f"experiment.description={spec.preset.description}")
    return command


def final_evaluation_output(spec: ExperimentSpec) -> Path:
    """Return the ignored JSON path for one final held-out evaluation."""
    return spec.run_dir / "test_metrics.json"


def final_checkpoint_path(spec: ExperimentSpec) -> Path:
    """Select the sole validation-best checkpoint written by one training run."""
    if spec.condition.name != "learned_lag":
        raise ValueError("final test evaluation is restricted to learned_lag")
    checkpoints = tuple(sorted((spec.run_dir / "checkpoints").glob("*.ckpt")))
    if len(checkpoints) != 1:
        raise ValueError(
            f"expected exactly one checkpoint for {spec.experiment_name}, found {len(checkpoints)}"
        )
    return checkpoints[0]


def evaluation_command(
    spec: ExperimentSpec, checkpoint_path: Path, python_executable: str
) -> list[str]:
    """Build the held-out evaluation command for one full-model seed."""
    if spec.condition.name != "learned_lag":
        raise ValueError("final test evaluation is restricted to learned_lag")
    command = [
        python_executable,
        "-m",
        "RiverLagNet.cli.evaluate",
        f"seed={spec.seed}",
    ]
    if spec.preset.data_override is not None:
        command.append(f"data={spec.preset.data_override}")
    if spec.preset.trainer_override is not None:
        command.append(f"trainer={spec.preset.trainer_override}")
    command.extend(
        [
        "model=riverlagnet",
        "model.graph_variant=directed",
        "model.lag_mode=learned_lag",
        f'checkpoint_path="{Path(checkpoint_path).as_posix()}"',
        f"evaluation_output={final_evaluation_output(spec).as_posix()}",
        "trainer.enable_progress_bar=false",
        ]
    )
    return command


def run_experiment_specs(
    specs: Iterable[ExperimentSpec],
    python_executable: str,
    dry_run: bool = False,
) -> list[list[str]]:
    """Run specs sequentially and stop immediately when a subprocess fails."""
    commands = [training_command(spec, python_executable) for spec in specs]
    if dry_run:
        return commands
    environment = {**os.environ, "PYTHONUTF8": "1"}
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    return commands


def run_final_evaluations(
    specs: Iterable[ExperimentSpec],
    python_executable: str,
    dry_run: bool = False,
) -> list[list[str]]:
    """Evaluate each full learned-lag seed once after validation decisions are final."""
    selected = tuple(spec for spec in specs if spec.condition.name == "learned_lag")
    commands = [
        evaluation_command(spec, final_checkpoint_path(spec), python_executable)
        for spec in selected
    ]
    if dry_run:
        return commands
    environment = {**os.environ, "PYTHONUTF8": "1"}
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    return commands
