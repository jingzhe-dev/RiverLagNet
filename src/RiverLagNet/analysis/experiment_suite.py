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
class ExperimentSpec:
    """One condition evaluated with one deterministic seed."""

    seed: int
    condition: SuiteCondition
    experiment_name: str
    run_dir: Path


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


def build_experiment_specs(seeds: Sequence[int]) -> tuple[ExperimentSpec, ...]:
    """Return condition-major specs while pairing every condition within a seed."""
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("suite seeds must be unique")
    specs: list[ExperimentSpec] = []
    for seed in normalized:
        for condition in CONDITIONS:
            name = f"robust_s{seed}_{condition.name}"
            specs.append(
                ExperimentSpec(
                    seed=seed,
                    condition=condition,
                    experiment_name=name,
                    run_dir=Path("runs") / name,
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
    return [
        python_executable,
        "-m",
        "RiverLagNet.cli.train",
        f"seed={spec.seed}",
        f"model={spec.condition.model}",
        f"experiment.name={spec.experiment_name}",
        f"run_dir={spec.run_dir.as_posix()}",
        "trainer.enable_progress_bar=false",
        "experiment.status=baseline",
        *spec.condition.model_overrides,
    ]


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
