"""Immutable successive-halving contracts for the v0.2 local model search."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


SUCCESS_STATUSES = frozenset({"baseline", "keep", "discard"})
ALLOWED_INPUT_WINDOWS = frozenset({90, 180, 365})
ALLOWED_HIDDEN_DIMS = frozenset({128, 256})
ALLOWED_LAYERS = frozenset({2, 4})
ALLOWED_DROPOUTS = frozenset({0.05, 0.10, 0.20})
ALLOWED_LEARNING_RATES = frozenset({3e-4, 6e-4, 1e-3})
ALLOWED_WEIGHT_DECAYS = frozenset({1e-5, 1e-4, 1e-3})
ALLOWED_NSE_AUX_WEIGHTS = frozenset({0.0, 0.05, 0.10})
EXPECTED_STAGE_COUNTS = (12, 4, 2)
EXPECTED_STAGE_NAMES = ("screen", "promote", "confirm")


@dataclass(frozen=True)
class LocalSearchConfig:
    """One fully explicit member of the preregistered local-model family."""

    config_id: str
    input_window: int
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    nse_aux_weight: float


@dataclass(frozen=True)
class SearchStage:
    """One immutable successive-halving stage and its equal run budget."""

    name: str
    candidate_count: int
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    max_epochs: int
    max_steps: int
    early_stopping_patience: int


@dataclass(frozen=True)
class SearchProtocol:
    """Complete versioned local search definition loaded from one YAML file."""

    protocol_version: str
    family: str
    dataset: str
    model: str
    trainer: str
    experiment: str
    ledger_path: Path
    run_root: Path
    effective_batch_size: int
    stages: tuple[SearchStage, ...]
    configs: tuple[LocalSearchConfig, ...]


@dataclass(frozen=True)
class SearchRunSpec:
    """One fold/seed training invocation derived without mutating a config."""

    config_id: str
    stage: str
    fold: str
    seed: int
    experiment_name: str
    run_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class SearchObservation:
    """Ranking fields for one completed validation-only search run."""

    config_id: str
    stage: str
    fold: str
    seed: int
    val_macro_nse: float
    duration_s: float
    trainable_parameters: int


def _as_mapping(path: Path) -> dict[str, object]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("local search protocol must be a YAML mapping")
    return raw


def _validate_config(config: LocalSearchConfig) -> None:
    checks = (
        (config.input_window, ALLOWED_INPUT_WINDOWS, "input_window"),
        (config.hidden_dim, ALLOWED_HIDDEN_DIMS, "hidden_dim"),
        (config.num_layers, ALLOWED_LAYERS, "num_layers"),
        (config.dropout, ALLOWED_DROPOUTS, "dropout"),
        (config.learning_rate, ALLOWED_LEARNING_RATES, "learning_rate"),
        (config.weight_decay, ALLOWED_WEIGHT_DECAYS, "weight_decay"),
        (config.nse_aux_weight, ALLOWED_NSE_AUX_WEIGHTS, "nse_aux_weight"),
    )
    for value, allowed, name in checks:
        if value not in allowed:
            raise ValueError(f"config {config.config_id} has disallowed {name}: {value}")


def _validate_protocol(protocol: SearchProtocol) -> None:
    if protocol.family != "local":
        raise ValueError("only the preregistered local family is supported")
    if len(protocol.configs) != 12:
        raise ValueError("local protocol must contain exactly 12 configs")
    ids = tuple(config.config_id for config in protocol.configs)
    if len(set(ids)) != len(ids):
        raise ValueError("local config IDs must be unique")
    if tuple(stage.name for stage in protocol.stages) != EXPECTED_STAGE_NAMES:
        raise ValueError("local stages must be screen, promote, confirm")
    if tuple(stage.candidate_count for stage in protocol.stages) != EXPECTED_STAGE_COUNTS:
        raise ValueError("local stage counts must be 12 -> 4 -> 2")
    for config in protocol.configs:
        _validate_config(config)
    fields = (
        ({config.input_window for config in protocol.configs}, ALLOWED_INPUT_WINDOWS),
        ({config.hidden_dim for config in protocol.configs}, ALLOWED_HIDDEN_DIMS),
        ({config.num_layers for config in protocol.configs}, ALLOWED_LAYERS),
        ({config.dropout for config in protocol.configs}, ALLOWED_DROPOUTS),
        ({config.learning_rate for config in protocol.configs}, ALLOWED_LEARNING_RATES),
        ({config.weight_decay for config in protocol.configs}, ALLOWED_WEIGHT_DECAYS),
        ({config.nse_aux_weight for config in protocol.configs}, ALLOWED_NSE_AUX_WEIGHTS),
    )
    if any(observed != set(allowed) for observed, allowed in fields):
        raise ValueError("the 12 configs must cover every approved value")
    if protocol.effective_batch_size != 96:
        raise ValueError("the selected hardware profile requires effective batch 96")
    expected_folds = ("v02_fold_a", "v02_fold_b", "v02_fold_c")
    expected = (
        (("v02_fold_a",), (42,), 25, 475),
        (expected_folds, (42,), 50, 950),
        (expected_folds, (42, 43, 44), 100, 1900),
    )
    for stage, (folds, seeds, epochs, steps) in zip(protocol.stages, expected, strict=True):
        if (stage.folds, stage.seeds, stage.max_epochs, stage.max_steps) != (
            folds,
            seeds,
            epochs,
            steps,
        ):
            raise ValueError(f"stage {stage.name} does not match the frozen budget")
        if stage.early_stopping_patience != 12:
            raise ValueError("all search stages use patience 12")


def load_search_protocol(path: str | Path) -> SearchProtocol:
    """Load and strictly validate the preregistered local-search YAML."""
    raw = _as_mapping(Path(path))
    raw_stages = raw.get("stages")
    raw_configs = raw.get("configs")
    if not isinstance(raw_stages, list) or not isinstance(raw_configs, list):
        raise ValueError("protocol stages/configs must be YAML lists")
    stages = tuple(
        SearchStage(
            name=str(item["name"]),
            candidate_count=int(item["candidate_count"]),
            folds=tuple(str(value) for value in item["folds"]),
            seeds=tuple(int(value) for value in item["seeds"]),
            max_epochs=int(item["max_epochs"]),
            max_steps=int(item["max_steps"]),
            early_stopping_patience=int(item["early_stopping_patience"]),
        )
        for item in raw_stages
    )
    configs = tuple(
        LocalSearchConfig(
            config_id=str(item["id"]),
            input_window=int(item["input_window"]),
            hidden_dim=int(item["hidden_dim"]),
            num_layers=int(item["num_layers"]),
            dropout=float(item["dropout"]),
            learning_rate=float(item["learning_rate"]),
            weight_decay=float(item["weight_decay"]),
            nse_aux_weight=float(item["nse_aux_weight"]),
        )
        for item in raw_configs
    )
    protocol = SearchProtocol(
        protocol_version=str(raw["protocol_version"]),
        family=str(raw["family"]),
        dataset=str(raw["dataset"]),
        model=str(raw["model"]),
        trainer=str(raw["trainer"]),
        experiment=str(raw["experiment"]),
        ledger_path=Path(str(raw["ledger_path"])),
        run_root=Path(str(raw["run_root"])),
        effective_batch_size=int(raw["effective_batch_size"]),
        stages=stages,
        configs=configs,
    )
    _validate_protocol(protocol)
    return protocol


def _registered_config(protocol: SearchProtocol, config: LocalSearchConfig) -> LocalSearchConfig:
    matches = tuple(item for item in protocol.configs if item.config_id == config.config_id)
    if len(matches) != 1 or matches[0] != config:
        raise ValueError(f"config {config.config_id} is not an unchanged preregistered config")
    return matches[0]


def _float_override(value: float) -> str:
    return format(value, ".10g")


def build_run_spec(
    protocol: SearchProtocol,
    config: LocalSearchConfig,
    stage: SearchStage,
    *,
    fold: str,
    seed: int,
    repo_root: str | Path,
    python_executable: str = "python",
) -> SearchRunSpec:
    """Derive one exact Hydra command from immutable protocol values."""
    config = _registered_config(protocol, config)
    if stage not in protocol.stages:
        raise ValueError(f"stage {stage.name} is not preregistered")
    if fold not in stage.folds or int(seed) not in stage.seeds:
        raise ValueError(f"fold/seed is outside stage {stage.name}")
    experiment_name = (
        f"v02_local_{stage.name}_{config.config_id}_{fold}_s{int(seed)}"
    )
    relative_run_dir = protocol.run_root / experiment_name
    description = experiment_name
    command = (
        str(python_executable),
        "-m",
        "RiverLagNet.cli.train",
        f"data={protocol.dataset}",
        f"model={protocol.model}",
        f"trainer={protocol.trainer}",
        f"experiment={protocol.experiment}",
        f"data.split_name={fold}",
        f"data.input_window={config.input_window}",
        f"model.hidden_dim={config.hidden_dim}",
        f"model.num_layers={config.num_layers}",
        f"model.dropout={_float_override(config.dropout)}",
        f"trainer.learning_rate={_float_override(config.learning_rate)}",
        f"trainer.weight_decay={_float_override(config.weight_decay)}",
        f"trainer.nse_aux_weight={_float_override(config.nse_aux_weight)}",
        f"trainer.effective_batch_size={protocol.effective_batch_size}",
        f"trainer.max_epochs={stage.max_epochs}",
        f"trainer.max_steps={stage.max_steps}",
        f"trainer.early_stopping_patience={stage.early_stopping_patience}",
        f"seed={int(seed)}",
        f"experiment.name={experiment_name}",
        f"experiment.description={description}",
        "experiment.status=baseline",
        "experiment.record_result=true",
        f"run_dir={relative_run_dir.as_posix()}",
    )
    return SearchRunSpec(
        config_id=config.config_id,
        stage=stage.name,
        fold=fold,
        seed=int(seed),
        experiment_name=experiment_name,
        run_dir=Path(repo_root) / relative_run_dir,
        command=command,
    )


def stage_run_specs(
    protocol: SearchProtocol,
    stage: SearchStage,
    configs: Sequence[LocalSearchConfig],
    *,
    repo_root: str | Path,
    python_executable: str = "python",
) -> tuple[SearchRunSpec, ...]:
    """Expand one stage into its frozen config/fold/seed run matrix."""
    if len(configs) != stage.candidate_count:
        raise ValueError(
            f"stage {stage.name} candidate count must be {stage.candidate_count}"
        )
    if len({config.config_id for config in configs}) != len(configs):
        raise ValueError("stage candidate IDs must be unique")
    return tuple(
        build_run_spec(
            protocol,
            config,
            stage,
            fold=fold,
            seed=seed,
            repo_root=repo_root,
            python_executable=python_executable,
        )
        for config in configs
        for fold in stage.folds
        for seed in stage.seeds
    )


def _successful_row(spec: SearchRunSpec, ledger_path: Path) -> dict[str, str] | None:
    if not ledger_path.is_file() or ledger_path.stat().st_size == 0:
        return None
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle, delimiter="\t"))
    for row in reversed(rows):
        if row.get("experiment") != spec.experiment_name:
            continue
        if row.get("seed") != str(spec.seed) or row.get("status") not in SUCCESS_STATUSES:
            continue
        try:
            nse = float(row.get("val_macro_nse", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(nse):
            return row
    return None


def is_completed_run(spec: SearchRunSpec, ledger_path: str | Path) -> bool:
    """Return true only for a valid ledger result with a saved checkpoint."""
    row = _successful_row(spec, Path(ledger_path))
    checkpoints = tuple((spec.run_dir / "checkpoints").glob("*.ckpt"))
    return row is not None and len(checkpoints) == 1


def read_completed_observation(
    spec: SearchRunSpec, ledger_path: str | Path
) -> SearchObservation:
    """Read ranking fields from durable training artifacts for one run."""
    row = _successful_row(spec, Path(ledger_path))
    checkpoints = tuple((spec.run_dir / "checkpoints").glob("*.ckpt"))
    if row is None or len(checkpoints) != 1:
        raise ValueError(f"run {spec.experiment_name} is not durably complete")
    hardware_path = spec.run_dir / "hardware.json"
    try:
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        parameters = int(hardware["trainable_parameters"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"run {spec.experiment_name} lacks valid parameter metadata"
        ) from error
    duration = float(row.get("duration_s", ""))
    nse = float(row["val_macro_nse"])
    if parameters <= 0 or not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"run {spec.experiment_name} has invalid ranking metadata")
    return SearchObservation(
        config_id=spec.config_id,
        stage=spec.stage,
        fold=spec.fold,
        seed=spec.seed,
        val_macro_nse=nse,
        duration_s=duration,
        trainable_parameters=parameters,
    )


def promote_configs(
    protocol: SearchProtocol,
    observations: Sequence[SearchObservation],
    *,
    candidate_count: int,
) -> tuple[LocalSearchConfig, ...]:
    """Rank config means with the frozen parameter/time/ID tie break."""
    grouped: dict[str, list[SearchObservation]] = {}
    for observation in observations:
        if observation.config_id not in {config.config_id for config in protocol.configs}:
            raise ValueError(f"unknown config observation: {observation.config_id}")
        if not math.isfinite(observation.val_macro_nse):
            raise ValueError("promotion observations require finite macro NSE")
        grouped.setdefault(observation.config_id, []).append(observation)
    if candidate_count <= 0 or candidate_count > len(grouped):
        raise ValueError("promotion candidate_count is outside observed configs")
    scores: list[tuple[float, int, float, str]] = []
    for config_id, values in grouped.items():
        parameter_values = {value.trainable_parameters for value in values}
        if len(parameter_values) != 1:
            raise ValueError(f"config {config_id} has inconsistent parameter counts")
        mean_nse = sum(value.val_macro_nse for value in values) / len(values)
        total_duration = sum(value.duration_s for value in values)
        scores.append((-mean_nse, parameter_values.pop(), total_duration, config_id))
    scores.sort()
    registry = {config.config_id: config for config in protocol.configs}
    return tuple(registry[score[3]] for score in scores[:candidate_count])
