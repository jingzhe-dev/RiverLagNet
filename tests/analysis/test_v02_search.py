from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re

import pytest

from RiverLagNet.analysis.v02_search import (
    SearchObservation,
    build_run_spec,
    is_completed_run,
    load_search_protocol,
    promote_configs,
    read_completed_observation,
    stage_run_specs,
)
from RiverLagNet.cli import run_v02_search
from RiverLagNet.training.experiment_log import ExperimentRecord, append_experiment_record


PROTOCOL_PATH = Path("experiments/v0.2_local_search.yaml")


def _protocol():
    return load_search_protocol(PROTOCOL_PATH)


def _record(experiment: str, *, status: str = "baseline", nse: float = 0.5) -> ExperimentRecord:
    return ExperimentRecord(
        timestamp="2026-07-17T00:00:00+00:00",
        commit="abc123",
        branch="research/v02",
        experiment=experiment,
        seed=42,
        val_macro_nse=nse if status != "crash" else None,
        val_macro_mae=0.2 if status != "crash" else None,
        val_macro_rmse=0.3 if status != "crash" else None,
        duration_s=12.0,
        peak_vram_gb=18.0,
        status=status,
        description="safe_description",
    )


def test_protocol_freezes_exactly_twelve_balanced_local_configs() -> None:
    protocol = _protocol()

    assert protocol.family == "local"
    assert len(protocol.configs) == 12
    assert len({config.config_id for config in protocol.configs}) == 12
    assert {config.input_window for config in protocol.configs} == {90, 180, 365}
    assert {config.hidden_dim for config in protocol.configs} == {128, 256}
    assert {config.num_layers for config in protocol.configs} == {2, 4}
    assert {config.dropout for config in protocol.configs} == {0.05, 0.10, 0.20}
    assert {config.learning_rate for config in protocol.configs} == {
        3e-4,
        6e-4,
        1e-3,
    }
    assert {config.weight_decay for config in protocol.configs} == {1e-5, 1e-4, 1e-3}
    assert {config.nse_aux_weight for config in protocol.configs} == {0.0, 0.05, 0.10}
    with pytest.raises(FrozenInstanceError):
        protocol.configs[0].hidden_dim = 999  # type: ignore[misc]


def test_protocol_freezes_successive_halving_as_twelve_four_two() -> None:
    protocol = _protocol()

    assert [stage.name for stage in protocol.stages] == ["screen", "promote", "confirm"]
    assert [stage.candidate_count for stage in protocol.stages] == [12, 4, 2]
    assert [stage.max_epochs for stage in protocol.stages] == [25, 50, 100]
    assert [stage.max_steps for stage in protocol.stages] == [475, 950, 1900]
    assert protocol.stages[0].folds == ("v02_fold_a",)
    assert protocol.stages[0].seeds == (42,)
    assert protocol.stages[1].folds == (
        "v02_fold_a",
        "v02_fold_b",
        "v02_fold_c",
    )
    assert protocol.stages[1].seeds == (42,)
    assert protocol.stages[2].folds == protocol.stages[1].folds
    assert protocol.stages[2].seeds == (42, 43, 44)
    assert protocol.stages[2].early_stopping_patience == 12


def test_stage_commands_include_frozen_budget_hardware_and_safe_description(tmp_path: Path) -> None:
    protocol = _protocol()
    stage = protocol.stages[0]
    specs = stage_run_specs(
        protocol,
        stage,
        protocol.configs,
        repo_root=tmp_path,
        python_executable="python",
    )

    assert len(specs) == 12
    command = specs[0].command
    assert command[:4] == ("python", "-m", "RiverLagNet.cli.train", "data=china_real_daily_contracted_1068_v02")
    assert "model=local_multiscale" in command
    assert "trainer=blackwell_96gb" in command
    assert "data.split_name=v02_fold_a" in command
    assert "seed=42" in command
    assert "trainer.effective_batch_size=96" in command
    assert "trainer.max_epochs=25" in command
    assert "trainer.max_steps=475" in command
    assert "trainer.early_stopping_patience=12" in command
    assert "data.input_window=90" in command
    assert "model.hidden_dim=128" in command
    assert "run_dir=runs/v02_local_search/v02_local_screen_local_01_v02_fold_a_s42" in command
    description = next(value.split("=", 1)[1] for value in command if value.startswith("experiment.description="))
    assert re.fullmatch(r"[A-Za-z0-9_]+", description)


def test_stage_run_specs_rejects_mutated_or_wrong_size_candidate_set(tmp_path: Path) -> None:
    protocol = _protocol()

    with pytest.raises(ValueError, match="candidate count"):
        stage_run_specs(
            protocol,
            protocol.stages[1],
            protocol.configs[:3],
            repo_root=tmp_path,
        )

    foreign = protocol.configs[0].__class__(
        config_id="foreign",
        input_window=90,
        hidden_dim=128,
        num_layers=2,
        dropout=0.05,
        learning_rate=3e-4,
        weight_decay=1e-5,
        nse_aux_weight=0.0,
    )
    with pytest.raises(ValueError, match="preregistered"):
        build_run_spec(
            protocol,
            foreign,
            protocol.stages[0],
            fold="v02_fold_a",
            seed=42,
            repo_root=tmp_path,
        )


def test_resume_requires_a_finite_ledger_row_and_checkpoint(tmp_path: Path) -> None:
    protocol = _protocol()
    spec = build_run_spec(
        protocol,
        protocol.configs[0],
        protocol.stages[0],
        fold="v02_fold_a",
        seed=42,
        repo_root=tmp_path,
    )
    ledger = tmp_path / "results.tsv"

    assert not is_completed_run(spec, ledger)
    append_experiment_record(ledger, _record(spec.experiment_name))
    assert not is_completed_run(spec, ledger)

    checkpoint = spec.run_dir / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (spec.run_dir / "hardware.json").write_text(
        json.dumps({"trainable_parameters": 1234}), encoding="utf-8"
    )
    assert is_completed_run(spec, ledger)
    observation = read_completed_observation(spec, ledger)
    assert observation.config_id == "local_01"
    assert observation.val_macro_nse == 0.5
    assert observation.trainable_parameters == 1234


@pytest.mark.parametrize("status,nse", [("crash", 0.5), ("baseline", float("nan"))])
def test_resume_rejects_crash_and_nonfinite_rows(
    tmp_path: Path, status: str, nse: float
) -> None:
    protocol = _protocol()
    spec = build_run_spec(
        protocol,
        protocol.configs[0],
        protocol.stages[0],
        fold="v02_fold_a",
        seed=42,
        repo_root=tmp_path,
    )
    ledger = tmp_path / "results.tsv"
    append_experiment_record(ledger, _record(spec.experiment_name, status=status, nse=nse))
    checkpoint = spec.run_dir / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert not is_completed_run(spec, ledger)


def test_promotion_uses_mean_nse_then_parameters_wall_time_and_id() -> None:
    protocol = _protocol()
    observations = (
        SearchObservation("local_01", "screen", "v02_fold_a", 42, 0.70, 20.0, 200),
        SearchObservation("local_02", "screen", "v02_fold_a", 42, 0.70, 10.0, 300),
        SearchObservation("local_03", "screen", "v02_fold_a", 42, 0.70, 10.0, 200),
        SearchObservation("local_04", "screen", "v02_fold_a", 42, 0.70, 10.0, 200),
        SearchObservation("local_05", "screen", "v02_fold_a", 42, 0.69, 1.0, 1),
    )

    promoted = promote_configs(protocol, observations, candidate_count=4)

    assert [config.config_id for config in promoted] == [
        "local_03",
        "local_04",
        "local_01",
        "local_02",
    ]
    assert all(config is protocol.configs[int(config.config_id[-2:]) - 1] for config in promoted)


def test_dry_run_prints_protocol_and_commands_without_starting_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry-run started training")

    monkeypatch.setattr("subprocess.run", unexpected_run)

    exit_code = run_v02_search.run(["--family", "local", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "stage_counts=12->4->2" in output
    assert output.count("RiverLagNet.cli.train") == 12
