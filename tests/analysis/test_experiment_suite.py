from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from RiverLagNet.analysis.experiment_suite import (
    CONDITION_NAMES,
    IDENTIFIABLE_FUSION_V2,
    IDENTIFIABLE_HORIZON_V5,
    IDENTIFIABLE_V1,
    REAL_LAG_V1,
    ROBUSTNESS_V1,
    build_experiment_specs,
    evaluation_command,
    final_checkpoint_path,
    final_evaluation_output,
    pending_experiment_specs,
    run_experiment_specs,
    successful_experiment_names,
    training_command,
)
from RiverLagNet.analysis.synthetic_identifiability import IdentifiabilityGateReport
from RiverLagNet.cli import run_experiment_suite as suite_cli
from RiverLagNet.training.experiment_log import (
    ExperimentRecord,
    append_experiment_record,
)


def _record(experiment: str, status: str) -> ExperimentRecord:
    return ExperimentRecord(
        timestamp="2026-07-13T00:00:00+00:00",
        commit="abc123",
        branch="research/test",
        experiment=experiment,
        seed=42,
        val_macro_nse=0.1 if status != "crash" else None,
        val_macro_mae=0.2 if status != "crash" else None,
        val_macro_rmse=0.3 if status != "crash" else None,
        duration_s=1.0,
        peak_vram_gb=0.1,
        status=status,
        description="test",
    )


def test_build_experiment_specs_creates_exact_paired_matrix() -> None:
    specs = build_experiment_specs([42, 43, 44, 45, 46])

    assert len(specs) == 45
    assert len({spec.experiment_name for spec in specs}) == 45
    assert set(CONDITION_NAMES) == {
        "persistence",
        "station_gru",
        "static_gat",
        "no_graph",
        "undirected_graph",
        "shuffled_graph",
        "no_lag",
        "fixed_lag",
        "learned_lag",
    }
    for seed in range(42, 47):
        assert {spec.condition.name for spec in specs if spec.seed == seed} == set(
            CONDITION_NAMES
        )


def test_training_command_contains_condition_specific_hydra_overrides() -> None:
    specs = build_experiment_specs([42])

    commands = {
        spec.condition.name: training_command(spec, "python") for spec in specs
    }

    assert commands["persistence"][:4] == [
        "python",
        "-m",
        "RiverLagNet.cli.train",
        "seed=42",
    ]
    assert "model=persistence" in commands["persistence"]
    assert "model.graph_variant=no_graph" in commands["no_graph"]
    assert "model.graph_variant=undirected" in commands["undirected_graph"]
    assert "model.graph_variant=shuffled" in commands["shuffled_graph"]
    assert "model.lag_mode=no_lag" in commands["no_lag"]
    assert "model.lag_mode=fixed_lag" in commands["fixed_lag"]
    assert "model.lag_mode=learned_lag" in commands["learned_lag"]
    assert "experiment.status=baseline" in commands["learned_lag"]


def test_successful_names_skip_completed_but_not_crashed_rows(tmp_path: Path) -> None:
    specs = build_experiment_specs([42])
    ledger = tmp_path / "results.tsv"
    statuses = ("baseline", "keep", "discard", "crash")
    for spec, status in zip(specs[:4], statuses, strict=True):
        append_experiment_record(ledger, _record(spec.experiment_name, status))

    successful = successful_experiment_names(ledger)
    pending = pending_experiment_specs(specs, successful)

    assert successful == {spec.experiment_name for spec in specs[:3]}
    assert specs[3] in pending
    assert all(spec not in pending for spec in specs[:3])


def test_dry_run_returns_commands_without_invoking_subprocess(
    monkeypatch,
) -> None:
    specs = build_experiment_specs([42])[:2]

    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry-run invoked subprocess")

    monkeypatch.setattr("subprocess.run", unexpected_run)

    commands = run_experiment_specs(specs, python_executable="python", dry_run=True)

    assert len(commands) == 2
    assert all(command[0] == "python" for command in commands)


def test_runner_does_not_force_utf8_mode_for_unicode_editable_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = build_experiment_specs([42])[:1]
    environments: list[dict[str, str]] = []

    def capture_run(command, check, env):
        environments.append(env)

    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setattr("subprocess.run", capture_run)

    run_experiment_specs(specs, python_executable="python")

    assert len(environments) == 1
    assert "PYTHONUTF8" not in environments[0]
    assert environments[0]["PYTHONIOENCODING"] == "utf-8"


def test_final_evaluation_command_uses_only_selected_learned_lag_checkpoint(
    tmp_path: Path,
) -> None:
    learned = next(
        spec
        for spec in build_experiment_specs([42])
        if spec.condition.name == "learned_lag"
    )
    learned = replace(learned, run_dir=tmp_path / learned.experiment_name)
    checkpoint = learned.run_dir / "checkpoints" / "epoch=004-val_nse=0.7000.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    selected = final_checkpoint_path(learned)
    command = evaluation_command(learned, selected, "python")

    assert selected == checkpoint
    assert "model=riverlagnet" in command
    assert "seed=42" in command
    assert "model.graph_variant=directed" in command
    assert "model.lag_mode=learned_lag" in command
    assert f'checkpoint_path="{checkpoint.as_posix()}"' in command
    assert f"evaluation_output={final_evaluation_output(learned).as_posix()}" in command


def test_final_checkpoint_path_requires_exactly_one_checkpoint(tmp_path: Path) -> None:
    spec = replace(build_experiment_specs([42])[-1], run_dir=tmp_path / "run")

    with pytest.raises(ValueError, match="exactly one"):
        final_checkpoint_path(spec)

    directory = spec.run_dir / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "a.ckpt").touch()
    (directory / "b.ckpt").touch()
    with pytest.raises(ValueError, match="exactly one"):
        final_checkpoint_path(spec)


def test_historical_preset_keeps_exact_names_and_command() -> None:
    spec = build_experiment_specs([42], ROBUSTNESS_V1)[0]

    assert spec.experiment_name == "robust_s42_persistence"
    command = training_command(spec, "python")
    assert "data=synthetic_identifiable_v1" not in command


def test_identifiable_preset_builds_unique_data_aware_specs() -> None:
    specs = build_experiment_specs([42, 43, 44, 45, 46], IDENTIFIABLE_V1)

    assert len(specs) == 45
    assert len({spec.experiment_name for spec in specs}) == 45
    assert specs[0].experiment_name == "ident_v1_s42_persistence"
    assert "data=synthetic_identifiable_v1" in training_command(specs[0], "python")
    learned = next(spec for spec in specs if spec.condition.name == "learned_lag")
    checkpoint = learned.run_dir / "checkpoints" / "best.ckpt"
    assert "data=synthetic_identifiable_v1" in evaluation_command(
        learned, checkpoint, "python"
    )


def test_identity_safe_fusion_preset_has_new_names_and_outputs() -> None:
    specs = build_experiment_specs([42, 43], IDENTIFIABLE_FUSION_V2)

    assert len(specs) == 18
    assert specs[0].experiment_name == "ident_fusion_v2_s42_persistence"
    assert "data=synthetic_identifiable_v1" in training_command(specs[0], "python")
    assert IDENTIFIABLE_FUSION_V2.summary_json.name == "identifiable_fusion_v2_summary.json"


def test_horizon_routing_preset_has_frozen_names_and_outputs() -> None:
    specs = build_experiment_specs([42, 43], IDENTIFIABLE_HORIZON_V5)

    assert len(specs) == 18
    assert specs[0].experiment_name == "ident_horizon_v5_s42_persistence"
    assert "data=synthetic_identifiable_v1" in training_command(specs[0], "python")
    assert IDENTIFIABLE_HORIZON_V5.summary_json.name == "identifiable_horizon_v5_summary.json"


def test_real_lag_preset_limits_matrix_and_uses_formal_real_data_config() -> None:
    specs = build_experiment_specs([42, 43], REAL_LAG_V1)

    assert len(specs) == 6
    assert {spec.condition.name for spec in specs} == {
        "no_lag",
        "fixed_lag",
        "learned_lag",
    }
    command = training_command(specs[0], "python")
    assert "data=china_real_daily" in command
    assert "trainer=formal_gpu" in command
    assert "experiment=china_real_daily" in command
    assert any(value.startswith("experiment.description=Real-source") for value in command)
    learned = next(spec for spec in specs if spec.condition.name == "learned_lag")
    checkpoint = learned.run_dir / "checkpoints" / "best.ckpt"
    evaluation = evaluation_command(learned, checkpoint, "python")
    assert "data=china_real_daily" in evaluation
    assert "trainer=formal_gpu" in evaluation
    assert REAL_LAG_V1.graph_data_root == Path("data/processed/china-real-daily-v0.1")
    assert REAL_LAG_V1.graph_figure_png is not None


@pytest.mark.parametrize(
    ("suite_name", "experiment_prefix"),
    [
        ("identifiable_v1", "ident_v1"),
        ("identifiable_fusion_v2", "ident_fusion_v2"),
        ("identifiable_horizon_v5", "ident_horizon_v5"),
    ],
)
def test_identifiable_cli_dry_run_gates_and_prints_preset_commands(
    suite_name: str,
    experiment_prefix: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = IdentifiabilityGateReport(
        passed=True, scenarios=(), pooled_lag_recovery_rate=1.0
    )
    calls: list[str] = []
    monkeypatch.setattr(suite_cli, "run_identifiability_gate", lambda seeds: gate)
    monkeypatch.setattr(
        suite_cli,
        "write_identifiability_gate",
        lambda report, json_path, markdown_path: calls.append(str(json_path)),
    )

    suite_cli.run(
        [
            "--suite",
            suite_name,
            "--seeds",
            "42",
            "--ledger",
            str(tmp_path / "results.tsv"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "suite_total=9" in output
    assert f"{experiment_prefix}_s42_persistence" in output
    assert "data=synthetic_identifiable_v1" in output
    assert calls == ["experiments\\identifiable_v1_data_gate.json"] or calls == [
        "experiments/identifiable_v1_data_gate.json"
    ]
