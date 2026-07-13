from __future__ import annotations

import json
from dataclasses import replace
from math import sqrt
from pathlib import Path

import pytest

from RiverLagNet.analysis.experiment_suite import build_experiment_specs
from RiverLagNet.analysis.robustness_summary import (
    load_successful_suite_rows,
    render_validation_markdown,
    summarize_validation,
    write_validation_summary,
)
from RiverLagNet.training.experiment_log import ExperimentRecord, append_experiment_record


def _record(name: str, seed: int, nse: float, *, commit: str = "suitecommit") -> ExperimentRecord:
    return ExperimentRecord(
        timestamp="2026-07-13T00:00:00+00:00",
        commit=commit,
        branch="research/test",
        experiment=name,
        seed=seed,
        val_macro_nse=nse,
        val_macro_mae=1.0 - nse / 2.0,
        val_macro_rmse=1.2 - nse / 2.0,
        duration_s=2.0 + seed,
        peak_vram_gb=0.2,
        status="baseline",
        description="synthetic robustness test",
    )


def _miniature_ledger(path: Path) -> tuple:
    specs = build_experiment_specs([42, 43])
    offsets = {
        "persistence": 0.1,
        "station_gru": 0.5,
        "static_gat": 0.55,
        "no_graph": 0.60,
        "undirected_graph": 0.62,
        "shuffled_graph": 0.58,
        "no_lag": 0.65,
        "fixed_lag": 0.67,
        "learned_lag": 0.70,
    }
    for spec in specs:
        seed_shift = 0.10 if spec.seed == 43 else 0.0
        append_experiment_record(
            path,
            _record(spec.experiment_name, spec.seed, offsets[spec.condition.name] + seed_shift),
        )
    return specs


def test_summarize_validation_calculates_statistics_and_paired_deltas(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "results.tsv"
    specs = _miniature_ledger(ledger)

    rows = load_successful_suite_rows(ledger, specs)
    summary = summarize_validation(rows, specs)

    learned = summary["conditions"]["learned_lag"]["val_macro_nse"]
    assert learned["count"] == 2
    assert learned["mean"] == pytest.approx(0.75)
    assert learned["std"] == pytest.approx(sqrt(0.005))
    assert learned["min"] == pytest.approx(0.70)
    assert learned["max"] == pytest.approx(0.80)
    no_graph = summary["paired_deltas"]["no_graph"]
    assert no_graph["mean_delta_macro_nse"] == pytest.approx(0.10)
    assert no_graph["wins"] == 2
    assert no_graph["seed_deltas"] == {"42": pytest.approx(0.10), "43": pytest.approx(0.10)}
    assert summary["seeds"] == [42, 43]
    assert summary["commit"] == "suitecommit"


def test_summary_outputs_json_and_decision_ready_markdown(tmp_path: Path) -> None:
    ledger = tmp_path / "results.tsv"
    specs = _miniature_ledger(ledger)
    summary = summarize_validation(load_successful_suite_rows(ledger, specs), specs)
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"

    write_validation_summary(summary, json_path, markdown_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["seeds"] == [42, 43]
    assert "Condition-level validation results" in markdown
    assert "Paired RiverLagNet mechanism deltas" in markdown
    assert "42, 43" in markdown
    assert "synthetic" in markdown.lower()
    assert render_validation_markdown(summary) == markdown


def test_load_successful_suite_rows_rejects_missing_or_duplicate_success(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "results.tsv"
    specs = _miniature_ledger(ledger)
    missing = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(missing[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_successful_suite_rows(ledger, specs)

    last_spec = specs[-1]
    append_experiment_record(
        ledger, _record(last_spec.experiment_name, last_spec.seed, 0.8)
    )
    append_experiment_record(
        ledger, _record(last_spec.experiment_name, last_spec.seed, 0.81)
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_successful_suite_rows(ledger, specs)


def test_load_successful_suite_rows_rejects_mixed_commits(tmp_path: Path) -> None:
    ledger = tmp_path / "results.tsv"
    specs = _miniature_ledger(ledger)
    extra = replace(
        _record(specs[0].experiment_name, specs[0].seed, 0.2),
        status="crash",
        commit="other",
        val_macro_nse=None,
        val_macro_mae=None,
        val_macro_rmse=None,
    )
    append_experiment_record(ledger, extra)
    rows = load_successful_suite_rows(ledger, specs)
    assert len(rows) == len(specs)

    lines = ledger.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split("\t")
    fields[1] = "other"
    lines[1] = "\t".join(fields)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commit"):
        load_successful_suite_rows(ledger, specs)
