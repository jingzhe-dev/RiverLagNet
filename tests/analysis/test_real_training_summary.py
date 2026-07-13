from __future__ import annotations

import json
from pathlib import Path

from RiverLagNet.analysis.real_training_summary import (
    REAL_RUNS,
    build_real_training_summary,
    render_real_training_markdown,
    write_real_training_summary,
)
from RiverLagNet.training.experiment_log import ExperimentRecord, append_experiment_record


def test_real_summary_is_derived_from_ledger_checkpoints_and_test_files(tmp_path: Path) -> None:
    ledger = tmp_path / "results.tsv"
    runs = tmp_path / "runs"
    for index, (condition, experiment) in enumerate(REAL_RUNS.items()):
        append_experiment_record(
            ledger,
            ExperimentRecord(
                timestamp="2026-07-14T00:00:00+00:00",
                commit="abc123",
                branch="research/test",
                experiment=experiment,
                seed=42,
                val_macro_nse=0.2 + index * 0.1,
                val_macro_mae=0.3,
                val_macro_rmse=0.4,
                duration_s=10.0,
                peak_vram_gb=0.5,
                status="baseline",
                description="real fixture",
            ),
        )
        run_dir = runs / experiment
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "best.ckpt").write_bytes(condition.encode())
        metrics = {
            "test_loss": 0.1,
            "test_macro_nse": 0.4 + index * 0.1,
            "test_macro_mae": 0.2,
            "test_macro_rmse": 0.3,
        }
        for target in ("NH3N", "CODMn", "TP"):
            metrics[f"test_nse_{target}"] = 0.5
            metrics[f"test_mae_{target}"] = 0.1
            metrics[f"test_rmse_{target}"] = 0.2
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    summary = build_real_training_summary(ledger, runs)
    markdown = render_real_training_markdown(summary)

    assert summary["commit"] == "abc123"
    assert summary["validation_selected_model"] == "riverlagnet"
    assert summary["descriptive_test_best_model"] == "riverlagnet"
    assert "Test data were not used" in markdown
    assert "Static Directed GAT" in markdown

    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    write_real_training_summary(summary, json_path, markdown_path)
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in markdown_path.read_bytes()
