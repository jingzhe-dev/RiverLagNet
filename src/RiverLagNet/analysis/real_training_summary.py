"""Ledger- and file-derived report for the fixed real daily seed-42 run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REAL_RUNS = {
    "persistence": "china_real_daily_persistence_seed42",
    "station_gru": "china_real_daily_station_gru_seed42_rerun",
    "static_gat": "china_real_daily_static_gat_seed42",
    "riverlagnet": "china_real_daily_riverlagnet_seed42",
}
SUCCESS_STATUSES = {"baseline", "keep", "discard"}
VALIDATION_FIELDS = (
    "val_macro_nse",
    "val_macro_mae",
    "val_macro_rmse",
    "duration_s",
    "peak_vram_gb",
)


def build_real_training_summary(
    ledger_path: Path, runs_root: Path
) -> dict[str, Any]:
    """Validate and combine exact ledger, checkpoint, and held-out test evidence."""
    ledger_path = Path(ledger_path)
    runs_root = Path(runs_root)
    rows: list[dict[str, str]]
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    conditions: dict[str, Any] = {}
    selected_rows: list[dict[str, str]] = []
    for condition, experiment in REAL_RUNS.items():
        matches = [
            row
            for row in rows
            if row.get("experiment") == experiment
            and row.get("status") in SUCCESS_STATUSES
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one successful ledger row for {experiment}")
        row = matches[0]
        selected_rows.append(row)
        validation = {field: _finite(row, field) for field in VALIDATION_FIELDS}
        run_dir = runs_root / experiment
        test_path = run_dir / "test_metrics.json"
        try:
            test = json.loads(test_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid held-out metrics: {test_path}") from error
        for name, value in test.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite held-out metric {name} for {experiment}")
        checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
        if len(checkpoints) != 1:
            raise ValueError(f"expected one validation checkpoint for {experiment}")
        conditions[condition] = {
            "experiment": experiment,
            "validation": validation,
            "test": test,
            "checkpoint": {
                "path": checkpoints[0].as_posix(),
                "sha256": _sha256(checkpoints[0]),
            },
        }
    commits = {row["commit"] for row in selected_rows}
    seeds = {int(row["seed"]) for row in selected_rows}
    if len(commits) != 1 or len(seeds) != 1:
        raise ValueError("real training rows must share one commit and seed")
    validation_winner = max(
        conditions, key=lambda name: conditions[name]["validation"]["val_macro_nse"]
    )
    descriptive_test_winner = max(
        conditions, key=lambda name: conditions[name]["test"]["test_macro_nse"]
    )
    full_test_nse = conditions["riverlagnet"]["test"]["test_macro_nse"]
    crash_rows = [
        {
            "timestamp": row.get("timestamp", ""),
            "experiment": row.get("experiment", ""),
            "description": row.get("description", ""),
        }
        for row in rows
        if row.get("status") == "crash"
        and row.get("experiment", "").startswith("china_real_daily_")
    ]
    return {
        "dataset": "china-real-daily-v0.1",
        "commit": next(iter(commits)),
        "seed": next(iter(seeds)),
        "protocol": {
            "input_days": 90,
            "output_days": 30,
            "split": [0.70, 0.15, 0.15],
            "selection_metric": "validation macro NSE",
            "precision": "bf16-mixed",
            "max_epochs": 50,
            "early_stopping_patience": 8,
        },
        "conditions": conditions,
        "validation_selected_model": validation_winner,
        "descriptive_test_best_model": descriptive_test_winner,
        "riverlagnet_test_macro_nse_deltas": {
            name: full_test_nse - conditions[name]["test"]["test_macro_nse"]
            for name in ("persistence", "station_gru", "static_gat")
        },
        "crash_audit": crash_rows,
    }


def render_real_training_markdown(summary: Mapping[str, Any]) -> str:
    """Render the fixed real training report without using test data for selection."""
    conditions = summary["conditions"]
    lines = [
        "# RiverLagNet real daily seed-42 training report",
        "",
        "## Result",
        "",
        (
            f"Validation selected **{summary['validation_selected_model']}**. On the held-out test period, "
            f"the descriptively highest macro NSE was **{summary['descriptive_test_best_model']}**. "
            "Test data were not used to select checkpoints or tune the model."
        ),
        "",
        "## Fixed protocol",
        "",
        f"- Training commit: `{summary['commit']}`",
        f"- Seed: {summary['seed']}",
        "- Daily history/forecast: 90/30 days",
        "- Chronological split: 70%/15%/15%",
        "- Selection: validation macro NSE; maximum 50 epochs; patience 8",
        "- Precision/device: BF16 mixed precision on one RTX PRO 6000",
        "- Imputed source values: excluded from scaling, loss, and metrics",
        "",
        "## Validation and held-out test",
        "",
        "MAE and RMSE are in mg/L because all three targets use that unit.",
        "",
        "| Model | Val NSE | Val MAE | Val RMSE | Test NSE | Test MAE | Test RMSE | Duration (s) | Peak VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "persistence": "Persistence",
        "station_gru": "Station GRU",
        "static_gat": "Static Directed GAT",
        "riverlagnet": "RiverLagNet",
    }
    for name in REAL_RUNS:
        validation = conditions[name]["validation"]
        test = conditions[name]["test"]
        lines.append(
            f"| {labels[name]} | {validation['val_macro_nse']:.4f} | "
            f"{validation['val_macro_mae']:.4f} | {validation['val_macro_rmse']:.4f} | "
            f"{test['test_macro_nse']:.4f} | {test['test_macro_mae']:.4f} | "
            f"{test['test_macro_rmse']:.4f} | {validation['duration_s']:.2f} | "
            f"{validation['peak_vram_gb']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-target held-out metrics",
            "",
            "| Model | NH3N NSE / MAE / RMSE | CODMn NSE / MAE / RMSE | TP NSE / MAE / RMSE |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in REAL_RUNS:
        test = conditions[name]["test"]
        cells = []
        for target in ("NH3N", "CODMn", "TP"):
            cells.append(
                f"{test[f'test_nse_{target}']:.4f} / {test[f'test_mae_{target}']:.4f} / "
                f"{test[f'test_rmse_{target}']:.4f}"
            )
        lines.append(f"| {labels[name]} | " + " | ".join(cells) + " |")
    deltas = summary["riverlagnet_test_macro_nse_deltas"]
    static_delta = deltas["static_gat"]
    static_comparison = (
        f"RiverLagNet was {static_delta:.4f} higher than Static Directed GAT"
        if static_delta >= 0
        else f"Static Directed GAT was {abs(static_delta):.4f} higher than RiverLagNet"
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"RiverLagNet improved held-out macro NSE over Persistence by {deltas['persistence']:.4f} "
            f"and over Station GRU by {deltas['station_gru']:.4f}. {static_comparison} "
            "on the held-out period, despite "
            "RiverLagNet having the best validation NSE. The defensible single-seed conclusion is that graph information helped, while a stable learned-lag advantage over a static directed graph is not yet established.",
            "",
            "## Execution audit and limitations",
            "",
        ]
    )
    crashes = summary["crash_audit"]
    if crashes:
        for crash in crashes:
            lines.append(
                f"- Retained crash row: `{crash['experiment']}` — {crash['description']}"
            )
    else:
        lines.append("- No crash rows were recorded.")
    lines.extend(
        [
            "- One station-to-river mapping remains flagged for manual review.",
            "- Travel-time priors use an assumed 30 km/day velocity, not measured hydraulics.",
            "- This report covers one seed and one regional graph; multi-seed runs and lag/no-lag ablations are required before a mechanism claim.",
            "- Attention weights are routing weights and must not be interpreted as causal contributions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_real_training_summary(
    summary: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> None:
    """Write reproducible LF-only JSON and Markdown outputs."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_real_training_markdown(summary))


def _finite(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in {row.get('experiment', '<unknown>')}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in {row.get('experiment', '<unknown>')}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
