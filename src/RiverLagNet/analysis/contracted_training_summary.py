"""Summarize and visualize the first contracted-graph formal benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("persistence", "station_gru", "static_gat", "riverlagnet")
MODEL_LABELS = {
    "persistence": "Persistence",
    "station_gru": "Station GRU",
    "static_gat": "Static directed GAT",
    "riverlagnet": "RiverLagNet",
}
EXPERIMENT_NAMES = {
    model: f"china_real_daily_contracted_{model}_seed42" for model in MODEL_ORDER
}

# 可调参数：画布、字体与导出
FIGURE_SIZE = (13.8, 4.5)
EXPORT_DPI = 300
TITLE_SIZE = 14.0
SUBTITLE_SIZE = 9.2
PANEL_TITLE_SIZE = 10.5
TICK_SIZE = 8.4
VALUE_SIZE = 8.1
FOOTNOTE_SIZE = 8.2
PANEL_WSPACE = 0.36

# 可调参数：两根学术配色，蓝色表示对照，橙色突出完整模型
MODEL_COLORS = {
    "persistence": "#C9D9E2",
    "station_gru": "#8FB2C4",
    "static_gat": "#4E819D",
    "riverlagnet": "#C47A2C",
}
TEXT_COLOR = "#252A2E"
MUTED_TEXT_COLOR = "#667078"
GRID_COLOR = "#E2E7EA"


def build_contracted_training_summary(results_path: str | Path) -> dict[str, Any]:
    """Select the latest valid seed-42 row for each required benchmark model."""
    results_path = Path(results_path)
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected: dict[str, dict[str, str]] = {}
    for model, experiment in EXPERIMENT_NAMES.items():
        candidates = [
            row
            for row in rows
            if row.get("experiment") == experiment and row.get("status") != "crash"
        ]
        if not candidates:
            raise ValueError(f"missing completed experiment row: {experiment}")
        selected[model] = max(candidates, key=lambda row: row["timestamp"])

    models: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row = selected[model]
        metrics = {
            name: _finite(row[f"val_macro_{name}"], f"{model}.{name}")
            for name in ("nse", "mae", "rmse")
        }
        models.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "experiment": row["experiment"],
                "seed": int(row["seed"]),
                "status": row["status"],
                "commit": row["commit"],
                "validation": metrics,
                "duration_s": _finite(row["duration_s"], f"{model}.duration_s"),
                "peak_vram_gb": _finite(
                    row["peak_vram_gb"], f"{model}.peak_vram_gb"
                ),
            }
        )
    by_model = {item["model"]: item for item in models}
    full_nse = by_model["riverlagnet"]["validation"]["nse"]
    return {
        "dataset": "china-real-daily-contracted-v0.2",
        "selection_metric": "validation macro NSE",
        "seed": 42,
        "test_set_used": False,
        "model_count": len(models),
        "models": models,
        "riverlagnet_nse_delta_vs_station_gru": full_nse
        - by_model["station_gru"]["validation"]["nse"],
        "riverlagnet_nse_delta_vs_static_gat": full_nse
        - by_model["static_gat"]["validation"]["nse"],
        "interpretation": (
            "single-seed pilot; requires paired seeds and lag ablations before a "
            "mechanism claim"
        ),
    }


def write_contracted_training_summary(
    summary: Mapping[str, Any], path: str | Path
) -> Path:
    """Write the benchmark summary with stable UTF-8 LF formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return path


def render_contracted_training_figure(
    summary: Mapping[str, Any], png_path: str | Path, pdf_path: str | Path
) -> tuple[Path, Path]:
    """Render validation NSE, MAE, and RMSE as aligned model comparisons."""
    models = summary.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_ORDER):
        raise ValueError("summary must contain the four benchmark models")
    model_names = [str(item["model"]) for item in models]
    if tuple(model_names) != MODEL_ORDER:
        raise ValueError("benchmark models are not in the declared order")
    labels = [MODEL_LABELS[name] for name in model_names]
    colors = [MODEL_COLORS[name] for name in model_names]
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.72, bottom=0.28, wspace=PANEL_WSPACE)
    metric_specs = (
        ("nse", "Validation macro NSE", "Higher is better", 4),
        ("mae", "Validation macro MAE", "Lower is better · physical units", 4),
        ("rmse", "Validation macro RMSE", "Lower is better · physical units", 4),
    )
    x_positions = np.arange(len(models))
    for axis, (metric, title, subtitle, decimals) in zip(
        axes, metric_specs, strict=True
    ):
        values = [float(item["validation"][metric]) for item in models]
        bars = axis.bar(
            x_positions,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.7,
            width=0.72,
            zorder=3,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.{decimals}f}" for value in values],
            padding=3,
            fontsize=VALUE_SIZE,
            color=TEXT_COLOR,
        )
        axis.set_title(
            title,
            fontsize=PANEL_TITLE_SIZE,
            fontweight="bold",
            loc="left",
            pad=26,
        )
        axis.text(
            0.0,
            1.015,
            subtitle,
            transform=axis.transAxes,
            fontsize=TICK_SIZE,
            color=MUTED_TEXT_COLOR,
            ha="left",
            va="bottom",
        )
        axis.set_xticks(x_positions, labels, rotation=25, ha="right", fontsize=TICK_SIZE)
        axis.tick_params(axis="y", labelsize=TICK_SIZE, colors=MUTED_TEXT_COLOR)
        axis.set_ylim(0.0, max(values) * 1.18)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_color("#AEB6BC")
        axis.spines["bottom"].set_color("#AEB6BC")
    figure.text(
        0.055,
        0.94,
        "Contracted real-data validation metrics",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.885,
        "Seed 42 · identical chronological split and training budget · test set not used",
        fontsize=SUBTITLE_SIZE,
        color=MUTED_TEXT_COLOR,
        ha="left",
        va="top",
    )
    station_delta = float(summary["riverlagnet_nse_delta_vs_station_gru"])
    graph_delta = float(summary["riverlagnet_nse_delta_vs_static_gat"])
    figure.text(
        0.055,
        0.06,
        (
            f"RiverLagNet NSE delta: {station_delta:+.4f} vs Station GRU; "
            f"{graph_delta:+.4f} vs Static directed GAT. Single-seed evidence only."
        ),
        fontsize=FOOTNOTE_SIZE,
        color=TEXT_COLOR,
        ha="left",
        va="bottom",
    )
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid metric value for {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric value for {label}")
    return number
