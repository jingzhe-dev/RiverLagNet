"""Publication-style visual summaries derived from experiment summary JSON."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 可调参数：画布与导出
FIGURE_SIZE = (12.0, 4.1)
EXPORT_DPI = 300
PANEL_WSPACE = 0.42

# 可调参数：字体与线条
FONT_SIZE = 9.0
LABEL_SIZE = 9.5
PANEL_LABEL_SIZE = 11.0
LINE_WIDTH = 0.8
ERROR_LINE_WIDTH = 1.1
MARKER_SIZE = 34.0
MEAN_MARKER_SIZE = 48.0
POINT_SPREAD = 0.13
AXIS_PADDING = 0.18

# 可调参数：学术配色
CONDITION_COLORS = {
    "no_lag": "#6F8FAF",
    "fixed_lag": "#D6A45F",
    "learned_lag": "#3E7468",
}
DEFAULT_COLOR = "#8D98A1"
MEAN_COLOR = "#202428"
TARGET_COLORS = {
    "Macro": "#30363B",
    "NH3N": "#3E7468",
    "CODMn": "#C58A45",
    "TP": "#80678F",
}
GRID_COLOR = "#D9DDE0"
ZERO_COLOR = "#666B70"


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid visualization value for {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite visualization value for {label}")
    return number


def _stats(payload: Mapping[str, Any], label: str) -> tuple[float, float]:
    return _number(payload.get("mean"), f"{label}.mean"), _number(
        payload.get("std"), f"{label}.std"
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=FONT_SIZE, width=LINE_WIDTH, length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(LINE_WIDTH)
    axis.spines["bottom"].set_linewidth(LINE_WIDTH)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.7, zorder=0)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
    )


def _set_padded_limits(axis: plt.Axes, values: list[float], errors: list[float]) -> None:
    lower = min(value - error for value, error in zip(values, errors, strict=True))
    upper = max(value + error for value, error in zip(values, errors, strict=True))
    span = max(upper - lower, 1e-4)
    axis.set_xlim(lower - AXIS_PADDING * span, upper + AXIS_PADDING * span)


def _validation_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping) or not conditions:
        raise ValueError("summary conditions are required for visualization")
    names = list(conditions)
    means: list[float] = []
    errors: list[float] = []
    for name in names:
        metrics = conditions[name]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"invalid condition metrics: {name}")
        nse = metrics.get("val_macro_nse")
        if not isinstance(nse, Mapping):
            raise ValueError(f"missing val_macro_nse statistics: {name}")
        mean, std = _stats(nse, f"{name}.val_macro_nse")
        means.append(mean)
        errors.append(std)
    positions = np.arange(len(names))
    colors = [CONDITION_COLORS.get(name, DEFAULT_COLOR) for name in names]
    for mean, error, position, color in zip(
        means, errors, positions, colors, strict=True
    ):
        axis.errorbar(
            [mean],
            [position],
            xerr=[error],
            fmt="none",
            ecolor=color,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=3,
            zorder=2,
        )
    axis.scatter(means, positions, c=colors, s=MARKER_SIZE, zorder=3, edgecolor="white")
    axis.set_yticks(positions, [name.replace("_", " ") for name in names])
    axis.invert_yaxis()
    axis.set_xlabel("Validation macro NSE (mean ± SD)", fontsize=LABEL_SIZE)
    _set_padded_limits(axis, means, errors)
    _style_axis(axis)
    _panel_label(axis, "a")


def _delta_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    deltas = summary.get("paired_deltas")
    if not isinstance(deltas, Mapping) or not deltas:
        raise ValueError("paired deltas are required for visualization")
    names = list(deltas)
    all_values: list[float] = []
    for position, name in enumerate(names):
        payload = deltas[name]
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid paired delta: {name}")
        seed_deltas = payload.get("seed_deltas")
        if not isinstance(seed_deltas, Mapping) or not seed_deltas:
            raise ValueError(f"missing seed deltas: {name}")
        values = [_number(value, f"{name}.{seed}") for seed, value in seed_deltas.items()]
        offsets = np.linspace(-POINT_SPREAD, POINT_SPREAD, len(values))
        color = CONDITION_COLORS.get(name, DEFAULT_COLOR)
        axis.scatter(
            values,
            position + offsets,
            color=color,
            s=MARKER_SIZE,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        mean = _number(payload.get("mean_delta_macro_nse"), f"{name}.mean_delta")
        axis.scatter(
            [mean],
            [position],
            marker="D",
            color=MEAN_COLOR,
            s=MEAN_MARKER_SIZE,
            zorder=4,
        )
        all_values.extend(values)
    limit = max(max(abs(value) for value in all_values), 1e-4) * (1.0 + AXIS_PADDING)
    axis.axvline(0.0, color=ZERO_COLOR, linewidth=LINE_WIDTH, linestyle="--", zorder=1)
    axis.set_xlim(-limit, limit)
    axis.set_yticks(np.arange(len(names)), [name.replace("_", " ") for name in names])
    axis.invert_yaxis()
    axis.set_xlabel("Learned lag − ablation macro NSE", fontsize=LABEL_SIZE)
    _style_axis(axis)
    _panel_label(axis, "b")


def _test_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    test = summary.get("test")
    if not isinstance(test, Mapping):
        axis.axis("off")
        axis.text(0.5, 0.5, "Held-out test not evaluated", ha="center", va="center")
        _panel_label(axis, "c")
        return
    metrics = test.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("test metrics are required for visualization")
    metric_names = (
        ("Macro", "test_macro_nse"),
        ("NH3N", "test_nse_NH3N"),
        ("CODMn", "test_nse_CODMn"),
        ("TP", "test_nse_TP"),
    )
    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    for label, metric_name in metric_names:
        payload = metrics.get(metric_name)
        if not isinstance(payload, Mapping):
            raise ValueError(f"missing test statistics: {metric_name}")
        mean, std = _stats(payload, metric_name)
        labels.append(label)
        means.append(mean)
        errors.append(std)
    positions = np.arange(len(labels))
    colors = [TARGET_COLORS[label] for label in labels]
    for mean, error, position, color in zip(
        means, errors, positions, colors, strict=True
    ):
        axis.errorbar(
            [mean],
            [position],
            xerr=[error],
            fmt="none",
            ecolor=color,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=3,
            zorder=2,
        )
    axis.scatter(means, positions, c=colors, s=MARKER_SIZE, edgecolor="white", zorder=3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Held-out NSE (mean ± SD)", fontsize=LABEL_SIZE)
    _set_padded_limits(axis, means, errors)
    _style_axis(axis)
    _panel_label(axis, "c")


def render_experiment_summary_figure(
    summary: Mapping[str, Any], png_path: Path, pdf_path: Path
) -> tuple[Path, Path]:
    """Render validation, paired-delta, and held-out-test panels from one summary."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.labelcolor": MEAN_COLOR,
            "xtick.color": MEAN_COLOR,
            "ytick.color": MEAN_COLOR,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    _validation_panel(axes[0], summary)
    _delta_panel(axes[1], summary)
    _test_panel(axes[2], summary)
    figure.subplots_adjust(wspace=PANEL_WSPACE)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def render_experiment_summary_figure_from_json(
    summary_path: Path, png_path: Path, pdf_path: Path
) -> tuple[Path, Path]:
    """Load a machine-readable summary and render matching PNG/PDF figures."""
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid experiment summary: {summary_path}") from error
    if not isinstance(summary, Mapping):
        raise ValueError("experiment summary must be a JSON object")
    return render_experiment_summary_figure(summary, png_path, pdf_path)
