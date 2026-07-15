"""Plot the audited validation gain and directed upstream propagation design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIGURE_SIZE = (13.6, 7.2)
EXPORT_DPI = 300
BASE_FONT_SIZE = 9.0
TEXT_COLOR = "#243238"
MUTED_COLOR = "#68777D"
GRID_COLOR = "#DCE4E7"
RIVER_COLOR = "#4A8292"
BEST_COLOR = "#176B79"
ADAPTIVE_COLOR = "#72A69A"
COUNTERFACTUAL_COLOR = "#4F8797"
NEGATIVE_COLOR = "#B8756D"
TARGET_COLOR = "#9B3E46"
NH3N_COLOR = "#2F7183"
CODMN_COLOR = "#72A69A"
TP_COLOR = "#C48A52"


def _relative_gain(baseline: float, graph: float) -> float:
    return 100.0 * (graph - baseline) / abs(baseline)


def _load_and_validate(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["held_out_test_opened"] is not False:
        raise ValueError("diagnostic figure must not use the held-out test split")
    for section_name in ("full_network",):
        section = payload[section_name]
        expected = _relative_gain(
            float(section["no_graph_macro_nse"]),
            float(section["graph_macro_nse"]),
        )
        if abs(expected - float(section["relative_gain_percent"])) > 1e-8:
            raise ValueError(f"{section_name} relative gain is inconsistent")
    for section_name in ("mainstem", "crossformer", "recurrent_crossformer"):
        section = payload[section_name]
        baseline = float(section["no_graph_macro_nse"])
        for row in section["models"]:
            expected = _relative_gain(baseline, float(row["macro_nse"]))
            if abs(expected - float(row["relative_gain_percent"])) > 1e-8:
                raise ValueError(f"{section_name} gain is inconsistent for {row['name']}")
        if "target_macro_nse" in section:
            expected_target = baseline * (
                1.0 + float(section["target_relative_gain_percent"]) / 100.0
            )
            if abs(expected_target - float(section["target_macro_nse"])) > 1e-8:
                raise ValueError(f"{section_name} target NSE is inconsistent")
    return payload


def _rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str = "white",
    size: float = 8.0,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=face,
            zorder=8,
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=TEXT_COLOR,
        zorder=9,
    )


def _plot_model_gain(axis: plt.Axes, payload: dict[str, object]) -> None:
    recurrent = payload["recurrent_crossformer"]
    rows = {row["name"]: row for row in recurrent["models"]}
    ordered = [
        rows["RCELA + GMRF fixed innovation"],
        rows["RCELA + GMRF adaptive DCUV"],
        rows["RCELA + GMRF counterfactual CLGF"],
        rows["RCELA + GMRF absolute state"],
    ]
    labels = [
        "Fixed innovation",
        "Adaptive DCUV",
        "Counterfactual CLGF",
        "Absolute state (best)",
    ]
    colors = [NEGATIVE_COLOR, ADAPTIVE_COLOR, COUNTERFACTUAL_COLOR, BEST_COLOR]
    gains = np.asarray([float(row["relative_gain_percent"]) for row in ordered])
    positions = np.arange(4)
    bars = axis.barh(
        positions,
        gains,
        height=0.54,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9)
    axis.set_yticks(positions, labels)
    axis.set_xlim(-0.42, 1.28)
    axis.set_xticks([-0.25, 0.0, 0.5, 1.0])
    axis.set_xlabel("Relative validation macro NSE gain (%)")
    axis.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=6)
    for bar, value in zip(bars, gains, strict=True):
        x = value + 0.035 if value >= 0 else value - 0.035
        axis.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8.8,
            fontweight="bold" if value == gains.max() else "normal",
            color=TEXT_COLOR,
        )
    best = float(gains.max())
    required = float(recurrent["target_relative_gain_percent"])
    axis.text(
        0.0,
        1.13,
        f"BEST  {best:.2f}%",
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=BEST_COLOR,
        va="center",
    )
    axis.text(
        0.26,
        1.13,
        f"REQUIRED  {required:.0f}%",
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=TARGET_COLOR,
        va="center",
    )
    axis.text(
        0.63,
        1.13,
        f"only {100.0 * best / required:.1f}% of target",
        transform=axis.transAxes,
        fontsize=8.4,
        color=MUTED_COLOR,
        va="center",
    )
    axis.set_title(
        "a  Controlled attention-value ablation",
        loc="left",
        pad=28,
        fontsize=10.8,
        fontweight="bold",
    )


def _plot_target_gain(axis: plt.Axes, payload: dict[str, object]) -> None:
    recurrent = payload["recurrent_crossformer"]
    best_row = next(
        row
        for row in recurrent["models"]
        if row["name"] == "RCELA + GMRF absolute state"
    )
    values = best_row["target_delta_nse"]
    names = ["NH3N", "CODMn", "TP"]
    deltas = np.asarray([float(values[name]) for name in names])
    colors = [NH3N_COLOR, CODMN_COLOR, TP_COLOR]
    positions = np.arange(3)
    axis.barh(
        positions,
        deltas,
        height=0.50,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(-0.0008, 0.0132)
    axis.set_xticks([0.0, 0.004, 0.008, 0.012])
    axis.set_xlabel("Absolute validation NSE change")
    axis.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=6)
    for position, value in zip(positions, deltas, strict=True):
        axis.text(
            value + 0.00022,
            position,
            f"+{value:.4f}",
            ha="left",
            va="center",
            fontsize=8.6,
            color=TEXT_COLOR,
        )
    axis.set_title(
        "b  Best graph model: gain by pollutant",
        loc="left",
        pad=10,
        fontsize=10.8,
        fontweight="bold",
    )


def _plot_river_flow(axis: plt.Axes, payload: dict[str, object]) -> None:
    mainstem = payload["mainstem"]
    nodes = int(mainstem["nodes"])
    x = np.linspace(0.055, 0.945, nodes)
    y = 0.40 + 0.040 * np.sin(np.linspace(0.0, 3.3 * np.pi, nodes))
    axis.plot(x, y, color=RIVER_COLOR, linewidth=2.3, zorder=1)
    axis.scatter(
        x,
        y,
        s=19,
        color=RIVER_COLOR,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for index in (9, 19, 29, 39, 49, 59):
        axis.add_patch(
            FancyArrowPatch(
                (x[index - 1], y[index - 1]),
                (x[index + 1], y[index + 1]),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.2,
                color=RIVER_COLOR,
                zorder=2,
            )
        )

    target = 42
    sources = (41, 38, 34)
    labels = ("1-hop", "4-hop", "8-hop")
    radii = (-0.45, -0.78, -1.08)
    colors = ("#9BBBC2", "#6D9EAA", BEST_COLOR)
    for source, label, radius, color in zip(
        sources, labels, radii, colors, strict=True
    ):
        axis.scatter(
            x[source], y[source], s=45, color="white", edgecolor=color,
            linewidth=1.4, zorder=6,
        )
        axis.add_patch(
            FancyArrowPatch(
                (x[source], y[source] + 0.012),
                (x[target], y[target] + 0.012),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={radius}",
                mutation_scale=10,
                linewidth=1.6,
                color=color,
                zorder=5,
            )
        )
        axis.text(
            (x[source] + x[target]) / 2,
            max(y[source], y[target]) + abs(radius) * 0.105,
            label,
            color=color,
            fontsize=7.7,
            ha="center",
            fontweight="bold",
        )
    axis.scatter(
        x[target], y[target], s=73, marker="D", color=BEST_COLOR,
        edgecolor="white", linewidth=0.9, zorder=7,
    )

    _rounded_box(
        axis, (0.56, 0.70), 0.19, 0.085, "Local Transformer\nstate at station i",
        edge="#839399", face="#F4F7F8",
    )
    _rounded_box(
        axis, (0.78, 0.70), 0.17, 0.085, "GMRF fused\nfuture state",
        edge=BEST_COLOR, face="#EEF6F6",
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.665, 0.70), (x[target], y[target] + 0.028),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
            color="#839399", zorder=4,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (x[target] + 0.012, y[target] + 0.032), (0.78, 0.74),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.4,
            color=BEST_COLOR, zorder=4,
        )
    )
    axis.text(
        0.755, 0.615,
        "RCELA jointly selects\nupstream path × positive lag",
        ha="center", va="center", fontsize=8.0, color=BEST_COLOR,
        fontweight="bold",
    )
    axis.text(
        0.055, 0.29, "UPSTREAM", fontsize=8.3, fontweight="bold",
        color=RIVER_COLOR, ha="left",
    )
    axis.text(
        0.945, 0.29, "DOWNSTREAM", fontsize=8.3, fontweight="bold",
        color=RIVER_COLOR, ha="right",
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.12, 0.275), (0.88, 0.275), arrowstyle="-|>",
            mutation_scale=12, linewidth=1.1, color=TEXT_COLOR,
        )
    )
    axis.text(
        0.5, 0.22,
        f"{nodes} stations  |  {mainstem['edges']} directed edges  |  "
        f"{mainstem['total_travel_time_days']:.1f} d cumulative travel-time prior",
        ha="center", va="center", fontsize=8.2, color=MUTED_COLOR,
    )
    axis.text(
        0.5, 0.12,
        "At forecast day h, attention can read observed history or earlier model\n"
        "states only; current/future target information is never available.",
        ha="center", va="center", fontsize=8.0, color=MUTED_COLOR,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.04, 0.86)
    axis.axis("off")
    axis.set_title(
        "c  Directed lag-aware Transformer–GNN recurrence",
        loc="left", pad=12, fontsize=10.8, fontweight="bold",
    )


def build_figure(audit_path: Path, output_stem: Path) -> tuple[Path, Path]:
    """Build PNG and PDF figures from one validation-only audit source."""
    payload = _load_and_validate(audit_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": BASE_FONT_SIZE,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    grid = GridSpec(
        2, 2, figure=figure, width_ratios=(0.93, 1.17),
        height_ratios=(1.0, 0.92), wspace=0.34, hspace=0.48,
    )
    gain_axis = figure.add_subplot(grid[0, 0])
    target_axis = figure.add_subplot(grid[1, 0])
    river_axis = figure.add_subplot(grid[:, 1])
    _plot_model_gain(gain_axis, payload)
    _plot_target_gain(target_axis, payload)
    _plot_river_flow(river_axis, payload)

    recurrent = payload["recurrent_crossformer"]
    figure.suptitle(
        "River-network information improves validation NSE, but the gain remains 1.00%",
        x=0.055, y=0.975, ha="left", fontsize=13.0, fontweight="bold",
    )
    figure.text(
        0.055, 0.925,
        f"Same-budget no-graph NSE {recurrent['no_graph_macro_nse']:.6f}  →  "
        f"best directed NSE {recurrent['models'][0]['macro_nse']:.6f}  |  "
        f"15% target NSE {recurrent['target_macro_nse']:.6f}",
        ha="left", va="center", fontsize=9.0, color=MUTED_COLOR,
    )
    figure.text(
        0.055, 0.018,
        "Chronological validation only; the held-out test split remains unopened. "
        "Attention weights are routing preferences, not causal effects.",
        ha="left", va="bottom", fontsize=7.7, color=MUTED_COLOR,
    )
    figure.subplots_adjust(top=0.84, bottom=0.12, left=0.15, right=0.985)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    png, pdf = build_figure(args.audit, args.output_stem)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
