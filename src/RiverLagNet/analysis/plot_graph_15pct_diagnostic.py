"""Plot the audited 15% river-network gain diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


# 可调参数：画布与导出
FIGURE_SIZE = (12.4, 4.9)
EXPORT_DPI = 300
# 可调参数：字体与线宽
BASE_FONT_SIZE = 9.0
PANEL_FONT_SIZE = 11.0
AXIS_LINE_WIDTH = 0.8
# 可调参数：统一配色
DEPLOYED_COLOR = "#2F7183"
DIAGNOSTIC_COLOR = "#73A39A"
ORACLE_COLOR = "#C18451"
NEGATIVE_COLOR = "#A96B64"
TARGET_COLOR = "#8D3F45"
TEXT_COLOR = "#263238"
MUTED_COLOR = "#6B767C"
GRID_COLOR = "#DCE2E5"
RIVER_COLOR = "#4F8797"
HEADWATER_COLOR = "#E8EFF1"
# 可调参数：柱宽、节点和面板留白
BAR_HEIGHT = 0.58
NODE_SIZE = 18.0
PANEL_WSPACE = 0.34


def _relative_gain(baseline: float, graph: float) -> float:
    return 100.0 * (graph - baseline) / abs(baseline)


def _load_and_validate(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["held_out_test_opened"] is not False:
        raise ValueError("diagnostic figure must not use the held-out test split")
    full = payload["full_network"]
    expected_full = _relative_gain(
        float(full["no_graph_macro_nse"]), float(full["graph_macro_nse"])
    )
    if abs(expected_full - float(full["relative_gain_percent"])) > 1e-8:
        raise ValueError("full-network relative gain is inconsistent")
    mainstem = payload["mainstem"]
    baseline = float(mainstem["no_graph_macro_nse"])
    for row in mainstem["models"]:
        expected = _relative_gain(baseline, float(row["macro_nse"]))
        if abs(expected - float(row["relative_gain_percent"])) > 1e-8:
            raise ValueError(f"mainstem gain is inconsistent for {row['name']}")
    return payload


def build_figure(audit_path: Path, output_stem: Path) -> tuple[Path, Path]:
    """Build PNG and PDF diagnostics from one audited JSON source."""
    payload = _load_and_validate(audit_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": BASE_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (gain_axis, graph_axis) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": PANEL_WSPACE},
    )

    full = payload["full_network"]
    mainstem = payload["mainstem"]
    diagnostic = payload["signal_diagnostics"]
    model_rows = {row["name"]: row for row in mainstem["models"]}
    labels = [
        "Full network\n8-hop graph",
        "Mainstem\nhidden trajectory",
        "Mainstem\noutput transport",
        "Train→validation\nlinear signal ceiling",
        "Training-selected\n12-station cohort",
        "True-future\n8-hop oracle*",
    ]
    gains = np.asarray(
        [
            full["relative_gain_percent"],
            model_rows["8-hop hidden trajectory"]["relative_gain_percent"],
            model_rows["8-hop output transport"]["relative_gain_percent"],
            diagnostic["deployable_train_fit_validation_eval"]["relative_gain_percent"],
            diagnostic["training_only_selected_cohort"]["relative_gain_percent"],
            diagnostic["non_deployable_true_future_oracle"]["relative_gain_percent"],
        ],
        dtype=float,
    )
    colors = [
        DEPLOYED_COLOR,
        DEPLOYED_COLOR,
        NEGATIVE_COLOR,
        DIAGNOSTIC_COLOR,
        DIAGNOSTIC_COLOR,
        ORACLE_COLOR,
    ]
    y = np.arange(len(labels))
    bars = gain_axis.barh(
        y, gains, height=BAR_HEIGHT, color=colors, edgecolor="white", linewidth=0.6
    )
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("white")
    gain_axis.axvline(
        15.0, color=TARGET_COLOR, linewidth=1.4, linestyle=(0, (4, 2)), zorder=0
    )
    gain_axis.text(
        15.0,
        -0.72,
        "required +15%",
        color=TARGET_COLOR,
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )
    gain_axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.8)
    gain_axis.set_yticks(y, labels)
    gain_axis.invert_yaxis()
    gain_axis.set_xlim(-1.2, 18.2)
    gain_axis.set_xticks([0, 5, 10, 15])
    gain_axis.set_xlabel("Relative validation macro NSE gain (%)")
    gain_axis.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    gain_axis.set_axisbelow(True)
    gain_axis.spines[["top", "right", "left"]].set_visible(False)
    gain_axis.tick_params(axis="y", length=0, pad=5)
    for bar, value in zip(bars, gains):
        label_x = value + 0.24 if value >= 0 else 0.24
        gain_axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold" if value >= 15 else "normal",
        )
    gain_axis.set_title(
        "Observed graph gain versus the project threshold",
        loc="left",
        pad=12,
        fontsize=10.4,
        fontweight="bold",
    )
    gain_axis.text(
        0.0,
        1.015,
        "*Uses true future upstream errors; diagnostic only, not deployable",
        transform=gain_axis.transAxes,
        color=MUTED_COLOR,
        fontsize=7.5,
        va="bottom",
    )

    nodes = int(mainstem["nodes"])
    x = np.linspace(0.04, 0.96, nodes)
    y_curve = 0.52 + 0.045 * np.sin(np.linspace(0.0, 3.4 * np.pi, nodes))
    graph_axis.plot(x, y_curve, color=RIVER_COLOR, linewidth=2.0, zorder=1)
    graph_axis.scatter(
        x[1:],
        y_curve[1:],
        s=NODE_SIZE,
        color=RIVER_COLOR,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    graph_axis.scatter(
        x[0],
        y_curve[0],
        s=NODE_SIZE * 1.35,
        color=HEADWATER_COLOR,
        edgecolor=RIVER_COLOR,
        linewidth=1.0,
        zorder=4,
    )
    for index in (9, 19, 29, 39, 49, 59):
        start = (x[index - 1], y_curve[index - 1])
        end = (x[index + 1], y_curve[index + 1])
        graph_axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.1,
                color=RIVER_COLOR,
                zorder=2,
            )
        )
    bracket_start, bracket_end = 24, 32
    bracket_y = 0.69
    graph_axis.plot(
        [x[bracket_start], x[bracket_end]],
        [bracket_y, bracket_y],
        color=DEPLOYED_COLOR,
        linewidth=1.2,
    )
    graph_axis.plot(
        [x[bracket_start], x[bracket_start]],
        [bracket_y - 0.018, bracket_y + 0.018],
        color=DEPLOYED_COLOR,
        linewidth=1.2,
    )
    graph_axis.plot(
        [x[bracket_end], x[bracket_end]],
        [bracket_y - 0.018, bracket_y + 0.018],
        color=DEPLOYED_COLOR,
        linewidth=1.2,
    )
    graph_axis.text(
        (x[bracket_start] + x[bracket_end]) / 2,
        bracket_y + 0.035,
        "8-hop message reach",
        ha="center",
        va="bottom",
        color=DEPLOYED_COLOR,
        fontsize=8.2,
        fontweight="bold",
    )
    graph_axis.text(
        x[0],
        0.39,
        f"Upstream\n{mainstem['start_node_id']}",
        ha="left",
        va="top",
        fontsize=8.4,
    )
    graph_axis.text(
        x[-1],
        0.39,
        f"Downstream\n{mainstem['end_node_id']}",
        ha="right",
        va="top",
        fontsize=8.4,
    )
    graph_axis.text(
        0.5,
        0.21,
        f"{nodes} stations  •  {mainstem['edges']} directed edges  •  "
        f"{mainstem['total_travel_time_days']:.1f} d cumulative prior",
        ha="center",
        va="center",
        transform=graph_axis.transAxes,
        fontsize=8.2,
        color=MUTED_COLOR,
    )
    graph_axis.add_patch(
        FancyArrowPatch(
            (0.16, 0.29),
            (0.84, 0.29),
            transform=graph_axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=TEXT_COLOR,
        )
    )
    graph_axis.text(
        0.5,
        0.305,
        "information flow",
        transform=graph_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=TEXT_COLOR,
    )
    graph_axis.set_xlim(0.0, 1.0)
    graph_axis.set_ylim(0.12, 0.84)
    graph_axis.axis("off")
    graph_axis.set_title(
        "Topology-only mainstem used for the controlled test",
        loc="left",
        pad=12,
        fontsize=10.4,
        fontweight="bold",
    )

    gain_axis.text(
        -0.17,
        1.10,
        "a",
        transform=gain_axis.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
    )
    graph_axis.text(
        -0.08,
        1.10,
        "b",
        transform=graph_axis.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
    )
    figure.suptitle(
        "River-network information does not support a deployable 15% NSE gain",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=12.2,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.015,
        "Chronological validation only; the held-out test split remains unopened. "
        "Bars quantify association and predictive gain, not causal contribution.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=MUTED_COLOR,
    )
    figure.subplots_adjust(top=0.79, bottom=0.19, left=0.20, right=0.985)
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
