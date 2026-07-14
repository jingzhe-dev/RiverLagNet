"""Build the five-seed attributable graph-gain audit and figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RiverLagNet.analysis.graph_gain_analysis import (
    build_graph_gain_summary,
    load_json,
    render_graph_gain_figure,
    write_graph_gain_outputs,
)


def main() -> None:
    """Run checkpoint inference on validation data and write audited artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/china-real-daily-contracted-v0.2/dataset.npz"),
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--ledger", type=Path, default=Path("experiments/results.tsv"))
    parser.add_argument(
        "--graph-summary",
        type=Path,
        default=Path("experiments/china_real_daily_contracted_graph_summary.json"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/real_contracted_graph_gain_v9_summary.json"),
    )
    parser.add_argument(
        "--node-output",
        type=Path,
        default=Path("experiments/real_contracted_graph_gain_v9_nodes.csv"),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=Path("docs/figures/real_contracted_graph_gain_v9.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("docs/figures/real_contracted_graph_gain_v9.pdf"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    summary = build_graph_gain_summary(
        args.dataset, args.runs, args.ledger, device=args.device
    )
    write_graph_gain_outputs(summary, args.summary_output, args.node_output)
    render_graph_gain_figure(
        summary,
        load_json(args.graph_summary),
        args.png_output,
        args.pdf_output,
    )
    print(json.dumps(summary["paired_global_delta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
