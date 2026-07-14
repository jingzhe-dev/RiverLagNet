"""Summarize and visualize the contracted real-data seed-42 benchmark."""

from __future__ import annotations

import argparse

from RiverLagNet.analysis.contracted_training_summary import (
    build_contracted_training_summary,
    render_contracted_training_figure,
    write_contracted_training_summary,
)


def main() -> None:
    """Create machine-readable and publication-style benchmark outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/results.tsv")
    parser.add_argument(
        "--summary", default="experiments/china_real_daily_contracted_seed42_summary.json"
    )
    parser.add_argument(
        "--png", default="docs/figures/china_real_daily_contracted_seed42_metrics.png"
    )
    parser.add_argument(
        "--pdf", default="docs/figures/china_real_daily_contracted_seed42_metrics.pdf"
    )
    args = parser.parse_args()
    summary = build_contracted_training_summary(args.results)
    write_contracted_training_summary(summary, args.summary)
    render_contracted_training_figure(summary, args.png, args.pdf)
    print(
        f"summary={args.summary} png={args.png} pdf={args.pdf} "
        f"riverlagnet_val_macro_nse={summary['models'][-1]['validation']['nse']:.4f}"
    )


if __name__ == "__main__":
    main()
