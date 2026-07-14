"""Generate an audited upstream-to-downstream real river-network figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.analysis.river_graph_visualization import (
    build_river_graph_summary,
    render_river_graph_figure,
    write_river_graph_summary,
)


def main() -> None:
    """CLI wrapper for graph audit JSON plus PNG/PDF outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/china-real-daily-v0.1"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/china_real_daily_graph_summary.json"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("docs/figures/china_real_daily_upstream_graph.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("docs/figures/china_real_daily_upstream_graph.pdf"),
    )
    args = parser.parse_args()
    summary = build_river_graph_summary(args.data_root)
    write_river_graph_summary(summary, args.summary)
    png_path, pdf_path = render_river_graph_figure(summary, args.png, args.pdf)
    print(
        f"graph_summary={args.summary} graph_png={png_path} graph_pdf={pdf_path} "
        f"nodes={summary['node_count']} edges={summary['edge_count']}"
    )


if __name__ == "__main__":
    main()
