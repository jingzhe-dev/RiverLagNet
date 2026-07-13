"""Generate PNG and PDF visualizations from an experiment summary JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.analysis.result_visualization import (
    render_experiment_summary_figure_from_json,
)


def main() -> None:
    """CLI wrapper for reproducible result visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    png_path, pdf_path = render_experiment_summary_figure_from_json(
        args.summary, args.png, args.pdf
    )
    print(f"figure_png={png_path} figure_pdf={pdf_path}")


if __name__ == "__main__":
    main()
