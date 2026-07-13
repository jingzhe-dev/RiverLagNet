"""Generate the versioned real daily training summary from raw run evidence."""

from pathlib import Path

from RiverLagNet.analysis.real_training_summary import (
    build_real_training_summary,
    write_real_training_summary,
)


def main() -> None:
    """Build the fixed repository report paths."""
    summary = build_real_training_summary(Path("experiments/results.tsv"), Path("runs"))
    write_real_training_summary(
        summary,
        Path("experiments/china_real_daily_seed42_summary.json"),
        Path("docs/real_training_report_2026-07-14.md"),
    )


if __name__ == "__main__":
    main()
