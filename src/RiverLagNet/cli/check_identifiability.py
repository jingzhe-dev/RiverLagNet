"""Evaluate and persist the identifiable synthetic training-data gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.analysis.synthetic_identifiability import (
    run_identifiability_gate,
    write_identifiability_gate,
)


def main() -> None:
    """Run the predeclared five-seed gate and fail closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("experiments/identifiable_v1_data_gate.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/identifiable_v1_data_gate_2026-07-13.md"),
    )
    args = parser.parse_args()
    report = run_identifiability_gate(args.seeds)
    write_identifiability_gate(report, args.json, args.markdown)
    for item in report.scenarios:
        print(
            f"seed={item.seed} routed_ratio={item.routed_variance_ratio:.4f} "
            f"direction_margin={item.direction_margin:.4f} "
            f"lag_recovery={item.lag_recovered}/{item.lag_pairs}"
        )
    print(
        f"passed={report.passed} "
        f"pooled_lag_recovery={report.pooled_lag_recovery_rate:.4f}"
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
