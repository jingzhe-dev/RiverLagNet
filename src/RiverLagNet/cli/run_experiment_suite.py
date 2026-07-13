"""Run the paired RiverLagNet robustness experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from RiverLagNet.analysis.experiment_suite import (
    build_experiment_specs,
    pending_experiment_specs,
    run_experiment_specs,
    successful_experiment_names,
    training_command,
)


def main() -> None:
    """Run only robustness jobs not already successful in the ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--ledger", type=Path, default=Path("experiments/results.tsv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = build_experiment_specs(args.seeds)
    successful = successful_experiment_names(args.ledger)
    pending = pending_experiment_specs(specs, successful)
    print(f"suite_total={len(specs)} successful={len(specs) - len(pending)} pending={len(pending)}")
    for spec in pending:
        print(subprocess.list2cmdline(training_command(spec, sys.executable)))
    run_experiment_specs(pending, sys.executable, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
