"""Run the paired RiverLagNet robustness experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from RiverLagNet.analysis.experiment_suite import (
    build_experiment_specs,
    final_evaluation_output,
    pending_experiment_specs,
    run_final_evaluations,
    run_experiment_specs,
    successful_experiment_names,
    training_command,
)
from RiverLagNet.analysis.robustness_summary import (
    aggregate_test_metrics,
    load_successful_suite_rows,
    summarize_validation,
    write_validation_summary,
)


def main() -> None:
    """Run only robustness jobs not already successful in the ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--ledger", type=Path, default=Path("experiments/results.tsv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/robustness_summary.json"),
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("docs/robustness_report_2026-07-13.md"),
    )
    args = parser.parse_args()

    specs = build_experiment_specs(args.seeds)
    if args.summarize:
        rows = load_successful_suite_rows(args.ledger, specs)
        summary = summarize_validation(rows, specs)
        learned = tuple(spec for spec in specs if spec.condition.name == "learned_lag")
        test_paths = {spec.seed: final_evaluation_output(spec) for spec in learned}
        existing = {seed: path for seed, path in test_paths.items() if path.is_file()}
        if existing and len(existing) != len(test_paths):
            raise ValueError("held-out test outputs are incomplete")
        if existing:
            summary["test"] = aggregate_test_metrics(existing)
        write_validation_summary(summary, args.summary_json, args.summary_markdown)
        print(f"summary_json={args.summary_json} summary_markdown={args.summary_markdown}")
        return
    if args.evaluate_final:
        load_successful_suite_rows(args.ledger, specs)
        commands = run_final_evaluations(specs, sys.executable, dry_run=True)
        for command in commands:
            print(subprocess.list2cmdline(command))
        if not args.dry_run:
            run_final_evaluations(specs, sys.executable)
        return
    successful = successful_experiment_names(args.ledger)
    pending = pending_experiment_specs(specs, successful)
    print(f"suite_total={len(specs)} successful={len(specs) - len(pending)} pending={len(pending)}")
    for spec in pending:
        print(subprocess.list2cmdline(training_command(spec, sys.executable)))
    run_experiment_specs(pending, sys.executable, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
