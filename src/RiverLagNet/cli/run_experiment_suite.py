"""Run the paired RiverLagNet robustness experiment matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from RiverLagNet.analysis.experiment_suite import (
    SUITE_PRESETS,
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
from RiverLagNet.analysis.result_visualization import render_experiment_summary_figure
from RiverLagNet.analysis.river_graph_visualization import (
    build_river_graph_summary,
    render_river_graph_figure,
    write_river_graph_summary,
)
from RiverLagNet.analysis.synthetic_identifiability import (
    run_identifiability_gate,
    write_identifiability_gate,
)


def run(argv: Sequence[str] | None = None) -> None:
    """Run only robustness jobs not already successful in the ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=tuple(SUITE_PRESETS), default="robustness_v1"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--ledger", type=Path, default=Path("experiments/results.tsv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=None,
    )
    parser.add_argument("--figure-png", type=Path, default=None)
    parser.add_argument("--figure-pdf", type=Path, default=None)
    args = parser.parse_args(argv)

    preset = SUITE_PRESETS[args.suite]
    summary_json = args.summary_json or preset.summary_json
    summary_markdown = args.summary_markdown or preset.summary_markdown
    specs = build_experiment_specs(args.seeds, preset)
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
        figure_stem = summary_json.stem
        figure_png = args.figure_png or Path("docs/figures") / f"{figure_stem}.png"
        figure_pdf = args.figure_pdf or Path("docs/figures") / f"{figure_stem}.pdf"
        summary["visualization"] = {
            "png": figure_png.as_posix(),
            "pdf": figure_pdf.as_posix(),
            "markdown_png": Path(
                os.path.relpath(figure_png, start=summary_markdown.parent)
            ).as_posix(),
        }
        graph_message = ""
        if preset.graph_data_root is not None:
            if not all(
                path is not None
                for path in (
                    preset.graph_summary_json,
                    preset.graph_figure_png,
                    preset.graph_figure_pdf,
                )
            ):
                raise ValueError("graph visualization preset paths are incomplete")
            graph_summary_path = preset.graph_summary_json
            graph_png = preset.graph_figure_png
            graph_pdf = preset.graph_figure_pdf
            assert graph_summary_path is not None and graph_png is not None and graph_pdf is not None
            graph_summary = build_river_graph_summary(preset.graph_data_root)
            write_river_graph_summary(graph_summary, graph_summary_path)
            render_river_graph_figure(graph_summary, graph_png, graph_pdf)
            summary["graph_visualization"] = {
                "summary": graph_summary_path.as_posix(),
                "png": graph_png.as_posix(),
                "pdf": graph_pdf.as_posix(),
                "markdown_png": Path(
                    os.path.relpath(graph_png, start=summary_markdown.parent)
                ).as_posix(),
                "markdown_summary": Path(
                    os.path.relpath(graph_summary_path, start=summary_markdown.parent)
                ).as_posix(),
                "node_count": graph_summary["node_count"],
                "edge_count": graph_summary["edge_count"],
                "component_count": graph_summary["component_count"],
                "rounded_prior_lag_counts": graph_summary[
                    "rounded_prior_lag_counts"
                ],
            }
            graph_message = (
                f" graph_summary={graph_summary_path} graph_png={graph_png} "
                f"graph_pdf={graph_pdf}"
            )
        render_experiment_summary_figure(summary, figure_png, figure_pdf)
        write_validation_summary(summary, summary_json, summary_markdown)
        print(
            f"summary_json={summary_json} summary_markdown={summary_markdown} "
            f"figure_png={figure_png} figure_pdf={figure_pdf}{graph_message}"
        )
        return
    if args.evaluate_final:
        load_successful_suite_rows(args.ledger, specs)
        commands = run_final_evaluations(specs, sys.executable, dry_run=True)
        for command in commands:
            print(subprocess.list2cmdline(command))
        if not args.dry_run:
            run_final_evaluations(specs, sys.executable)
        return
    if preset.data_override == "synthetic_identifiable_v1":
        gate = run_identifiability_gate(args.seeds)
        write_identifiability_gate(
            gate,
            Path("experiments/identifiable_v1_data_gate.json"),
            Path("docs/identifiable_v1_data_gate_2026-07-13.md"),
        )
        if not gate.passed:
            raise RuntimeError(f"{preset.name} data gate failed")
    successful = successful_experiment_names(args.ledger)
    pending = pending_experiment_specs(specs, successful)
    print(f"suite_total={len(specs)} successful={len(specs) - len(pending)} pending={len(pending)}")
    for spec in pending:
        print(subprocess.list2cmdline(training_command(spec, sys.executable)))
    run_experiment_specs(pending, sys.executable, dry_run=args.dry_run)


def main() -> None:
    """Command-line wrapper."""
    run()


if __name__ == "__main__":
    main()
