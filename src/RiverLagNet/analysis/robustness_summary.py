"""Strict ledger-derived summaries for paired robustness experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .experiment_suite import ExperimentSpec, SUCCESS_STATUSES


SUMMARY_METRICS = (
    "val_macro_nse",
    "val_macro_mae",
    "val_macro_rmse",
    "duration_s",
    "peak_vram_gb",
)
TEST_METRICS = (
    "test_macro_nse",
    "test_macro_mae",
    "test_macro_rmse",
    "test_nse_NH3N",
    "test_nse_CODMn",
    "test_nse_TP",
)


def load_successful_suite_rows(
    path: Path, specs: Sequence[ExperimentSpec]
) -> list[dict[str, str]]:
    """Load exactly one successful row per expected experiment specification."""
    expected = {spec.experiment_name: spec for spec in specs}
    if len(expected) != len(specs):
        raise ValueError("experiment specifications contain duplicate names")
    selected: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("experiment ledger has no header")
        for row in reader:
            name = row.get("experiment", "")
            if name not in expected or row.get("status") not in SUCCESS_STATUSES:
                continue
            if name in selected:
                raise ValueError(f"duplicate successful suite row: {name}")
            spec = expected[name]
            try:
                row_seed = int(row["seed"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid seed for suite row: {name}") from error
            if row_seed != spec.seed:
                raise ValueError(f"unexpected seed for suite row: {name}")
            selected[name] = row
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"missing successful suite rows: {', '.join(missing)}")
    commits = {row.get("commit", "") for row in selected.values()}
    if len(commits) != 1 or not next(iter(commits)):
        raise ValueError("successful suite rows must share one commit")
    return [selected[spec.experiment_name] for spec in specs]


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _number(row: Mapping[str, str], metric: str) -> float:
    try:
        return float(row[metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"suite row {row.get('experiment', '<unknown>')} has invalid {metric}"
        ) from error


def summarize_validation(
    rows: Sequence[dict[str, str]], specs: Sequence[ExperimentSpec]
) -> dict[str, object]:
    """Calculate condition statistics and learned-lag paired deltas."""
    if len(rows) != len(specs):
        raise ValueError("suite rows and specifications must have equal length")
    paired = list(zip(specs, rows, strict=True))
    seeds = sorted({spec.seed for spec in specs})
    commits = {row["commit"] for row in rows}
    if len(commits) != 1:
        raise ValueError("suite rows must share one commit")
    presets = {spec.preset for spec in specs}
    if len(presets) != 1:
        raise ValueError("suite specifications must share one preset")
    preset = next(iter(presets))

    conditions: dict[str, dict[str, dict[str, float | int]]] = {}
    for condition_name in dict.fromkeys(spec.condition.name for spec in specs):
        condition_rows = [row for spec, row in paired if spec.condition.name == condition_name]
        conditions[condition_name] = {
            metric: _statistics([_number(row, metric) for row in condition_rows])
            for metric in SUMMARY_METRICS
        }

    nse_by_condition_seed = {
        (spec.condition.name, spec.seed): _number(row, "val_macro_nse")
        for spec, row in paired
    }
    paired_deltas: dict[str, dict[str, Any]] = {}
    for ablation in preset.primary_comparisons:
        seed_deltas = {
            str(seed): nse_by_condition_seed[("learned_lag", seed)]
            - nse_by_condition_seed[(ablation, seed)]
            for seed in seeds
        }
        values = list(seed_deltas.values())
        paired_deltas[ablation] = {
            "mean_delta_macro_nse": statistics.fmean(values),
            "std_delta_macro_nse": statistics.stdev(values) if len(values) > 1 else 0.0,
            "wins": sum(value > 0.0 for value in values),
            "seed_deltas": seed_deltas,
        }
    return {
        "suite": preset.name,
        "report_title": preset.report_title,
        "seeds": seeds,
        "commit": next(iter(commits)),
        "experiment_count": len(rows),
        "conditions": conditions,
        "paired_deltas": paired_deltas,
    }


def aggregate_test_metrics(paths_by_seed: Mapping[int, Path]) -> dict[str, object]:
    """Aggregate held-out full-model metrics without feeding them into selection."""
    if not paths_by_seed:
        raise ValueError("at least one held-out test result is required")
    values: dict[str, list[float]] = {metric: [] for metric in TEST_METRICS}
    for seed, path in sorted(paths_by_seed.items()):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid test result for seed {seed}: {path}") from error
        for metric in TEST_METRICS:
            try:
                value = float(payload[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"seed {seed} has invalid {metric}") from error
            if not math.isfinite(value):
                raise ValueError(f"seed {seed} has non-finite {metric}")
            values[metric].append(value)
    return {
        "seeds": sorted(paths_by_seed),
        "metrics": {metric: _statistics(metric_values) for metric, metric_values in values.items()},
    }


def render_validation_markdown(summary: Mapping[str, object]) -> str:
    """Render an answer-first technical report from a validation summary."""
    seeds = summary["seeds"]
    conditions = summary["conditions"]
    deltas = summary["paired_deltas"]
    assert isinstance(seeds, list) and isinstance(conditions, dict) and isinstance(deltas, dict)
    identifiable = str(summary["suite"]).startswith("identifiable_")
    real_world = str(summary["suite"]).startswith("real_")
    if identifiable:
        evidence_note = (
            "This report is generated from validation-selected checkpoints in the "
            "append-only experiment ledger. It tests an identifiable synthetic benchmark; "
            "it is not a real-world water-quality result or a formal significance test."
        )
    elif real_world:
        evidence_note = (
            "This report is generated from validation-selected checkpoints trained on "
            "real-source daily China observations. Imputed values are excluded by masks; "
            "the five-seed comparison quantifies training variability but is not a formal "
            "significance test."
        )
    else:
        evidence_note = (
            "This report is generated from validation-selected checkpoints in the "
            "append-only experiment ledger. It tests engineering robustness on synthetic "
            "data; it is not a real-world water-quality result or a formal significance test."
        )
    visualization = summary.get("visualization")
    visualization_lines: list[str] = []
    if isinstance(visualization, Mapping):
        markdown_png = visualization.get("markdown_png")
        if isinstance(markdown_png, str) and markdown_png:
            visualization_lines = [
                "## Result visualization",
                "",
                f"![Experiment result summary]({markdown_png})",
                "",
            ]
    graph_visualization = summary.get("graph_visualization")
    graph_visualization_lines: list[str] = []
    if isinstance(graph_visualization, Mapping):
        graph_png = graph_visualization.get("markdown_png")
        graph_summary_path = graph_visualization.get("markdown_summary")
        prior_counts = graph_visualization.get("rounded_prior_lag_counts")
        if (
            isinstance(graph_png, str)
            and graph_png
            and isinstance(graph_summary_path, str)
            and graph_summary_path
            and isinstance(prior_counts, Mapping)
        ):
            lag_counts = ", ".join(
                f"{count} edges at {lag} d" for lag, count in prior_counts.items()
            )
            graph_visualization_lines = [
                "## Monitored upstream-downstream graph",
                "",
                f"![Monitored upstream-to-downstream river graph]({graph_png})",
                "",
                (
                    f"The audited graph contains {graph_visualization['node_count']} monitored "
                    f"segments, {graph_visualization['edge_count']} upstream-to-downstream edges, "
                    f"and {graph_visualization['component_count']} disjoint components. Rounded "
                    f"travel-time priors comprise {lag_counts}."
                ),
                "",
                (
                    "This concentration near zero makes `fixed_lag` structurally close to "
                    "`no_lag` and is a plausible explanation for the small validation deltas; "
                    "it is a mechanism diagnostic, not a causal claim. The component-card "
                    "layout is a schematic topology, not river-line geometry."
                ),
                "",
                f"Machine-readable graph audit: [{graph_summary_path}]({graph_summary_path})",
                "",
            ]
    lines = [
        f"# {summary['report_title']}",
        "",
        "## Technical summary",
        "",
        evidence_note,
        "",
        *visualization_lines,
        *graph_visualization_lines,
        "## Scope and evidence",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Training commit: `{summary['commit']}`",
        f"- Successful jobs: {summary['experiment_count']}",
        "- Selection metric: validation macro NSE",
        "- Shared budget: maximum 50 epochs with patience-8 early stopping",
        "",
        "## Condition-level validation results",
        "",
        "| Condition | Runs | Macro NSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | Duration mean (s) | Peak VRAM mean (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in conditions.items():
        nse = metrics["val_macro_nse"]
        mae = metrics["val_macro_mae"]
        rmse = metrics["val_macro_rmse"]
        duration = metrics["duration_s"]
        vram = metrics["peak_vram_gb"]
        lines.append(
            f"| {name} | {nse['count']} | {nse['mean']:.4f} ± {nse['std']:.4f} | "
            f"{mae['mean']:.4f} ± {mae['std']:.4f} | {rmse['mean']:.4f} ± {rmse['std']:.4f} | "
            f"{duration['mean']:.2f} | {vram['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired RiverLagNet mechanism deltas",
            "",
            "Positive values favor the full directed learned-lag model over the named ablation on the same seed.",
            "",
            "| Ablation | Mean learned-minus-ablation macro NSE | Paired SD | Full-model wins | Directionally supported |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, values in deltas.items():
        supported = values["mean_delta_macro_nse"] > 0 and values["wins"] >= 3
        lines.append(
            f"| {name} | {values['mean_delta_macro_nse']:.4f} | "
            f"{values['std_delta_macro_nse']:.4f} | {values['wins']}/{len(seeds)} | "
            f"{'yes' if supported else 'no'} |"
        )
    if real_world:
        no_lag = deltas["no_lag"]
        fixed_lag = deltas["fixed_lag"]
        lines.extend(
            [
                "",
                "## Validation decision",
                "",
                (
                    "The learned-lag model does not beat `no_lag` under the predeclared "
                    f"rule: its paired mean macro-NSE delta is "
                    f"{no_lag['mean_delta_macro_nse']:.4f}, with wins in "
                    f"{no_lag['wins']}/{len(seeds)} seeds."
                ),
                "",
                (
                    "It is directionally above `fixed_lag`, but only by "
                    f"{fixed_lag['mean_delta_macro_nse']:.4f} macro NSE with wins in "
                    f"{fixed_lag['wins']}/{len(seeds)} seeds. Because `no_lag` is stronger "
                    "on average and both effect sizes are small, these results do not "
                    "establish a stable learned propagation-time advantage on this dataset."
                ),
            ]
        )
    if identifiable:
        lines.extend(
            [
                "",
                "## Comparator interpretation",
                "",
                "The fixed-lag condition receives the exact synthetic travel-time prior and is therefore an oracle-like comparator. The undirected condition contains every correct edge plus reverse edges, so it is reported but is not a primary directionality decision in this suite.",
            ]
        )
    test = summary.get("test")
    if test is not None:
        assert isinstance(test, dict)
        test_metrics = test["metrics"]
        assert isinstance(test_metrics, dict)
        lines.extend(
            [
                "",
                "## Held-out test results",
                "",
                f"These metrics summarize only the {len(seeds)} full learned-lag checkpoints after all validation comparisons were fixed.",
                "",
                "| Metric | Runs | Mean ± SD | Min | Max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in TEST_METRICS:
            stats = test_metrics[metric]
            lines.append(
                f"| {metric} | {stats['count']} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The predeclared engineering rule calls a mechanism directionally supported only when the paired mean delta is positive and the full model wins at least three of five seeds. Test data are excluded from these decisions. Five seeds quantify pipeline variability but do not establish statistical significance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_summary(
    summary: Mapping[str, object], json_path: Path, markdown_path: Path
) -> None:
    """Write machine-readable and technical-report views of one summary."""
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_lf(
        json_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    _write_text_lf(markdown_path, render_validation_markdown(summary))


def _write_text_lf(path: Path, content: str) -> None:
    """Write version-controlled text with stable LF line endings on every OS."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
