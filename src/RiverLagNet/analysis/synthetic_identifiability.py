"""Training-only diagnostics for the identifiable synthetic benchmark."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from RiverLagNet.data.synthetic_identifiable import (
    SyntheticScenario,
    generate_identifiable_synthetic_scenario,
)


@dataclass(frozen=True)
class ScenarioIdentifiability:
    """Direction, lag, and routed-signal diagnostics for one seed."""

    seed: int
    routed_variance_ratio: float
    true_direction_correlation: float
    reverse_direction_correlation: float
    shifted_direction_correlation: float
    direction_margin: float
    lag_recovered: int
    lag_pairs: int


@dataclass(frozen=True)
class IdentifiabilityGateReport:
    """Aggregate pass/fail result over a predeclared seed set."""

    passed: bool
    scenarios: tuple[ScenarioIdentifiability, ...]
    pooled_lag_recovery_rate: float


def evaluate_scenario_identifiability(
    scenario: SyntheticScenario,
    train_end: int,
    seed: int = 0,
    max_lag: int = 14,
) -> ScenarioIdentifiability:
    """Measure identifiable direction and lag using only the training period."""
    values = scenario.data.values[:train_end]
    if train_end < max_lag + 3 or train_end > scenario.data.values.shape[0]:
        raise ValueError("train_end cannot support the requested lag search")
    delta = values[1:] - values[:-1]
    source, destination = scenario.data.graph.edge_index
    true_lag = scenario.true_lag_days.long()
    shifted_source = source.roll(1)
    true_correlations: list[float] = []
    reverse_correlations: list[float] = []
    shifted_correlations: list[float] = []
    recovered = 0
    pairs = 0
    for edge in range(source.numel()):
        lag = int(true_lag[edge].item())
        for target in range(values.shape[-1]):
            source_delta = delta[:, source[edge], target]
            destination_delta = delta[:, destination[edge], target]
            true_correlations.append(
                _lagged_correlation(source_delta, destination_delta, lag)
            )
            reverse_correlations.append(
                _lagged_correlation(destination_delta, source_delta, lag)
            )
            shifted_correlations.append(
                _lagged_correlation(
                    delta[:, shifted_source[edge], target], destination_delta, lag
                )
            )
            candidates = [
                _lagged_correlation(source_delta, destination_delta, candidate)
                for candidate in range(max_lag + 1)
            ]
            best_lag = max(range(max_lag + 1), key=candidates.__getitem__)
            recovered += int(abs(best_lag - lag) <= 1)
            pairs += 1

    non_root = destination.unique(sorted=True)
    routed_variance = scenario.routed_load[:train_end, non_root].var(unbiased=False)
    final_variance = values[:, non_root].var(unbiased=False)
    variance_ratio = float(routed_variance / final_variance.clamp_min(torch.finfo(values.dtype).eps))
    true_mean = statistics.fmean(true_correlations)
    reverse_mean = statistics.fmean(reverse_correlations)
    shifted_mean = statistics.fmean(shifted_correlations)
    return ScenarioIdentifiability(
        seed=int(seed),
        routed_variance_ratio=variance_ratio,
        true_direction_correlation=true_mean,
        reverse_direction_correlation=reverse_mean,
        shifted_direction_correlation=shifted_mean,
        direction_margin=true_mean - max(reverse_mean, shifted_mean),
        lag_recovered=recovered,
        lag_pairs=pairs,
    )


def run_identifiability_gate(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> IdentifiabilityGateReport:
    """Generate default scenarios and evaluate all predeclared gates."""
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("gate seeds must be nonempty and unique")
    scenarios = tuple(
        evaluate_scenario_identifiability(
            generate_identifiable_synthetic_scenario(seed=seed),
            train_end=int(520 * 0.70),
            seed=seed,
        )
        for seed in normalized
    )
    recovered = sum(item.lag_recovered for item in scenarios)
    pairs = sum(item.lag_pairs for item in scenarios)
    recovery_rate = recovered / pairs
    passed = (
        all(0.20 <= item.routed_variance_ratio <= 0.60 for item in scenarios)
        and all(item.direction_margin >= 0.15 for item in scenarios)
        and recovery_rate >= 0.80
    )
    return IdentifiabilityGateReport(
        passed=passed,
        scenarios=scenarios,
        pooled_lag_recovery_rate=recovery_rate,
    )


def write_identifiability_gate(
    report: IdentifiabilityGateReport, json_path: Path, markdown_path: Path
) -> None:
    """Write machine-readable and technical views of one gate result."""
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Identifiable synthetic v1 data gate",
        "",
        f"Overall status: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "All metrics use only the chronological training period of complete synthetic signals.",
        "",
        "| Seed | Routed variance ratio | True correlation | Reverse correlation | Shifted correlation | Direction margin | Lag recovered |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.scenarios:
        lines.append(
            f"| {item.seed} | {item.routed_variance_ratio:.4f} | "
            f"{item.true_direction_correlation:.4f} | "
            f"{item.reverse_direction_correlation:.4f} | "
            f"{item.shifted_direction_correlation:.4f} | "
            f"{item.direction_margin:.4f} | {item.lag_recovered}/{item.lag_pairs} |"
        )
    lines.extend(
        [
            "",
            f"Pooled lag recovery within one day: `{report.pooled_lag_recovery_rate:.4f}`.",
            "",
            "Thresholds: routed variance ratio 0.20--0.60 per seed; direction margin at least 0.15 per seed; pooled lag recovery at least 0.80.",
            "",
            "These diagnostics establish synthetic benchmark identifiability, not real-world causality.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _lagged_correlation(source_delta: Tensor, destination_delta: Tensor, lag: int) -> float:
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    if lag == 0:
        left, right = source_delta, destination_delta
    else:
        left, right = source_delta[:-lag], destination_delta[lag:]
    left = left.flatten().float()
    right = right.flatten().float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    if denominator <= torch.finfo(left.dtype).eps:
        return 0.0
    return float((left * right).mean() / denominator)
