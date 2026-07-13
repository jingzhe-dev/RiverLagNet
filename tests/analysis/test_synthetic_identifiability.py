from __future__ import annotations

import json
from pathlib import Path

import torch

from RiverLagNet.analysis.synthetic_identifiability import (
    IdentifiabilityGateReport,
    ScenarioIdentifiability,
    evaluate_scenario_identifiability,
    run_identifiability_gate,
    write_identifiability_gate,
)
from RiverLagNet.data.schema import RiverGraph, TimeSeriesData
from RiverLagNet.data.synthetic_identifiable import SyntheticScenario


def test_lag_metric_recovers_a_hand_constructed_three_day_shift() -> None:
    days = 80
    source = torch.zeros(days)
    source[[5, 20, 40, 60]] = torch.tensor([1.0, 0.8, 1.2, 0.7])
    values = torch.zeros(days, 2, 3)
    values[:, 0] = source[:, None]
    values[3:, 1] = 0.7 * source[:-3, None]
    graph = RiverGraph(
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[10.0, 0.01, 3.0]]),
        torch.zeros(2, 3),
    )
    data = TimeSeriesData(values, torch.ones_like(values, dtype=torch.bool), None, graph)
    scenario = SyntheticScenario(
        data=data,
        true_lag_days=torch.tensor([3]),
        local_background=torch.zeros_like(values),
        local_events=values.clone(),
        routed_load=torch.zeros_like(values),
    )

    metrics = evaluate_scenario_identifiability(scenario, train_end=70, seed=0)

    assert metrics.lag_recovered == metrics.lag_pairs
    assert metrics.true_direction_correlation > metrics.reverse_direction_correlation


def test_default_five_seed_scenarios_pass_predeclared_gate() -> None:
    report = run_identifiability_gate(seeds=(42, 43, 44, 45, 46))

    assert report.passed
    assert len(report.scenarios) == 5
    assert all(0.20 <= item.routed_variance_ratio <= 0.60 for item in report.scenarios)
    assert all(item.direction_margin >= 0.15 for item in report.scenarios)
    assert report.pooled_lag_recovery_rate >= 0.80


def test_gate_report_writes_machine_and_reader_outputs(tmp_path: Path) -> None:
    scenario = ScenarioIdentifiability(
        seed=42,
        routed_variance_ratio=0.3,
        true_direction_correlation=0.6,
        reverse_direction_correlation=0.1,
        shifted_direction_correlation=0.2,
        direction_margin=0.4,
        lag_recovered=18,
        lag_pairs=21,
    )
    report = IdentifiabilityGateReport(
        passed=True,
        scenarios=(scenario,),
        pooled_lag_recovery_rate=18 / 21,
    )
    json_path = tmp_path / "gate.json"
    markdown_path = tmp_path / "gate.md"

    write_identifiability_gate(report, json_path, markdown_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["passed"] is True
    assert saved["scenarios"][0]["seed"] == 42
    assert "PASS" in markdown
    assert "0.4000" in markdown
