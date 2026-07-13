# Identifiable Synthetic Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe synthetic scenario whose directed topology and 1--7 day travel lags pass predeclared training-only identifiability gates, then run the existing nine-condition five-seed experiment matrix without changing model code.

**Architecture:** Keep the legacy generator untouched and place the new scenario plus synthetic-only truth in a focused module. A separate analysis module computes and persists data gates before the DataModule or suite can train the new preset. Generalize the existing suite through immutable presets so historical `robust_s*` commands remain byte-for-byte compatible while `ident_v1_s*` uses the new data override and output paths.

**Tech Stack:** Python 3.10, PyTorch, Lightning, Hydra/OmegaConf, pytest, Git; existing Conda environment `DeepWater` only.

## Global Constraints

- Preserve `generate_synthetic_river_data`, `data=synthetic`, all historical experiment rows, and all RiverLagNet model files unchanged.
- Use exactly 520 days, 8 nodes, 3 variables, 8% missingness, `T_in=90`, `T_out=30`, chronological `70% / 15% / 15%`, and seeds `42..46` for formal experiments.
- Fit normalization only on the chronological training period and never include synthetic truth in a batch.
- Gate thresholds are routed variance ratio `0.20..0.60` per seed, direction margin at least `0.15` per seed, and pooled lag recovery at least `0.80` within one day.
- Once the first `ident_v1` model job starts, generator parameters are frozen. Never use held-out test metrics for selection or tuning.
- Keep root Hydra YAML and `src/RiverLagNet/configs` mirrors identical.
- Use `$env:PYTHONUTF8='1'` and `C:\Program Files\ANACONDA\envs\DeepWater\python.exe`; do not create another environment.
- Do not commit checkpoints, run logs, generated training artifacts, or raw data.

---

## File map

- Create `src/RiverLagNet/data/synthetic_identifiable.py`: scenario dataclass, event generation, causal multi-hop routing, final observations.
- Create `src/RiverLagNet/analysis/synthetic_identifiability.py`: metric calculation, gate validation, JSON/Markdown serialization.
- Create `src/RiverLagNet/cli/check_identifiability.py`: reproducible training-only gate entrypoint.
- Modify `src/RiverLagNet/data/datamodule.py`: select legacy or identifiable scenario while keeping batches unchanged.
- Create `configs/data/synthetic_identifiable_v1.yaml` and packaged mirror.
- Modify `configs/data/synthetic.yaml` and packaged mirror to declare `scenario: legacy` explicitly.
- Modify `src/RiverLagNet/analysis/experiment_suite.py`: immutable suite presets and data-aware training/evaluation commands.
- Modify `src/RiverLagNet/cli/run_experiment_suite.py`: `--suite` selection, preset paths, and mandatory identifiable gate.
- Modify `src/RiverLagNet/analysis/robustness_summary.py`: preset-specific report title, comparison subset, and scope language.
- Create focused tests under `tests/data`, `tests/analysis`, and extend existing integration tests.
- Generate versioned gate, validation summary, and report files only from executed code.

---

### Task 1: Deterministic event and routing generator

**Files:**
- Create: `src/RiverLagNet/data/synthetic_identifiable.py`
- Create: `tests/data/test_synthetic_identifiable.py`
- Preserve: `src/RiverLagNet/data/synthetic.py`

**Interfaces:**
- Produces: `SyntheticScenario`, `route_pollutant_events(local_events: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]`, and `generate_identifiable_synthetic_scenario(num_days: int = 520, num_nodes: int = 8, num_variables: int = 3, missing_rate: float = 0.08, seed: int = 42) -> SyntheticScenario`.
- Consumes: `RiverGraph` and `TimeSeriesData` from `RiverLagNet.data.schema`.

- [ ] **Step 1: Write failing deterministic, reconstruction, and routing tests**

```python
import torch

from RiverLagNet.data.synthetic import generate_synthetic_river_data
from RiverLagNet.data.synthetic_identifiable import (
    generate_identifiable_synthetic_scenario,
    route_pollutant_events,
)


def test_identifiable_scenario_is_deterministic_and_reconstructs_values() -> None:
    first = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=11)
    second = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=11)
    different = generate_identifiable_synthetic_scenario(num_days=180, num_nodes=6, seed=12)

    first.data.validate()
    assert torch.equal(first.data.graph.edge_index, second.data.graph.edge_index)
    assert torch.allclose(first.data.values, second.data.values)
    assert not torch.allclose(first.data.values, different.data.values)
    reconstructed = (
        first.local_background + first.local_events + first.routed_load
    ).clamp_min(0.0)
    assert torch.allclose(first.data.values, reconstructed)
    assert torch.equal(
        first.true_lag_days,
        first.data.graph.edge_attr[:, -1].round().long(),
    )
    assert torch.all(first.data.graph.edge_index[0] < first.data.graph.edge_index[1])


def test_route_pollutant_events_applies_exact_lags_and_multihop_attenuation() -> None:
    events = torch.zeros(7, 3, 3)
    events[0, 0] = 1.0
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    attrs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

    total, routed = route_pollutant_events(events, edges, attrs)

    first = 0.75 * torch.tensor([0.85, 1.0, 0.75])
    assert torch.allclose(routed[1, 1], first)
    assert torch.allclose(routed[3, 2], first.square())
    assert torch.count_nonzero(routed[:1, 1]) == 0
    assert torch.count_nonzero(routed[:3, 2]) == 0
    assert torch.allclose(total, events + routed)


def test_legacy_generator_remains_deterministic() -> None:
    first = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)
    second = generate_synthetic_river_data(num_days=160, num_nodes=6, seed=11)
    assert torch.equal(first.graph.edge_index, second.graph.edge_index)
    assert torch.allclose(first.values, second.values)
```

- [ ] **Step 2: Run the tests and confirm the import fails**

Run:

```powershell
$env:PYTHONUTF8='1'
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/data/test_synthetic_identifiable.py -q
```

Expected: collection fails because `RiverLagNet.data.synthetic_identifiable` does not exist.

- [ ] **Step 3: Implement the immutable scenario and routing helper**

Implement these exact public contracts and constants:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .schema import RiverGraph, TimeSeriesData


EVENT_TARGET_MULTIPLIERS = torch.tensor([0.8, 1.0, 0.6])
ROUTING_TARGET_MULTIPLIERS = torch.tensor([0.85, 1.0, 0.75])


@dataclass(frozen=True)
class SyntheticScenario:
    data: TimeSeriesData
    true_lag_days: Tensor
    local_background: Tensor
    local_events: Tensor
    routed_load: Tensor


def route_pollutant_events(
    local_events: Tensor, edge_index: Tensor, edge_attr: Tensor
) -> tuple[Tensor, Tensor]:
    days, nodes, variables = local_events.shape
    if variables != 3:
        raise ValueError("identifiable v1 requires exactly three target variables")
    source, destination = edge_index
    lag_days = edge_attr[:, -1].round().long()
    if bool(((lag_days < 1) | (lag_days > 7)).any()):
        raise ValueError("identifiable v1 lags must be in 1..7 days")
    distance = edge_attr[:, 0]
    slope = edge_attr[:, 1]
    base = 0.75 * torch.exp(-distance / 100.0) * torch.exp(-20.0 * slope)
    attenuation = base[:, None] * ROUTING_TARGET_MULTIPLIERS.to(edge_attr)
    total = local_events.clone()
    routed = torch.zeros_like(local_events)
    for day in range(days):
        for node in range(nodes):
            incoming = torch.nonzero(destination == node, as_tuple=False).flatten()
            for edge in incoming.tolist():
                lag = int(lag_days[edge])
                if day >= lag:
                    contribution = attenuation[edge] * total[day - lag, source[edge]]
                    total[day, node] += contribution
                    routed[day, node] += contribution
    return total, routed
```

Build the remaining private helpers and generator with these concrete operations:

```python
def _generate_graph(num_nodes: int, generator: torch.Generator) -> RiverGraph:
    destination = torch.arange(1, num_nodes, dtype=torch.long)
    source = torch.tensor(
        [
            int(torch.randint(0, node, (1,), generator=generator))
            for node in range(1, num_nodes)
        ],
        dtype=torch.long,
    )
    edge_index = torch.stack((source, destination))
    distance = 10.0 + 40.0 * torch.rand(num_nodes - 1, generator=generator)
    slope = 0.001 + 0.02 * torch.rand(num_nodes - 1, generator=generator)
    lag = torch.randint(1, 8, (num_nodes - 1,), generator=generator).float()
    edge_attr = torch.stack((distance, slope, lag), dim=-1)
    position = torch.linspace(0.0, 1.0, num_nodes)
    static = torch.stack(
        (position, torch.linspace(0.2, 1.0, num_nodes), 1.0 - position), dim=-1
    )
    return RiverGraph(edge_index=edge_index, edge_attr=edge_attr, static=static)


def _generate_background(
    num_days: int, num_nodes: int, generator: torch.Generator
) -> Tensor:
    background = torch.zeros(num_days, num_nodes, 3)
    phases = 2.0 * math.pi * torch.rand(num_nodes, 3, generator=generator)
    offsets = 0.5 + 1.5 * torch.rand(num_nodes, 3, generator=generator)
    for day in range(num_days):
        seasonal = torch.sin(torch.tensor(2.0 * math.pi * day / 30.0) + phases)
        innovation = 0.05 * torch.randn(num_nodes, 3, generator=generator)
        previous = background[day - 1] if day else offsets
        background[day] = (
            0.45 * previous + 0.45 * offsets + 0.05 * seasonal + innovation
        )
    return background


def _generate_events(
    num_days: int, num_nodes: int, generator: torch.Generator
) -> Tensor:
    events = torch.zeros(num_days, num_nodes, 3)
    target_scale = EVENT_TARGET_MULTIPLIERS.to(events)
    for node in range(num_nodes):
        for day in range(num_days):
            if float(torch.rand((), generator=generator)) >= 0.025:
                continue
            duration = int(torch.randint(12, 25, (1,), generator=generator))
            amplitude = 0.5 + 0.5 * float(torch.rand((), generator=generator))
            length = min(duration, num_days - day)
            age = torch.arange(length, dtype=events.dtype)
            shape = amplitude * torch.exp(-3.0 * age / duration)
            events[day : day + length, node] += shape[:, None] * target_scale
    return events


def generate_identifiable_synthetic_scenario(
    num_days: int = 520,
    num_nodes: int = 8,
    num_variables: int = 3,
    missing_rate: float = 0.08,
    seed: int = 42,
) -> SyntheticScenario:
    if num_days < 2 or num_nodes < 2:
        raise ValueError("identifiable v1 requires at least two days and two nodes")
    if num_variables != 3:
        raise ValueError("identifiable v1 requires exactly three target variables")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    graph = _generate_graph(num_nodes, generator)
    background = _generate_background(num_days, num_nodes, generator)
    events = _generate_events(num_days, num_nodes, generator)
    transported, routed = route_pollutant_events(events, graph.edge_index, graph.edge_attr)
    values = (background + transported).clamp_min(0.0)
    observed = torch.rand(values.shape, generator=generator) >= missing_rate
    quality = observed.to(values.dtype) * (
        0.8 + 0.2 * torch.rand(values.shape, generator=generator)
    )
    data = TimeSeriesData(values, observed, quality, graph)
    data.validate()
    return SyntheticScenario(
        data=data,
        true_lag_days=graph.edge_attr[:, -1].round().long(),
        local_background=background,
        local_events=events,
        routed_load=routed,
    )
```

- [ ] **Step 4: Run focused and legacy generator tests**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/data/test_synthetic_identifiable.py tests/data/test_synthetic.py -q
```

Expected: all tests pass and `src/RiverLagNet/data/synthetic.py` has no diff.

- [ ] **Step 5: Commit the generator**

```powershell
git add src/RiverLagNet/data/synthetic_identifiable.py tests/data/test_synthetic_identifiable.py
git commit -m "feat: add identifiable synthetic generator"
```

---

### Task 2: Training-only identifiability metrics and hard gate

**Files:**
- Create: `src/RiverLagNet/analysis/synthetic_identifiability.py`
- Create: `src/RiverLagNet/cli/check_identifiability.py`
- Create: `tests/analysis/test_synthetic_identifiability.py`

**Interfaces:**
- Consumes: `SyntheticScenario` and `generate_identifiable_synthetic_scenario` from Task 1.
- Produces: `ScenarioIdentifiability`, `IdentifiabilityGateReport`, `evaluate_scenario_identifiability`, `run_identifiability_gate`, and `write_identifiability_gate`.

- [ ] **Step 1: Write failing metric-alignment and real-gate tests**

```python
import pytest
import torch

from RiverLagNet.analysis.synthetic_identifiability import (
    evaluate_scenario_identifiability,
    run_identifiability_gate,
)
from RiverLagNet.data.synthetic_identifiable import SyntheticScenario
from RiverLagNet.data.schema import RiverGraph, TimeSeriesData


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
```

- [ ] **Step 2: Run the tests and confirm the analysis module is missing**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/analysis/test_synthetic_identifiability.py -q
```

Expected: collection fails because `synthetic_identifiability` does not exist.

- [ ] **Step 3: Implement correlation alignment and gate dataclasses**

Use these exact result types:

```python
@dataclass(frozen=True)
class ScenarioIdentifiability:
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
    passed: bool
    scenarios: tuple[ScenarioIdentifiability, ...]
    pooled_lag_recovery_rate: float
```

Implement first-difference correlation with explicit lag alignment:

```python
def _lagged_correlation(source_delta: Tensor, destination_delta: Tensor, lag: int) -> float:
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
```

Implement `evaluate_scenario_identifiability(scenario: SyntheticScenario, train_end: int, seed: int = 0, max_lag: int = 14) -> ScenarioIdentifiability`. It must use only `scenario.data.values[:train_end]`, compute `delta = values[1:] - values[:-1]`, roll edge sources by one for the shifted control, search lags 0--14 independently for each edge-target pair, and calculate routed variance only over non-root nodes. Calculate `direction_margin = true_mean - max(reverse_mean, shifted_mean)` and count a lag recovered when `abs(best_lag - true_lag) <= 1`.

Implement `run_identifiability_gate(seeds: Sequence[int] = (42, 43, 44, 45, 46)) -> IdentifiabilityGateReport` by generating a 520-day scenario per seed, using `train_end=364`, pooling `lag_recovered` and `lag_pairs`, and setting `passed` to the conjunction of:

```python
all(0.20 <= item.routed_variance_ratio <= 0.60 for item in scenarios)
and all(item.direction_margin >= 0.15 for item in scenarios)
and pooled_lag_recovery_rate >= 0.80
```

Do not drop or coerce a failed metric.

- [ ] **Step 4: Implement durable JSON/Markdown output and CLI failure semantics**

`write_identifiability_gate` writes dataclass fields to JSON and a compact technical Markdown table. The CLI accepts `--seeds`, `--json`, and `--markdown`, defaults to:

```text
experiments/identifiable_v1_data_gate.json
docs/identifiable_v1_data_gate_2026-07-13.md
```

It writes the report, prints the three gate outcomes, and exits nonzero with `SystemExit(1)` when `report.passed` is false.

- [ ] **Step 5: Run focused tests and the real five-seed gate**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/analysis/test_synthetic_identifiability.py -q
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.check_identifiability
```

Expected: tests pass; CLI exits 0; every seed passes contribution and direction gates; pooled lag recovery is at least `0.80`. If it fails, stop before any model training, diagnose the failed formula or fixed generator parameters, add a regression test, and commit the correction before rerunning.

- [ ] **Step 6: Commit the gate implementation and generated gate evidence**

```powershell
git add src/RiverLagNet/analysis/synthetic_identifiability.py src/RiverLagNet/cli/check_identifiability.py tests/analysis/test_synthetic_identifiability.py experiments/identifiable_v1_data_gate.json docs/identifiable_v1_data_gate_2026-07-13.md
git commit -m "feat: gate identifiable synthetic data"
```

---

### Task 3: DataModule, Hydra, packaging, and one-batch training

**Files:**
- Modify: `src/RiverLagNet/data/datamodule.py`
- Modify: `configs/data/synthetic.yaml`
- Modify: `src/RiverLagNet/configs/data/synthetic.yaml`
- Create: `configs/data/synthetic_identifiable_v1.yaml`
- Create: `src/RiverLagNet/configs/data/synthetic_identifiable_v1.yaml`
- Modify: `tests/data/test_datamodule.py`
- Modify: `tests/integration/test_fast_dev_run.py`
- Verify: `tests/integration/test_packaging.py`

**Interfaces:**
- Consumes: generator and scenario dataclass from Task 1.
- Produces: the existing `RiverDataModule` constructor extended with `scenario: str = "legacy"`, plus diagnostic-only `synthetic_scenario: SyntheticScenario | None`.

- [ ] **Step 1: Write failing DataModule and batch-isolation tests**

```python
def test_identifiable_datamodule_uses_scenario_without_leaking_truth() -> None:
    module = RiverDataModule(
        scenario="identifiable_v1",
        num_days=180,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=4,
        seed=5,
    )
    module.setup("fit")
    assert module.synthetic_scenario is not None
    assert module.data is module.synthetic_scenario.data
    batch = next(iter(module.train_dataloader()))
    assert set(batch).isdisjoint(
        {"true_lag_days", "local_background", "local_events", "routed_load"}
    )
    expected = []
    for feature in range(module.data.values.shape[-1]):
        values = module.data.values[: module.train_end, :, feature]
        mask = module.data.observed[: module.train_end, :, feature]
        expected.append(values[mask].mean())
    assert torch.allclose(module.scaler.mean, torch.stack(expected))


def test_datamodule_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="scenario"):
        RiverDataModule(scenario="unknown")
```

Extend the existing fast-dev parametrization with `scenario="identifiable_v1"` for `station_gru` and `riverlagnet` on CPU.

- [ ] **Step 2: Run focused tests and verify the constructor fails**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/data/test_datamodule.py tests/integration/test_fast_dev_run.py tests/integration/test_packaging.py -q
```

Expected: the new scenario tests fail because `RiverDataModule` has no `scenario` parameter/config.

- [ ] **Step 3: Add explicit scenario selection without changing batch construction**

Modify the constructor and setup using this branch:

```python
if scenario not in {"legacy", "identifiable_v1"}:
    raise ValueError("scenario must be legacy or identifiable_v1")
self.synthetic_scenario: SyntheticScenario | None = None

# inside setup, only when self.data is None
if self.hparams.scenario == "identifiable_v1":
    self.synthetic_scenario = generate_identifiable_synthetic_scenario(
        num_days=self.hparams.num_days,
        num_nodes=self.hparams.num_nodes,
        num_variables=self.hparams.num_variables,
        missing_rate=self.hparams.missing_rate,
        seed=self.hparams.seed,
    )
    self.data = self.synthetic_scenario.data
else:
    self.data = generate_synthetic_river_data(
        num_days=self.hparams.num_days,
        num_nodes=self.hparams.num_nodes,
        num_variables=self.hparams.num_variables,
        missing_rate=self.hparams.missing_rate,
        seed=self.hparams.seed,
    )
```

Do not modify `RiverWindowDataset` or `river_collate`.

- [ ] **Step 4: Add root and packaged Hydra configs**

Both new YAML files contain exactly:

```yaml
scenario: identifiable_v1
num_days: 520
num_nodes: 8
num_variables: 3
missing_rate: 0.08
input_window: 90
output_window: 30
batch_size: 16
num_workers: 0
pin_memory: false
seed: ${seed}
```

Add `scenario: legacy` as the first key in both existing synthetic YAML files. Do not change any other legacy value.

- [ ] **Step 5: Run data, packaging, and CPU fast-dev tests**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/data tests/integration/test_fast_dev_run.py tests/integration/test_packaging.py -q
```

Expected: all tests pass, scaler assertions use only the first 70%, and no truth key reaches a batch.

- [ ] **Step 6: Commit DataModule and configuration integration**

```powershell
git add src/RiverLagNet/data/datamodule.py configs/data/synthetic.yaml src/RiverLagNet/configs/data/synthetic.yaml configs/data/synthetic_identifiable_v1.yaml src/RiverLagNet/configs/data/synthetic_identifiable_v1.yaml tests/data/test_datamodule.py tests/integration/test_fast_dev_run.py
git commit -m "feat: integrate identifiable synthetic data"
```

---

### Task 4: Immutable experiment-suite presets

**Files:**
- Modify: `src/RiverLagNet/analysis/experiment_suite.py`
- Modify: `src/RiverLagNet/cli/run_experiment_suite.py`
- Modify: `src/RiverLagNet/analysis/robustness_summary.py`
- Modify: `tests/analysis/test_experiment_suite.py`
- Modify: `tests/analysis/test_robustness_summary.py`

**Interfaces:**
- Produces: `SuitePreset`, `ROBUSTNESS_V1`, `IDENTIFIABLE_V1`, `SUITE_PRESETS`, and preset-aware `build_experiment_specs`.
- Preserves: `build_experiment_specs(seeds)` defaults to the historical `robust_s*` matrix and existing commands omit a data override.

- [ ] **Step 1: Write failing historical-compatibility and identifiable-preset tests**

```python
from RiverLagNet.analysis.experiment_suite import (
    IDENTIFIABLE_V1,
    ROBUSTNESS_V1,
    build_experiment_specs,
    training_command,
)


def test_historical_preset_keeps_exact_names_and_command() -> None:
    spec = build_experiment_specs([42], ROBUSTNESS_V1)[0]
    assert spec.experiment_name == "robust_s42_persistence"
    command = training_command(spec, "python")
    assert "data=synthetic_identifiable_v1" not in command


def test_identifiable_preset_builds_45_unique_data_aware_specs() -> None:
    specs = build_experiment_specs([42, 43, 44, 45, 46], IDENTIFIABLE_V1)
    assert len(specs) == 45
    assert len({spec.experiment_name for spec in specs}) == 45
    assert specs[0].experiment_name == "ident_v1_s42_persistence"
    assert "data=synthetic_identifiable_v1" in training_command(specs[0], "python")
    learned = next(spec for spec in specs if spec.condition.name == "learned_lag")
    checkpoint = learned.run_dir / "checkpoints" / "best.ckpt"
    assert "data=synthetic_identifiable_v1" in evaluation_command(
        learned, checkpoint, "python"
    )
```

Add a CLI parsing test by exposing `run(args: Sequence[str] | None = None)` from the suite CLI and monkeypatching execution. Assert `--suite identifiable_v1 --dry-run` prints `ident_v1_s42_*` and the data override, while default dry-run preserves `robust_s42_*`.

- [ ] **Step 2: Run analysis tests and confirm preset imports fail**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/analysis/test_experiment_suite.py tests/analysis/test_robustness_summary.py -q
```

Expected: collection fails because suite preset symbols do not exist.

- [ ] **Step 3: Add preset and spec metadata**

Use these fields:

```python
@dataclass(frozen=True)
class SuitePreset:
    name: str
    experiment_prefix: str
    data_override: str | None
    summary_json: Path
    summary_markdown: Path
    report_title: str
    primary_comparisons: tuple[str, ...]


ROBUSTNESS_V1 = SuitePreset(
    name="robustness_v1",
    experiment_prefix="robust",
    data_override=None,
    summary_json=Path("experiments/robustness_summary.json"),
    summary_markdown=Path("docs/robustness_report_2026-07-13.md"),
    report_title="RiverLagNet multi-seed robustness and ablation report",
    primary_comparisons=(
        "no_graph", "undirected_graph", "shuffled_graph", "no_lag", "fixed_lag"
    ),
)

IDENTIFIABLE_V1 = SuitePreset(
    name="identifiable_v1",
    experiment_prefix="ident_v1",
    data_override="synthetic_identifiable_v1",
    summary_json=Path("experiments/identifiable_v1_summary.json"),
    summary_markdown=Path("docs/identifiable_v1_report_2026-07-13.md"),
    report_title="RiverLagNet identifiable synthetic benchmark report",
    primary_comparisons=("no_graph", "shuffled_graph", "no_lag"),
)
```

Add `preset: SuitePreset` to `ExperimentSpec`. Name specs as `robust_s<seed>_<condition>` for `ROBUSTNESS_V1` and `ident_v1_s<seed>_<condition>` for `IDENTIFIABLE_V1`. Insert `data=<override>` immediately after `seed=<seed>` only when a preset has a data override. Apply the same override to final evaluation commands.

- [ ] **Step 4: Generalize report rendering without altering historical numeric content**

Pass `preset.report_title` and `preset.primary_comparisons` into summary/render functions. The identifiable report must state that fixed lag receives the exact prior and undirected contains both correct and reverse edges. The robustness default must reproduce the existing title, all five paired deltas, and JSON field values.

- [ ] **Step 5: Add CLI suite selection and mandatory pre-training gate**

Add:

```python
parser.add_argument(
    "--suite", choices=tuple(SUITE_PRESETS), default="robustness_v1"
)
```

When selected preset is identifiable and the operation is training rather than summary/final evaluation, call `run_identifiability_gate(args.seeds)` before reading pending experiments. Write gate outputs through `write_identifiability_gate`. If `passed` is false, raise `RuntimeError("identifiable_v1 data gate failed")` before spawning a training subprocess. Derive default summary paths from the preset only when the user did not supply explicit paths.

- [ ] **Step 6: Run focused tests and both dry-runs**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest tests/analysis -q
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --dry-run
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --dry-run
```

Expected: analysis tests pass; default shows historical `robust_s*`; identifiable shows 45 pending `ident_v1_s*` commands with `data=synthetic_identifiable_v1`.

- [ ] **Step 7: Commit suite presets**

```powershell
git add src/RiverLagNet/analysis/experiment_suite.py src/RiverLagNet/analysis/robustness_summary.py src/RiverLagNet/cli/run_experiment_suite.py tests/analysis/test_experiment_suite.py tests/analysis/test_robustness_summary.py
git commit -m "feat: add identifiable experiment preset"
```

---

### Task 5: Full verification, documentation, and training freeze

**Files:**
- Modify: `README.md`
- Modify only if an agent-caused error occurred: `docs/agent_errors.md`
- Verify all files modified in Tasks 1--4.

**Interfaces:**
- Produces: one clean, pushed implementation commit that all formal training rows will reference.

- [ ] **Step 1: Add README commands and interpretation boundary**

Document these commands:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.check_identifiability
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --summarize
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --evaluate-final
```

State that v1 uses exact lag priors, fixed lag is oracle-like, undirected includes true edges, and results remain synthetic engineering evidence.

- [ ] **Step 2: Run the complete test suite**

Run:

```powershell
$env:PYTHONUTF8='1'
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest -q
```

Expected: zero failures and no deleted or weakened tests.

- [ ] **Step 3: Run the formal data gate from source**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.check_identifiability
```

Expected: five per-seed contribution/direction gates pass and pooled lag recovery is at least `0.80`.

- [ ] **Step 4: Install the non-editable package and verify packaged CLI/configs**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pip install .
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --dry-run
```

Expected: installation exits 0 and installed-package dry-run shows exactly 45 identifiable commands.

- [ ] **Step 5: Run nine-condition CUDA mixed-precision preflight without ledger writes**

For each of the nine preset conditions, run the generated training command with these appended overrides:

```text
trainer.fast_dev_run=true
trainer.enable_progress_bar=false
experiment.record_result=false
```

Expected: every command exits 0 on the available RTX PRO 6000, uses `16-mixed`, and appends no `ident_v1_s*` ledger row.

- [ ] **Step 6: Verify diff, commit documentation or corrections, and push the frozen branch**

Run:

```powershell
git diff --check
git status --short
git add README.md docs/agent_errors.md
git commit -m "docs: document identifiable benchmark workflow"
git push -u origin research/20260713-identifiable-synthetic
git status -sb
```

If `docs/agent_errors.md` has no new factual entry, omit it from `git add`. If README already committed with another task and no files changed, do not create an empty commit. Record the resulting `git rev-parse HEAD`; it is the required training commit.

---

### Task 6: Formal 45-job training, frozen validation decisions, final test, and result commit

**Files generated or modified by execution:**
- Modify append-only: `experiments/results.tsv`
- Generate: `experiments/identifiable_v1_summary.json`
- Generate: `docs/identifiable_v1_report_2026-07-13.md`
- Update after results: `README.md`
- Preserve under ignored paths: `runs/ident_v1_s*/`

**Interfaces:**
- Consumes: the frozen implementation commit and passing gate from Task 5.
- Produces: 45 validation rows sharing one commit, five final test JSON files under ignored runs, one strict summary, one technical report, and one pushed result commit.

- [ ] **Step 1: Confirm the ledger has no successful identifiable rows and the worktree is clean**

Run:

```powershell
git status --porcelain
Import-Csv experiments/results.tsv -Delimiter "`t" |
  Where-Object { $_.experiment -like 'ident_v1_s*' } |
  Format-Table experiment,status,commit
```

Expected: clean worktree and zero matching rows. If resumable rows exist, verify they use the current training commit before continuing; never delete them.

- [ ] **Step 2: Execute the resumable formal suite**

Run:

```powershell
$env:PYTHONUTF8='1'
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1
```

Expected: 45 successful jobs, no crash row, and one append-only row per experiment.

- [ ] **Step 3: Freeze and inspect validation-only summary before test evaluation**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --summarize
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -c "import json; s=json.load(open('experiments/identifiable_v1_summary.json',encoding='utf-8')); assert s['experiment_count']==45; assert 'test' not in s; print(s['paired_deltas'])"
```

Expected: strict loader accepts exactly 45 successful rows sharing one commit, and the summary contains no test section. Record validation decisions without changing generator/model/configuration.

- [ ] **Step 4: Evaluate exactly five full-model checkpoints once**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --evaluate-final
```

Expected: exactly five `runs/ident_v1_s*_learned_lag/test_metrics.json` files and no new training ledger row.

- [ ] **Step 5: Regenerate final report and validate experiment integrity**

Run:

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --summarize
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest -q
git diff --check
```

Then run a Python integrity assertion that checks 45 unique `ident_v1_s*` rows, one nonempty commit, finite validation metrics, five test seeds, count five for every aggregated test metric, and no incomplete test output set.

Expected: all tests pass and every integrity assertion succeeds.

- [ ] **Step 6: Update README with actual results and limitations**

Report condition means, primary paired deltas/wins, target test NSE values, training commit, gate values, and the explicit statement that this benchmark was constructed for identifiability and is not field evidence. Do not claim attention weights are causal.

- [ ] **Step 7: Commit and push immutable experiment evidence**

```powershell
git add experiments/results.tsv experiments/identifiable_v1_summary.json docs/identifiable_v1_report_2026-07-13.md README.md docs/agent_errors.md
git commit -m "exp: record identifiable synthetic results"
git push
git status -sb
git rev-parse HEAD
```

Omit `docs/agent_errors.md` if it has no new factual entry. Expected: clean branch tracking `origin/research/20260713-identifiable-synthetic`.

---

## Plan self-review result

- Spec coverage: generator isolation, exact signal equations, multi-hop routing, ground-truth isolation, all three data gates, leakage, Hydra mirrors, suite versioning, nine conditions, five seeds, validation freeze, final test, documentation, commits, and push each map to a task above.
- Scope: one benchmark-identifiability hypothesis; no architecture or hyperparameter work is included.
- Type consistency: `SyntheticScenario`, `ScenarioIdentifiability`, `IdentifiabilityGateReport`, `SuitePreset`, and preset names are defined once and consumed under the same names.
- Failure handling: a failed data gate stops before training; a post-start implementation correction requires a new versioned preset rather than mixed commits.
