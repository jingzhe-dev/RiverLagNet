# Identifiable directed-lag synthetic benchmark design

## Objective and single hypothesis

Build a second synthetic scenario in which correct upstream topology and nonzero travel lag are observable from the training period and useful for 90-to-30 forecasting, without changing RiverLagNet model code.

The single experiment hypothesis is:

> Independent, slowly varying pollutant events propagated only along the true directed river tree will make upstream topology and travel lag identifiable, while retaining local dynamics, noise, missingness, and a 30-day forecast horizon.

This phase tests benchmark identifiability. It does not tune the RiverLagNet architecture and does not claim real-world predictive performance.

## Evidence motivating the change

The completed five-seed robustness suite showed the full model was not better on average than `undirected_graph` or `shuffled_graph`. A source-derived diagnostic over seeds 42--46 found:

- mean within-station lag-1 correlation: `0.9728`;
- mean true-edge level correlation at the configured lag: `0.4503`;
- mean shifted-source level correlation: `0.2930`;
- mean correlation between the true lagged source and downstream one-day change: `0.0241`;
- mean corresponding shifted-source correlation: `-0.0250`.

The current generator therefore provides strong local persistence but only a small incremental graph signal after a local one-step baseline. Level correlation alone is not enough because autocorrelated, shared-frequency signals make incorrect sources useful proxies.

The controlling sources for this diagnosis are `src/RiverLagNet/data/synthetic.py` at commit `b0afb7cb6dffa24bc26fe85b3f2150540405984f` and the 45 immutable rows in `experiments/results.tsv` from training commit `e4d41dd954b9559f96f68a7c2744fbcfbc89beb3`.

## Alternatives considered

### Selected: independent slow events with causal routing

Generate independent node-level pollutant events and propagate their load through the true tree using edge-specific lags and attenuation. This supplies auditable ground truth, supports multi-hop propagation, and makes incorrect topology distinguishable.

### Rejected: increase only the upstream coefficient

Increasing the existing `0.15` coefficient is a smaller code change, but all nodes retain shared periodic structure and high autocorrelation. A wrong source may remain a useful proxy, so a larger coefficient would not resolve the identifiability defect.

### Rejected: latent continuous autoregressive drivers

Independent latent AR drivers could improve topology specificity, but they are harder to audit and make lag recovery less direct. They remain a possible later realism extension after the event benchmark works.

## Isolation and compatibility

- Preserve `generate_synthetic_river_data` and `data=synthetic` unchanged so previous tests and experiment rows remain reproducible.
- Add `data=synthetic_identifiable_v1` as a separate Hydra configuration.
- Keep `T_in=90`, `T_out=30`, the chronological `70% / 15% / 15%` split, the three target order, masks, quality values, graph schema, model configurations, optimizer, epoch budget, and validation checkpoint selection unchanged.
- Use 520 days, 8 nodes, 3 target variables, and 8% missingness. With a 90-to-30 window this yields 401 eligible windows, providing enough event transitions in each chronological split without changing the window contract.
- No synthetic-only ground truth tensor may enter a training or evaluation batch.

## Public data structures and interfaces

Add an immutable `SyntheticScenario` dataclass with:

```text
data: TimeSeriesData
true_lag_days: Tensor[E]
local_background: Tensor[T,N,V]
local_events: Tensor[T,N,V]
routed_load: Tensor[T,N,V]
```

Add:

```text
generate_identifiable_synthetic_scenario(...) -> SyntheticScenario
```

`RiverDataModule` accepts a `scenario` option with values `legacy` and `identifiable_v1`. It stores the returned ground truth only on a diagnostic property and passes `scenario.data` into the existing normalization, split, and window pipeline. Batch keys and tensor shapes do not change.

Both root Hydra configs and packaged mirror configs must remain structurally identical.

## Graph generation

Use the existing deterministic tree construction:

- destination nodes are `1..N-1`;
- every destination samples one source from a lower node index;
- node indices are therefore a valid topological order;
- edges remain `source -> destination`;
- distance, slope, and integer travel-time prior remain the three edge attributes;
- the integer travel time is sampled from 1--7 days and is also the true lag in v1.

Because the prior is exact in v1, `fixed_lag` is an oracle-like reference. The design does not require learned lag to outperform fixed lag. Prior error or time-varying travel time is a separate future hypothesis.

## Signal generation

All random draws use one CPU `torch.Generator` seeded from the configured seed.

### Local background

For every node and target, generate a positive local background:

```text
b[t] = 0.45 * b[t-1]
     + 0.45 * node_offset
     + 0.05 * sin(2*pi*t/30 + node_target_phase)
     + epsilon[t]

epsilon ~ Normal(0, 0.05)
```

Offsets and phases remain node- and target-specific. Clamp only the final combined values, not intermediate diagnostic components.

### Independent local events

For each node, draw event starts independently with daily probability `0.025`. An event has:

- integer duration uniformly sampled from 12--24 days;
- base amplitude uniformly sampled from `0.5--1.0`;
- target multipliers `[0.8, 1.0, 0.6]` for `NH3N`, `CODMn`, and `TP`;
- exponentially decaying shape `amplitude * exp(-3*k/duration)` for event day `k`;
- additive superposition when events overlap.

The duration is longer than the maximum seven-day edge lag, so recent upstream history contains information about load that will reach downstream nodes after the forecast origin.

### Directed multi-hop routing

Process days in ascending order and nodes in topological order. Define transported load as:

```text
load[t,n,v] = local_event[t,n,v]
            + sum over edges j->n of attenuation[e,v] * load[t-true_lag[e],j,v]
```

Terms with negative time indices are zero. Edge attenuation is:

```text
base[e] = 0.75 * exp(-distance_km[e] / 100) * exp(-20 * slope[e])
attenuation[e,:] = base[e] * [0.85, 1.0, 0.75]
```

The routed contribution stored in `routed_load` excludes each node's own `local_events`; it contains only the incoming sum. Final values are:

```text
values = clamp_min(local_background + local_events + routed_load, 0)
```

Observed masks and quality scores use the existing deterministic mechanism after the complete values are generated.

## Training-only identifiability gates

Model training is forbidden until all gates pass for seeds 42--46. Gates use only the chronological training portion of each generated time series, before missing values are applied where a complete signal is required.

### Structural and component gates

- Shapes match the documented contracts and all components are finite.
- `values` exactly reconstruct from the three stored components after final clamping.
- Every graph edge is directed from a lower topological index to a higher index.
- All true lags are in 1--7 days and equal the rounded travel-time prior.
- Non-root routed-load variance divided by final-value variance is between `0.20` and `0.60` for every seed after pooling nodes and targets.

### Observed-signal direction gate

For every edge and target, correlate the lagged source first difference with the downstream first difference. Differencing removes the slow local level and aligns event onsets and decays across the travel lag. Compute the same statistic for:

1. the reversed edge; and
2. a deterministic shifted-source edge list.

Average edge-target correlations within each seed. For every seed:

```text
true_mean - max(reverse_mean, shifted_mean) >= 0.15
```

This is a benchmark identifiability gate, not a causal estimate from observational data.

### Lag-recovery gate

For every true edge and target, search candidate lags 0--14 and choose the lag with maximum positive correlation between source and downstream first differences. At least 80% of edge-target pairs, pooled over seeds, must recover a lag within one day of ground truth.

### Leakage gate

Reuse the existing chronological window assertions. No input or normalization statistic may use observations after its forecast origin, and normalization remains fitted only on the training period.

If any data gate fails, no neural training starts. A correction requires a new generator commit and rerunning all gates. Once the gates pass and the first model run starts, v1 parameters are frozen and may not be adjusted from validation or test performance.

## Experiment matrix and decision rules

Run seeds `42, 43, 44, 45, 46` with the same nine conditions and training budget as the completed robustness suite. Use experiment names:

```text
ident_v1_s<seed>_<condition>
```

and run directories:

```text
runs/ident_v1_s<seed>_<condition>/
```

The runner must pass `data=synthetic_identifiable_v1`. Successful rows must share one training commit. Write separate generated artifacts:

- `experiments/identifiable_v1_summary.json`;
- `docs/identifiable_v1_report_2026-07-13.md`.

Primary validation comparisons are full `learned_lag` RiverLagNet minus `no_graph`, `shuffled_graph`, and `no_lag`. A mechanism is directionally supported when the paired mean macro-NSE delta is positive and the full model wins at least three of five seeds.

Report `station_gru`, `static_gat`, `undirected_graph`, and `fixed_lag` without changing the rule after results are observed. In particular:

- `undirected_graph` contains all true edges plus reverse edges, so failure to beat it does not by itself refute directionality;
- `fixed_lag` receives the exact v1 lag prior, so learned-lag parity is acceptable and superiority is not required.

Do not inspect held-out test metrics until all validation summaries and decisions are written. Then evaluate exactly the five validation-selected full-model checkpoints once.

## Experiment runner changes

Generalize the existing suite definitions without changing the already completed `robust_s*` matrix:

- a suite preset owns its prefix, data override, conditions, summary paths, and expected seeds;
- `robustness_v1` reproduces the current commands and names exactly;
- `identifiable_v1` adds only `data=synthetic_identifiable_v1` and new paths/names;
- ledger parsing remains append-only and strict for the selected preset;
- dry-run, resume, validation summary, final evaluation, and partial-test-output rejection work for both presets.

If implementation changes are required after some `ident_v1` jobs have succeeded, do not mix commits or duplicate successful names. Create a new versioned preset and prefix, retain all prior ledger rows, and classify the interrupted suite accurately.

## Tests

Add tests before implementation for:

- deterministic identical output for the same seed and different output for another seed;
- preservation of the legacy generator interface and behavior;
- component reconstruction and finite nonnegative final values;
- exact directed multi-hop routing on a tiny hand-constructed tree;
- event duration, causal time indexing, attenuation, and lag indexing;
- all five-seed structural, contribution, direction, and lag-recovery gates;
- absence of synthetic truth tensors from model batches;
- Hydra root/package mirror compatibility for `synthetic_identifiable_v1`;
- unchanged window leakage and train-only normalization behavior;
- exact suite commands, prefix, data override, resume behavior, and separate summary outputs;
- Lightning one-batch training for the new scenario on CPU and CUDA mixed precision when available;
- full `pytest` and nine-condition fast-dev preflight before formal training.

Tests must not weaken existing assertions or alter historical ledger rows.

## Documentation and result boundaries

Update README with the new scenario, commands, data gates, and links after results exist. The report must distinguish:

- verified generator identifiability;
- validation evidence about model mechanisms;
- held-out test performance;
- synthetic engineering evidence versus real water-quality evidence.

Attention weights remain routing weights and must not be described as causal contributions.

## Explicit non-goals

- No RiverLagNet architecture or hyperparameter change.
- No probability forecasting, physical loss, or extreme-event module.
- No tuning from held-out test results.
- No replacement of missing real Chinese NH3N observations with proxies.
- No claim that success on the identifiable scenario establishes field skill.
- No noisy lag prior or time-varying travel time in v1.
