# Multi-seed robustness and ablation design

## Objective

Determine whether RiverLagNet's current synthetic advantage is robust across random seeds and whether it depends specifically on directed upstream routing and learned discrete lag selection.

## Data decision

The available real-data audit does not pass the fixed China three-target gate: Caravan-Qual has zero Chinese NH3N observations, and its global three-target stations are temporally sparse. No real-data training claim will be made in this phase. The robustness suite uses the leakage-tested synthetic generator and paired seeds so every condition within a seed receives the same graph, signals, missingness, initialization seed, split, and budget.

## Conditions and seeds

Use seeds `42, 43, 44, 45, 46`. Each condition uses the default 50-epoch maximum, early-stopping patience 8, `16-mixed`, batch size 16, and validation macro NSE checkpoint selection.

The nine unique conditions are:

| Condition | Model/configuration | Purpose |
|---|---|---|
| `persistence` | Persistence | non-learned baseline |
| `station_gru` | Station GRU | local temporal baseline |
| `static_gat` | Static Directed GAT | directed graph without lag selection |
| `no_graph` | RiverLagNet, graph bypassed | contribution of all upstream information |
| `undirected_graph` | RiverLagNet, reverse edges added | importance of hydrological direction |
| `shuffled_graph` | RiverLagNet, destinations shuffled deterministically | importance of correct topology |
| `no_lag` | directed RiverLagNet, lag fixed to zero | contribution of lagged states |
| `fixed_lag` | directed RiverLagNet, edge-prior lag | learned versus prior-only lag |
| `learned_lag` | full directed RiverLagNet | proposed model and ablation reference |

This creates 45 training jobs. The full `learned_lag` condition is also the required RiverLagNet baseline, so it is not duplicated under a second name.

## Reproducible execution

Add a resumable suite CLI that generates the exact condition × seed matrix, invokes the existing Hydra training entrypoint as a separate process, and skips only experiment names already present with status `baseline`, `keep`, or `discard`. `crash` rows are rerun. Each run directory is `runs/robust_s<seed>_<condition>/`, and each ledger experiment name matches the directory name.

The runner must stop on the first new failure by default. A dry-run prints the pending matrix without training. All model code and runner code are committed before execution so the 45 successful rows share one Git commit.

## Summary and decision rules

Generate the summary from `experiments/results.tsv`, never by transcribing console values. For each condition report count, mean, sample standard deviation, minimum, and maximum for validation macro NSE, MAE, RMSE, duration, and peak VRAM.

For each ablation and seed, compute:

```text
delta_macro_nse = learned_lag_macro_nse - ablation_macro_nse
```

Report the paired mean delta and the number of seeds on which the full model wins. The mechanism evidence is considered directionally supported when the full model has positive mean delta and wins at least three of five paired seeds. This is an engineering rule, not a formal significance test.

Do not inspect held-out test results until all validation summaries and mechanism decisions are frozen. Then evaluate only the five validation-selected `learned_lag` checkpoints once and aggregate their test metrics automatically.

## Components

1. `analysis/experiment_suite.py` owns immutable condition definitions, experiment names, ledger completion checks, command generation, and sequential execution.
2. `cli/run_experiment_suite.py` exposes seeds, dry-run, training, and summary modes.
3. `analysis/robustness_summary.py` parses and validates the exact suite ledger rows and produces machine-readable JSON plus a Markdown technical report.
4. `cli/evaluate.py` gains a reusable `run()` function and optional JSON output so final learned-lag test metrics can be aggregated without manual copying.

## Testing and failure handling

- Unit-test the exact 45-spec matrix, unique names, and correct Hydra overrides.
- Unit-test skip logic: successful rows skip; crash and absent rows remain pending.
- Unit-test summary statistics and paired deltas with a miniature ledger.
- Integration-test one subprocess-free dry-run and one JSON evaluation output using a one-epoch checkpoint fixture.
- Preserve automatic `crash` ledger rows from the training entrypoint.
- Never delete a failed run or edit its numeric result.

## Scope boundaries

- No hyperparameter search.
- No use of test data for model or mechanism selection.
- No claim of statistical significance from five seeds.
- No proxy replacement for missing China NH3N.
- No commit of checkpoints, raw archives, or run logs.
