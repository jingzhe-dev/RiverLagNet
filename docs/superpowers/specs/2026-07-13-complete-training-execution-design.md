# Complete Training Execution Design

## Objective

Complete the first reproducible RiverLagNet training comparison on the validated synthetic dataset. Train Persistence, Station GRU, Static Directed GAT, and RiverLagNet with the same chronological split, seed, input variables, metrics, and 50-epoch/early-stopping budget. Select checkpoints only by validation macro NSE and evaluate the held-out test split only after training.

## Data decision

The imported HydroWQ China bundles remain excluded from the main comparison. Their 45-history/46-forecast annual windows and basin-holdout split are incompatible with the fixed 90-history/30-forecast chronological protocol. Re-windowing them would invent temporal continuity that the source bundle does not contain.

The complete training run therefore uses `data=synthetic`, seed 42, 260 daily steps, eight nodes, and the repository's leakage-safe 70/15/15 target split. Results are engineering baselines and make no real-world performance claim.

## Experiment recording

Each non-smoke training run appends exactly one row to `experiments/results.tsv` after `Trainer.fit` completes. The row captures UTC timestamp, current commit, branch, experiment name, seed, final best validation macro NSE/MAE/RMSE, fit duration, peak allocated VRAM, status, and description.

The checkpoint callback selects the maximum validation macro NSE. Metrics for the ledger are obtained by validating the selected best checkpoint, not by copying an arbitrary final epoch. Persistence uses the same path even though it has no learned forecasting parameters.

Rows are written through a focused `ExperimentRecord` API. TSV field order and allowed statuses are validated. Smoke runs never alter the ledger. If a full CLI run raises after it has started, a `crash` row is written with empty numeric metrics and the concise exception description, then the exception is re-raised.

## Run layout

Each model receives a unique ignored directory:

```text
runs/synthetic_seed42_persistence/
runs/synthetic_seed42_station_gru/
runs/synthetic_seed42_static_gat/
runs/synthetic_seed42_riverlagnet/
```

Each directory contains the best checkpoint, CSV logs, TensorBoard logs, Hydra/runtime output, and timing/VRAM metrics. Version-controlled results contain only the compact TSV ledger.

## Verification

Before training, unit tests cover TSV schema, valid append behavior, invalid status rejection, and smoke-run suppression. After implementation is committed, run all four models sequentially. For every model verify that a best checkpoint exists and can be loaded for held-out testing. Finally run the complete pytest suite, inspect the four ledger rows, confirm `runs/` remains ignored, update README with the exact commands and limitations, commit, and push `main`.

## Scope boundaries

- No hyperparameter search or test-set model selection.
- No claim that synthetic metrics demonstrate empirical water-quality skill.
- No resampling or double-normalization of imported HydroWQ data.
- No raw data, checkpoints, TensorBoard logs, or large runtime output in Git.
