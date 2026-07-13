# RiverLagNet v0.1

RiverLagNet is a daily, multi-station water-quality forecasting system centered on **Directed Lag-aware River Message Passing**. It predicts `NH3N`, `CODMn`, and `TP` for the next 30 days from 90 historical days while preserving explicit upstream-to-downstream river direction and discrete travel lags.

This repository validates the engineering and training loop with deterministic synthetic river data. A checksum-verified catalog of source-normalized HydroWQ China sample bundles can also be imported locally for data-interface and model-forward validation. It does not claim real-world predictive performance.

## Environment

Use the existing Conda environment `DeepWater`:

```powershell
conda run -n DeepWater python -m pip install .
conda run -n DeepWater python -m pytest -q
```

On Windows checkouts whose path contains non-ASCII characters, use a regular installation (`pip install .`) rather than editable installation because Python 3.10 may read editable `.pth` files with the system code page.

## Import reusable HydroWQ China assets

The seventh-paper workspace contains hundreds of GiB of raw climate and raster data. RiverLagNet imports only the small processed China multi-basin manifest whitelist (sample NPZ files, river graphs, static attributes, and three audit metadata files):

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.import_hydrowq `
  --source-processed "D:\05.Paper\07.第七篇论文\Code\data\processed" `
  --destination "data\processed\hydrowq-china-multibasin-v0.1"
```

The destination is Git-ignored. Every manifest asset is SHA-256 verified before and after copying; raw rasters, archives, caches, logs, and run outputs are excluded. The imported arrays were already normalized using the source project's training-basin statistics. Do not normalize them again.

These samples have 45 history days and 46 forecast days, so they are intentionally not wired into the default 90-to-30 chronological experiment. `HydroWQChinaCatalog.compatibility()` exposes this mismatch explicitly. See the [import audit](docs/data/hydrowq-china-import-audit-2026-07-13.md) for provenance and quality results.

## Train

Full RiverLagNet:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.train
```

One-batch synthetic validation:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.train trainer.fast_dev_run=true data=synthetic
```

Station GRU baseline:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.train model=station_gru
```

Evaluate only a validation-selected checkpoint:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.evaluate checkpoint_path="runs/.../checkpoints/model.ckpt"
```

Hydra groups are in `configs/data`, `configs/model`, `configs/trainer`, and `configs/experiment`. Trainer precision defaults to `16-mixed` on GPU and automatically falls back to `32-true` on CPU.

## Completed synthetic comparison

The seed-42 engineering comparison has been completed for all four required models with the same chronological synthetic split and training budget. Each full run automatically validates its best validation-macro-NSE checkpoint and appends one immutable row to `experiments/results.tsv`:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.train model=persistence experiment.name=synthetic_seed42_persistence run_dir=runs/synthetic_seed42_persistence trainer.enable_progress_bar=false
conda run -n DeepWater python -m RiverLagNet.cli.train model=station_gru experiment.name=synthetic_seed42_station_gru run_dir=runs/synthetic_seed42_station_gru trainer.enable_progress_bar=false
conda run -n DeepWater python -m RiverLagNet.cli.train model=static_gat experiment.name=synthetic_seed42_static_gat run_dir=runs/synthetic_seed42_static_gat trainer.enable_progress_bar=false
conda run -n DeepWater python -m RiverLagNet.cli.train model=riverlagnet experiment.name=synthetic_seed42_riverlagnet run_dir=runs/synthetic_seed42_riverlagnet trainer.enable_progress_bar=false
```

RiverLagNet achieved validation/test macro NSE of `0.6697/0.6283` in this single-seed synthetic run. Checkpoints and full logs remain under ignored `runs/` directories. See the [complete training report](docs/training_run_2026-07-13.md) for all metrics and limitations; these values are not empirical water-quality results.

## Multi-seed robustness suite

The paired robustness matrix covers five seeds (`42`–`46`) and nine conditions: three baselines, five RiverLagNet ablations, and the full learned-lag model. The runner is resumable from successful rows in `experiments/results.tsv`:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --summarize
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --evaluate-final
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --summarize
```

All 45 training jobs completed without a crash. Full RiverLagNet validation macro NSE was `0.7067 ± 0.0404`; its held-out test macro NSE across the five validation-selected checkpoints was `0.6865 ± 0.0561`. The predeclared validation rule directionally supported learned lag over `no_graph`, `no_lag`, and `fixed_lag`, but did not support the full model over `undirected_graph` or `shuffled_graph`. See the [multi-seed robustness report](docs/robustness_report_2026-07-13.md) and machine-readable [summary](experiments/robustness_summary.json).

These are synthetic engineering results, not evidence of field predictive skill. A separate [Caravan-Qual readiness audit](docs/data/caravan-qual-readiness-audit-2026-07-13.md) found that the available three-target observations are too sparse for the fixed daily 90-to-30 main experiment without changing the scientific task.

## Identifiable directed-lag benchmark

`data=synthetic_identifiable_v1` preserves the 90-to-30 task while adding independent slowly varying pollutant events that propagate only along the true directed tree. Synthetic truth components are retained for data diagnostics but never enter model batches.

Run the training-only data gate before any benchmark training:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.check_identifiability
```

The frozen v1 gate passes all five seeds: routed contribution accounts for `0.247–0.491` of non-root variance, true direction exceeds reversed/shuffled controls by `0.207–0.492`, and all `105/105` edge-target lags are recovered within one day. The gate uses only the chronological training period.

Run, summarize, and finally evaluate the versioned matrix with:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --summarize
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --evaluate-final
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --suite identifiable_v1 --summarize
```

In v1 the edge travel-time prior equals the true lag, so `fixed_lag` is an oracle-like comparator and learned lag is not required to outperform it. The undirected graph contains every correct edge plus reverse edges, so it is reported but is not a primary directionality decision. See the [data-gate report](docs/identifiable_v1_data_gate_2026-07-13.md). This constructed benchmark tests engineering identifiability, not field predictive skill or observational causality.

## Models and ablations

The common training/evaluation path supports:

1. Persistence
2. Station GRU without a graph
3. Static Directed GAT without lag selection
4. RiverLagNet with joint incoming-edge × lag attention

Required ablations are `no_graph`, `undirected_graph`, `shuffled_graph`, `no_lag`, `fixed_lag`, and `learned_lag`. Attention weights are routing weights and must not be interpreted as causal contributions.

## Outputs

Run artifacts, checkpoints, CSV logs, TensorBoard logs, and Hydra outputs are stored under ignored run directories. Only the experiment ledger `experiments/results.tsv` is version controlled. Test-set results must not be used for model selection or hyperparameter tuning.

See [data schema](docs/data_schema.md) and [architecture](docs/architecture.md) for tensor contracts and implementation details.
