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
