# RiverLagNet v0.1

RiverLagNet is a daily, multi-station water-quality forecasting system centered on **Directed Lag-aware River Message Passing**. It predicts `NH3N`, `CODMn`, and `TP` for the next 30 days from 90 historical days while preserving explicit upstream-to-downstream river direction and discrete travel lags.

This repository currently validates the engineering and training loop with deterministic synthetic river data. It does not claim real-world predictive performance.

## Environment

Use the existing Conda environment `DeepWater`:

```powershell
conda run -n DeepWater python -m pip install .
conda run -n DeepWater python -m pytest -q
```

On Windows checkouts whose path contains non-ASCII characters, use a regular installation (`pip install .`) rather than editable installation because Python 3.10 may read editable `.pth` files with the system code page.

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
