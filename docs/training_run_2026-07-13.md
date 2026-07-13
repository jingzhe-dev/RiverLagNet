# Complete synthetic training run — 2026-07-13

## Run contract

- Training code commit: `d477e054d95fcef907923dbcf11654f15f14c723`
- Branch: `main`
- Seed: 42
- Data: deterministic synthetic river network, 260 days, 8 nodes, 3 targets
- Window: 90 history days to 30 forecast days
- Split: leakage-safe chronological 70%/15%/15% target periods
- Budget: maximum 50 epochs, early-stopping patience 8, AdamW, gradient clipping 1.0
- Selection: maximum validation macro NSE only
- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, Lightning `16-mixed`

All models used the same data generation, normalization, batches, target masks, evaluation code, seed, and training budget. No test metric was used to select a checkpoint or tune a setting.

## Validation-selected results

These rows were written automatically by the training entrypoint to `experiments/results.tsv` after reloading and validating each best checkpoint.

| Model | Best epoch | Validation macro NSE | Validation macro MAE | Validation macro RMSE | Fit duration (s) | Peak allocated VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0 | 0.410460 | 0.594890 | 0.783872 | 0.670 | 0.0007 |
| Station GRU | 26 | 0.663368 | 0.493393 | 0.592373 | 2.682 | 0.1241 |
| Static Directed GAT | 27 | 0.650530 | 0.502702 | 0.603242 | 14.398 | 0.1242 |
| RiverLagNet | 32 | **0.669733** | **0.492618** | **0.586582** | 8.937 | 0.1246 |

## One-time held-out test evaluation

After all training jobs finished, each validation-selected checkpoint was evaluated once on the held-out chronological test split.

| Model | Test macro NSE | Test macro MAE | Test macro RMSE | NH3N NSE | CODMn NSE | TP NSE |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.363543 | 0.614886 | 0.796779 | 0.636377 | 0.005574 | 0.448679 |
| Station GRU | 0.597346 | 0.517547 | 0.633963 | 0.766538 | 0.341284 | 0.684216 |
| Static Directed GAT | 0.599256 | 0.522567 | 0.633455 | 0.702100 | 0.400980 | 0.694690 |
| RiverLagNet | **0.628303** | **0.506118** | **0.608981** | **0.790144** | 0.393738 | **0.701027** |

RiverLagNet has the strongest validation and test macro NSE in this run. Static Directed GAT has a slightly higher test CODMn NSE, so the result does not imply uniform superiority for every target. With only one seed and synthetic data, the comparison verifies the training system and provides an initial mechanism check; it does not establish statistical significance or real-world water-quality skill.

## Artifacts

The four best checkpoints are stored locally under:

```text
runs/synthetic_seed42_persistence/checkpoints/epoch=000-val_nse=0.4105.ckpt
runs/synthetic_seed42_station_gru/checkpoints/epoch=026-val_nse=0.6634.ckpt
runs/synthetic_seed42_static_gat/checkpoints/epoch=027-val_nse=0.6505.ckpt
runs/synthetic_seed42_riverlagnet/checkpoints/epoch=032-val_nse=0.6697.ckpt
```

The entire `runs/` tree is intentionally ignored by Git. CSV logs, TensorBoard events, checkpoints, and runtime artifacts are not committed.

## Data boundary

The imported HydroWQ China bundle was not used for this comparison. Its 45-to-46 annual windows and basin-holdout split cannot be substituted for the fixed 90-to-30 chronological protocol without a new experiment design. No temporal continuity was fabricated and no source-normalized values were normalized again.
