# RiverLagNet real daily seed-42 training report

## Result

Validation selected **riverlagnet**. On the held-out test period, the descriptively highest macro NSE was **static_gat**. Test data were not used to select checkpoints or tune the model.

## Fixed protocol

- Training commit: `ba67183679c297d111c4a03d556d9f3741b355f1`
- Seed: 42
- Daily history/forecast: 90/30 days
- Chronological split: 70%/15%/15%
- Selection: validation macro NSE; maximum 50 epochs; patience 8
- Precision/device: BF16 mixed precision on one RTX PRO 6000
- Imputed source values: excluded from scaling, loss, and metrics

## Validation and held-out test

MAE and RMSE are in mg/L because all three targets use that unit.

| Model | Val NSE | Val MAE | Val RMSE | Test NSE | Test MAE | Test RMSE | Duration (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.2925 | 0.2170 | 0.5633 | 0.5979 | 0.2007 | 0.4913 | 2.10 | 0.0719 |
| Station GRU | 0.5632 | 0.2087 | 0.4932 | 0.7196 | 0.1907 | 0.4218 | 30.55 | 0.5887 |
| Static Directed GAT | 0.5702 | 0.2011 | 0.4836 | 0.7335 | 0.1843 | 0.4159 | 316.80 | 0.5888 |
| RiverLagNet | 0.5782 | 0.2061 | 0.4904 | 0.7284 | 0.1968 | 0.4354 | 97.22 | 1.0422 |

## Per-target held-out metrics

| Model | NH3N NSE / MAE / RMSE | CODMn NSE / MAE / RMSE | TP NSE / MAE / RMSE |
|---|---:|---:|---:|
| Persistence | 0.2844 / 0.1234 / 0.3300 | 0.8294 / 0.4599 / 0.7834 | 0.6799 / 0.0189 / 0.0400 |
| Station GRU | 0.5232 / 0.1159 / 0.2694 | 0.8721 / 0.4381 / 0.6783 | 0.7635 / 0.0179 / 0.0343 |
| Static Directed GAT | 0.5440 / 0.1074 / 0.2634 | 0.8753 / 0.4285 / 0.6697 | 0.7812 / 0.0170 / 0.0330 |
| RiverLagNet | 0.5497 / 0.1130 / 0.2618 | 0.8612 / 0.4596 / 0.7065 | 0.7743 / 0.0178 / 0.0336 |

## Interpretation

RiverLagNet improved held-out macro NSE over Persistence by 0.1305 and over Station GRU by 0.0088. Static Directed GAT was 0.0051 higher than RiverLagNet on the held-out period, despite RiverLagNet having the best validation NSE. The defensible single-seed conclusion is that graph information helped, while a stable learned-lag advantage over a static directed graph is not yet established.

## Execution audit and limitations

- Retained crash row: `china_real_daily_station_gru_seed42` — OSError: [Errno 22] Invalid argument
- One station-to-river mapping remains flagged for manual review.
- Travel-time priors use an assumed 30 km/day velocity, not measured hydraulics.
- This report covers one seed and one regional graph; multi-seed runs and lag/no-lag ablations are required before a mechanism claim.
- Attention weights are routing weights and must not be interpreted as causal contributions.
