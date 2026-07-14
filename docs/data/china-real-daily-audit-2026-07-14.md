# China real daily data audit — 2026-07-14

## Scope and provenance

This audit covers the local, Git-ignored `china-real-daily-v0.1` artifact used by the fixed RiverLagNet 90-history/30-forecast protocol. The continuous value table and per-value imputation flags come from the sixth-paper water-quality imputation workspace. Station-to-HydroRIVERS mappings and the 10 previously checksum-verified monitored graph components come from the seventh-paper HydroWQ workspace.

The preparation CLI records SHA256 for the 8.4 GB value table, 606 MB flag table, station mapping, and each generated artifact. The local manifest is `data/processed/china-real-daily-v0.1/manifest.json`; raw and prepared arrays remain excluded from Git.

## Prepared contract

- Dates: 2014-04-01 through 2025-02-13, 3,972 contiguous days.
- Graph: 36 monitored river-segment nodes in 10 disjoint components; 26 directed upstream-to-downstream edges.
- Source monitoring stations: 59; 36 graph nodes receive one to seven stations.
- Targets and channel order: `NH3N`, `CODMn`, `TP`.
- Tensor shape: `[3972, 36, 3]`.
- Non-imputed segment-day observations: 132,715 per target, or 92.8129% of 142,992 possible positions.
- Mapping quality: 58 `nearest_reach_good`, one `nearest_reach_review` (station 2649, 3.687 km from segment 40376480).
- Directed-graph visualization: [PNG](../figures/china_real_daily_upstream_graph.png), [PDF](../figures/china_real_daily_upstream_graph.pdf), and [machine-readable graph audit](../../experiments/china_real_daily_graph_summary.json).

When multiple stations map to a segment, only values with `*_is_imputed=0` are averaged. A segment-day with no such value is null in the review Parquet and has `value=0`, `mask=false`, and `quality=0` in the tensor artifact. Therefore model input, train-only scaling, target loss, and metrics exclude all source-imputed values.

## Chronological split and leakage controls

| Split | Dates | Observed positions per target |
|---|---|---:|
| Train | 2014-04-01 to 2021-11-09 | 90,940 |
| Validation | 2021-11-10 to 2023-06-28 | 20,979 |
| Test | 2023-06-29 to 2025-02-13 | 20,796 |

Scaler statistics use only masked-in training-period observations. Forecast targets are wholly contained in their split; validation and test timestamps are never used for checkpoint selection beyond validation, and test metrics are computed only after training is complete.

## River attributes and travel-time prior

The imported graph stores source-normalized edge attributes. Its former `travel_time_proxy` equals normalized river length and is not a value in days. Preparation restores raw `length_km` using the source normalization metadata and derives:

```text
travel_time_prior_days = length_km / 30 km day^-1
```

The assumed velocity is recorded in the manifest and should be tested in later sensitivity experiments. It is neither measured travel time nor causal evidence. Attention weights likewise remain learned routing weights, not causal contributions.

For the 26 retained edges, this proxy ranges from 0.035 to 1.471 days and rounds to 0 days on 23 edges and 1 day on three edges. Consequently, the current `fixed_lag` ablation is structurally close to `no_lag`. This is a concrete mechanism diagnostic for the small observed validation deltas, not a causal interpretation of routing weights.

## Limitations

- One station-to-river mapping is flagged for manual review.
- Graph nodes aggregate stations to monitored river segments, so the model does not retain within-segment station heterogeneity.
- The travel-speed assumption is a prior rather than a calibrated hydraulic quantity.
- Observation coverage is uneven by node (minimum approximately 12.46%, maximum 100%); macro metrics can therefore be influenced by high-coverage nodes.
- This is one regional dataset. Five paired seeds quantify optimization variability but do not by themselves establish statistical significance or transferable field performance.
