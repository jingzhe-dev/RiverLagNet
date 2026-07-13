# RiverLagNet horizon-aligned routing benchmark report

## Technical summary

This report is generated from validation-selected checkpoints in the append-only experiment ledger. It tests an identifiable synthetic benchmark; it is not a real-world water-quality result or a formal significance test.

## Scope and evidence

- Seeds: 42, 43, 44, 45, 46
- Training commit: `167c1f17f3ad2d8c246cee2844f429f4b1306bf9`
- Successful jobs: 45
- Selection metric: validation macro NSE
- Shared budget: maximum 50 epochs with patience-8 early stopping

## Condition-level validation results

| Condition | Runs | Macro NSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | Duration mean (s) | Peak VRAM mean (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 5 | 0.1787 ± 0.1215 | 0.6894 ± 0.1159 | 0.9335 ± 0.1477 | 1.49 | 0.0011 |
| station_gru | 5 | 0.4894 ± 0.0724 | 0.5463 ± 0.0860 | 0.7386 ± 0.1233 | 4.38 | 0.1243 |
| static_gat | 5 | 0.4740 ± 0.0954 | 0.5596 ± 0.0934 | 0.7497 ± 0.1391 | 22.48 | 0.1244 |
| no_graph | 5 | 0.4828 ± 0.0723 | 0.5473 ± 0.0880 | 0.7427 ± 0.1224 | 3.61 | 0.1244 |
| undirected_graph | 5 | 0.4869 ± 0.0820 | 0.5497 ± 0.0856 | 0.7399 ± 0.1284 | 11.45 | 0.2746 |
| shuffled_graph | 5 | 0.4852 ± 0.0779 | 0.5502 ± 0.0850 | 0.7400 ± 0.1242 | 9.94 | 0.1850 |
| no_lag | 5 | 0.4877 ± 0.0736 | 0.5489 ± 0.0769 | 0.7390 ± 0.1226 | 12.15 | 0.1783 |
| fixed_lag | 5 | 0.4895 ± 0.0754 | 0.5465 ± 0.0788 | 0.7376 ± 0.1238 | 11.63 | 0.1850 |
| learned_lag | 5 | 0.4893 ± 0.0741 | 0.5466 ± 0.0783 | 0.7375 ± 0.1228 | 11.94 | 0.1850 |

## Paired RiverLagNet mechanism deltas

Positive values favor the full directed learned-lag model over the named ablation on the same seed.

| Ablation | Mean learned-minus-ablation macro NSE | Paired SD | Full-model wins | Directionally supported |
|---|---:|---:|---:|---|
| no_graph | 0.0065 | 0.0100 | 3/5 | yes |
| shuffled_graph | 0.0042 | 0.0117 | 4/5 | yes |
| no_lag | 0.0016 | 0.0010 | 5/5 | yes |

## Comparator interpretation

The fixed-lag condition receives the exact synthetic travel-time prior and is therefore an oracle-like comparator. The undirected condition contains every correct edge plus reverse edges, so it is reported but is not a primary directionality decision in this suite.

## Held-out test results

These metrics summarize only the five full learned-lag checkpoints after all validation comparisons were fixed.

| Metric | Runs | Mean ± SD | Min | Max |
|---|---:|---:|---:|---:|
| test_macro_nse | 5 | 0.4024 ± 0.1600 | 0.2614 | 0.6005 |
| test_macro_mae | 5 | 0.5593 ± 0.1065 | 0.4470 | 0.6893 |
| test_macro_rmse | 5 | 0.7656 ± 0.2049 | 0.5550 | 1.0978 |
| test_nse_NH3N | 5 | 0.3909 ± 0.2293 | 0.0043 | 0.5730 |
| test_nse_CODMn | 5 | 0.1462 ± 0.2997 | -0.2265 | 0.5092 |
| test_nse_TP | 5 | 0.6701 ± 0.1263 | 0.4998 | 0.7913 |

## Interpretation boundary

The predeclared engineering rule calls a mechanism directionally supported only when the paired mean delta is positive and the full model wins at least three of five seeds. Test data are excluded from these decisions. Five seeds quantify pipeline variability but do not establish statistical significance.
