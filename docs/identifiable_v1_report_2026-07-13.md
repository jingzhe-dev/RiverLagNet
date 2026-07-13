# RiverLagNet identifiable synthetic benchmark report

## Technical summary

This report is generated from validation-selected checkpoints in the append-only experiment ledger. It tests an identifiable synthetic benchmark; it is not a real-world water-quality result or a formal significance test.

## Scope and evidence

- Seeds: 42, 43, 44, 45, 46
- Training commit: `3834a4cb9ed00e72a2107ca0c573652754764aa6`
- Successful jobs: 45
- Selection metric: validation macro NSE
- Shared budget: maximum 50 epochs with patience-8 early stopping

## Condition-level validation results

| Condition | Runs | Macro NSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | Duration mean (s) | Peak VRAM mean (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 5 | 0.1787 ± 0.1215 | 0.6894 ± 0.1159 | 0.9335 ± 0.1477 | 1.40 | 0.0011 |
| station_gru | 5 | 0.4894 ± 0.0724 | 0.5463 ± 0.0860 | 0.7386 ± 0.1233 | 4.25 | 0.1243 |
| static_gat | 5 | 0.4740 ± 0.0954 | 0.5596 ± 0.0934 | 0.7497 ± 0.1391 | 18.97 | 0.1244 |
| no_graph | 5 | 0.4914 ± 0.0690 | 0.5497 ± 0.0819 | 0.7365 ± 0.1199 | 4.69 | 0.1244 |
| undirected_graph | 5 | 0.4625 ± 0.0958 | 0.5605 ± 0.0805 | 0.7577 ± 0.1396 | 8.50 | 0.1247 |
| shuffled_graph | 5 | 0.4958 ± 0.0805 | 0.5452 ± 0.0797 | 0.7338 ± 0.1276 | 8.38 | 0.1247 |
| no_lag | 5 | 0.4814 ± 0.0878 | 0.5505 ± 0.0820 | 0.7441 ± 0.1326 | 8.11 | 0.1247 |
| fixed_lag | 5 | 0.4789 ± 0.0901 | 0.5497 ± 0.0845 | 0.7464 ± 0.1339 | 11.38 | 0.1247 |
| learned_lag | 5 | 0.4720 ± 0.0918 | 0.5557 ± 0.0889 | 0.7507 ± 0.1357 | 7.79 | 0.1247 |

## Paired RiverLagNet mechanism deltas

Positive values favor the full directed learned-lag model over the named ablation on the same seed.

| Ablation | Mean learned-minus-ablation macro NSE | Paired SD | Full-model wins | Directionally supported |
|---|---:|---:|---:|---|
| no_graph | -0.0194 | 0.0236 | 1/5 | no |
| shuffled_graph | -0.0238 | 0.0163 | 0/5 | no |
| no_lag | -0.0095 | 0.0091 | 0/5 | no |

## Comparator interpretation

The fixed-lag condition receives the exact synthetic travel-time prior and is therefore an oracle-like comparator. The undirected condition contains every correct edge plus reverse edges, so it is reported but is not a primary directionality decision in this suite.

## Interpretation boundary

The predeclared engineering rule calls a mechanism directionally supported only when the paired mean delta is positive and the full model wins at least three of five seeds. Test data are excluded from these decisions. Five seeds quantify pipeline variability but do not establish statistical significance.
