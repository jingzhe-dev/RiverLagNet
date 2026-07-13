# RiverLagNet identity-safe fusion benchmark report

## Technical summary

This report is generated from validation-selected checkpoints in the append-only experiment ledger. It tests an identifiable synthetic benchmark; it is not a real-world water-quality result or a formal significance test.

## Scope and evidence

- Seeds: 42, 43, 44, 45, 46
- Training commit: `fa5cea77859a647c7794ce140fac5879f5a86c1d`
- Successful jobs: 45
- Selection metric: validation macro NSE
- Shared budget: maximum 50 epochs with patience-8 early stopping

## Condition-level validation results

| Condition | Runs | Macro NSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | Duration mean (s) | Peak VRAM mean (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 5 | 0.1787 ± 0.1215 | 0.6894 ± 0.1159 | 0.9335 ± 0.1477 | 1.40 | 0.0011 |
| station_gru | 5 | 0.4894 ± 0.0724 | 0.5463 ± 0.0860 | 0.7386 ± 0.1233 | 4.24 | 0.1243 |
| static_gat | 5 | 0.4740 ± 0.0954 | 0.5596 ± 0.0934 | 0.7497 ± 0.1391 | 20.17 | 0.1244 |
| no_graph | 5 | 0.4901 ± 0.0645 | 0.5491 ± 0.0827 | 0.7370 ± 0.1171 | 4.02 | 0.1244 |
| undirected_graph | 5 | 0.4756 ± 0.0864 | 0.5535 ± 0.0854 | 0.7489 ± 0.1333 | 8.50 | 0.1248 |
| shuffled_graph | 5 | 0.4983 ± 0.0699 | 0.5405 ± 0.0843 | 0.7309 ± 0.1157 | 10.82 | 0.1248 |
| no_lag | 5 | 0.4842 ± 0.0756 | 0.5450 ± 0.0823 | 0.7417 ± 0.1241 | 8.46 | 0.1248 |
| fixed_lag | 5 | 0.4927 ± 0.0665 | 0.5419 ± 0.0810 | 0.7356 ± 0.1158 | 11.47 | 0.1248 |
| learned_lag | 5 | 0.4842 ± 0.0751 | 0.5479 ± 0.0834 | 0.7417 ± 0.1210 | 9.48 | 0.1248 |

## Paired RiverLagNet mechanism deltas

Positive values favor the full directed learned-lag model over the named ablation on the same seed.

| Ablation | Mean learned-minus-ablation macro NSE | Paired SD | Full-model wins | Directionally supported |
|---|---:|---:|---:|---|
| no_graph | -0.0058 | 0.0360 | 3/5 | no |
| shuffled_graph | -0.0141 | 0.0102 | 1/5 | no |
| no_lag | 0.0000 | 0.0217 | 2/5 | no |

## Comparator interpretation

The fixed-lag condition receives the exact synthetic travel-time prior and is therefore an oracle-like comparator. The undirected condition contains every correct edge plus reverse edges, so it is reported but is not a primary directionality decision in this suite.

## Interpretation boundary

The predeclared engineering rule calls a mechanism directionally supported only when the paired mean delta is positive and the full model wins at least three of five seeds. Test data are excluded from these decisions. Five seeds quantify pipeline variability but do not establish statistical significance.
