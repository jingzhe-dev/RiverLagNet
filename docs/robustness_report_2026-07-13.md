# RiverLagNet multi-seed robustness and ablation report

## Technical summary

This report is generated from validation-selected checkpoints in the append-only experiment ledger. It tests engineering robustness on synthetic data; it is not a real-world water-quality result or a formal significance test.

## Scope and evidence

- Seeds: 42, 43, 44, 45, 46
- Training commit: `e4d41dd954b9559f96f68a7c2744fbcfbc89beb3`
- Successful jobs: 45
- Selection metric: validation macro NSE
- Shared budget: maximum 50 epochs with patience-8 early stopping

## Condition-level validation results

| Condition | Runs | Macro NSE mean ± SD | MAE mean ± SD | RMSE mean ± SD | Duration mean (s) | Peak VRAM mean (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 5 | 0.4257 ± 0.0798 | 0.5804 ± 0.0409 | 0.7494 ± 0.0482 | 0.67 | 0.0007 |
| station_gru | 5 | 0.7037 ± 0.0400 | 0.4498 ± 0.0358 | 0.5385 ± 0.0392 | 2.90 | 0.1241 |
| static_gat | 5 | 0.6947 ± 0.0478 | 0.4549 ± 0.0413 | 0.5462 ± 0.0452 | 16.91 | 0.1242 |
| no_graph | 5 | 0.7028 ± 0.0377 | 0.4503 ± 0.0350 | 0.5392 ± 0.0364 | 3.04 | 0.1242 |
| undirected_graph | 5 | 0.7087 ± 0.0375 | 0.4472 ± 0.0353 | 0.5339 ± 0.0369 | 6.10 | 0.1246 |
| shuffled_graph | 5 | 0.7078 ± 0.0412 | 0.4471 ± 0.0364 | 0.5346 ± 0.0402 | 6.10 | 0.1246 |
| no_lag | 5 | 0.7056 ± 0.0407 | 0.4482 ± 0.0373 | 0.5366 ± 0.0400 | 6.41 | 0.1246 |
| fixed_lag | 5 | 0.7053 ± 0.0417 | 0.4491 ± 0.0378 | 0.5369 ± 0.0408 | 5.83 | 0.1246 |
| learned_lag | 5 | 0.7067 ± 0.0404 | 0.4484 ± 0.0371 | 0.5356 ± 0.0393 | 6.35 | 0.1246 |

## Paired RiverLagNet mechanism deltas

Positive values favor the full directed learned-lag model over the named ablation on the same seed.

| Ablation | Mean learned-minus-ablation macro NSE | Paired SD | Full-model wins | Directionally supported |
|---|---:|---:|---:|---|
| no_graph | 0.0039 | 0.0061 | 3/5 | yes |
| undirected_graph | -0.0020 | 0.0057 | 2/5 | no |
| shuffled_graph | -0.0011 | 0.0015 | 1/5 | no |
| no_lag | 0.0011 | 0.0028 | 3/5 | yes |
| fixed_lag | 0.0014 | 0.0025 | 3/5 | yes |

## Held-out test results

These metrics summarize only the five full learned-lag checkpoints after all validation comparisons were fixed.

| Metric | Runs | Mean ± SD | Min | Max |
|---|---:|---:|---:|---:|
| test_macro_nse | 5 | 0.6865 ± 0.0561 | 0.6283 | 0.7614 |
| test_macro_mae | 5 | 0.4650 ± 0.0427 | 0.3978 | 0.5061 |
| test_macro_rmse | 5 | 0.5551 ± 0.0519 | 0.4778 | 0.6090 |
| test_nse_NH3N | 5 | 0.7317 ± 0.1168 | 0.5448 | 0.8584 |
| test_nse_CODMn | 5 | 0.5992 ± 0.1416 | 0.3937 | 0.7214 |
| test_nse_TP | 5 | 0.7287 ± 0.0499 | 0.6886 | 0.8132 |

## Interpretation boundary

The predeclared engineering rule calls a mechanism directionally supported only when the paired mean delta is positive and the full model wins at least three of five seeds. Test data are excluded from these decisions. Five seeds quantify pipeline variability but do not establish statistical significance.
