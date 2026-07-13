# HydroWQ China processed-data import audit

Audit date: 2026-07-13

## Decision

Import only the processed `hydrowq-china-multibasin-v0.1` sample-bundle whitelist. Do not copy the source workspace's raw ERA5-Land, CHIRPS, raster, archive, cache, log, or run directories. This keeps the repository-facing data footprint small and avoids treating derived data as raw observations.

The imported destination is `data/processed/hydrowq-china-multibasin-v0.1`, which is ignored by Git. Code, tests, provenance, and this audit are version controlled; the copied arrays are not.

## Provenance and integrity

- Source: `D:\05.Paper\07.第七篇论文\Code\data\processed`
- Manifest: `_manifests/sample-bundles-china-multibasin-v0.1.json`
- Manifest SHA-256: `ef45aba685c09a9023d474cc57020098fc427fea30d2e8fdc964f87133f69155`
- Unique manifest assets: 130 (110 daily samples, 10 graph files, 10 static files)
- Additional metadata: `sample-index.parquet`, `normalization-train.json`, `build-report.json`
- Imported files excluding receipt: 134
- Imported bytes excluding receipt: 5,537,499
- Missing referenced assets: 0
- Failed source or destination SHA-256 checks: 0

The importer completes all source checksum and path-containment checks before creating the destination, then verifies every copied manifest asset again.

## Coverage and structure

- Samples: 110 unique sample IDs
- Basin components: 10
- Source split: 66 train, 22 validation, 22 test samples across 6/2/2 held-out basins; no basin overlap was found
- Window coverage: eleven annual windows from 2014-04-01 through 2024-06-27
- Daily contract: 45 history days and 46 forecast days
- Graph sizes: 3–5 nodes and 2–4 directed edges; no self-loops or invalid node indices found
- Static features: 201
- Edge features: 4
- Available dynamic drivers: 16
- Available water-quality variables: 11, including `NH3N`, `CODMn`, and `TP`

The annual windows are not temporally continuous enough to construct the project's default 90-history/30-forecast samples. The source basin holdout also differs from RiverLagNet's required chronological 70/15/15 evaluation protocol. The data must therefore remain isolated from default model-selection experiments unless a new, explicit experiment contract is added.

## Target quality checks

All 110 sample NPZ files were readable, finite, and shape-consistent with their graphs. The required target masks were fully observed in this processed bundle. Values below are source-normalized, not physical concentrations:

| Target | Observed fraction | Count | Min | Median | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| NH3N | 1.000 | 36,036 | -0.4137 | -0.2575 | 2.8647 | 17.2508 |
| CODMn | 1.000 | 36,036 | -2.1505 | 0.0004 | 4.0129 | 6.2842 |
| TP | 1.000 | 36,036 | -0.4060 | -0.1343 | 1.7752 | 77.3345 |

Negative values arise from source normalization and must not be interpreted as negative concentrations. The large normalized TP maximum is retained rather than silently clipped; any robust-loss or outlier treatment requires a separately recorded experiment.

## Engineering validation

`HydroWQChinaCatalog` loaded all 110 samples and all 10 graphs, reordered source channels to `NH3N, CODMn, TP`, and reported the 45-to-46/default-90-to-30 incompatibility. A no-gradient RiverLagNet forward pass on the first imported sample produced finite output with shape `[1, 46, 5, 3]`.

This is an interface smoke test only. It is not a training result and does not establish predictive performance.
