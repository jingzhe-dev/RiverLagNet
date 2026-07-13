# HydroWQ China Data Import Design

## Objective

Reuse the small, processed China multi-basin assets from `D:/05.Paper/07.第七篇论文/Code/data` without copying the source repository's roughly 533 GiB of raw climate rasters, archives, caches, or logs into RiverLagNet.

## Source quality findings

The selected source is `processed/_manifests/sample-bundles-china-multibasin-v0.1.json` plus the assets it references under `processed/hydrowq-v0.1`. It contains 110 unique samples across 10 disjoint basin components, 10 valid directed graphs, 201 static features, 16 drivers, and 11 water-quality variables including `NH3N`, `CODMn`, and `TP`. All inspected sample arrays are finite, every checked sample digest matches the manifest, and the source train/validation/test basin sets do not overlap.

The source samples use a 45-day history and 46-day forecast. Their numeric arrays are already normalized using source-project training-basin statistics. Therefore they are suitable for graph/schema integration and engineering smoke tests, but not as the default RiverLagNet v0.1 `90 -> 30` training dataset and not for a second normalization pass. Their source basin-holdout split also does not replace RiverLagNet's required chronological 70/15/15 split.

## Import boundary

The importer copies only:

- the China multi-basin bundle manifest;
- its 110 referenced daily NPZ assets;
- its 10 referenced graph NPZ assets;
- its 10 referenced static NPZ assets;
- `sample-index.parquet`, `normalization-train.json`, and `build-report.json`.

Files are copied to ignored local storage at `data/processed/hydrowq-china-multibasin-v0.1`. Raw ERA5/CHIRPS files, public raw archives, HydroRIVERS caches, logs, and unrelated processed datasets are excluded by construction.

## Integrity and provenance

The importer reads the manifest rather than glob-copying directories. Every referenced asset must exist and match its manifest SHA-256 before copying. Each copied asset is hashed again at the destination. A local import receipt records source, destination, manifest digest, file count, byte count, and compatibility warnings. The receipt is stored with ignored data; durable quality findings and reproduction commands are committed under `docs/data`.

## Read-only catalog

`HydroWQChinaCatalog` exposes:

- `sample_ids` and source window metadata;
- `load_graph(basin_id) -> RiverGraph` using source `edge_index`, four edge attributes, and 201 static features;
- `load_sample(sample_id) -> HydroWQChinaSample`, selecting and reordering source channels to `NH3N`, `CODMn`, `TP`;
- `compatibility(input_window=90, output_window=30)` with explicit boolean status and reasons.

Loaded sample tensors retain `[time, node, target]` axes. The catalog never silently inverse-transforms or renormalizes source values. A caller must deliberately use source normalization metadata or rebuild samples from raw observations before formal training.

## Testing

Unit tests create a miniature manifest and NPZ assets in temporary directories. They verify manifest-driven copying, SHA rejection, exclusion of unrelated files, graph direction and shapes, target-channel reordering, and default-window incompatibility. After local import, an actual source sample and graph are loaded and passed through a small RiverLagNet forward smoke test. The full repository test suite remains required before commit and push.
