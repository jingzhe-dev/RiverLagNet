# Caravan-Qual readiness audit for RiverLagNet

Audit date: 2026-07-13  
Audience: technical  
Decision: do not use Caravan-Qual as the default real-data training set yet; proceed with multi-seed and ablation experiments while building a separate global sparse-observation adapter.

## Technical summary

Caravan-Qual is the strongest real-data candidate in the available seventh-paper workspace, but it does not satisfy the current China-focused, three-target daily training contract. The global archive contains stations with all three targets and long calendar spans, yet observations are mostly monthly or sparser and same-station/day duplicates require quality-aware aggregation. China has no usable NH3N observations in the linkage inventory, so a China `NH3N, CODMn, TP` model cannot be trained from this source without changing the target definition or adding another source.

The immediate modeling decision is therefore:

1. keep the verified synthetic 90-to-30 benchmark as the mechanism-development dataset;
2. run five paired seeds and all six required graph/lag ablations;
3. treat Caravan-Qual as a separate global sparse-observation experiment after an explicit ingestion, duplicate-resolution, graph-construction, and missingness design.

## Source inventory and grain

Primary sources inspected without extracting the archives:

- `Caravan-Qual_linkages.parquet`: 151,814 unique WQMS stations and 716 metadata/coverage columns;
- `wqms-csv.zip`: 100 parameter CSV files, 4.366 GiB uncompressed;
- `Caravan-Qual_lite.zarr.zip`: 16,034 files, 8.239 GiB uncompressed, daily axis length 16,710;
- `wqms_site_info.csv`: station coordinates and HydroRIVERS linkage;
- `HydroRIVERS_v10_shp.zip`: global river reach network needed to construct direction;
- the previously imported China multi-basin bundle: 110 prebuilt 45-to-46 samples.

The raw WQMS grain is station × parameter × observation date. It is not a dense daily table. Multiple measurements can occur for the same station, parameter, and date.

## Target availability

| Scope | Stations | NH3N observed | CODMn observed | TP observed | Stations with all 3 targets | Each target ≥30 observations | Each target ≥120 observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global linkage inventory | 151,814 | 106,200 | 105,733 | 2,389,252 | 286 quality-observed | 45 | 0 |
| People's Republic of China | 480 | 0 | 27,333 | 196 | 0 | 0 | 0 |

The raw CSV station-ID intersection is 374 because it includes flagged, imputed, or otherwise non-`observed` records that the linkage inventory excludes from its quality-observed counts. Seventy-eight global stations have at least 30 unique dates per target and a joint calendar span of at least 120 days. Their best-covered examples have approximately 95–108 unique dates per target across roughly eleven years, confirming monthly rather than daily target density.

The largest level-12 HydroBASINS group with all three targets contains five Italian stations. Within that group, quality-observed per-station counts range from 9–59 for NH3N, 6–35 for CODMn, and 20–65 for TP over multi-year spans. This is enough for a sparse masked pilot, not for claiming dense daily supervision.

## Raw parameter quality

| Target file | Rows | Stations | Date range | Duplicate rows on station/date | Duplicate station/date keys | Missing values | Negative values | Imputed rows |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| NH3N | 157,457 | 4,273 | 1982-01-11 to 2025-07-23 | 12,224 | 5,708 | 856 | 0 | 45,496 |
| CODMn | 124,498 | 4,653 | 1971-08-31 to 2023-12-29 | 24,893 | 8,171 | 0 | 0 | 2,274 |
| TP | 3,005,281 | 67,005 | 1900-09-06 to 2025-11-06 | 370,474 | 142,702 | 611 | 47 | 156,979 |

All three files declare `mg/l`, so no unit conversion is needed within these target files. The duplicate rates, imputation flags, detection-limit flags, and 47 negative TP values mean raw rows cannot be pivoted directly into model tensors.

## Findings and risk assessment

### Critical — China target incompleteness

No Chinese station has an NH3N observation in the inspected Caravan-Qual linkage inventory. This breaks the fixed three-target contract and makes a China three-target loss undefined for NH3N. Confidence: high.

Remediation: obtain NH3N from a separate compatible China source or scope Caravan-Qual to an explicitly global experiment. Do not silently substitute NH4N or TAN for NH3N.

### High — target supervision is sparse in time

Even the best three-target stations have roughly monthly observations over long spans. Daily reindexing is possible because RiverLagNet is mask-aware, but most forecast positions would be unobserved and many rolling windows would contribute little or no target loss. Confidence: high.

Remediation: construct windows only when the future 30-day block contains a minimum observed-target count, report mask coverage by split and target, and compare against monthly-scale alternatives before treating the daily experiment as decision-ready.

### High — duplicate station/day measurements need a declared rule

The three target files contain 407,591 rows participating in duplicate station/date keys. Choosing the first row would be order-dependent and could mix detection-limit imputations with observed values. Confidence: high.

Remediation: preserve raw flags, prefer valid unflagged observations, aggregate multiple valid same-day values with a documented robust statistic, and unit-test the resolution hierarchy.

### Medium — validity and quality flags affect usable counts

NH3N contains 856 missing values and 45,496 imputed rows; TP contains 47 negative values, 611 missing values, and 156,979 imputed rows. Raw station intersections exceed quality-observed intersections. Confidence: high.

Remediation: keep value, observed mask, detection-limit metadata, imputation method, and quality code separate. Never convert rejected values to real zero.

### Medium — graph construction is available but not yet packaged

Every selected station needs a HydroRIVERS reach, directed downstream topology, edge attributes, and a connected-component assignment. The source workspace contains the global HydroRIVERS archive and WQMS reach identifiers, but RiverLagNet does not yet have a tested Caravan graph adapter. Confidence: high.

Remediation: build and validate the adapter as a separate feature after the mechanism robustness suite. Test direction, self-loops, orphan stations, connected components, and graph/data station-order agreement.

## Automated gates for a future real-data pilot

A Caravan-based training configuration should remain disabled until all gates pass:

- exact target names are `NH3N, CODMn, TP` with no proxy substitution;
- station/parameter/date is unique after a tested quality-aware aggregation;
- all values have explicit observed and quality masks;
- no negative physical target value is marked observed;
- every node maps to one graph station order and a valid directed reach;
- each chronological split has nonzero observations for every target;
- every retained 90-to-30 window has a declared minimum future-target coverage;
- normalization is fitted only on observed training-period values;
- no station, timestamp, or target row crosses split boundaries through preprocessing.

## Assumptions and open boundary

This audit assesses the files currently present in the seventh-paper workspace. It does not claim that no other China NH3N source exists. The conclusion is specifically that the inspected Caravan-Qual assets are not sufficient for the current China three-target contract. The global data remain promising for a later sparse masked pilot.
