# Data schema

## Dynamic observations

Real observations should use a long Parquet table with one row per date and station. Required columns are:

```text
date
station_id
NH3N
CODMn
TP
NH3N_observed
CODMn_observed
TP_observed
```

Optional dynamic variables follow the same `<variable>`, `<variable>_observed`, and `<variable>_quality` convention. Values that fill missing positions are never treated as observations; every model receives the mask alongside filled values.

The in-memory schema is:

```text
values:   [T, N, V]
observed: [T, N, V] bool
quality:  [T, N, V] optional
```

The first three variable channels are always `NH3N`, `CODMn`, and `TP` in that order.

## River graph

The edge table uses:

```text
src_station_id              # upstream
dst_station_id              # downstream
distance_km
slope
travel_time_prior_days
```

Tensor form is `edge_index [2,E]`, `edge_attr [E,A]`, and `static [N,S]`. Row 0 of `edge_index` is the source; row 1 is the destination.

## Windows and splits

Batches retain all public axes:

```text
x:             [B, T_in, N, V]
x_mask:        [B, T_in, N, V]
x_quality:     [B, T_in, N, V]
time_features: [B, T_in, 4]
y:             [B, T_out, N, 3]
y_mask:        [B, T_out, N, 3]
```

Forecast target timestamps define the chronological 70/15/15 splits. Every target window lies wholly inside one split and all inputs precede its first target. Validation/test inputs may use earlier history, but target timestamps never overlap across splits. Feature means and standard deviations are fitted from observed values before the training boundary only.

## Synthetic data

`generate_synthetic_river_data` creates a deterministic directed tree with `source < destination`, edge travel-time priors, static station attributes, seasonal/autoregressive signals, lagged upstream influence, quality scores, and missing-observation masks. It is solely an engineering fixture, not an empirical dataset.
