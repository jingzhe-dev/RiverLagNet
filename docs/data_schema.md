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

Prepared NPZ files persist the complete ordered `variable_names` array. Loading
resolves an immutable `FeatureRoles` object by name: targets must occupy channels
0–2, every later channel is exogenous, flow uses the fixed priority
`discharge`, `river_discharge`, `streamflow`, `flow`, then `dis24`, and rainfall
aliases are collected explicitly. If no recognized flow variable exists,
`flow_index` is `None`; another covariate is never silently substituted.

The graph-free v0.2 backbone exposes an intermediate tensor contract without
flattening public axes:

```text
history_states [B,T_in,N,D]
scale_states   [B,K,N,D]
horizon_states [B,30,N,D]
prediction     [B,30,N,3]
```

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

RiverLagNet v0.2 replaces the legacy development split with three expanding
rolling-origin folds: train/validation `[0,55%)/[55%,65%)`,
`[0,65%)/[65%,75%)`, and `[0,75%)/[75%,85%)`. The final `[85%,100%)`
interval is sealed until Session D. Each fold fits its scaler only on its own
training interval, and the v0.2 data module does not construct a test dataset.

## Versioned v0.2 manifest

`python -m RiverLagNet.cli.write_data_manifest` writes the versioned
`experiments/v0.2_data_manifest.json`. It records the complete NPZ SHA-256,
dimensions, feature and target roles, date bounds, per-variable observation
fractions, graph direction, and hashes of node order, edge order, and the
observation mask. It deliberately contains no raw observations. Any content
change therefore requires a new dataset version and a new set of paired runs.

## Prepared China real daily dataset

`RiverLagNet.data.prepare_china_real_daily` converts the continuous China source into a reviewable long Parquet table and a compact NPZ tensor artifact. Graph nodes are monitored HydroRIVERS segments; the retained mapping table connects each node to one or more source monitoring stations. If several stations map to one segment, the daily node value is the mean of values whose source flag says they were not imputed.

The generated artifact contains:

```text
values:                 [T,N,3] float32; zero only at masked positions
observed:               [T,N,3] bool
quality:                [T,N,3] fraction of mapped stations observed
static:                 [N,S]
edge_index:             [2,E] upstream -> downstream
edge_attr:              [E,A]
dates:                  [T] contiguous ISO dates
node_ids:               [N] monitored segment IDs
source_station_count:   [N]
```

`prepare_china_contracted_real_daily` is the preferred graph-construction path for the formal real-data pilot. It follows HydroRIVERS `NEXT_DOWN` through unmonitored reaches and connects each selected monitored reach to its first selected downstream reach. Edge attributes contain standardized accumulated path length, hop count, source/destination stream order, and the raw `travel_time_prior_days` as the final channel. The coverage profile and contracted-graph construction report are stored next to the ignored prepared dataset.

Source values flagged `*_is_imputed=1` become null in `observations.parquet`, zero plus `observed=false` in `dataset.npz`, and never enter scaler fitting, loss, or metrics. The first edge features retain their source z-scores; the final feature is an explicit `travel_time_prior_days` derived from restored river length and the manifest-declared velocity assumption. This prior is a routing regularizer, not evidence of causality or measured travel time.

## Synthetic data

`generate_synthetic_river_data` creates a deterministic directed tree with `source < destination`, edge travel-time priors, static station attributes, seasonal/autoregressive signals, lagged upstream influence, quality scores, and missing-observation masks. It is solely an engineering fixture, not an empirical dataset.

## Imported HydroWQ China sample bundles

`RiverLagNet.data.HydroWQChinaCatalog` reads the locally imported `hydrowq-china-multibasin-v0.1` manifest. It preserves graph edges exactly as stored (`edge_index[0]` upstream, `edge_index[1]` downstream) and selects/reorders water-quality channels to the fixed RiverLagNet order:

```text
source:      ..., CODMn, NH3N, TP, ...
RiverLagNet: NH3N, CODMn, TP
```

The returned sample tensors are:

```text
x:         [45, N, 3]
x_mask:    [45, N, 3] bool
x_quality: [45, N, 3] optional source quality codes
y:         [46, N, 3]
y_mask:    [46, N, 3] bool
```

These values are already source-normalized from training-basin statistics. The source manifest uses basin holdouts and one 91-day window per sampled year, rather than RiverLagNet's required chronological 70/15/15 split. Consequently, the catalog is currently an adapter for interface checks and controlled follow-up experiments, not a replacement for the default 90-history/30-forecast training dataset.
