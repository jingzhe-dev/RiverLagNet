"""Prepare and load leakage-safe real daily China water-quality data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from .feature_roles import resolve_feature_roles
from .schema import TARGET_NAMES, RiverGraph, TimeSeriesData


DATASET_VERSION = "china-real-daily-v0.1"
DEFAULT_TRAVEL_SPEED_KM_PER_DAY = 30.0
EXTENDED_DYNAMIC_COVARIATES = (
    "Temp",
    "pH",
    "DO",
    "EC",
    "Tur",
    "TN",
    "Chl_a",
    "Algae_Density",
    "dewpoint_temperature_2m",
    "potential_evaporation",
    "snow_depth_water_equivalent",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_pressure",
    "temperature_2m",
    "total_precipitation",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "rowe",
    "dis24",
)


@dataclass(frozen=True)
class RealDailyPreparationSummary:
    """Auditable summary of one prepared real-data artifact."""

    dataset_path: Path
    observations_path: Path
    edges_path: Path
    station_mapping_path: Path
    manifest_path: Path
    start_date: str
    end_date: str
    num_days: int
    num_nodes: int
    num_edges: int
    num_source_stations: int
    observed_counts: dict[str, int]
    observed_rates: dict[str, float]
    mapping_quality_counts: dict[str, int]


@dataclass(frozen=True)
class _CombinedGraph:
    node_ids: np.ndarray
    component_ids: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_attr_names: np.ndarray
    static: np.ndarray
    static_names: np.ndarray


def prepare_china_real_daily(
    dynamic_path: str | Path,
    flags_path: str | Path,
    mapping_path: str | Path,
    graph_root: str | Path,
    output_dir: str | Path,
    *,
    edge_normalization_path: str | Path | None = None,
    travel_speed_km_per_day: float = DEFAULT_TRAVEL_SPEED_KM_PER_DAY,
    hash_sources: bool = True,
    dataset_id: str = DATASET_VERSION,
    graph_construction: dict[str, Any] | None = None,
    dynamic_covariates: Sequence[str] = (),
) -> RealDailyPreparationSummary:
    """Build a chronological daily panel using original observations only.

    Values marked ``*_is_imputed=1`` are removed before aggregation. The compact
    tensor artifact stores zero at those missing positions together with an
    explicit false observation mask; the reviewable Parquet table retains nulls.
    """
    dynamic_path = Path(dynamic_path).expanduser().resolve()
    flags_path = Path(flags_path).expanduser().resolve()
    mapping_path = Path(mapping_path).expanduser().resolve()
    graph_root = Path(graph_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for source in (dynamic_path, flags_path, mapping_path):
        if not source.is_file():
            raise FileNotFoundError(f"required real-data source not found: {source}")
    if not graph_root.is_dir():
        raise FileNotFoundError(f"graph root not found: {graph_root}")
    if travel_speed_km_per_day <= 0:
        raise ValueError("travel_speed_km_per_day must be positive")
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")

    normalization_path = _resolve_edge_normalization_path(
        graph_root, edge_normalization_path
    )
    graph = _load_combined_graph(
        graph_root,
        normalization_path=normalization_path,
        travel_speed_km_per_day=travel_speed_km_per_day,
    )
    mapping = _load_selected_mapping(mapping_path, graph.node_ids)
    source_station_ids = (
        mapping.get_column("source_station_id").unique().sort().to_list()
    )
    covariates = tuple(dict.fromkeys(str(name) for name in dynamic_covariates))
    if any(name in TARGET_NAMES for name in covariates):
        raise ValueError("dynamic_covariates must not repeat target names")
    variable_names = (*TARGET_NAMES, *covariates)
    panel = _build_observed_panel(
        dynamic_path, flags_path, mapping, source_station_ids, variable_names
    )

    dates = panel.get_column("date").unique().sort()
    _validate_daily_dates(dates)
    node_order = pl.DataFrame(
        {
            "station_id": graph.node_ids.tolist(),
            "node_index": np.arange(graph.node_ids.size, dtype=np.int64),
        }
    )
    panel = panel.join(node_order, on="station_id", how="inner").sort(
        ["date", "node_index"]
    )
    expected_rows = len(dates) * graph.node_ids.size
    if panel.height != expected_rows:
        raise ValueError(
            f"daily panel is incomplete: found {panel.height} rows, expected {expected_rows}"
        )

    num_days = len(dates)
    num_nodes = graph.node_ids.size
    values = np.stack(
        [panel.get_column(name).fill_null(0.0).to_numpy() for name in variable_names],
        axis=-1,
    ).reshape(num_days, num_nodes, len(variable_names)).astype(np.float32)
    observed = np.stack(
        [panel.get_column(f"{name}_observed").to_numpy() for name in variable_names],
        axis=-1,
    ).reshape(num_days, num_nodes, len(variable_names)).astype(bool)
    quality = np.stack(
        [panel.get_column(f"{name}_quality").to_numpy() for name in variable_names],
        axis=-1,
    ).reshape(num_days, num_nodes, len(variable_names)).astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(quality).all():
        raise ValueError("prepared real-data tensors contain non-finite values")

    station_counts = (
        mapping.group_by("station_id")
        .agg(pl.len().alias("source_station_count"))
        .join(node_order, on="station_id", how="inner")
        .sort("node_index")
    )
    if station_counts.height != num_nodes:
        raise ValueError("every graph node must have at least one mapped source station")
    source_station_count = station_counts.get_column("source_station_count").to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.npz"
    observations_path = output_dir / "observations.parquet"
    edges_path = output_dir / "edges.parquet"
    station_mapping_path = output_dir / "station_mapping.parquet"
    manifest_path = output_dir / "manifest.json"

    date_strings = np.asarray([str(value) for value in dates.to_list()], dtype="U10")
    with dataset_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            values=values,
            observed=observed,
            quality=quality,
            static=graph.static.astype(np.float32),
            edge_index=graph.edge_index.astype(np.int64),
            edge_attr=graph.edge_attr.astype(np.float32),
            dates=date_strings,
            node_ids=graph.node_ids.astype("U"),
            component_ids=graph.component_ids.astype("U"),
            target_names=np.asarray(TARGET_NAMES, dtype="U"),
            variable_names=np.asarray(variable_names, dtype="U"),
            static_names=graph.static_names.astype("U"),
            edge_attr_names=graph.edge_attr_names.astype("U"),
            source_station_count=source_station_count.astype(np.int64),
        )

    review_columns = ["date", "station_id"]
    for name in variable_names:
        review_columns.extend([name, f"{name}_observed", f"{name}_quality"])
    panel.select(review_columns).write_parquet(observations_path)
    mapping.write_parquet(station_mapping_path)
    _edge_frame(graph).write_parquet(edges_path)

    observed_counts = {
        name: int(observed[..., index].sum())
        for index, name in enumerate(TARGET_NAMES)
    }
    denominator = float(num_days * num_nodes)
    observed_rates = {name: count / denominator for name, count in observed_counts.items()}
    mapping_quality_counts = {
        str(row["mapping_quality_flag"]): int(row["len"])
        for row in mapping.group_by("mapping_quality_flag").len().to_dicts()
    }
    source_hashes = {
        str(path): _sha256(path) if hash_sources else None
        for path in (dynamic_path, flags_path, mapping_path)
    }
    artifact_hashes = {
        path.name: _sha256(path)
        for path in (dataset_path, observations_path, edges_path, station_mapping_path)
    }
    split_train_end = int(num_days * 0.70)
    split_val_end = int(num_days * 0.85)
    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_hashes,
        "artifact_sha256": artifact_hashes,
        "graph_root": str(graph_root),
        "edge_normalization_path": str(normalization_path) if normalization_path else None,
        "target_names": list(TARGET_NAMES),
        "dynamic_input_names": list(variable_names),
        "dynamic_input_observed_rates": {
            name: float(observed[..., index].mean())
            for index, name in enumerate(variable_names)
        },
        "date_range": [date_strings[0], date_strings[-1]],
        "shape": list(values.shape),
        "num_edges": int(graph.edge_index.shape[1]),
        "num_source_stations": len(source_station_ids),
        "observed_counts": observed_counts,
        "observed_rates": observed_rates,
        "imputed_values_used_as_observations": False,
        "aggregation": "mean of non-imputed source-station observations per segment-day",
        "node_definition": "monitored HydroRIVERS segment",
        "edge_direction": "upstream_to_downstream",
        "travel_time_prior": {
            "method": "length_km / assumed_travel_speed_km_per_day",
            "assumed_travel_speed_km_per_day": travel_speed_km_per_day,
            "causal_interpretation": False,
        },
        "chronological_split": {
            "ratios": [0.70, 0.15, 0.15],
            "train": [date_strings[0], date_strings[split_train_end - 1]],
            "validation": [date_strings[split_train_end], date_strings[split_val_end - 1]],
            "test": [date_strings[split_val_end], date_strings[-1]],
        },
    }
    if graph_construction is not None:
        manifest["graph_construction"] = graph_construction
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return RealDailyPreparationSummary(
        dataset_path=dataset_path,
        observations_path=observations_path,
        edges_path=edges_path,
        station_mapping_path=station_mapping_path,
        manifest_path=manifest_path,
        start_date=date_strings[0],
        end_date=date_strings[-1],
        num_days=num_days,
        num_nodes=num_nodes,
        num_edges=int(graph.edge_index.shape[1]),
        num_source_stations=len(source_station_ids),
        observed_counts=observed_counts,
        observed_rates=observed_rates,
        mapping_quality_counts=mapping_quality_counts,
    )


def load_real_daily_dataset(path: str | Path) -> TimeSeriesData:
    """Load a prepared real daily NPZ into the shared tensor contract."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prepared real-data artifact not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "values",
            "observed",
            "quality",
            "static",
            "edge_index",
            "edge_attr",
            "dates",
            "node_ids",
            "target_names",
            "variable_names",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prepared real-data artifact lacks {sorted(missing)}")
        target_names = tuple(archive["target_names"].astype(str).tolist())
        if target_names != TARGET_NAMES:
            raise ValueError(
                f"target order must be {TARGET_NAMES}, found {target_names}"
            )
        variable_names = tuple(archive["variable_names"].astype(str).tolist())
        resolve_feature_roles(variable_names)
        dates = archive["dates"].astype("datetime64[D]")
        if dates.size < 2 or not np.all(np.diff(dates).astype(int) == 1):
            raise ValueError("prepared dates must be a contiguous daily sequence")
        values = np.asarray(archive["values"], dtype=np.float32)
        observed = np.asarray(archive["observed"], dtype=bool)
        quality = np.asarray(archive["quality"], dtype=np.float32)
        static = np.asarray(archive["static"], dtype=np.float32)
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(archive["edge_attr"], dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(quality).all():
        raise ValueError("prepared real-data values and quality must be finite")
    data = TimeSeriesData(
        values=torch.from_numpy(values),
        observed=torch.from_numpy(observed),
        quality=torch.from_numpy(quality),
        graph=RiverGraph(
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            static=torch.from_numpy(static).float(),
        ),
        variable_names=variable_names,
    )
    data.validate()
    return data


def summary_as_dict(summary: RealDailyPreparationSummary) -> dict[str, Any]:
    """Return a JSON-safe representation of a preparation summary."""
    payload = asdict(summary)
    for name in (
        "dataset_path",
        "observations_path",
        "edges_path",
        "station_mapping_path",
        "manifest_path",
    ):
        payload[name] = str(payload[name])
    return payload


def _load_combined_graph(
    graph_root: Path,
    *,
    normalization_path: Path | None,
    travel_speed_km_per_day: float,
) -> _CombinedGraph:
    graph_dirs = sorted(path for path in graph_root.iterdir() if (path / "graph.npz").is_file())
    if not graph_dirs:
        raise ValueError(f"no graph.npz assets found under {graph_root}")
    node_ids: list[np.ndarray] = []
    component_ids: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    edge_attrs: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    edge_names: tuple[str, ...] | None = None
    static_names: tuple[str, ...] | None = None
    offset = 0
    for directory in graph_dirs:
        with np.load(directory / "graph.npz", allow_pickle=False) as archive:
            segments = archive["segment_id"].astype(str)
            edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(archive["edge_attr"], dtype=np.float32)
            names = tuple(archive["edge_attr_names"].astype(str).tolist())
            mask = np.asarray(archive["edge_attr_mask"], dtype=bool)
        with np.load(directory / "static.npz", allow_pickle=False) as archive:
            static = np.asarray(archive["static_node"], dtype=np.float32)
            names_static = tuple(archive["static_names"].astype(str).tolist())
            static_mask = np.asarray(archive["static_mask"], dtype=bool)
            static_segments = archive["segment_id"].astype(str)
        if not np.array_equal(segments, static_segments):
            raise ValueError(f"graph/static node order differs in {directory}")
        if not mask.all() or not static_mask.all():
            raise ValueError(f"graph/static assets contain missing values in {directory}")
        if edge_names is not None and names != edge_names:
            raise ValueError("graph components use inconsistent edge attributes")
        if static_names is not None and names_static != static_names:
            raise ValueError("graph components use inconsistent static attributes")
        edge_names = names
        static_names = names_static
        node_ids.append(segments)
        component_name = directory.name.rsplit("-v", 1)[0]
        component_ids.append(np.full(segments.size, component_name, dtype="U64"))
        edges.append(edge_index + offset)
        edge_attrs.append(edge_attr)
        statics.append(static)
        offset += segments.size
    combined_nodes = np.concatenate(node_ids)
    if np.unique(combined_nodes).size != combined_nodes.size:
        raise ValueError("graph components contain duplicate segment IDs")
    combined_edge_attr = np.concatenate(edge_attrs, axis=0)
    assert edge_names is not None and static_names is not None
    combined_edge_attr, output_edge_names = _restore_travel_time_prior(
        combined_edge_attr,
        edge_names,
        normalization_path,
        travel_speed_km_per_day,
    )
    return _CombinedGraph(
        node_ids=combined_nodes,
        component_ids=np.concatenate(component_ids),
        edge_index=np.concatenate(edges, axis=1),
        edge_attr=combined_edge_attr,
        edge_attr_names=np.asarray(output_edge_names, dtype="U64"),
        static=np.concatenate(statics, axis=0),
        static_names=np.asarray(static_names, dtype="U128"),
    )


def _restore_travel_time_prior(
    edge_attr: np.ndarray,
    edge_names: tuple[str, ...],
    normalization_path: Path | None,
    travel_speed_km_per_day: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if edge_names[-1] == "travel_time_prior_days":
        return edge_attr, edge_names
    if edge_names[-1] != "travel_time_proxy":
        raise ValueError("last edge attribute must be a travel-time field")
    if normalization_path is None or not normalization_path.is_file():
        raise FileNotFoundError(
            "edge normalization metadata is required to recover raw river length"
        )
    payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    edge_stats = payload.get("edge", {})
    names = tuple(edge_stats.get("feature_names", []))
    if names != edge_names:
        raise ValueError("edge normalization feature order does not match graph assets")
    mean = np.asarray(edge_stats["mean"], dtype=np.float32)
    scale = np.asarray(edge_stats["std"], dtype=np.float32)
    length_index = edge_names.index("length_km")
    raw_length_km = edge_attr[:, length_index] * scale[length_index] + mean[length_index]
    restored = edge_attr.copy()
    restored[:, -1] = np.maximum(raw_length_km / travel_speed_km_per_day, 0.0)
    output_names = tuple(
        f"{name}_z" if index < len(edge_names) - 1 else "travel_time_prior_days"
        for index, name in enumerate(edge_names)
    )
    return restored, output_names


def _resolve_edge_normalization_path(
    graph_root: Path, explicit: str | Path | None
) -> Path | None:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    inferred = (
        graph_root.parents[1]
        / "datasets"
        / "china-multibasin-v0.1"
        / "normalization-train.json"
    )
    return inferred if inferred.is_file() else None


def _load_selected_mapping(mapping_path: Path, node_ids: np.ndarray) -> pl.DataFrame:
    required = {
        "station_id",
        "segment_id",
        "mapped_flag",
        "distance_km",
        "mapping_quality_flag",
    }
    mapping = pl.read_csv(mapping_path)
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"station mapping lacks {sorted(missing)}")
    mapping = (
        mapping.with_columns(
            pl.col("station_id").cast(pl.Int64),
            pl.col("segment_id").cast(pl.String).str.replace(r"\.0$", ""),
            pl.col("mapped_flag").cast(pl.Boolean),
        )
        .filter(pl.col("mapped_flag") & pl.col("segment_id").is_in(node_ids.tolist()))
        .rename({"segment_id": "station_id", "station_id": "source_station_id"})
        .sort(["station_id", "source_station_id"])
    )
    covered = set(mapping.get_column("station_id").to_list())
    missing_nodes = sorted(set(node_ids.tolist()).difference(covered))
    if missing_nodes:
        raise ValueError(f"graph nodes lack mapped observations: {missing_nodes}")
    return mapping


def _build_observed_panel(
    dynamic_path: Path,
    flags_path: Path,
    mapping: pl.DataFrame,
    station_ids: list[int],
    variable_names: Sequence[str] = TARGET_NAMES,
) -> pl.DataFrame:
    value_scan = pl.scan_csv(dynamic_path)
    value_columns = set(value_scan.collect_schema().names())
    missing_values = set(variable_names).difference(value_columns)
    if missing_values:
        raise ValueError(f"dynamic source lacks {sorted(missing_values)}")
    values = (
        value_scan
        .select(["id", "time", *variable_names])
        .filter(pl.col("id").is_in(station_ids))
        .with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("time").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
        )
        .drop("time")
    )
    flag_scan = pl.scan_csv(flags_path)
    flag_columns = set(flag_scan.collect_schema().names())
    flagged_variables = [
        name for name in variable_names if f"{name}_is_imputed" in flag_columns
    ]
    flag_names = [f"{name}_is_imputed" for name in flagged_variables]
    flags = (
        flag_scan
        .select(["id", "time", *flag_names])
        .filter(pl.col("id").is_in(station_ids))
        .with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("time").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
        )
        .drop("time")
    )
    station_map = mapping.select(
        pl.col("source_station_id").alias("id"), pl.col("station_id")
    )
    joined = values.join(flags, on=["id", "date"], how="inner").join(
        station_map.lazy(), on="id", how="inner"
    )
    aggregations: list[pl.Expr] = []
    for name in variable_names:
        is_observed = pl.col(name).is_not_null() & pl.col(name).is_finite()
        if name in flagged_variables:
            is_observed = is_observed & (pl.col(f"{name}_is_imputed") == 0)
        aggregations.extend(
            [
                pl.when(is_observed).then(pl.col(name)).mean().alias(name),
                is_observed.any().alias(f"{name}_observed"),
                is_observed.mean().cast(pl.Float32).alias(f"{name}_quality"),
            ]
        )
    panel = joined.group_by(["date", "station_id"]).agg(aggregations).collect(
        engine="streaming"
    )
    if panel.is_empty():
        raise ValueError("selected graph has no matching daily water-quality rows")
    return panel


def _validate_daily_dates(dates: pl.Series) -> None:
    as_numpy = dates.to_numpy().astype("datetime64[D]")
    if as_numpy.size < 2 or not np.all(np.diff(as_numpy).astype(int) == 1):
        raise ValueError("source dates must form one contiguous daily range")


def _edge_frame(graph: _CombinedGraph) -> pl.DataFrame:
    source = graph.edge_index[0]
    destination = graph.edge_index[1]
    columns: dict[str, Any] = {
        "src_station_id": graph.node_ids[source],
        "dst_station_id": graph.node_ids[destination],
    }
    for index, name in enumerate(graph.edge_attr_names.tolist()):
        columns[name] = graph.edge_attr[:, index]
    return pl.DataFrame(columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
