"""Build monitored-reach graphs contracted across unmonitored river segments."""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import polars as pl

from .schema import TARGET_NAMES


HYDRORIVERS_COLUMNS = (
    "HYRIV_ID",
    "NEXT_DOWN",
    "HYBAS_L12",
    "LENGTH_KM",
    "DIST_DN_KM",
    "CATCH_SKM",
    "UPLAND_SKM",
    "ORD_STRA",
)
STATIC_NAMES = (
    "log1p_length_km_z",
    "log1p_distance_downstream_km_z",
    "log1p_catchment_area_km2_z",
    "log1p_upstream_area_km2_z",
    "stream_order_z",
)
EDGE_ATTR_NAMES = (
    "log1p_path_length_km_z",
    "log1p_hop_count_z",
    "source_stream_order_z",
    "destination_stream_order_z",
    "travel_time_prior_days",
)


@dataclass(frozen=True)
class DbfField:
    """Minimal DBF field descriptor used by the HydroRIVERS reader."""

    name: str
    field_type: str
    offset: int
    length: int
    decimal_count: int


@dataclass(frozen=True)
class ContractedEdge:
    """One upstream-to-downstream edge spanning a HydroRIVERS path."""

    source: str
    destination: str
    path_length_km: float
    hop_count: int


@dataclass(frozen=True)
class ContractedGraphAssets:
    """Selected graph arrays plus an auditable construction report."""

    node_ids: np.ndarray
    component_ids: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    static: np.ndarray
    report: dict[str, Any]


def load_mapped_station_table(mapping_path: str | Path) -> pl.DataFrame:
    """Load valid station-to-HydroRIVERS mappings with normalized IDs."""
    mapping_path = Path(mapping_path).expanduser().resolve()
    mapping = pl.read_csv(mapping_path)
    required = {
        "station_id",
        "segment_id",
        "mapped_flag",
        "mapping_quality_flag",
    }
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"station mapping lacks {sorted(missing)}")
    return (
        mapping.with_columns(
            pl.col("station_id").cast(pl.Int64),
            pl.col("segment_id").cast(pl.String).str.replace(r"\.0$", ""),
            pl.col("mapped_flag").cast(pl.Boolean),
        )
        .filter(pl.col("mapped_flag"))
        .sort(["segment_id", "station_id"])
    )


def profile_segment_observation_coverage(
    flags_path: str | Path,
    mapping_path: str | Path,
) -> pl.DataFrame:
    """Return per-segment original-observation coverage for all three targets."""
    flags_path = Path(flags_path).expanduser().resolve()
    if not flags_path.is_file():
        raise FileNotFoundError(f"imputation flags not found: {flags_path}")
    mapping = load_mapped_station_table(mapping_path)
    station_ids = mapping.get_column("station_id").unique().sort().to_list()
    flag_names = [f"{target}_is_imputed" for target in TARGET_NAMES]
    schema_names = set(pl.scan_csv(flags_path).collect_schema().names())
    required = {"id", "time", *flag_names}
    missing = required.difference(schema_names)
    if missing:
        raise ValueError(f"imputation flags lack {sorted(missing)}")

    station_map = mapping.select("station_id", "segment_id").lazy()
    segment_days = (
        pl.scan_csv(flags_path)
        .select("id", "time", *flag_names)
        .filter(pl.col("id").is_in(station_ids))
        .with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("time").str.slice(0, 10).alias("date"),
        )
        .join(station_map, left_on="id", right_on="station_id", how="inner")
        .group_by("segment_id", "date")
        .agg(
            *[
                (pl.col(flag_name) == 0)
                .any()
                .alias(flag_name.replace("_is_imputed", "_observed"))
                for flag_name in flag_names
            ]
        )
        .collect(engine="streaming")
    )
    if segment_days.is_empty():
        raise ValueError("mapped stations have no imputation-flag rows")
    num_days = segment_days.get_column("date").n_unique()
    coverage = segment_days.group_by("segment_id").agg(
        pl.len().alias("available_day_count"),
        *[
            (
                pl.col(f"{target}_observed").sum().cast(pl.Float64)
                / pl.lit(float(num_days))
            ).alias(f"{target}_coverage")
            for target in TARGET_NAMES
        ],
    )
    mapping_profile = mapping.group_by("segment_id").agg(
        pl.len().alias("source_station_count"),
        pl.col("mapping_quality_flag")
        .str.contains("review")
        .any()
        .alias("mapping_review"),
    )
    fill_columns = ["available_day_count", *[f"{target}_coverage" for target in TARGET_NAMES]]
    return (
        mapping_profile.join(coverage, on="segment_id", how="left")
        .with_columns(pl.col(fill_columns).fill_null(0))
        .with_columns(
            pl.min_horizontal([f"{target}_coverage" for target in TARGET_NAMES]).alias(
                "min_target_coverage"
            ),
            pl.lit(num_days).cast(pl.Int64).alias("expected_day_count"),
        )
        .sort("segment_id")
    )


def read_hydrorivers_attributes(path: str | Path) -> pl.DataFrame:
    """Read the required HydroRIVERS DBF attributes directly from its ZIP."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HydroRIVERS ZIP not found: {path}")
    with zipfile.ZipFile(path) as archive:
        dbf_names = sorted(
            name for name in archive.namelist() if name.lower().endswith(".dbf")
        )
        if not dbf_names:
            raise ValueError("HydroRIVERS ZIP contains no DBF table")
        preferred = next(
            (
                name
                for name in dbf_names
                if Path(name).name.lower() == "hydrorivers_v10_as.dbf"
            ),
            dbf_names[0],
        )
        with archive.open(preferred) as handle:
            raw = _read_dbf_columns(handle, HYDRORIVERS_COLUMNS)

    frame = pl.DataFrame(raw)
    return (
        frame.select(
            _normalized_id("HYRIV_ID").alias("segment_id"),
            _normalized_optional_id("NEXT_DOWN").alias("downstream_segment_id"),
            _normalized_id("HYBAS_L12").alias("basin_id"),
            pl.col("LENGTH_KM").cast(pl.Float64, strict=False).alias("length_km"),
            pl.col("DIST_DN_KM")
            .cast(pl.Float64, strict=False)
            .alias("distance_downstream_km"),
            pl.col("CATCH_SKM")
            .cast(pl.Float64, strict=False)
            .alias("catchment_area_km2"),
            pl.col("UPLAND_SKM")
            .cast(pl.Float64, strict=False)
            .alias("upstream_area_km2"),
            pl.col("ORD_STRA").cast(pl.Float64, strict=False).alias("stream_order"),
        )
        .with_columns(
            pl.when(pl.col(name) >= 0).then(pl.col(name)).otherwise(None).alias(name)
            for name in (
                "length_km",
                "distance_downstream_km",
                "catchment_area_km2",
                "upstream_area_km2",
                "stream_order",
            )
        )
    )


def contract_monitored_paths(
    segments: pl.DataFrame,
    selected_node_ids: set[str],
) -> list[ContractedEdge]:
    """Connect each node to its first selected downstream node."""
    required = {"segment_id", "downstream_segment_id", "length_km"}
    missing = required.difference(segments.columns)
    if missing:
        raise ValueError(f"HydroRIVERS attributes lack {sorted(missing)}")
    segment_ids = segments.get_column("segment_id").to_list()
    downstream_ids = segments.get_column("downstream_segment_id").to_list()
    lengths = segments.get_column("length_km").to_list()
    next_by_id = dict(zip(segment_ids, downstream_ids, strict=True))
    length_by_id = dict(zip(segment_ids, lengths, strict=True))
    unknown = selected_node_ids.difference(next_by_id)
    if unknown:
        raise ValueError(f"selected mappings contain unknown HydroRIVERS IDs: {sorted(unknown)[:5]}")

    edges: list[ContractedEdge] = []
    for source in sorted(selected_node_ids, key=_id_sort_key):
        current = source
        visited = {source}
        path_length_km = 0.0
        hop_count = 0
        while True:
            destination = next_by_id.get(current)
            if destination is None:
                break
            if destination in visited:
                raise ValueError(f"HydroRIVERS path contains a cycle at {destination}")
            current_length = length_by_id.get(current)
            if current_length is None or not math.isfinite(float(current_length)):
                raise ValueError(f"HydroRIVERS path has missing length at {current}")
            path_length_km += max(float(current_length), 0.0)
            hop_count += 1
            if destination in selected_node_ids:
                edges.append(
                    ContractedEdge(
                        source=source,
                        destination=destination,
                        path_length_km=path_length_km,
                        hop_count=hop_count,
                    )
                )
                break
            visited.add(destination)
            current = destination
    if len({(edge.source, edge.destination) for edge in edges}) != len(edges):
        raise ValueError("contracted graph contains duplicate directed edges")
    return edges


def build_contracted_graph(
    segments: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    min_target_coverage: float = 0.90,
    min_component_nodes: int = 3,
    max_component_nodes: int = 256,
    component_limit: int = 1,
    travel_speed_km_per_day: float = 30.0,
    max_lag_days: int = 14,
) -> ContractedGraphAssets:
    """Select trainable connected components from the contracted monitored graph."""
    if not 0 <= min_target_coverage <= 1:
        raise ValueError("min_target_coverage must lie in [0,1]")
    if min_component_nodes <= 0 or max_component_nodes < min_component_nodes:
        raise ValueError("component node limits are invalid")
    if component_limit <= 0:
        raise ValueError("component_limit must be positive")
    if travel_speed_km_per_day <= 0 or max_lag_days <= 0:
        raise ValueError("travel speed and max lag must be positive")
    required_coverage = {
        "segment_id",
        "source_station_count",
        "mapping_review",
        "min_target_coverage",
    }
    missing = required_coverage.difference(coverage.columns)
    if missing:
        raise ValueError(f"coverage profile lacks {sorted(missing)}")

    all_mapped_ids = set(coverage.get_column("segment_id").to_list())
    all_edges = contract_monitored_paths(segments, all_mapped_ids)
    all_components = _weak_components(all_mapped_ids, all_edges)
    coverage_ids = set(
        coverage.filter(pl.col("min_target_coverage") >= min_target_coverage)
        .get_column("segment_id")
        .to_list()
    )
    if not coverage_ids:
        raise ValueError("coverage threshold removes every mapped river segment")
    coverage_edges = contract_monitored_paths(segments, coverage_ids)
    coverage_components = _weak_components(coverage_ids, coverage_edges)
    station_count_by_node = dict(
        zip(
            coverage.get_column("segment_id").to_list(),
            coverage.get_column("source_station_count").to_list(),
            strict=True,
        )
    )
    coverage_components.sort(
        key=lambda component: (
            -len(component),
            -sum(int(station_count_by_node[node]) for node in component),
            _id_sort_key(min(component, key=_id_sort_key)),
        )
    )
    eligible = [
        component
        for component in coverage_components
        if min_component_nodes <= len(component) <= max_component_nodes
    ]
    if len(eligible) < component_limit:
        raise ValueError(
            f"only {len(eligible)} components satisfy the configured node limits"
        )
    selected_components = eligible[:component_limit]
    selected_ids = set().union(*selected_components)
    selected_edges = [
        edge
        for edge in coverage_edges
        if edge.source in selected_ids and edge.destination in selected_ids
    ]
    if len(selected_edges) != len(selected_ids) - len(selected_components):
        raise ValueError("selected contracted graph must be an acyclic directed forest")

    ordered_nodes = sorted(selected_ids, key=_id_sort_key)
    node_index = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    component_by_node: dict[str, str] = {}
    for index, component in enumerate(selected_components, start=1):
        component_id = f"contracted-component-{index:02d}"
        component_by_node.update({node_id: component_id for node_id in component})
    ordered_edges = sorted(
        selected_edges,
        key=lambda edge: (node_index[edge.source], node_index[edge.destination]),
    )
    edge_index = np.asarray(
        [
            [node_index[edge.source] for edge in ordered_edges],
            [node_index[edge.destination] for edge in ordered_edges],
        ],
        dtype=np.int64,
    )
    selected_segment_table = (
        segments.filter(pl.col("segment_id").is_in(ordered_nodes))
        .with_columns(
            pl.col("segment_id")
            .replace_strict(node_index, return_dtype=pl.Int64)
            .alias("node_index")
        )
        .sort("node_index")
    )
    static = _static_matrix(selected_segment_table)
    edge_attr = _edge_matrix(ordered_edges, selected_segment_table, travel_speed_km_per_day)
    travel_days = edge_attr[:, -1]
    all_sizes = sorted((len(component) for component in all_components), reverse=True)
    coverage_sizes = sorted(
        (len(component) for component in coverage_components), reverse=True
    )
    coverage_values = coverage.get_column("min_target_coverage")
    selected_profile = coverage.filter(pl.col("segment_id").is_in(ordered_nodes))
    selected_station_count = int(selected_profile.get_column("source_station_count").sum())
    selected_coverage_summary = {
        target: {
            "min": float(selected_profile.get_column(f"{target}_coverage").min()),
            "mean": float(selected_profile.get_column(f"{target}_coverage").mean()),
        }
        for target in TARGET_NAMES
        if f"{target}_coverage" in selected_profile.columns
    }
    report: dict[str, Any] = {
        "construction": "first_downstream_selected_reach_path_contraction",
        "edge_direction": "upstream_to_downstream",
        "selection": {
            "min_target_coverage": min_target_coverage,
            "min_component_nodes": min_component_nodes,
            "max_component_nodes": max_component_nodes,
            "component_limit": component_limit,
            "strategy": "largest_components_within_compute_cap",
        },
        "travel_time_prior": {
            "method": "contracted_path_length_km / assumed_travel_speed_km_per_day",
            "assumed_travel_speed_km_per_day": travel_speed_km_per_day,
            "model_max_lag_days": max_lag_days,
            "causal_interpretation": False,
        },
        "mapped_station_count": int(coverage.get_column("source_station_count").sum()),
        "mapped_segment_count": len(all_mapped_ids),
        "all_mapped_contracted_edge_count": len(all_edges),
        "all_mapped_component_count": len(all_components),
        "all_mapped_largest_component_sizes": all_sizes[:10],
        "coverage_passing_segment_count": len(coverage_ids),
        "coverage_passing_edge_count": len(coverage_edges),
        "coverage_passing_component_count": len(coverage_components),
        "coverage_passing_largest_component_sizes": coverage_sizes[:10],
        "undersized_component_count": sum(
            len(component) < min_component_nodes for component in coverage_components
        ),
        "oversized_component_count": sum(
            len(component) > max_component_nodes for component in coverage_components
        ),
        "eligible_component_count": len(eligible),
        "selected_component_sizes": [len(component) for component in selected_components],
        "selected_segment_count": len(ordered_nodes),
        "selected_edge_count": len(ordered_edges),
        "selected_source_station_count": selected_station_count,
        "selected_mapping_review_segment_count": int(
            selected_profile.get_column("mapping_review").sum()
        ),
        "selected_graph_is_directed_forest": True,
        "selected_duplicate_edge_count": 0,
        "selected_min_target_coverage": {
            "min": float(selected_profile.get_column("min_target_coverage").min()),
            "mean": float(selected_profile.get_column("min_target_coverage").mean()),
        },
        "selected_target_coverage": selected_coverage_summary,
        "coverage_quantiles": {
            str(quantile): float(coverage_values.quantile(quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "contracted_path_hops": _numeric_summary(
            np.asarray([edge.hop_count for edge in ordered_edges], dtype=np.float64)
        ),
        "contracted_path_length_km": _numeric_summary(
            np.asarray([edge.path_length_km for edge in ordered_edges], dtype=np.float64)
        ),
        "travel_time_prior_days": _numeric_summary(travel_days.astype(np.float64)),
        "travel_time_prior_above_max_lag_count": int((travel_days > max_lag_days).sum()),
        "component_size_distribution_after_coverage": {
            str(size): count
            for size, count in sorted(Counter(coverage_sizes).items())
        },
    }
    return ContractedGraphAssets(
        node_ids=np.asarray(ordered_nodes, dtype="U32"),
        component_ids=np.asarray(
            [component_by_node[node_id] for node_id in ordered_nodes], dtype="U64"
        ),
        edge_index=edge_index,
        edge_attr=edge_attr.astype(np.float32),
        static=static.astype(np.float32),
        report=report,
    )


def write_contracted_graph_assets(
    assets: ContractedGraphAssets,
    graph_root: str | Path,
) -> Path:
    """Write component NPZ fixtures consumable by the real-data preparation path."""
    graph_root = Path(graph_root).expanduser().resolve()
    graph_root.mkdir(parents=True, exist_ok=True)
    component_ids = list(dict.fromkeys(assets.component_ids.tolist()))
    expected_directories = {f"{component_id}-v0.2" for component_id in component_ids}
    unexpected = [
        path.name
        for path in graph_root.iterdir()
        if path.is_dir() and path.name not in expected_directories
    ]
    if unexpected:
        raise ValueError(
            f"contracted graph root contains stale component directories: {sorted(unexpected)}"
        )
    for component_id in component_ids:
        global_indices = np.flatnonzero(assets.component_ids == component_id)
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(global_indices.tolist())
        }
        edge_mask = np.isin(assets.edge_index[0], global_indices) & np.isin(
            assets.edge_index[1], global_indices
        )
        global_edges = assets.edge_index[:, edge_mask]
        local_edges = np.asarray(
            [
                [global_to_local[int(index)] for index in global_edges[0]],
                [global_to_local[int(index)] for index in global_edges[1]],
            ],
            dtype=np.int64,
        )
        directory = graph_root / f"{component_id}-v0.2"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "graph.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                segment_id=assets.node_ids[global_indices],
                edge_index=local_edges,
                edge_attr=assets.edge_attr[edge_mask],
                edge_attr_names=np.asarray(EDGE_ATTR_NAMES, dtype="U64"),
                edge_attr_mask=np.ones((int(edge_mask.sum()), len(EDGE_ATTR_NAMES)), dtype=bool),
            )
        with (directory / "static.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                static_node=assets.static[global_indices],
                static_names=np.asarray(STATIC_NAMES, dtype="U64"),
                static_mask=np.ones((len(global_indices), len(STATIC_NAMES)), dtype=bool),
                segment_id=assets.node_ids[global_indices],
            )
    report_path = graph_root / "contracted_graph_report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(assets.report, indent=2, ensure_ascii=False) + "\n")
    return report_path


def _read_dbf_columns(
    handle: BinaryIO,
    columns: tuple[str, ...],
) -> dict[str, list[object]]:
    header = handle.read(32)
    if len(header) != 32:
        raise ValueError("HydroRIVERS DBF header is incomplete")
    record_count = int.from_bytes(header[4:8], "little")
    header_length = int.from_bytes(header[8:10], "little")
    record_length = int.from_bytes(header[10:12], "little")
    fields, consumed = _read_dbf_fields(handle)
    if header_length > consumed:
        handle.read(header_length - consumed)
    by_name = {field.name: field for field in fields}
    missing = [column for column in columns if column not in by_name]
    if missing:
        raise ValueError(f"HydroRIVERS DBF lacks {missing}")
    selected = [by_name[column] for column in columns]
    values: dict[str, list[object]] = {column: [] for column in columns}
    for _ in range(record_count):
        record = handle.read(record_length)
        if not record:
            break
        if record[:1] == b"*":
            continue
        for field in selected:
            raw = record[field.offset : field.offset + field.length]
            values[field.name].append(_parse_dbf_value(raw, field))
    return values


def _read_dbf_fields(handle: BinaryIO) -> tuple[list[DbfField], int]:
    fields: list[DbfField] = []
    offset = 1
    consumed = 32
    while True:
        first = handle.read(1)
        consumed += 1
        if first == b"\r":
            break
        descriptor = first + handle.read(31)
        consumed += 31
        if len(descriptor) != 32:
            raise ValueError("HydroRIVERS DBF field descriptor is incomplete")
        name = (
            descriptor[:11]
            .split(b"\x00", 1)[0]
            .decode("ascii", errors="ignore")
            .strip()
        )
        field = DbfField(
            name=name,
            field_type=chr(descriptor[11]),
            offset=offset,
            length=descriptor[16],
            decimal_count=descriptor[17],
        )
        fields.append(field)
        offset += field.length
    return fields, consumed


def _parse_dbf_value(raw: bytes, field: DbfField) -> object:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text:
        return None
    if field.field_type in {"N", "F"}:
        try:
            if field.decimal_count == 0 and "." not in text and "E" not in text.upper():
                return int(text)
            return float(text)
        except ValueError:
            return None
    if field.field_type == "L":
        return text.upper() in {"Y", "T"}
    return text


def _normalized_id(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Int64, strict=False).cast(pl.String)


def _normalized_optional_id(column: str) -> pl.Expr:
    numeric = pl.col(column).cast(pl.Int64, strict=False)
    return pl.when(numeric > 0).then(numeric.cast(pl.String)).otherwise(None)


def _weak_components(
    nodes: set[str], edges: list[ContractedEdge]
) -> list[set[str]]:
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        adjacency[edge.source].add(edge.destination)
        adjacency[edge.destination].add(edge.source)
    components: list[set[str]] = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen, key=_id_sort_key)
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _static_matrix(segments: pl.DataFrame) -> np.ndarray:
    columns = (
        "length_km",
        "distance_downstream_km",
        "catchment_area_km2",
        "upstream_area_km2",
        "stream_order",
    )
    values = np.column_stack(
        [segments.get_column(column).to_numpy().astype(np.float64) for column in columns]
    )
    values[:, :4] = np.log1p(np.maximum(values[:, :4], 0.0))
    return _impute_and_standardize(values)


def _edge_matrix(
    edges: list[ContractedEdge],
    segments: pl.DataFrame,
    travel_speed_km_per_day: float,
) -> np.ndarray:
    stream_order = dict(
        zip(
            segments.get_column("segment_id").to_list(),
            segments.get_column("stream_order").to_list(),
            strict=True,
        )
    )
    features = np.asarray(
        [
            [
                math.log1p(edge.path_length_km),
                math.log1p(edge.hop_count),
                _optional_float(stream_order[edge.source]),
                _optional_float(stream_order[edge.destination]),
            ]
            for edge in edges
        ],
        dtype=np.float64,
    )
    standardized = _impute_and_standardize(features)
    travel_days = np.asarray(
        [edge.path_length_km / travel_speed_km_per_day for edge in edges],
        dtype=np.float64,
    )
    return np.column_stack((standardized, travel_days))


def _impute_and_standardize(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        if not finite.any():
            raise ValueError(f"graph feature column {column} has no finite values")
        median = float(np.median(output[finite, column]))
        output[~finite, column] = median
        mean = float(output[:, column].mean())
        scale = float(output[:, column].std())
        output[:, column] = 0.0 if scale < 1e-8 else (output[:, column] - mean) / scale
    return output


def _optional_float(value: object) -> float:
    return float(value) if value is not None else float("nan")


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("min", "median", "p90", "max", "mean")}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _id_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**63 - 1, value
