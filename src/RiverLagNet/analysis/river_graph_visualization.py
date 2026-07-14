"""Audit and visualize the real monitored upstream-to-downstream river graph."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# 可调参数：画布与导出
FIGURE_SIZE = (13.0, 7.2)
EXPORT_DPI = 300
WIDTH_RATIOS = (1.35, 1.0)
PANEL_WSPACE = 0.15

# 可调参数：节点、箭头与标签
BASE_NODE_SIZE = 34.0
STATION_SIZE_STEP = 12.0
MAX_NODE_SIZE = 110.0
EDGE_LINE_WIDTH = 0.9
ARROW_SCALE = 9.0
ARROW_SHRINK = 5.5
NODE_LABEL_DIGITS = 4
NODE_LABEL_SIZE = 6.7
COMPONENT_LABEL_SIZE = 8.0
AXIS_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 8.5
PANEL_LABEL_SIZE = 11.0
TOPOLOGY_BRANCH_SPREAD = 0.18
TOPOLOGY_X_MIN = 0.20
TOPOLOGY_X_MAX = 0.92
COMPONENT_LABEL_OFFSETS = (
    (-28, 10),
    (25, 18),
    (0, 15),
    (0, 15),
    (28, -8),
    (-20, 14),
    (16, 14),
    (-12, 14),
    (16, -10),
    (0, 15),
)

# 可调参数：学术配色
COMPONENT_PALETTE = (
    "#4E79A7",
    "#D79A45",
    "#5B8E7D",
    "#8E6C8A",
    "#7D8F5B",
    "#B46A63",
    "#5F8FA3",
    "#A27C4C",
    "#687A9A",
    "#85706B",
)
PRIOR_COLORS = {0: "#7C8790", 1: "#D08B3E"}
DEFAULT_EDGE_COLOR = "#B46A63"
REVIEW_COLOR = "#B23A3A"
GRID_COLOR = "#DCE0E3"
TEXT_COLOR = "#252A2E"


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid graph value for {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite graph value for {label}")
    return number


def _topological_depths(node_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    adjacency = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for source, destination in edges:
        adjacency[source].append(destination)
        indegree[destination] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    depths = {node_id: 0 for node_id in node_ids}
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for destination in sorted(adjacency[source]):
            depths[destination] = max(depths[destination], depths[source] + 1)
            indegree[destination] -= 1
            if indegree[destination] == 0:
                queue.append(destination)
    if visited != len(node_ids):
        raise ValueError("river graph must be acyclic")
    return depths


def build_river_graph_summary(data_root: Path) -> dict[str, object]:
    """Build a JSON-safe audited graph summary from prepared real-data artifacts."""
    data_root = Path(data_root)
    dataset_path = data_root / "dataset.npz"
    mapping_path = data_root / "station_mapping.parquet"
    edges_path = data_root / "edges.parquet"
    if not all(path.is_file() for path in (dataset_path, mapping_path, edges_path)):
        raise FileNotFoundError(f"prepared graph assets are incomplete: {data_root}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {"node_ids", "component_ids", "edge_index", "source_station_count"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"dataset is missing graph arrays: {sorted(missing)}")
        node_ids = [str(value) for value in archive["node_ids"].tolist()]
        component_ids = [str(value) for value in archive["component_ids"].tolist()]
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        source_station_count = np.asarray(archive["source_station_count"], dtype=np.int64)
    if len(node_ids) != len(set(node_ids)) or len(node_ids) != len(component_ids):
        raise ValueError("node IDs and component IDs are inconsistent")
    if source_station_count.shape != (len(node_ids),) or np.any(source_station_count <= 0):
        raise ValueError("source station counts must be positive per node")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(node_ids)):
        raise ValueError("edge_index contains an invalid node index")

    mapping = pl.read_parquet(mapping_path)
    required_mapping = {
        "station_id",
        "source_station_id",
        "station_lon",
        "station_lat",
        "mapping_quality_flag",
    }
    if not required_mapping.issubset(mapping.columns):
        raise ValueError("station mapping is missing required visualization columns")
    coordinates = (
        mapping.with_columns(pl.col("station_id").cast(pl.String))
        .group_by("station_id")
        .agg(
            pl.col("station_lon").mean().alias("longitude"),
            pl.col("station_lat").mean().alias("latitude"),
            pl.len().alias("mapped_station_count"),
            pl.col("mapping_quality_flag")
            .str.contains("review")
            .any()
            .alias("mapping_review"),
        )
    )
    coordinate_rows = {row["station_id"]: row for row in coordinates.to_dicts()}
    if set(coordinate_rows) != set(node_ids):
        raise ValueError("station mapping must cover every graph node exactly by segment ID")

    edge_table = pl.read_parquet(edges_path)
    required_edges = {"src_station_id", "dst_station_id", "travel_time_prior_days"}
    if not required_edges.issubset(edge_table.columns):
        raise ValueError("edge table is missing required visualization columns")
    edge_rows = edge_table.select(
        pl.col("src_station_id").cast(pl.String),
        pl.col("dst_station_id").cast(pl.String),
        pl.col("travel_time_prior_days").cast(pl.Float64),
    ).to_dicts()
    indexed_pairs = [
        (node_ids[int(source)], node_ids[int(destination)])
        for source, destination in edge_index.T.tolist()
    ]
    table_pairs = [
        (row["src_station_id"], row["dst_station_id"]) for row in edge_rows
    ]
    if indexed_pairs != table_pairs:
        raise ValueError("edge table order or direction does not match dataset edge_index")

    component_names = sorted(set(component_ids))
    component_labels = {
        component: f"C{index + 1:02d}" for index, component in enumerate(component_names)
    }
    component_by_node = dict(zip(node_ids, component_ids, strict=True))
    indegree = Counter(destination for _, destination in indexed_pairs)
    outdegree = Counter(source for source, _ in indexed_pairs)
    nodes: list[dict[str, object]] = []
    for index, node_id in enumerate(node_ids):
        component = component_by_node[node_id]
        incoming = indegree[node_id]
        outgoing = outdegree[node_id]
        role = (
            "isolated"
            if incoming == 0 and outgoing == 0
            else "headwater"
            if incoming == 0
            else "outlet"
            if outgoing == 0
            else "internal"
        )
        coordinate = coordinate_rows[node_id]
        mapped_count = int(coordinate["mapped_station_count"])
        if mapped_count != int(source_station_count[index]):
            raise ValueError(f"source station count mismatch for node {node_id}")
        nodes.append(
            {
                "node_id": node_id,
                "node_index": index,
                "component_id": component,
                "component_label": component_labels[component],
                "longitude": _finite(coordinate["longitude"], f"{node_id}.longitude"),
                "latitude": _finite(coordinate["latitude"], f"{node_id}.latitude"),
                "source_station_count": mapped_count,
                "mapping_review": bool(coordinate["mapping_review"]),
                "role": role,
                "indegree": incoming,
                "outdegree": outgoing,
            }
        )

    edges: list[dict[str, object]] = []
    for row in edge_rows:
        source = row["src_station_id"]
        destination = row["dst_station_id"]
        if source == destination:
            raise ValueError("river graph cannot contain self edges")
        if component_by_node[source] != component_by_node[destination]:
            raise ValueError("river edge cannot cross a disjoint graph component")
        prior = _finite(row["travel_time_prior_days"], "travel_time_prior_days")
        rounded = int(round(prior))
        edges.append(
            {
                "src_station_id": source,
                "dst_station_id": destination,
                "component_id": component_by_node[source],
                "component_label": component_labels[component_by_node[source]],
                "travel_time_prior_days": prior,
                "rounded_prior_lag_days": rounded,
            }
        )

    components: list[dict[str, object]] = []
    for component in component_names:
        component_nodes = [
            node["node_id"] for node in nodes if node["component_id"] == component
        ]
        component_edges = [
            (edge["src_station_id"], edge["dst_station_id"])
            for edge in edges
            if edge["component_id"] == component
        ]
        depths = _topological_depths(component_nodes, component_edges)
        components.append(
            {
                "component_id": component,
                "component_label": component_labels[component],
                "node_count": len(component_nodes),
                "edge_count": len(component_edges),
                "headwater_nodes": sorted(
                    node_id for node_id in component_nodes if indegree[node_id] == 0
                ),
                "outlet_nodes": sorted(
                    node_id for node_id in component_nodes if outdegree[node_id] == 0
                ),
                "topological_depth": max(depths.values(), default=0),
            }
        )

    prior_counts = Counter(int(edge["rounded_prior_lag_days"]) for edge in edges)
    return {
        "direction": "upstream_to_downstream",
        "node_definition": "monitored HydroRIVERS segment",
        "coordinate_semantics": (
            "mean longitude/latitude of source monitoring stations mapped to each segment; "
            "not river-line geometry"
        ),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(components),
        "mapping_review_node_count": sum(bool(node["mapping_review"]) for node in nodes),
        "rounded_prior_lag_counts": {
            str(lag): count for lag, count in sorted(prior_counts.items())
        },
        "nodes": nodes,
        "edges": edges,
        "components": components,
    }


def write_river_graph_summary(summary: Mapping[str, object], path: Path) -> None:
    """Write the graph audit summary with stable UTF-8 LF formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def _node_size(count: int) -> float:
    return min(BASE_NODE_SIZE + STATION_SIZE_STEP * (count - 1), MAX_NODE_SIZE)


def _arrow(
    axis: plt.Axes,
    source: tuple[float, float],
    destination: tuple[float, float],
    color: str,
    *,
    shrink: float = ARROW_SHRINK,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            source,
            destination,
            arrowstyle="-|>",
            mutation_scale=ARROW_SCALE,
            linewidth=EDGE_LINE_WIDTH,
            color=color,
            shrinkA=shrink,
            shrinkB=shrink,
            connectionstyle="arc3,rad=0.025",
            zorder=1,
        )
    )


def _component_colors(summary: Mapping[str, Any]) -> dict[str, str]:
    components = summary["components"]
    if not isinstance(components, list) or len(components) > len(COMPONENT_PALETTE):
        raise ValueError("component palette does not cover the graph")
    return {
        str(component["component_id"]): COMPONENT_PALETTE[index]
        for index, component in enumerate(components)
    }


def _geographic_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    nodes = summary["nodes"]
    edges = summary["edges"]
    assert isinstance(nodes, list) and isinstance(edges, list)
    node_by_id = {str(node["node_id"]): node for node in nodes}
    component_colors = _component_colors(summary)
    for edge in edges:
        source = node_by_id[str(edge["src_station_id"])]
        destination = node_by_id[str(edge["dst_station_id"])]
        lag = int(edge["rounded_prior_lag_days"])
        _arrow(
            axis,
            (_finite(source["longitude"], "longitude"), _finite(source["latitude"], "latitude")),
            (
                _finite(destination["longitude"], "longitude"),
                _finite(destination["latitude"], "latitude"),
            ),
            PRIOR_COLORS.get(lag, DEFAULT_EDGE_COLOR),
        )
    for node in nodes:
        longitude = _finite(node["longitude"], "longitude")
        latitude = _finite(node["latitude"], "latitude")
        color = component_colors[str(node["component_id"])]
        size = _node_size(int(node["source_station_count"]))
        axis.scatter(
            [longitude],
            [latitude],
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        if bool(node["mapping_review"]):
            axis.scatter(
                [longitude],
                [latitude],
                s=size + 28,
                facecolor="none",
                edgecolor=REVIEW_COLOR,
                linewidth=1.2,
                zorder=4,
            )
    for component_index, component in enumerate(summary["components"]):
        component_nodes = [
            node for node in nodes if node["component_id"] == component["component_id"]
        ]
        longitude = float(np.mean([node["longitude"] for node in component_nodes]))
        latitude = float(np.mean([node["latitude"] for node in component_nodes]))
        offset = COMPONENT_LABEL_OFFSETS[component_index]
        axis.annotate(
            str(component["component_label"]),
            xy=(longitude, latitude),
            xytext=offset,
            textcoords="offset points",
            color=TEXT_COLOR,
            fontsize=COMPONENT_LABEL_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            arrowprops={
                "arrowstyle": "-",
                "color": component_colors[str(component["component_id"])],
                "lw": 0.65,
                "shrinkA": 1.5,
                "shrinkB": 2.0,
            },
            zorder=5,
        )
    latitudes = [float(node["latitude"]) for node in nodes]
    axis.set_aspect(1.0 / math.cos(math.radians(float(np.mean(latitudes)))))
    axis.set_xlabel("Longitude (°E)", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Latitude (°N)", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(labelsize=TICK_LABEL_SIZE, width=0.7, length=3)
    axis.grid(color=GRID_COLOR, linewidth=0.55, alpha=0.75, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        -0.08,
        1.02,
        "a",
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color=PRIOR_COLORS[0], lw=1.4, marker=">", markevery=[1], label="Prior lag 0 d"),
        Line2D([0], [0], color=PRIOR_COLORS[1], lw=1.4, marker=">", markevery=[1], label="Prior lag 1 d"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=REVIEW_COLOR,
            label="Mapping review",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=TICK_LABEL_SIZE,
        ncol=3,
        bbox_to_anchor=(0.0, -0.15),
    )


def _topology_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    nodes = summary["nodes"]
    edges = summary["edges"]
    components = summary["components"]
    assert isinstance(nodes, list) and isinstance(edges, list) and isinstance(components, list)
    node_by_id = {str(node["node_id"]): node for node in nodes}
    component_colors = _component_colors(summary)
    for row_index, component in enumerate(components):
        row = len(components) - 1 - row_index
        component_id = str(component["component_id"])
        component_nodes = sorted(
            str(node["node_id"]) for node in nodes if node["component_id"] == component_id
        )
        component_edges = [
            (str(edge["src_station_id"]), str(edge["dst_station_id"]))
            for edge in edges
            if edge["component_id"] == component_id
        ]
        depths = _topological_depths(component_nodes, component_edges)
        max_depth = max(depths.values(), default=0)
        positions: dict[str, tuple[float, float]] = {}
        for depth in range(max_depth + 1):
            level_nodes = sorted(node for node in component_nodes if depths[node] == depth)
            offsets = np.linspace(
                -TOPOLOGY_BRANCH_SPREAD,
                TOPOLOGY_BRANCH_SPREAD,
                len(level_nodes),
            )
            x_position = (
                TOPOLOGY_X_MIN
                if max_depth == 0
                else TOPOLOGY_X_MIN
                + (TOPOLOGY_X_MAX - TOPOLOGY_X_MIN) * depth / max_depth
            )
            for node_id, offset in zip(level_nodes, offsets, strict=True):
                positions[node_id] = (x_position, row + float(offset))
        edge_lookup = {
            (str(edge["src_station_id"]), str(edge["dst_station_id"])): edge
            for edge in edges
            if edge["component_id"] == component_id
        }
        for source, destination in component_edges:
            lag = int(edge_lookup[(source, destination)]["rounded_prior_lag_days"])
            _arrow(
                axis,
                positions[source],
                positions[destination],
                PRIOR_COLORS.get(lag, DEFAULT_EDGE_COLOR),
                shrink=4.5,
            )
        for node_id in component_nodes:
            node = node_by_id[node_id]
            x_position, y_position = positions[node_id]
            axis.scatter(
                [x_position],
                [y_position],
                s=_node_size(int(node["source_station_count"])) * 0.72,
                color=component_colors[component_id],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            if bool(node["mapping_review"]):
                axis.scatter(
                    [x_position],
                    [y_position],
                    s=_node_size(int(node["source_station_count"])) * 0.72 + 22,
                    facecolor="none",
                    edgecolor=REVIEW_COLOR,
                    linewidth=1.0,
                    zorder=4,
                )
            axis.text(
                x_position,
                y_position + 0.24,
                node_id[-NODE_LABEL_DIGITS:],
                ha="center",
                va="bottom",
                fontsize=NODE_LABEL_SIZE,
                color=TEXT_COLOR,
            )
        axis.text(
            0.03,
            row,
            str(component["component_label"]),
            ha="left",
            va="center",
            fontsize=COMPONENT_LABEL_SIZE,
            fontweight="bold",
            color=component_colors[component_id],
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.55, len(components) - 0.25)
    axis.axis("off")
    axis.annotate(
        "",
        xy=(TOPOLOGY_X_MAX, 1.03),
        xytext=(TOPOLOGY_X_MIN, 1.03),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEXT_COLOR, "lw": 0.8},
    )
    axis.text(TOPOLOGY_X_MIN, 1.055, "Upstream", transform=axis.transAxes, ha="center", fontsize=AXIS_LABEL_SIZE)
    axis.text(TOPOLOGY_X_MAX, 1.055, "Downstream", transform=axis.transAxes, ha="center", fontsize=AXIS_LABEL_SIZE)
    axis.text(
        -0.02,
        1.02,
        "b",
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )


def render_river_graph_figure(
    summary: Mapping[str, Any], png_path: Path, pdf_path: Path
) -> tuple[Path, Path]:
    """Render geographic and topology panels for upstream-downstream evidence."""
    if summary.get("direction") != "upstream_to_downstream":
        raise ValueError("graph direction must be upstream_to_downstream")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TICK_LABEL_SIZE,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": WIDTH_RATIOS},
    )
    _geographic_panel(axes[0], summary)
    _topology_panel(axes[1], summary)
    figure.subplots_adjust(wspace=PANEL_WSPACE)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path
