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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# 可调参数：画布、卡片与导出
FIGURE_SIZE = (15.0, 7.8)
FIGURE_WIDTH_PER_COLUMN = 3.0
FIGURE_HEIGHT_PER_ROW = 3.9
EXPORT_DPI = 300
CARD_ROWS = 2
CARD_COLUMNS = 5
CARD_LEFT = 0.035
CARD_RIGHT = 0.985
CARD_TOP = 0.855
CARD_BOTTOM = 0.145
CARD_WSPACE = 0.13
CARD_HSPACE = 0.22
CARD_FRAME_VISIBLE = True
CARD_FRAME_COLOR = "#D7DDE1"
CARD_FRAME_WIDTH = 0.8
CARD_FACE_COLOR = "#FBFCFD"
CARD_CORNER_RADIUS = 0.035

# 可调参数：节点、箭头与标签
NODE_SIZE = 510.0
NODE_EDGE_WIDTH = 1.0
NODE_COUNT_SIZE = 8.4
NODE_LABEL_SIZE = 7.6
COMPONENT_LABEL_SIZE = 10.0
COMPONENT_META_SIZE = 7.5
EDGE_LINE_WIDTH = 1.25
ARROW_SCALE = 12.0
ARROW_SHRINK = 16.0
EDGE_LABEL_SIZE = 7.0
TOPOLOGY_BRANCH_SPREAD = 0.17
TOPOLOGY_X_MIN = 0.12
TOPOLOGY_X_MAX = 0.88
TOPOLOGY_Y_CENTER = 0.50
NODE_LABEL_OFFSET = 0.125

# 可调参数：标题、图例与方向标识
TITLE_SIZE = 15.0
SUBTITLE_SIZE = 9.2
DIRECTION_LABEL_SIZE = 9.2
LEGEND_SIZE = 8.3
FOOTNOTE_SIZE = 8.0

# 可调参数：学术配色
ROLE_COLORS = {
    "headwater": "#D9E8F0",
    "internal": "#7FA8BD",
    "outlet": "#2F6482",
    "isolated": "#C9D0D5",
}
ROLE_TEXT_COLORS = {
    "headwater": "#24343C",
    "internal": "#FFFFFF",
    "outlet": "#FFFFFF",
    "isolated": "#24343C",
}
PRIOR_COLORS = {0: "#8B969E", 1: "#D48835"}
DEFAULT_EDGE_COLOR = "#B46A63"
REVIEW_COLOR = "#B23A3A"
TEXT_COLOR = "#252A2E"
MUTED_TEXT_COLOR = "#667078"

# 可调参数：大规模收缩河网布局（上游在左、下游在右）
LARGE_GRAPH_NODE_THRESHOLD = 80
LARGE_FIGURE_SIZE = (15.2, 8.6)
LARGE_NETWORK_RECT = (0.035, 0.12, 0.735, 0.76)
LARGE_STATS_RECT = (0.805, 0.53, 0.17, 0.31)
LARGE_LAG_RECT = (0.805, 0.16, 0.17, 0.27)
LARGE_NODE_SIZE = 13.0
LARGE_HEADWATER_SIZE = 21.0
LARGE_OUTLET_SIZE = 70.0
LARGE_EDGE_WIDTH = 0.72
LARGE_ARROW_SCALE = 5.8
LARGE_X_MARGIN = 0.025
LARGE_Y_MARGIN = 0.018
LARGE_LABEL_SIZE = 7.2

# 可调参数：时滞区间使用单一蓝色根系；超过模型上限的边用橙色虚线警示
LAG_BIN_SPECS = (
    ("<1 d", 0.0, 1.0, "#C9D9E2"),
    ("1–3 d", 1.0, 3.0, "#8FB2C4"),
    ("3–7 d", 3.0, 7.0, "#4E819D"),
    ("7–14 d", 7.0, 14.0, "#255A78"),
    (">14 d", 14.0, float("inf"), "#C47A2C"),
)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid graph value for {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite graph value for {label}")
    return number


def _lag_bin(prior_days: float) -> tuple[str, str]:
    """Return the declared label and color for one travel-time prior."""
    for label, lower, upper, color in LAG_BIN_SPECS:
        if lower <= prior_days < upper or (
            math.isinf(upper) and prior_days >= lower
        ):
            return label, color
    raise ValueError(f"travel-time prior must be non-negative, found {prior_days}")


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
    manifest_path = data_root / "manifest.json"
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
    lag_bin_counts = Counter(
        _lag_bin(float(edge["travel_time_prior_days"]))[0] for edge in edges
    )
    prior_values = np.asarray(
        [float(edge["travel_time_prior_days"]) for edge in edges], dtype=np.float64
    )
    graph_construction: dict[str, object] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest.get("graph_construction")
        if isinstance(candidate, dict):
            graph_construction = candidate
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
        "prior_lag_bin_counts": {
            label: lag_bin_counts.get(label, 0)
            for label, _, _, _ in LAG_BIN_SPECS
        },
        "travel_time_prior_days": {
            "min": float(prior_values.min()) if prior_values.size else 0.0,
            "median": float(np.median(prior_values)) if prior_values.size else 0.0,
            "p90": float(np.quantile(prior_values, 0.9)) if prior_values.size else 0.0,
            "max": float(prior_values.max()) if prior_values.size else 0.0,
        },
        "graph_construction": graph_construction,
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


def _card_positions(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> dict[str, tuple[float, float]]:
    depths = _topological_depths(node_ids, edges)
    max_depth = max(depths.values(), default=0)
    positions: dict[str, tuple[float, float]] = {}
    for depth in range(max_depth + 1):
        level_nodes = sorted(node_id for node_id in node_ids if depths[node_id] == depth)
        offsets = (
            np.array([0.0])
            if len(level_nodes) == 1
            else np.linspace(-TOPOLOGY_BRANCH_SPREAD, TOPOLOGY_BRANCH_SPREAD, len(level_nodes))
        )
        x_position = (
            (TOPOLOGY_X_MIN + TOPOLOGY_X_MAX) / 2.0
            if max_depth == 0
            else TOPOLOGY_X_MIN
            + (TOPOLOGY_X_MAX - TOPOLOGY_X_MIN) * depth / max_depth
        )
        for node_id, offset in zip(level_nodes, offsets, strict=True):
            positions[node_id] = (x_position, TOPOLOGY_Y_CENTER + float(offset))
    return positions


def _component_card(
    axis: plt.Axes,
    component: Mapping[str, Any],
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
) -> None:
    component_id = str(component["component_id"])
    component_nodes = [node for node in nodes if node["component_id"] == component_id]
    component_edges = [edge for edge in edges if edge["component_id"] == component_id]
    node_ids = sorted(str(node["node_id"]) for node in component_nodes)
    edge_pairs = [
        (str(edge["src_station_id"]), str(edge["dst_station_id"]))
        for edge in component_edges
    ]
    positions = _card_positions(node_ids, edge_pairs)
    node_by_id = {str(node["node_id"]): node for node in component_nodes}
    edge_by_pair = {
        (str(edge["src_station_id"]), str(edge["dst_station_id"])): edge
        for edge in component_edges
    }

    frame = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=axis.transAxes,
        boxstyle=f"round,pad=0.012,rounding_size={CARD_CORNER_RADIUS}",
        facecolor=CARD_FACE_COLOR,
        edgecolor=CARD_FRAME_COLOR if CARD_FRAME_VISIBLE else "none",
        linewidth=CARD_FRAME_WIDTH,
        clip_on=False,
        zorder=-5,
    )
    axis.add_patch(frame)
    axis.text(
        0.045,
        0.915,
        str(component["component_label"]),
        transform=axis.transAxes,
        fontsize=COMPONENT_LABEL_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    axis.text(
        0.955,
        0.915,
        f"{component['node_count']} segments · {component['edge_count']} edges",
        transform=axis.transAxes,
        fontsize=COMPONENT_META_SIZE,
        color=MUTED_TEXT_COLOR,
        ha="right",
        va="top",
    )

    for source, destination in edge_pairs:
        edge = edge_by_pair[(source, destination)]
        lag = int(edge["rounded_prior_lag_days"])
        color = PRIOR_COLORS.get(lag, DEFAULT_EDGE_COLOR)
        axis.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[destination],
                arrowstyle="-|>",
                mutation_scale=ARROW_SCALE,
                linewidth=EDGE_LINE_WIDTH,
                color=color,
                shrinkA=ARROW_SHRINK,
                shrinkB=ARROW_SHRINK,
                connectionstyle="arc3,rad=0.0",
                zorder=1,
            )
        )
        if lag != 0:
            midpoint_x = (positions[source][0] + positions[destination][0]) / 2.0
            midpoint_y = (positions[source][1] + positions[destination][1]) / 2.0
            axis.text(
                midpoint_x,
                midpoint_y + 0.055,
                f"{lag} d",
                fontsize=EDGE_LABEL_SIZE,
                color=color,
                ha="center",
                va="bottom",
                bbox={"facecolor": CARD_FACE_COLOR, "edgecolor": "none", "pad": 0.5},
                zorder=2,
            )

    for node_id in node_ids:
        node = node_by_id[node_id]
        x_position, y_position = positions[node_id]
        role = str(node["role"])
        facecolor = ROLE_COLORS.get(role, ROLE_COLORS["isolated"])
        text_color = ROLE_TEXT_COLORS.get(role, ROLE_TEXT_COLORS["isolated"])
        axis.scatter(
            [x_position],
            [y_position],
            s=NODE_SIZE,
            facecolor=facecolor,
            edgecolor="white",
            linewidth=NODE_EDGE_WIDTH,
            zorder=3,
        )
        if bool(node["mapping_review"]):
            axis.scatter(
                [x_position],
                [y_position],
                s=NODE_SIZE + 170,
                facecolor="none",
                edgecolor=REVIEW_COLOR,
                linewidth=1.4,
                zorder=4,
            )
        axis.text(
            x_position,
            y_position,
            str(node["source_station_count"]),
            fontsize=NODE_COUNT_SIZE,
            fontweight="bold",
            color=text_color,
            ha="center",
            va="center",
            zorder=5,
        )
        display_id = f"{node_id}{'*' if bool(node['mapping_review']) else ''}"
        axis.text(
            x_position,
            y_position - NODE_LABEL_OFFSET,
            display_id,
            fontsize=NODE_LABEL_SIZE,
            color=REVIEW_COLOR if bool(node["mapping_review"]) else TEXT_COLOR,
            ha="center",
            va="top",
            zorder=5,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")


def _large_graph_positions(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> dict[str, tuple[float, float]]:
    """Lay out an in-arborescence with upstream leaves left and outlet right."""
    depths = _topological_depths(node_ids, edges)
    maximum_depth = max(depths.values(), default=0)
    upstream = {node_id: [] for node_id in node_ids}
    downstream = {node_id: [] for node_id in node_ids}
    for source, destination in edges:
        upstream[destination].append(source)
        downstream[source].append(destination)
    outlets = sorted(node_id for node_id in node_ids if not downstream[node_id])
    leaf_order: dict[str, float] = {}
    cursor = 0

    def assign_vertical(node_id: str) -> float:
        nonlocal cursor
        parents = sorted(upstream[node_id])
        if not parents:
            value = float(cursor)
            cursor += 1
        else:
            value = float(np.mean([assign_vertical(parent) for parent in parents]))
        leaf_order[node_id] = value
        return value

    for outlet in outlets:
        assign_vertical(outlet)
        cursor += 1
    if set(leaf_order) != set(node_ids):
        raise ValueError("large-graph layout did not visit every node")
    minimum_y = min(leaf_order.values(), default=0.0)
    maximum_y = max(leaf_order.values(), default=0.0)
    y_span = maximum_y - minimum_y
    positions: dict[str, tuple[float, float]] = {}
    for node_id in node_ids:
        x_position = (
            0.5
            if maximum_depth == 0
            else LARGE_X_MARGIN
            + (1.0 - 2.0 * LARGE_X_MARGIN) * depths[node_id] / maximum_depth
        )
        y_position = (
            0.5
            if y_span == 0
            else LARGE_Y_MARGIN
            + (1.0 - 2.0 * LARGE_Y_MARGIN)
            * (leaf_order[node_id] - minimum_y)
            / y_span
        )
        positions[node_id] = (x_position, y_position)
    return positions


def _large_network_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ROLE_COLORS[role],
            markeredgecolor="white",
            markersize=size,
            label=label,
        )
        for role, size, label in (
            ("headwater", 5.0, "Headwater segment"),
            ("internal", 4.2, "Internal segment"),
            ("outlet", 7.0, "Outlet segment"),
        )
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=REVIEW_COLOR,
            markersize=6.5,
            label="Mapping review",
        )
    ]


def _render_large_graph_figure(
    summary: Mapping[str, Any], png_path: Path, pdf_path: Path
) -> tuple[Path, Path]:
    """Render a dense contracted river tree and its travel-time distribution."""
    nodes = list(summary["nodes"])
    edges = list(summary["edges"])
    node_ids = [str(node["node_id"]) for node in nodes]
    edge_pairs = [
        (str(edge["src_station_id"]), str(edge["dst_station_id"]))
        for edge in edges
    ]
    positions = _large_graph_positions(node_ids, edge_pairs)
    node_by_id = {str(node["node_id"]): node for node in nodes}
    figure = plt.figure(figsize=LARGE_FIGURE_SIZE, facecolor="white")
    network_axis = figure.add_axes(LARGE_NETWORK_RECT)
    stats_axis = figure.add_axes(LARGE_STATS_RECT)
    lag_axis = figure.add_axes(LARGE_LAG_RECT)

    for edge in edges:
        source = str(edge["src_station_id"])
        destination = str(edge["dst_station_id"])
        prior_days = float(edge["travel_time_prior_days"])
        _, color = _lag_bin(prior_days)
        network_axis.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[destination],
                arrowstyle="-|>",
                mutation_scale=LARGE_ARROW_SCALE,
                linewidth=LARGE_EDGE_WIDTH,
                linestyle="--" if prior_days > 14.0 else "-",
                color=color,
                alpha=0.78,
                shrinkA=0.8,
                shrinkB=1.2,
                zorder=1,
            )
        )

    for role, size in (
        ("headwater", LARGE_HEADWATER_SIZE),
        ("internal", LARGE_NODE_SIZE),
        ("outlet", LARGE_OUTLET_SIZE),
    ):
        role_ids = [node_id for node_id in node_ids if node_by_id[node_id]["role"] == role]
        if not role_ids:
            continue
        network_axis.scatter(
            [positions[node_id][0] for node_id in role_ids],
            [positions[node_id][1] for node_id in role_ids],
            s=size,
            facecolor=ROLE_COLORS[role],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    review_ids = [
        node_id for node_id in node_ids if bool(node_by_id[node_id]["mapping_review"])
    ]
    if review_ids:
        network_axis.scatter(
            [positions[node_id][0] for node_id in review_ids],
            [positions[node_id][1] for node_id in review_ids],
            s=LARGE_HEADWATER_SIZE + 20.0,
            facecolor="none",
            edgecolor=REVIEW_COLOR,
            linewidth=0.75,
            zorder=4,
        )
    for node_id in node_ids:
        if node_by_id[node_id]["role"] == "outlet":
            x_position, y_position = positions[node_id]
            network_axis.text(
                x_position - 0.008,
                y_position - 0.025,
                f"Outlet\n{node_id}",
                fontsize=LARGE_LABEL_SIZE,
                fontweight="bold",
                color=TEXT_COLOR,
                ha="right",
                va="top",
            )

    network_axis.annotate(
        "",
        xy=(0.96, 0.985),
        xytext=(0.72, 0.985),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEXT_COLOR, "linewidth": 1.0},
        annotation_clip=False,
    )
    network_axis.text(
        0.70,
        0.985,
        "UPSTREAM",
        transform=network_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        fontweight="bold",
        ha="right",
        va="center",
    )
    network_axis.text(
        0.98,
        0.985,
        "DOWNSTREAM",
        transform=network_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="center",
    )
    network_axis.legend(
        handles=_large_network_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.0, -0.105),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    network_axis.set_xlim(0.0, 1.0)
    network_axis.set_ylim(0.0, 1.0)
    network_axis.axis("off")

    headwater_count = sum(node["role"] == "headwater" for node in nodes)
    review_count = sum(bool(node["mapping_review"]) for node in nodes)
    source_station_count = sum(int(node["source_station_count"]) for node in nodes)
    prior_summary = summary.get("travel_time_prior_days", {})
    components = list(summary["components"])
    maximum_depth = max(int(component["topological_depth"]) for component in components)
    above_max_lag = sum(float(edge["travel_time_prior_days"]) > 14.0 for edge in edges)
    stats_axis.text(
        0.0,
        1.0,
        "Network audit",
        fontsize=11.0,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    stats_axis.text(
        0.0,
        0.86,
        (
            f"{len(nodes)} monitored segments\n"
            f"{len(edges)} contracted directed edges\n"
            f"{source_station_count} source stations\n"
            f"{headwater_count} headwaters · depth {maximum_depth}\n"
            f"{review_count} mapping-review segments"
        ),
        fontsize=9.0,
        color=TEXT_COLOR,
        ha="left",
        va="top",
        linespacing=1.55,
    )
    stats_axis.text(
        0.0,
        0.42,
        "Travel-time prior",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    stats_axis.text(
        0.0,
        0.32,
        (
            f"Median  {float(prior_summary.get('median', 0.0)):.2f} d\n"
            f"P90       {float(prior_summary.get('p90', 0.0)):.2f} d\n"
            f"Maximum {float(prior_summary.get('max', 0.0)):.2f} d\n"
            f">14 d      {above_max_lag} edges"
        ),
        fontsize=9.0,
        color=TEXT_COLOR,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    stats_axis.axis("off")

    lag_counts = Counter(
        _lag_bin(float(edge["travel_time_prior_days"]))[0] for edge in edges
    )
    labels = [spec[0] for spec in LAG_BIN_SPECS]
    colors = [spec[3] for spec in LAG_BIN_SPECS]
    values = [lag_counts[label] for label in labels]
    y_positions = np.arange(len(labels))
    lag_axis.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.7,
    )
    for y_position, value in zip(y_positions, values, strict=True):
        lag_axis.text(
            value + max(values) * 0.025,
            y_position,
            str(value),
            fontsize=8.2,
            color=TEXT_COLOR,
            ha="left",
            va="center",
        )
    lag_axis.set_title(
        "Edges by travel-time prior",
        fontsize=10.0,
        fontweight="bold",
        color=TEXT_COLOR,
        loc="left",
        pad=8,
    )
    lag_axis.set_yticks(y_positions, labels, fontsize=8.2)
    lag_axis.set_xlim(0.0, max(values) * 1.22)
    lag_axis.tick_params(axis="x", labelsize=7.5, colors=MUTED_TEXT_COLOR)
    lag_axis.grid(axis="x", color="#E5E9EC", linewidth=0.7)
    lag_axis.set_axisbelow(True)
    lag_axis.spines[["top", "right", "left"]].set_visible(False)
    lag_axis.spines["bottom"].set_color("#AEB6BC")
    lag_axis.tick_params(axis="y", length=0, colors=TEXT_COLOR)

    figure.text(
        0.035,
        0.955,
        "Contracted monitored river network",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    figure.text(
        0.035,
        0.918,
        (
            "Edges follow the first downstream selected reach; layout emphasizes "
            "information flow rather than geographic distance."
        ),
        fontsize=SUBTITLE_SIZE,
        color=MUTED_TEXT_COLOR,
        ha="left",
        va="top",
    )
    figure.text(
        0.035,
        0.035,
        (
            "Blue shades encode path-length travel priors; dashed orange edges exceed the "
            "14-day candidate lag. Priors are routing assumptions, not causal effects."
        ),
        fontsize=FOOTNOTE_SIZE,
        color=MUTED_TEXT_COLOR,
        ha="left",
        va="bottom",
    )
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _network_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ROLE_COLORS["headwater"],
            markeredgecolor="white",
            markersize=9,
            label="Headwater",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ROLE_COLORS["internal"],
            markeredgecolor="white",
            markersize=9,
            label="Internal",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ROLE_COLORS["outlet"],
            markeredgecolor="white",
            markersize=9,
            label="Outlet",
        ),
        Line2D([0], [0], color=PRIOR_COLORS[0], lw=1.8, label="0-day prior"),
        Line2D([0], [0], color=PRIOR_COLORS[1], lw=1.8, label="1-day prior"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=REVIEW_COLOR,
            markersize=9,
            label="Mapping review",
        ),
    ]


def render_river_graph_figure(
    summary: Mapping[str, Any], png_path: Path, pdf_path: Path
) -> tuple[Path, Path]:
    """Render each directed river component as an intuitive upstream-flow card."""
    if summary.get("direction") != "upstream_to_downstream":
        raise ValueError("graph direction must be upstream_to_downstream")
    nodes = summary.get("nodes")
    edges = summary.get("edges")
    components = summary.get("components")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(components, list):
        raise ValueError("graph summary nodes, edges, and components are required")
    if not components:
        raise ValueError("graph summary must contain at least one component")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": LEGEND_SIZE,
            "text.color": TEXT_COLOR,
        }
    )
    if int(summary.get("node_count", 0)) > LARGE_GRAPH_NODE_THRESHOLD:
        return _render_large_graph_figure(summary, png_path, pdf_path)
    card_columns = min(CARD_COLUMNS, len(components))
    card_rows = math.ceil(len(components) / card_columns)
    figure_size = (
        FIGURE_SIZE[0]
        if card_columns == CARD_COLUMNS
        else FIGURE_WIDTH_PER_COLUMN * card_columns,
        FIGURE_SIZE[1]
        if card_rows == CARD_ROWS
        else FIGURE_HEIGHT_PER_ROW * card_rows,
    )
    figure, axes = plt.subplots(
        card_rows,
        card_columns,
        figsize=figure_size,
        squeeze=False,
    )
    for axis, component in zip(axes.flat, components, strict=False):
        _component_card(axis, component, nodes, edges)
    for axis in axes.flat[len(components) :]:
        axis.axis("off")
    figure.subplots_adjust(
        left=CARD_LEFT,
        right=CARD_RIGHT,
        top=CARD_TOP,
        bottom=CARD_BOTTOM,
        wspace=CARD_WSPACE,
        hspace=CARD_HSPACE,
    )
    figure.text(
        CARD_LEFT,
        0.958,
        "Directed monitored river network",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )
    figure.text(
        CARD_LEFT,
        0.920,
        "Each card is one disjoint component; arrows show the modelled information-flow direction.",
        fontsize=SUBTITLE_SIZE,
        ha="left",
        va="top",
        color=MUTED_TEXT_COLOR,
    )
    figure.add_artist(
        FancyArrowPatch(
            (0.735, 0.931),
            (0.890, 0.931),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.0,
            color=TEXT_COLOR,
        )
    )
    figure.text(
        0.720,
        0.931,
        "UPSTREAM",
        fontsize=DIRECTION_LABEL_SIZE,
        fontweight="bold",
        ha="right",
        va="center",
    )
    figure.text(
        0.905,
        0.931,
        "DOWNSTREAM",
        fontsize=DIRECTION_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="center",
    )
    figure.legend(
        handles=_network_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(CARD_LEFT, 0.025),
        ncol=6,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.0,
        columnspacing=1.3,
    )
    figure.text(
        CARD_LEFT,
        0.105,
        "Number inside node = mapped source stations · label below node = HydroRIVERS segment ID · * = mapping review",
        fontsize=FOOTNOTE_SIZE,
        ha="left",
        va="center",
        color=MUTED_TEXT_COLOR,
    )
    prior_counts = summary.get("rounded_prior_lag_counts")
    if not isinstance(prior_counts, Mapping):
        raise ValueError("rounded prior-lag counts are required")
    figure.text(
        CARD_RIGHT,
        0.105,
        (
            f"{summary['node_count']} segments  |  {summary['edge_count']} directed edges  |  "
            f"prior lag: {prior_counts.get('0', 0)} × 0 d, {prior_counts.get('1', 0)} × 1 d"
        ),
        fontsize=FOOTNOTE_SIZE,
        fontweight="bold",
        ha="right",
        va="center",
        color=TEXT_COLOR,
    )
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path
