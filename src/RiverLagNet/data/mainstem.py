"""Create topology-selected mainstem panels from prepared river datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class MainstemPreparationSummary:
    """Auditable summary of a topology-only mainstem selection."""

    dataset_path: Path
    manifest_path: Path
    nodes_path: Path
    edges_path: Path
    num_nodes: int
    num_edges: int
    total_travel_time_days: float
    start_node_id: str
    end_node_id: str


def longest_directed_path(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    edge_attr_names: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return node and edge indices on the longest travel-time directed path.

    Selection depends only on graph topology and the predefined travel-time
    prior. Water-quality values, masks, split membership, and model outcomes are
    deliberately absent from this interface.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    names = tuple(np.asarray(edge_attr_names).astype(str).tolist())
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have shape [E,A]")
    if num_nodes < 2:
        raise ValueError("mainstem selection requires at least two nodes")
    try:
        travel_column = names.index("travel_time_prior_days")
    except ValueError as error:
        raise ValueError("edge attributes lack travel_time_prior_days") from error
    source, destination = edge_index
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains an invalid node")
    travel_time = edge_attr[:, travel_column].astype(np.float64)
    if not np.isfinite(travel_time).all() or np.any(travel_time < 0):
        raise ValueError("travel-time priors must be finite and non-negative")

    children: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    indegree = np.zeros(num_nodes, dtype=np.int64)
    for edge_id, (src, dst) in enumerate(zip(source.tolist(), destination.tolist())):
        children[src].append((dst, edge_id))
        indegree[dst] += 1
    for outgoing in children:
        outgoing.sort()
    queue = sorted(np.flatnonzero(indegree == 0).tolist(), reverse=True)
    order: list[int] = []
    while queue:
        node = queue.pop()
        order.append(node)
        for child, _ in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort(reverse=True)
    if len(order) != num_nodes:
        raise ValueError("mainstem selection requires a directed acyclic graph")

    score = np.zeros(num_nodes, dtype=np.float64)
    predecessor = np.full(num_nodes, -1, dtype=np.int64)
    predecessor_edge = np.full(num_nodes, -1, dtype=np.int64)
    for src in order:
        for dst, edge_id in children[src]:
            candidate = score[src] + travel_time[edge_id]
            if candidate > score[dst]:
                score[dst] = candidate
                predecessor[dst] = src
                predecessor_edge[dst] = edge_id
    end = int(np.argmax(score))
    path_nodes: list[int] = []
    path_edges: list[int] = []
    node = end
    while node >= 0:
        path_nodes.append(node)
        edge_id = int(predecessor_edge[node])
        if edge_id >= 0:
            path_edges.append(edge_id)
        node = int(predecessor[node])
    path_nodes.reverse()
    path_edges.reverse()
    if len(path_nodes) < 2:
        raise ValueError("graph does not contain a positive-travel-time directed path")
    return np.asarray(path_nodes, dtype=np.int64), np.asarray(path_edges, dtype=np.int64)


def prepare_mainstem_dataset(
    source_dataset: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
) -> MainstemPreparationSummary:
    """Slice a prepared NPZ to its topology-defined longest mainstem."""
    source_dataset = Path(source_dataset).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_dataset.is_file():
        raise FileNotFoundError(f"prepared source dataset not found: {source_dataset}")
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    with np.load(source_dataset, allow_pickle=False) as archive:
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
            "static_names",
            "edge_attr_names",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prepared dataset lacks {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    num_nodes = int(arrays["values"].shape[1])
    path_nodes, path_edges = longest_directed_path(
        arrays["edge_index"],
        arrays["edge_attr"],
        arrays["edge_attr_names"],
        num_nodes,
    )
    selected: dict[str, np.ndarray] = dict(arrays)
    for name in ("values", "observed", "quality"):
        selected[name] = arrays[name][:, path_nodes]
    for name in ("static", "node_ids", "component_ids", "source_station_count"):
        if name in arrays:
            selected[name] = arrays[name][path_nodes]
    selected["edge_index"] = np.vstack(
        (
            np.arange(path_nodes.size - 1, dtype=np.int64),
            np.arange(1, path_nodes.size, dtype=np.int64),
        )
    )
    selected["edge_attr"] = arrays["edge_attr"][path_edges]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.npz"
    with dataset_path.open("wb") as handle:
        np.savez_compressed(handle, **selected)

    node_ids = selected["node_ids"].astype(str)
    nodes_path = output_dir / "nodes.parquet"
    pl.DataFrame(
        {
            "mainstem_index": np.arange(path_nodes.size, dtype=np.int64),
            "source_node_index": path_nodes,
            "station_id": node_ids,
        }
    ).write_parquet(nodes_path)
    edge_names = selected["edge_attr_names"].astype(str).tolist()
    edge_frame: dict[str, object] = {
        "src_station_id": node_ids[:-1],
        "dst_station_id": node_ids[1:],
        "source_edge_index": path_edges,
    }
    for index, name in enumerate(edge_names):
        edge_frame[name] = selected["edge_attr"][:, index]
    edges_path = output_dir / "edges.parquet"
    pl.DataFrame(edge_frame).write_parquet(edges_path)

    travel_column = edge_names.index("travel_time_prior_days")
    total_travel_time = float(selected["edge_attr"][:, travel_column].sum())
    observed = selected["observed"]
    target_names = selected["target_names"].astype(str).tolist()
    source_manifest = source_dataset.parent / "manifest.json"
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_manifest_sha256": (
            _sha256(source_manifest) if source_manifest.is_file() else None
        ),
        "selection": {
            "rule": "longest directed path by cumulative travel_time_prior_days",
            "uses_water_quality_values": False,
            "uses_validation_or_test_outcomes": False,
            "tie_break": "lowest deterministic node/edge order",
        },
        "shape": list(selected["values"].shape),
        "num_nodes": int(path_nodes.size),
        "num_edges": int(path_edges.size),
        "start_node_id": str(node_ids[0]),
        "end_node_id": str(node_ids[-1]),
        "total_travel_time_days": total_travel_time,
        "edge_direction": "upstream_to_downstream",
        "target_names": target_names,
        "target_observed_rates": {
            name: float(observed[..., index].mean())
            for index, name in enumerate(target_names)
        },
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in (dataset_path, nodes_path, edges_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return MainstemPreparationSummary(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
        num_nodes=int(path_nodes.size),
        num_edges=int(path_edges.size),
        total_travel_time_days=total_travel_time,
        start_node_id=str(node_ids[0]),
        end_node_id=str(node_ids[-1]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
