"""Versioned, raw-value-free fingerprints for prepared RiverLagNet datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .schema import TARGET_NAMES


@dataclass(frozen=True)
class DatasetManifest:
    """Auditable metadata that freezes one prepared NPZ dataset contract."""

    path: str
    sha256: str
    num_days: int
    num_nodes: int
    num_edges: int
    variables: tuple[str, ...]
    targets: tuple[str, ...]
    static_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    start_date: str
    end_date: str
    observed_fraction: tuple[float, ...]
    graph_direction: str
    node_order_sha256: str
    edge_index_sha256: str
    observed_mask_sha256: str


def inspect_dataset(path: Path) -> DatasetManifest:
    """Inspect a prepared NPZ without serializing any observation values."""
    source = Path(path)
    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"prepared dataset not found: {resolved}")

    required = {
        "values",
        "observed",
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
    with np.load(resolved, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prepared dataset lacks {sorted(missing)}")
        values = archive["values"]
        observed = np.asarray(archive["observed"], dtype=bool)
        static = archive["static"]
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        edge_attr = archive["edge_attr"]
        dates = archive["dates"].astype(str)
        node_ids = archive["node_ids"].astype(str)
        targets = tuple(archive["target_names"].astype(str).tolist())
        variables = tuple(archive["variable_names"].astype(str).tolist())
        static_features = tuple(archive["static_names"].astype(str).tolist())
        edge_features = tuple(archive["edge_attr_names"].astype(str).tolist())

    if values.ndim != 3:
        raise ValueError("values must have shape [T, N, V]")
    num_days, num_nodes, num_variables = values.shape
    if observed.shape != values.shape:
        raise ValueError("observed must match values shape")
    if targets != TARGET_NAMES:
        raise ValueError(f"target order must be {TARGET_NAMES}, found {targets}")
    if len(variables) != num_variables or variables[: len(targets)] != targets:
        raise ValueError("variable names must match values and begin with the targets")
    if static.shape != (num_nodes, len(static_features)):
        raise ValueError("static feature names or node dimension do not match")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    num_edges = edge_index.shape[1]
    if edge_attr.shape != (num_edges, len(edge_features)):
        raise ValueError("edge feature names or edge dimension do not match")
    if node_ids.shape != (num_nodes,) or len(set(node_ids.tolist())) != num_nodes:
        raise ValueError("node_ids must be unique and match the node dimension")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains an invalid node index")
    if dates.shape != (num_days,):
        raise ValueError("dates must match the time dimension")
    parsed_dates = dates.astype("datetime64[D]")
    if num_days < 2 or not np.all(np.diff(parsed_dates).astype(int) == 1):
        raise ValueError("dates must form one contiguous daily sequence")

    observed_fraction = tuple(float(value) for value in observed.mean(axis=(0, 1)))
    return DatasetManifest(
        path=source.as_posix(),
        sha256=_sha256_file(resolved),
        num_days=num_days,
        num_nodes=num_nodes,
        num_edges=num_edges,
        variables=variables,
        targets=targets,
        static_features=static_features,
        edge_features=edge_features,
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        observed_fraction=observed_fraction,
        graph_direction="upstream_to_downstream",
        node_order_sha256=_sha256_array(node_ids),
        edge_index_sha256=_sha256_array(edge_index),
        observed_mask_sha256=_sha256_array(observed),
    )


def verify_dataset(path: Path, expected_sha256: str) -> DatasetManifest:
    """Inspect a dataset only after its complete-file digest matches expectation."""
    resolved = Path(path).expanduser().resolve()
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    return inspect_dataset(path)


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write a deterministic UTF-8 JSON manifest with LF line endings."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True, ensure_ascii=False)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
