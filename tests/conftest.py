from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def hydrowq_source(tmp_path: Path) -> Path:
    """Build a minimal HydroWQ processed-data tree for import tests."""
    processed = tmp_path / "source" / "processed"
    data_root = processed / "hydrowq-v0.1"
    dataset = data_root / "datasets" / "china-multibasin-v0.1"
    sample_dir = dataset / "samples" / "sample-001"
    graph_dir = data_root / "fixtures" / "river_graph" / "basin-001-v0.1"
    sample_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)

    variables = ["Tw", "CODMn", "NH3N", "TP"]
    x_obs = np.zeros((2, 2, 4), dtype=np.float32)
    x_obs[..., 1] = 10.0
    x_obs[..., 2] = 20.0
    x_obs[..., 3] = 30.0
    target = np.zeros((3, 2, 4), dtype=np.float32)
    target[..., 1] = 11.0
    target[..., 2] = 21.0
    target[..., 3] = 31.0
    daily_path = sample_dir / "daily.npz"
    np.savez_compressed(
        daily_path,
        x_obs=x_obs,
        mask_obs=np.ones_like(x_obs, dtype=bool),
        target=target,
        target_mask=np.ones_like(target, dtype=bool),
        quality_code=np.ones_like(x_obs, dtype=np.int64),
    )

    graph_path = graph_dir / "graph.npz"
    np.savez_compressed(
        graph_path,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_attr=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    static_path = graph_dir / "static.npz"
    np.savez_compressed(
        static_path,
        static_node=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

    daily_rel = daily_path.relative_to(data_root).as_posix()
    graph_rel = graph_path.relative_to(data_root).as_posix()
    static_rel = static_path.relative_to(data_root).as_posix()

    def asset(asset_id: str, relative_path: str, digest: str, key: str) -> dict[str, object]:
        return {
            "asset_id": asset_id,
            "format": "npz",
            "relative_path": relative_path,
            "sha256": digest,
            "metadata": {"npz_key": key},
        }

    daily_sha = _sha256(daily_path)
    manifest = {
        "manifest_id": "hydrowq-china-multibasin-v0.1",
        "layout_version": "hydrowq-sample-bundle-manifest-v0.1",
        "data_root": "hydrowq-v0.1",
        "bundles": [
            {
                "sample_id": "sample-001",
                "basin_id": "basin-001",
                "segment_count": 2,
                "graph_ref": asset("graph", graph_rel, _sha256(graph_path), "edge_index"),
                "static_ref": asset("static", static_rel, _sha256(static_path), "static_node"),
                "scales": {
                    "daily": {
                        "history_length": 2,
                        "forecast_length": 3,
                        "variables": variables,
                        "target_variables": variables,
                        "x_obs_ref": asset("x_obs", daily_rel, daily_sha, "x_obs"),
                        "mask_obs_ref": asset("mask_obs", daily_rel, daily_sha, "mask_obs"),
                        "target_ref": asset("target", daily_rel, daily_sha, "target"),
                        "target_mask_ref": asset(
                            "target_mask", daily_rel, daily_sha, "target_mask"
                        ),
                        "quality_code_ref": asset(
                            "quality_code", daily_rel, daily_sha, "quality_code"
                        ),
                    }
                },
            }
        ],
    }
    manifest_dir = processed / "_manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "sample-bundles-china-multibasin-v0.1.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for name in ("sample-index.parquet", "normalization-train.json", "build-report.json"):
        (dataset / name).write_bytes(f"fixture:{name}".encode())
    (processed / "unrelated-large-file.bin").write_bytes(b"do-not-copy")
    return processed
