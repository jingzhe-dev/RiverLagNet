from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from RiverLagNet.data.manifest import inspect_dataset, verify_dataset, write_manifest


def _write_tiny_dataset(path: Path) -> None:
    values = np.arange(32, dtype=np.float32).reshape(4, 2, 4)
    observed = np.ones_like(values, dtype=bool)
    observed[0, 0, 0] = False
    observed[:, 1, 3] = False
    np.savez_compressed(
        path,
        values=values,
        observed=observed,
        quality=observed.astype(np.float32),
        static=np.ones((2, 2), dtype=np.float32),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        edge_attr=np.asarray([[1.5, 2.0]], dtype=np.float32),
        dates=np.asarray(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        node_ids=np.asarray(["upstream", "downstream"]),
        target_names=np.asarray(["NH3N", "CODMn", "TP"]),
        variable_names=np.asarray(["NH3N", "CODMn", "TP", "flow"]),
        static_names=np.asarray(["elevation", "area"]),
        edge_attr_names=np.asarray(["distance_km", "travel_time_prior_days"]),
    )


def test_manifest_fingerprints_dataset_contract_without_raw_values(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    output_path = tmp_path / "manifest.json"
    _write_tiny_dataset(dataset_path)

    manifest = inspect_dataset(dataset_path)
    write_manifest(manifest, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest.sha256 == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert (manifest.num_days, manifest.num_nodes, manifest.num_edges) == (4, 2, 1)
    assert manifest.variables == ("NH3N", "CODMn", "TP", "flow")
    assert manifest.targets == ("NH3N", "CODMn", "TP")
    assert manifest.static_features == ("elevation", "area")
    assert manifest.edge_features == ("distance_km", "travel_time_prior_days")
    assert (manifest.start_date, manifest.end_date) == ("2020-01-01", "2020-01-04")
    assert manifest.observed_fraction == pytest.approx((0.875, 1.0, 1.0, 0.5))
    assert manifest.graph_direction == "upstream_to_downstream"
    assert payload["sha256"] == manifest.sha256
    assert "values" not in payload
    assert output_path.read_bytes().endswith(b"\n")


def test_manifest_verification_rejects_changed_file_hash(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    _write_tiny_dataset(dataset_path)
    expected_sha256 = inspect_dataset(dataset_path).sha256
    dataset_path.write_bytes(dataset_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_dataset(dataset_path, expected_sha256)


def test_manifest_rejects_wrong_target_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    _write_tiny_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["target_names"] = np.asarray(["CODMn", "NH3N", "TP"])
    np.savez_compressed(dataset_path, **arrays)

    with pytest.raises(ValueError, match="target order"):
        inspect_dataset(dataset_path)
