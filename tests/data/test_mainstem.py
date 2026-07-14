from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from RiverLagNet.data.mainstem import longest_directed_path, prepare_mainstem_dataset


def _source_dataset(path: Path) -> Path:
    dataset = path / "source.npz"
    edge_index = np.asarray([[0, 1, 3, 2], [1, 2, 2, 4]], dtype=np.int64)
    edge_attr = np.asarray([[1.0], [2.0], [8.0], [3.0]], dtype=np.float32)
    np.savez_compressed(
        dataset,
        values=np.arange(12 * 5 * 4, dtype=np.float32).reshape(12, 5, 4),
        observed=np.ones((12, 5, 4), dtype=bool),
        quality=np.ones((12, 5, 4), dtype=np.float32),
        static=np.arange(10, dtype=np.float32).reshape(5, 2),
        edge_index=edge_index,
        edge_attr=edge_attr,
        dates=np.asarray([f"2020-01-{day:02d}" for day in range(1, 13)]),
        node_ids=np.asarray(["a", "b", "c", "d", "e"]),
        component_ids=np.asarray(["x"] * 5),
        target_names=np.asarray(["NH3N", "CODMn", "TP"]),
        variable_names=np.asarray(["NH3N", "CODMn", "TP", "Temp"]),
        static_names=np.asarray(["s1", "s2"]),
        edge_attr_names=np.asarray(["travel_time_prior_days"]),
        source_station_count=np.ones(5, dtype=np.int64),
    )
    return dataset


def test_longest_directed_path_uses_travel_time_not_water_quality(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path)
    with np.load(source, allow_pickle=False) as archive:
        nodes, edges = longest_directed_path(
            archive["edge_index"],
            archive["edge_attr"],
            archive["edge_attr_names"],
            num_nodes=5,
        )
    assert nodes.tolist() == [3, 2, 4]
    assert edges.tolist() == [2, 3]


def test_prepare_mainstem_remaps_chain_and_records_selection(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path)
    summary = prepare_mainstem_dataset(
        source, tmp_path / "mainstem", dataset_id="test-mainstem-v0.1"
    )
    assert (summary.num_nodes, summary.num_edges) == (3, 2)
    assert summary.total_travel_time_days == 11.0
    assert (summary.start_node_id, summary.end_node_id) == ("d", "e")
    with np.load(summary.dataset_path, allow_pickle=False) as archive:
        assert archive["values"].shape == (12, 3, 4)
        assert archive["node_ids"].tolist() == ["d", "c", "e"]
        assert archive["edge_index"].tolist() == [[0, 1], [1, 2]]
        assert archive["edge_attr"].ravel().tolist() == [8.0, 3.0]
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["uses_water_quality_values"] is False
    assert manifest["selection"]["uses_validation_or_test_outcomes"] is False
    assert manifest["num_nodes"] == 3
