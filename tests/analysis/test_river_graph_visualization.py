from pathlib import Path

import numpy as np
import polars as pl

from RiverLagNet.analysis.river_graph_visualization import (
    _large_graph_positions,
    build_river_graph_summary,
    render_river_graph_figure,
)


def _write_graph_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "dataset.npz",
        node_ids=np.array(["100", "200", "300"]),
        component_ids=np.array(["component-a", "component-a", "component-a"]),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        source_station_count=np.array([1, 2, 1], dtype=np.int64),
    )
    pl.DataFrame(
        {
            "station_id": ["100", "200", "200", "300"],
            "source_station_id": [1, 2, 3, 4],
            "station_lon": [110.0, 110.4, 110.6, 111.0],
            "station_lat": [30.0, 30.4, 30.6, 31.0],
            "mapping_quality_flag": [
                "nearest_reach_good",
                "nearest_reach_good",
                "nearest_reach_review",
                "nearest_reach_good",
            ],
        }
    ).write_parquet(root / "station_mapping.parquet")
    pl.DataFrame(
        {
            "src_station_id": ["100", "200"],
            "dst_station_id": ["200", "300"],
            "travel_time_prior_days": [0.2, 0.8],
        }
    ).write_parquet(root / "edges.parquet")


def test_graph_summary_preserves_direction_roles_and_prior_lags(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_graph_fixture(root)

    summary = build_river_graph_summary(root)

    assert summary["direction"] == "upstream_to_downstream"
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 2
    assert summary["component_count"] == 1
    assert summary["rounded_prior_lag_counts"] == {"0": 1, "1": 1}
    assert summary["prior_lag_bin_counts"] == {
        "<1 d": 2,
        "1–3 d": 0,
        "3–7 d": 0,
        "7–14 d": 0,
        ">14 d": 0,
    }
    assert [
        (edge["src_station_id"], edge["dst_station_id"])
        for edge in summary["edges"]
    ] == [("100", "200"), ("200", "300")]
    roles = {node["node_id"]: node["role"] for node in summary["nodes"]}
    assert roles == {"100": "headwater", "200": "internal", "300": "outlet"}
    review = {node["node_id"]: node["mapping_review"] for node in summary["nodes"]}
    assert review == {"100": False, "200": True, "300": False}


def test_large_graph_layout_places_every_edge_upstream_to_downstream() -> None:
    node_ids = ["1", "2", "3", "4", "5"]
    edges = [("1", "3"), ("2", "3"), ("3", "5"), ("4", "5")]

    positions = _large_graph_positions(node_ids, edges)

    assert set(positions) == set(node_ids)
    assert all(positions[source][0] < positions[destination][0] for source, destination in edges)
    assert all(0.0 < x < 1.0 and 0.0 < y < 1.0 for x, y in positions.values())


def test_graph_visualization_writes_png_and_pdf(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_graph_fixture(root)
    summary = build_river_graph_summary(root)
    png_path = tmp_path / "graph.png"
    pdf_path = tmp_path / "graph.pdf"

    rendered = render_river_graph_figure(summary, png_path, pdf_path)

    assert rendered == (png_path, pdf_path)
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.stat().st_size > 10_000
    assert pdf_path.stat().st_size > 1_000
