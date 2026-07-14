from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from RiverLagNet.data.contracted_graph import (
    EDGE_ATTR_NAMES,
    STATIC_NAMES,
    build_contracted_graph,
    contract_monitored_paths,
    profile_segment_observation_coverage,
    read_hydrorivers_attributes,
    write_contracted_graph_assets,
)


def _segments() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": ["1", "2", "3", "4", "10", "11"],
            "downstream_segment_id": ["2", "3", None, "3", "11", None],
            "basin_id": ["100"] * 4 + ["200"] * 2,
            "length_km": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5],
            "distance_downstream_km": [6.0, 5.0, 3.0, 7.0, 4.0, 2.5],
            "catchment_area_km2": [10.0, 20.0, 40.0, 15.0, 8.0, 16.0],
            "upstream_area_km2": [10.0, 30.0, 70.0, 15.0, 8.0, 24.0],
            "stream_order": [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        }
    )


def test_contract_monitored_paths_crosses_unmonitored_reaches() -> None:
    edges = contract_monitored_paths(_segments(), {"1", "3", "4"})

    assert [(edge.source, edge.destination) for edge in edges] == [
        ("1", "3"),
        ("4", "3"),
    ]
    assert edges[0].hop_count == 2
    assert edges[0].path_length_km == 3.0
    assert edges[1].hop_count == 1
    assert edges[1].path_length_km == 4.0


def test_build_and_write_contracted_graph_selects_trainable_component(
    tmp_path: Path,
) -> None:
    coverage = pl.DataFrame(
        {
            "segment_id": ["1", "2", "3", "4", "10", "11"],
            "source_station_count": [1, 1, 2, 1, 1, 1],
            "mapping_review": [False, False, True, False, False, False],
            "min_target_coverage": [1.0, 0.4, 0.95, 0.91, 1.0, 1.0],
        }
    )

    assets = build_contracted_graph(
        _segments(),
        coverage,
        min_target_coverage=0.9,
        min_component_nodes=3,
        max_component_nodes=3,
        component_limit=1,
        travel_speed_km_per_day=10.0,
        max_lag_days=14,
    )

    assert assets.node_ids.tolist() == ["1", "3", "4"]
    assert assets.edge_index.tolist() == [[0, 2], [1, 1]]
    assert assets.edge_attr.shape == (2, len(EDGE_ATTR_NAMES))
    assert assets.static.shape == (3, len(STATIC_NAMES))
    assert np.isfinite(assets.edge_attr).all()
    assert np.isfinite(assets.static).all()
    np.testing.assert_allclose(assets.edge_attr[:, -1], [0.3, 0.4])
    assert assets.report["selected_edge_count"] == 2
    assert assets.report["selected_segment_count"] == 3
    assert assets.report["selected_source_station_count"] == 4
    assert assets.report["selected_mapping_review_segment_count"] == 1

    report_path = write_contracted_graph_assets(assets, tmp_path / "graph")
    component = tmp_path / "graph" / "contracted-component-01-v0.2"
    with np.load(component / "graph.npz") as graph:
        assert graph["segment_id"].tolist() == ["1", "3", "4"]
        assert graph["edge_index"].tolist() == [[0, 2], [1, 1]]
        assert graph["edge_attr_names"].tolist() == list(EDGE_ATTR_NAMES)
    with np.load(component / "static.npz") as static:
        assert static["static_names"].tolist() == list(STATIC_NAMES)
    assert report_path.is_file()


def test_profile_segment_coverage_uses_original_observations(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    pl.DataFrame(
        {
            "station_id": [1, 2, 3],
            "segment_id": [10, 10, 20],
            "mapped_flag": [True, True, True],
            "mapping_quality_flag": ["ok", "review", "ok"],
        }
    ).write_csv(mapping_path)
    flags_path = tmp_path / "flags.csv"
    pl.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 3],
            "time": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
            ],
            "NH3N_is_imputed": [0, 1, 0, 1, 1, 0],
            "CODMn_is_imputed": [0, 0, 1, 0, 0, 1],
            "TP_is_imputed": [1, 1, 0, 0, 1, 0],
        }
    ).write_csv(flags_path)

    profile = profile_segment_observation_coverage(flags_path, mapping_path)
    segment_10 = profile.filter(pl.col("segment_id") == "10").row(0, named=True)
    segment_20 = profile.filter(pl.col("segment_id") == "20").row(0, named=True)

    assert segment_10["source_station_count"] == 2
    assert segment_10["mapping_review"]
    assert segment_10["NH3N_coverage"] == 0.5
    assert segment_10["CODMn_coverage"] == 1.0
    assert segment_10["TP_coverage"] == 0.5
    assert segment_10["min_target_coverage"] == 0.5
    assert segment_20["NH3N_coverage"] == 1.0
    assert segment_20["CODMn_coverage"] == 0.0
    assert segment_20["TP_coverage"] == 1.0
    assert segment_20["expected_day_count"] == 2


def test_read_hydrorivers_attributes_from_zip(tmp_path: Path) -> None:
    fields = [
        ("HYRIV_ID", "N", 12, 0),
        ("NEXT_DOWN", "N", 12, 0),
        ("HYBAS_L12", "N", 12, 0),
        ("LENGTH_KM", "N", 10, 3),
        ("DIST_DN_KM", "N", 12, 3),
        ("CATCH_SKM", "N", 12, 3),
        ("UPLAND_SKM", "N", 12, 3),
        ("ORD_STRA", "N", 4, 0),
    ]
    rows = [
        (1, 2, 100, 1.5, 4.0, 10.0, 20.0, 1),
        (2, 0, 100, 2.5, 2.5, 20.0, 40.0, 2),
    ]
    archive_path = tmp_path / "hydrorivers.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("HydroRIVERS_v10_as.dbf", _dbf_bytes(fields, rows))

    frame = read_hydrorivers_attributes(archive_path)

    assert frame.get_column("segment_id").to_list() == ["1", "2"]
    assert frame.get_column("downstream_segment_id").to_list() == ["2", None]
    assert frame.get_column("basin_id").to_list() == ["100", "100"]
    assert frame.get_column("length_km").to_list() == [1.5, 2.5]
    assert frame.get_column("stream_order").to_list() == [1.0, 2.0]


def _dbf_bytes(
    fields: list[tuple[str, str, int, int]],
    rows: list[tuple[object, ...]],
) -> bytes:
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    header = bytearray(32)
    header[0] = 3
    header[4:8] = len(rows).to_bytes(4, "little")
    header[8:10] = header_length.to_bytes(2, "little")
    header[10:12] = record_length.to_bytes(2, "little")
    descriptors = bytearray()
    for name, field_type, width, decimals in fields:
        descriptor = bytearray(32)
        encoded_name = name.encode("ascii")[:11]
        descriptor[: len(encoded_name)] = encoded_name
        descriptor[11] = ord(field_type)
        descriptor[16] = width
        descriptor[17] = decimals
        descriptors.extend(descriptor)
    records = bytearray()
    for row in rows:
        records.extend(b" ")
        for value, (_, _, width, decimals) in zip(row, fields, strict=True):
            text = f"{value:.{decimals}f}" if decimals else str(value)
            records.extend(text.rjust(width).encode("ascii"))
    return bytes(header + descriptors + b"\r" + records + b"\x1a")
