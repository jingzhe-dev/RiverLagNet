from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.data.real_daily import load_real_daily_dataset, prepare_china_real_daily


def _write_real_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(20)]
    value_rows: list[dict[str, object]] = []
    flag_rows: list[dict[str, object]] = []
    for station_id in (1, 2, 3):
        for day_index, current_date in enumerate(dates):
            row = {"id": station_id, "time": str(current_date)}
            flags = {"id": station_id, "time": str(current_date)}
            for target_index, target in enumerate(("NH3N", "CODMn", "TP")):
                row[target] = float(100 * station_id + 10 * target_index + day_index)
                flags[f"{target}_is_imputed"] = int(
                    (station_id == 1 and day_index == 0)
                    or (station_id in (1, 2) and day_index == 1)
                )
            row["Temp"] = float(15 + station_id + day_index / 10)
            value_rows.append(row)
            flag_rows.append(flags)
    dynamic_path = tmp_path / "dynamic.csv"
    flags_path = tmp_path / "flags.csv"
    pl.DataFrame(value_rows).write_csv(dynamic_path)
    pl.DataFrame(flag_rows).write_csv(flags_path)

    mapping_path = tmp_path / "mapping.csv"
    pl.DataFrame(
        {
            "station_id": [1, 2, 3],
            "segment_id": [100, 100, 200],
            "mapped_flag": [True, True, True],
            "distance_km": [0.1, 0.2, 0.3],
            "mapping_quality_flag": [
                "nearest_reach_good",
                "nearest_reach_good",
                "nearest_reach_review",
            ],
        }
    ).write_csv(mapping_path)

    data_root = tmp_path / "hydrowq-v0.1"
    graph_root = data_root / "fixtures" / "river_graph"
    component = graph_root / "component-01-v0.2"
    component.mkdir(parents=True)
    np.savez_compressed(
        component / "graph.npz",
        segment_id=np.asarray(["100", "200"]),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        edge_attr=np.asarray([[0.5, 0.0, 0.0, 0.5]], dtype=np.float32),
        edge_attr_names=np.asarray(
            ["length_km", "distance_downstream_km", "stream_order", "travel_time_proxy"]
        ),
        edge_attr_mask=np.ones((1, 4), dtype=bool),
    )
    np.savez_compressed(
        component / "static.npz",
        static_node=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        static_names=np.asarray(["a", "b"]),
        static_mask=np.ones((2, 2), dtype=bool),
        segment_id=np.asarray(["100", "200"]),
    )
    normalization_path = data_root / "datasets" / "china-multibasin-v0.1"
    normalization_path.mkdir(parents=True)
    (normalization_path / "normalization-train.json").write_text(
        json.dumps(
            {
                "edge": {
                    "feature_names": [
                        "length_km",
                        "distance_downstream_km",
                        "stream_order",
                        "travel_time_proxy",
                    ],
                    "mean": [10.0, 0.0, 0.0, 10.0],
                    "std": [2.0, 1.0, 1.0, 2.0],
                }
            }
        ),
        encoding="utf-8",
    )
    return dynamic_path, flags_path, mapping_path, graph_root, tmp_path / "prepared"


def test_real_daily_preparation_excludes_imputed_values_and_preserves_direction(
    tmp_path: Path,
) -> None:
    sources = _write_real_sources(tmp_path)
    summary = prepare_china_real_daily(
        *sources,
        travel_speed_km_per_day=11.0,
        hash_sources=False,
        dataset_id="test-contracted-v0.2",
        graph_construction={"construction": "test_path_contraction"},
    )
    data = load_real_daily_dataset(summary.dataset_path)

    assert (summary.num_days, summary.num_nodes, summary.num_edges) == (20, 2, 1)
    assert summary.num_source_stations == 3
    assert data.values.shape == (20, 2, 3)
    assert data.graph.edge_index.tolist() == [[0], [1]]
    assert torch.allclose(data.graph.edge_attr[:, -1], torch.tensor([1.0]))
    assert data.values[0, 0, 0].item() == 200.0
    assert data.quality[0, 0, 0].item() == 0.5
    assert not data.observed[1, 0].any()
    assert torch.equal(data.values[1, 0], torch.zeros(3))
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "test-contracted-v0.2"
    assert manifest["graph_construction"]["construction"] == "test_path_contraction"
    with np.load(summary.dataset_path) as archive:
        assert archive["component_ids"].tolist() == ["component-01", "component-01"]

    review = pl.read_parquet(summary.observations_path).filter(
        (pl.col("date") == date(2020, 1, 2)) & (pl.col("station_id") == "100")
    )
    assert review.get_column("NH3N").null_count() == 1
    assert review.get_column("NH3N_observed").item() is False


def test_real_daily_preparation_can_append_original_dynamic_covariates(
    tmp_path: Path,
) -> None:
    sources = _write_real_sources(tmp_path)
    summary = prepare_china_real_daily(
        *sources,
        travel_speed_km_per_day=11.0,
        hash_sources=False,
        dynamic_covariates=("Temp",),
    )
    data = load_real_daily_dataset(summary.dataset_path)

    assert data.values.shape == (20, 2, 4)
    assert data.observed[..., 3].all()
    with np.load(summary.dataset_path) as archive:
        assert archive["variable_names"].tolist() == ["NH3N", "CODMn", "TP", "Temp"]
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dynamic_input_names"] == ["NH3N", "CODMn", "TP", "Temp"]
    assert manifest["dynamic_input_observed_rates"]["Temp"] == 1.0


def test_real_daily_datamodule_uses_train_only_observed_statistics(tmp_path: Path) -> None:
    sources = _write_real_sources(tmp_path)
    summary = prepare_china_real_daily(*sources, travel_speed_km_per_day=11.0, hash_sources=False)
    module = RiverDataModule(
        scenario="real_daily",
        dataset_path=str(summary.dataset_path),
        input_window=3,
        output_window=2,
        batch_size=2,
    )
    module.setup("fit")

    assert module.data is not None and module.scaler is not None
    assert (module.train_end, module.val_end) == (14, 17)
    for feature in range(3):
        values = module.data.values[: module.train_end, :, feature]
        mask = module.data.observed[: module.train_end, :, feature]
        assert torch.allclose(module.scaler.mean[feature], values[mask].mean())
    assert set(module.train_dataset.all_target_indices()).isdisjoint(
        module.val_dataset.all_target_indices()
    )
    batch = next(iter(module.train_dataloader()))
    assert batch["x"].shape == (2, 3, 2, 3)
    assert batch["y"].shape == (2, 2, 2, 3)
