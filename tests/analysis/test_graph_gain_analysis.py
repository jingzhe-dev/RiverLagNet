from __future__ import annotations

from pathlib import Path

from RiverLagNet.analysis.graph_gain_analysis import (
    _summary,
    render_graph_gain_figure,
    write_graph_gain_outputs,
)


def _payload() -> dict[str, object]:
    seeds = []
    for seed, local, graph in ((42, 0.50, 0.51), (43, 0.52, 0.525)):
        seeds.append(
            {
                "seed": seed,
                "no_graph": {"macro_nse": local},
                "static_gat": {"macro_nse": graph - 0.003},
                "uncalibrated_directed_graph": {"macro_nse": graph - 0.001},
                "directed_graph": {"macro_nse": graph},
                "learned_lag": {"macro_nse": graph + 0.00001},
            }
        )
    return {
        "test_split_used": False,
        "seeds": seeds,
        "paired_global_delta": {"macro_nse": _summary([0.01, 0.005])},
        "paired_horizon_calibration_delta": {
            "macro_nse": _summary([0.001, 0.001])
        },
        "paired_lag_refinement_delta": {
            "macro_nse": _summary([0.00001, 0.00001])
        },
        "paired_selected_model_delta": {
            "macro_nse": _summary([0.01001, 0.00501])
        },
        "paired_static_gat_delta": {
            "macro_nse": _summary([0.00301, 0.00301])
        },
        "downstream_target_delta_nse": {
            name: _summary(values)
            for name, values in {
                "NH3N": [0.02, 0.01],
                "CODMn": [0.01, 0.02],
                "TP": [-0.01, 0.005],
            }.items()
        },
        "downstream_horizon_delta_macro_nse": {
            name: _summary(values)
            for name, values in {
                "days_1_7": [0.0, 0.005],
                "days_8_14": [0.01, 0.015],
                "days_15_30": [0.02, 0.025],
            }.items()
        },
        "node_delta_macro_nse": [
            {
                "node_id": "a",
                "node_index": 0,
                "has_upstream": False,
                "mean_delta_macro_nse": 0.0,
                "seed_deltas": {"42": 0.0, "43": 0.0},
            },
            {
                "node_id": "b",
                "node_index": 1,
                "has_upstream": True,
                "mean_delta_macro_nse": 0.02,
                "seed_deltas": {"42": 0.01, "43": 0.03},
            },
            {
                "node_id": "c",
                "node_index": 2,
                "has_upstream": True,
                "mean_delta_macro_nse": -0.01,
                "seed_deltas": {"42": -0.02, "43": 0.0},
            },
        ],
    }


def _graph_payload() -> dict[str, object]:
    return {
        "nodes": [
            {"node_id": "a", "role": "headwater"},
            {"node_id": "b", "role": "internal"},
            {"node_id": "c", "role": "outlet"},
        ],
        "edges": [
            {"src_station_id": "a", "dst_station_id": "b"},
            {"src_station_id": "b", "dst_station_id": "c"},
        ],
    }


def test_summary_reports_sample_spread_and_positive_count() -> None:
    result = _summary([0.01, 0.02, -0.005])
    assert result["n"] == 3
    assert result["positive_count"] == 2
    assert result["min"] == -0.005
    assert float(result["sample_sd"]) > 0.0


def test_graph_gain_outputs_and_figure_render(tmp_path: Path) -> None:
    summary = _payload()
    json_path, csv_path = write_graph_gain_outputs(
        summary, tmp_path / "summary.json", tmp_path / "nodes.csv"
    )
    png_path, pdf_path = render_graph_gain_figure(
        summary,
        _graph_payload(),
        tmp_path / "figure.png",
        tmp_path / "figure.pdf",
    )
    assert json_path.stat().st_size > 100
    assert csv_path.read_text(encoding="utf-8").count("\n") == 4
    assert b"\r" not in json_path.read_bytes()
    assert b"\r" not in csv_path.read_bytes()
    assert png_path.stat().st_size > 1_000
    assert pdf_path.stat().st_size > 1_000
