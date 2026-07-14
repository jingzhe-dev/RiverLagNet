from __future__ import annotations

import csv
from pathlib import Path

from RiverLagNet.analysis.contracted_training_summary import (
    EXPERIMENT_NAMES,
    MODEL_ORDER,
    build_contracted_training_summary,
    render_contracted_training_figure,
)


def test_contracted_training_summary_and_figure(tmp_path: Path) -> None:
    ledger = tmp_path / "results.tsv"
    fieldnames = [
        "timestamp",
        "commit",
        "branch",
        "experiment",
        "seed",
        "val_macro_nse",
        "val_macro_mae",
        "val_macro_rmse",
        "duration_s",
        "peak_vram_gb",
        "status",
        "description",
    ]
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index, model in enumerate(MODEL_ORDER):
            writer.writerow(
                {
                    "timestamp": f"2026-07-14T00:0{index}:00+00:00",
                    "commit": "abc123",
                    "branch": "research/test",
                    "experiment": EXPERIMENT_NAMES[model],
                    "seed": 42,
                    "val_macro_nse": 0.2 + 0.1 * index,
                    "val_macro_mae": 0.4 - 0.05 * index,
                    "val_macro_rmse": 0.7 - 0.05 * index,
                    "duration_s": 10 + index,
                    "peak_vram_gb": 1 + index,
                    "status": "keep" if model == "riverlagnet" else "baseline",
                    "description": "test fixture",
                }
            )

    summary = build_contracted_training_summary(ledger)

    assert summary["test_set_used"] is False
    assert [item["model"] for item in summary["models"]] == list(MODEL_ORDER)
    assert abs(summary["riverlagnet_nse_delta_vs_station_gru"] - 0.2) < 1e-12
    assert abs(summary["riverlagnet_nse_delta_vs_static_gat"] - 0.1) < 1e-12

    png_path = tmp_path / "metrics.png"
    pdf_path = tmp_path / "metrics.pdf"
    rendered = render_contracted_training_figure(summary, png_path, pdf_path)
    assert rendered == (png_path, pdf_path)
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")
