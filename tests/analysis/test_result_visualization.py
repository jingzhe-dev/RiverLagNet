from pathlib import Path

from RiverLagNet.analysis.result_visualization import (
    render_experiment_summary_figure,
)


def _statistics(mean: float, std: float = 0.01) -> dict[str, float | int]:
    return {
        "count": 5,
        "mean": mean,
        "std": std,
        "min": mean - std,
        "max": mean + std,
    }


def _summary() -> dict[str, object]:
    condition_metrics = {
        name: {
            "val_macro_nse": _statistics(nse, 0.005),
            "val_macro_mae": _statistics(0.2),
            "val_macro_rmse": _statistics(0.4),
            "duration_s": _statistics(90.0),
            "peak_vram_gb": _statistics(1.0),
        }
        for name, nse in (
            ("no_lag", 0.575),
            ("fixed_lag", 0.573),
            ("learned_lag", 0.574),
        )
    }
    test_metrics = {
        "test_macro_nse": _statistics(0.72),
        "test_macro_mae": _statistics(0.19),
        "test_macro_rmse": _statistics(0.43),
        "test_nse_NH3N": _statistics(0.53),
        "test_nse_CODMn": _statistics(0.87),
        "test_nse_TP": _statistics(0.77),
    }
    return {
        "suite": "real_lag_v1",
        "report_title": "test",
        "seeds": [42, 43, 44, 45, 46],
        "commit": "abc123",
        "experiment_count": 15,
        "conditions": condition_metrics,
        "paired_deltas": {
            "no_lag": {
                "mean_delta_macro_nse": -0.001,
                "std_delta_macro_nse": 0.001,
                "wins": 1,
                "seed_deltas": {
                    "42": 0.0002,
                    "43": -0.0004,
                    "44": -0.0015,
                    "45": -0.0005,
                    "46": -0.0031,
                },
            },
            "fixed_lag": {
                "mean_delta_macro_nse": 0.0006,
                "std_delta_macro_nse": 0.002,
                "wins": 3,
                "seed_deltas": {
                    "42": -0.0002,
                    "43": 0.0017,
                    "44": 0.00004,
                    "45": 0.0036,
                    "46": -0.0021,
                },
            },
        },
        "test": {"seeds": [42, 43, 44, 45, 46], "metrics": test_metrics},
    }


def test_result_visualization_writes_valid_png_and_pdf(tmp_path: Path) -> None:
    png_path = tmp_path / "summary.png"
    pdf_path = tmp_path / "summary.pdf"

    rendered = render_experiment_summary_figure(_summary(), png_path, pdf_path)

    assert rendered == (png_path, pdf_path)
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.stat().st_size > 10_000
    assert pdf_path.stat().st_size > 1_000
