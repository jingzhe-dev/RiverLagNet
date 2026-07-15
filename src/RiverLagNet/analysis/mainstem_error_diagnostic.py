"""Diagnose which validation errors can be corrected by the selected river graph."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.data.schema import TARGET_NAMES
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model
from RiverLagNet.training.metrics import masked_metric_dict, masked_nse


def _metric_dict(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in masked_metric_dict(prediction, target, mask).items()
    }


def _restore_module(
    datamodule: RiverDataModule,
    checkpoint: Path,
    *,
    graph: bool,
) -> RiverForecastModule:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    hyperparameters = payload.get("hyper_parameters")
    if not isinstance(state_dict, dict) or not isinstance(hyperparameters, dict):
        raise ValueError(f"invalid checkpoint: {checkpoint}")
    if graph:
        model = build_model(
            "riverlagnet",
            datamodule.data_spec,
            output_window=30,
            hidden_dim=64,
            target_dim=3,
            max_lag=30,
            graph_variant="directed",
            lag_mode="learned_lag",
            dropout=0.1,
            graph_seed=42,
            lag_prior_scale_days=1.0,
            lag_prior_strength=8.0,
            propagation_mode="trajectory",
            trajectory_steps=8,
        )
    else:
        model = build_model(
            "station_gru",
            datamodule.data_spec,
            output_window=30,
            hidden_dim=64,
            target_dim=3,
        )
    module = RiverForecastModule(model, **hyperparameters)
    module.load_state_dict(state_dict, strict=True)
    return module


def _paired_predictions(
    baseline: RiverForecastModule,
    graph: RiverForecastModule,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    baseline = baseline.to(device).eval()
    graph = graph.to(device).eval()
    baseline_predictions: list[Tensor] = []
    graph_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in batches:
            device_batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
                if isinstance(value, Tensor)
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                baseline_prediction = baseline(device_batch)
                graph_prediction = graph(device_batch)
            mean = baseline.target_mean.to(device)
            scale = baseline.target_scale.to(device)
            baseline_predictions.append(
                (baseline_prediction.float() * scale + mean).cpu()
            )
            graph_predictions.append((graph_prediction.float() * scale + mean).cpu())
            targets.append((device_batch["y"].float() * scale + mean).cpu())
            masks.append(device_batch["y_mask"].cpu())
    return (
        torch.cat(baseline_predictions),
        torch.cat(graph_predictions),
        torch.cat(targets),
        torch.cat(masks),
    )


def _fit_scale(
    correction: Tensor,
    residual: Tensor,
    mask: Tensor,
    reduce_dims: tuple[int, ...],
    *,
    ridge_fraction: float = 1e-4,
    clip: float = 4.0,
) -> Tensor:
    weights = mask.to(correction.dtype)
    numerator = (correction * residual * weights).sum(dim=reduce_dims, keepdim=True)
    denominator = (correction.square() * weights).sum(
        dim=reduce_dims, keepdim=True
    )
    positive = denominator[denominator > 0]
    ridge = (
        positive.median() * ridge_fraction
        if positive.numel()
        else correction.new_tensor(ridge_fraction)
    )
    scale = torch.where(
        denominator > 0,
        numerator / (denominator + ridge),
        torch.zeros_like(numerator),
    )
    return scale.clamp(-clip, clip)


def _calibration_diagnostics(
    train: tuple[Tensor, Tensor, Tensor, Tensor],
    validation: tuple[Tensor, Tensor, Tensor, Tensor],
) -> dict[str, object]:
    train_base, train_graph, train_target, train_mask = train
    val_base, val_graph, val_target, val_mask = validation
    train_correction = train_graph - train_base
    train_residual = train_target - train_base
    val_correction = val_graph - val_base
    schemes = {
        "global": (0, 1, 2, 3),
        "target": (0, 1, 2),
        "horizon_target": (0, 2),
        "node_target": (0, 1),
        "horizon_node_target": (0,),
    }
    rows: dict[str, object] = {}
    for name, reduce_dims in schemes.items():
        scale = _fit_scale(
            train_correction, train_residual, train_mask, reduce_dims
        )
        calibrated = val_base + scale * val_correction
        rows[name] = {
            "validation_metrics": _metric_dict(calibrated, val_target, val_mask),
            "scale_summary": {
                "min": float(scale.min()),
                "median": float(scale.median()),
                "max": float(scale.max()),
                "negative_fraction": float((scale < 0).float().mean()),
            },
        }
    oracle_scale = _fit_scale(
        val_correction,
        val_target - val_base,
        val_mask,
        (0,),
        ridge_fraction=0.0,
    )
    rows["same_validation_horizon_node_target_oracle"] = {
        "validation_metrics": _metric_dict(
            val_base + oracle_scale * val_correction, val_target, val_mask
        ),
        "uses_validation_targets_for_fit": True,
    }
    return rows


def _target_and_horizon_diagnostics(
    baseline: Tensor,
    graph: Tensor,
    target: Tensor,
    mask: Tensor,
) -> dict[str, object]:
    target_rows: dict[str, object] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        target_mask = mask[..., target_index : target_index + 1]
        target_values = target[..., target_index : target_index + 1]
        baseline_nse = float(
            masked_nse(
                baseline[..., target_index : target_index + 1],
                target_values,
                target_mask,
            )[0]
        )
        graph_nse = float(
            masked_nse(
                graph[..., target_index : target_index + 1],
                target_values,
                target_mask,
            )[0]
        )
        target_rows[target_name] = {
            "baseline_nse": baseline_nse,
            "graph_nse": graph_nse,
            "delta_nse": graph_nse - baseline_nse,
        }
    horizon_rows: dict[str, object] = {}
    for name, start, stop in (
        ("days_1_7", 0, 7),
        ("days_8_14", 7, 14),
        ("days_15_30", 14, 30),
    ):
        base_metrics = _metric_dict(
            baseline[:, start:stop], target[:, start:stop], mask[:, start:stop]
        )
        graph_metrics = _metric_dict(
            graph[:, start:stop], target[:, start:stop], mask[:, start:stop]
        )
        horizon_rows[name] = {
            "baseline_nse": base_metrics["macro_nse"],
            "graph_nse": graph_metrics["macro_nse"],
            "delta_nse": graph_metrics["macro_nse"] - base_metrics["macro_nse"],
        }
    return {"targets": target_rows, "horizon_bands": horizon_rows}


def _node_diagnostics(
    baseline: Tensor,
    graph: Tensor,
    target: Tensor,
    mask: Tensor,
) -> dict[str, object]:
    deltas: list[float] = []
    for node in range(baseline.shape[2]):
        base_metrics = _metric_dict(
            baseline[:, :, node : node + 1],
            target[:, :, node : node + 1],
            mask[:, :, node : node + 1],
        )
        graph_metrics = _metric_dict(
            graph[:, :, node : node + 1],
            target[:, :, node : node + 1],
            mask[:, :, node : node + 1],
        )
        deltas.append(graph_metrics["macro_nse"] - base_metrics["macro_nse"])
    values = torch.tensor(deltas)
    top = torch.argsort(values, descending=True)[:10]
    bottom = torch.argsort(values)[:10]
    return {
        "positive_nodes": int((values > 0).sum()),
        "negative_nodes": int((values < 0).sum()),
        "median_delta_nse": float(values.median()),
        "mean_delta_nse": float(values.mean()),
        "correlation_with_downstream_depth": float(
            torch.corrcoef(
                torch.stack((values, torch.arange(values.numel()).float()))
            )[0, 1]
        ),
        "top_nodes": [
            {"node_index": int(index), "delta_nse": float(values[index])}
            for index in top
        ],
        "bottom_nodes": [
            {"node_index": int(index), "delta_nse": float(values[index])}
            for index in bottom
        ],
    }


def build_diagnostic(
    dataset_path: Path,
    baseline_checkpoint: Path,
    graph_checkpoint: Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Build a validation-only graph-error diagnostic."""
    datamodule = RiverDataModule(
        scenario="real_daily",
        dataset_path=str(dataset_path),
        input_window=90,
        output_window=30,
        batch_size=16,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        seed=42,
    )
    datamodule.setup("fit")
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    baseline = _restore_module(
        datamodule, baseline_checkpoint, graph=False
    )
    graph = _restore_module(datamodule, graph_checkpoint, graph=True)
    train = _paired_predictions(
        baseline, graph, datamodule.train_dataloader(), selected_device
    )
    validation = _paired_predictions(
        baseline, graph, datamodule.val_dataloader(), selected_device
    )
    val_base, val_graph, val_target, val_mask = validation
    baseline_metrics = _metric_dict(val_base, val_target, val_mask)
    graph_metrics = _metric_dict(val_graph, val_target, val_mask)
    correction = val_graph - val_base
    residual = val_target - val_base
    valid_correction = correction[val_mask]
    valid_residual = residual[val_mask]
    correlation = float(
        torch.corrcoef(torch.stack((valid_correction, valid_residual)))[0, 1]
    )
    return {
        "split_used": ["train_fit", "validation_evaluation"],
        "held_out_test_opened": False,
        "dataset_path": str(dataset_path),
        "baseline_checkpoint": str(baseline_checkpoint),
        "graph_checkpoint": str(graph_checkpoint),
        "baseline_metrics": baseline_metrics,
        "graph_metrics": graph_metrics,
        "absolute_delta_macro_nse": graph_metrics["macro_nse"]
        - baseline_metrics["macro_nse"],
        "relative_gain_percent": 100.0
        * (graph_metrics["macro_nse"] - baseline_metrics["macro_nse"])
        / abs(baseline_metrics["macro_nse"]),
        "graph_correction_diagnostic": {
            "correlation_with_baseline_residual": correlation,
            "rms_correction": float(valid_correction.square().mean().sqrt()),
            "rms_baseline_residual": float(valid_residual.square().mean().sqrt()),
        },
        **_target_and_horizon_diagnostics(
            val_base, val_graph, val_target, val_mask
        ),
        "nodes": _node_diagnostics(val_base, val_graph, val_target, val_mask),
        "train_fitted_correction_scaling": _calibration_diagnostics(
            train, validation
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    result = build_diagnostic(
        args.dataset,
        args.baseline_checkpoint,
        args.graph_checkpoint,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
