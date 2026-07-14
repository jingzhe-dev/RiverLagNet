"""Paired validation audit for attributable upstream river-network gains."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from torch import Tensor

from RiverLagNet.data.datamodule import RiverDataModule
from RiverLagNet.data.schema import TARGET_NAMES
from RiverLagNet.training.lightning_module import RiverForecastModule, build_model
from RiverLagNet.training.metrics import masked_metric_dict

from .river_graph_visualization import _large_graph_positions


RUN_PAIRS: dict[int, tuple[str, str]] = {
    42: (
        "real_contracted_nseaux_v4_s42_no_graph_w010",
        "real_contracted_residual_v5_s42_no_lag",
    ),
    **{
        seed: (
            f"real_contracted_multiseed_v9_s{seed}_no_graph",
            f"real_contracted_multiseed_v9_s{seed}_no_lag",
        )
        for seed in range(43, 47)
    },
}
HORIZON_RUNS = {
    seed: f"real_contracted_horizon_v10_s{seed}_linear_lr020"
    for seed in range(42, 47)
}
LAG_RUNS = {
    seed: f"real_contracted_lagrefine_v11_s{seed}" for seed in range(42, 47)
}
STATIC_GAT_RUNS = {
    42: "real_contracted_nseaux_v4_s42_static_gat_w010",
    **{
        seed: f"real_contracted_baseline_v12_s{seed}_static_gat"
        for seed in range(43, 47)
    },
}
HORIZON_BANDS = {"days_1_7": (0, 7), "days_8_14": (7, 14), "days_15_30": (14, 30)}

FIGURE_SIZE = (17.2, 6.0)
EXPORT_DPI = 300
TEXT_COLOR = "#24292D"
MUTED_COLOR = "#67727A"
GRID_COLOR = "#DDE3E6"
NO_GRAPH_COLOR = "#9AA7AE"
STATIC_GAT_COLOR = "#C18451"
RAW_GRAPH_COLOR = "#7196A5"
GRAPH_COLOR = "#2F7183"
LAG_COLOR = "#80678F"


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return stable descriptive statistics for one paired metric."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("summary values must include at least one finite number")
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "positive_count": sum(value > 0.0 for value in finite),
    }


def _metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    values = masked_metric_dict(prediction, target, mask)
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _best_checkpoint(run_root: Path, run_name: str) -> Path:
    checkpoints = sorted((Path(run_root) / run_name / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one best checkpoint for {run_name}")
    return checkpoints[0]


def _build_module(
    datamodule: RiverDataModule,
    checkpoint: Path,
    *,
    graph_variant: str,
    seed: int,
    lag_mode: str = "no_lag",
    lag_bias_mode: str = "none",
    lag_residual_max_mix: float = 1.0,
    horizon_gate_mode: str = "none",
) -> RiverForecastModule:
    model = build_model(
        "riverlagnet",
        datamodule.data_spec,
        output_window=30,
        hidden_dim=64,
        target_dim=3,
        max_lag=14,
        graph_variant=graph_variant,
        lag_mode=lag_mode,
        dropout=0.1,
        graph_seed=seed,
        lag_prior_scale_days=1.0,
        lag_prior_strength=8.0,
        lag_residual_max_mix=lag_residual_max_mix,
        lag_bias_mode=lag_bias_mode,
        horizon_gate_mode=horizon_gate_mode,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"invalid Lightning checkpoint: {checkpoint}")
    module = RiverForecastModule(model, **hyperparameters)
    incompatible = module.load_state_dict(state_dict, strict=False)
    allowed_missing = {"model.message_passing.lag_residual_scale"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "validation checkpoint is incompatible: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return module


def _build_static_gat_module(
    datamodule: RiverDataModule, checkpoint: Path
) -> RiverForecastModule:
    """Restore the independently trained Static Directed GAT baseline."""
    model = build_model(
        "static_gat",
        datamodule.data_spec,
        output_window=30,
        hidden_dim=64,
        target_dim=3,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"invalid Lightning checkpoint: {checkpoint}")
    module = RiverForecastModule(model, **hyperparameters)
    module.load_state_dict(state_dict, strict=True)
    return module


def _horizon_gate_parameters(checkpoint: Path) -> dict[str, object]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    offset = float(state["model.horizon_gate.offset"])
    slope = float(state["model.horizon_gate.slope"])
    normalized_lead = state["model.horizon_gate.normalized_lead"]
    scales = 2.0 * torch.sigmoid(offset + slope * normalized_lead)
    return {
        "offset": offset,
        "slope": slope,
        "scales": [float(value) for value in scales],
    }


def _lag_refinement_parameters(
    checkpoint: Path, max_mix: float = 0.1
) -> dict[str, object]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    raw_mix = float(state["model.message_passing.lag_residual_scale"])
    relative_bias = state["model.message_passing.lag_offset_bias"].to(torch.float32)
    biases = torch.cat((torch.zeros(1), relative_bias))
    return {
        "raw_mix": raw_mix,
        "effective_mix": max_mix * math.tanh(raw_mix),
        "lag_biases": [float(value) for value in biases],
        "peak_global_bias_lag": int(torch.argmax(biases)),
    }


def _validation_predictions(
    module: RiverForecastModule,
    datamodule: RiverDataModule,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    module = module.to(device).eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in datamodule.val_dataloader():
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
                prediction = module(device_batch)
            mean = module.target_mean.to(device)
            scale = module.target_scale.to(device)
            predictions.append((prediction.float() * scale + mean).cpu())
            targets.append((device_batch["y"].float() * scale + mean).cpu())
            masks.append(device_batch["y_mask"].cpu())
    del module
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def _ledger_metrics(ledger_path: Path, experiments: set[str]) -> dict[str, dict[str, float]]:
    with Path(ledger_path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        selected = {
            row["experiment"]: {
                "macro_nse": float(row["val_macro_nse"]),
                "macro_mae": float(row["val_macro_mae"]),
                "macro_rmse": float(row["val_macro_rmse"]),
            }
            for row in rows
            if row["experiment"] in experiments and row["status"] != "crash"
        }
    missing = experiments.difference(selected)
    if missing:
        raise ValueError(f"experiment ledger is missing runs: {sorted(missing)}")
    return selected


def _assert_ledger_match(
    calculated: Mapping[str, float], recorded: Mapping[str, float], *, tolerance: float = 5e-5
) -> None:
    for metric in ("macro_nse", "macro_mae", "macro_rmse"):
        if abs(float(calculated[metric]) - float(recorded[metric])) > tolerance:
            raise ValueError(f"checkpoint {metric} does not match the experiment ledger")


def _subset_metrics(
    prediction: Tensor, target: Tensor, mask: Tensor, node_indices: Tensor
) -> dict[str, float]:
    return _metrics(
        prediction.index_select(2, node_indices),
        target.index_select(2, node_indices),
        mask.index_select(2, node_indices),
    )


def build_graph_gain_summary(
    dataset_path: Path,
    run_root: Path,
    ledger_path: Path,
    *,
    device: str | None = None,
    run_pairs: Mapping[int, tuple[str, str]] = RUN_PAIRS,
    horizon_runs: Mapping[int, str] = HORIZON_RUNS,
    lag_runs: Mapping[int, str] = LAG_RUNS,
    static_gat_runs: Mapping[int, str] = STATIC_GAT_RUNS,
) -> dict[str, object]:
    """Audit paired validation predictions without touching the held-out test split."""
    if not run_pairs:
        raise ValueError("at least one paired seed is required")
    stage_seed_sets = (set(horizon_runs), set(lag_runs), set(static_gat_runs))
    if any(set(run_pairs) != seed_set for seed_set in stage_seed_sets):
        raise ValueError("all compared models must contain the same seeds")
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    datamodule = RiverDataModule(
        scenario="real_daily",
        dataset_path=str(dataset_path),
        input_window=90,
        output_window=30,
        batch_size=16,
        num_workers=0,
        pin_memory=selected_device.type == "cuda",
        seed=min(run_pairs),
    )
    datamodule.setup("fit")
    assert datamodule.data is not None
    edge_index = datamodule.data.graph.edge_index
    downstream_indices = edge_index[1].unique(sorted=True)
    all_indices = torch.arange(datamodule.data_spec.num_nodes)
    downstream_set = set(downstream_indices.tolist())
    headwater_indices = torch.tensor(
        [index for index in all_indices.tolist() if index not in downstream_set],
        dtype=torch.long,
    )
    with np.load(dataset_path, allow_pickle=False) as archive:
        node_ids = [str(value) for value in archive["node_ids"].tolist()]
    if len(node_ids) != datamodule.data_spec.num_nodes:
        raise ValueError("dataset node IDs do not match the tensor node count")

    experiment_names = (
        {name for pair in run_pairs.values() for name in pair}
        | set(horizon_runs.values())
        | set(lag_runs.values())
        | set(static_gat_runs.values())
    )
    ledger = _ledger_metrics(ledger_path, experiment_names)
    seed_rows: list[dict[str, object]] = []
    downstream_target_deltas = {target: [] for target in TARGET_NAMES}
    horizon_deltas = {band: [] for band in HORIZON_BANDS}
    lag_target_deltas = {target: [] for target in TARGET_NAMES}
    lag_horizon_deltas = {band: [] for band in HORIZON_BANDS}
    node_deltas: list[list[float]] = [[] for _ in node_ids]
    headwater_max_differences: list[float] = []

    reference_target: Tensor | None = None
    reference_mask: Tensor | None = None
    for seed, (no_graph_run, graph_run) in sorted(run_pairs.items()):
        horizon_run = horizon_runs[seed]
        lag_run = lag_runs[seed]
        static_gat_run = static_gat_runs[seed]
        no_graph = _validation_predictions(
            _build_module(
                datamodule,
                _best_checkpoint(run_root, no_graph_run),
                graph_variant="no_graph",
                seed=seed,
            ),
            datamodule,
            selected_device,
        )
        graph = _validation_predictions(
            _build_module(
                datamodule,
                _best_checkpoint(run_root, graph_run),
                graph_variant="directed",
                seed=seed,
            ),
            datamodule,
            selected_device,
        )
        horizon_checkpoint = _best_checkpoint(run_root, horizon_run)
        horizon_graph = _validation_predictions(
            _build_module(
                datamodule,
                horizon_checkpoint,
                graph_variant="directed",
                seed=seed,
                horizon_gate_mode="linear",
            ),
            datamodule,
            selected_device,
        )
        lag_checkpoint = _best_checkpoint(run_root, lag_run)
        lag_graph = _validation_predictions(
            _build_module(
                datamodule,
                lag_checkpoint,
                graph_variant="directed",
                seed=seed,
                lag_mode="learned_lag",
                lag_bias_mode="global",
                lag_residual_max_mix=0.1,
                horizon_gate_mode="linear",
            ),
            datamodule,
            selected_device,
        )
        static_gat = _validation_predictions(
            _build_static_gat_module(
                datamodule,
                _best_checkpoint(run_root, static_gat_run),
            ),
            datamodule,
            selected_device,
        )
        no_prediction, target, mask = no_graph
        raw_graph_prediction, graph_target, graph_mask = graph
        graph_prediction, horizon_target, horizon_mask = horizon_graph
        lag_prediction, lag_target, lag_mask = lag_graph
        static_prediction, static_target, static_mask = static_gat
        paired_targets = (
            (graph_target, graph_mask),
            (horizon_target, horizon_mask),
            (lag_target, lag_mask),
            (static_target, static_mask),
        )
        if any(
            not torch.equal(target, paired_target)
            or not torch.equal(mask, paired_mask)
            for paired_target, paired_mask in paired_targets
        ):
            raise ValueError("paired checkpoints must use identical validation targets")
        if reference_target is None:
            reference_target, reference_mask = target, mask
        elif not torch.equal(target, reference_target) or not torch.equal(mask, reference_mask):
            raise ValueError("all seeds must use identical real validation targets")

        no_global = _metrics(no_prediction, target, mask)
        raw_graph_global = _metrics(raw_graph_prediction, target, mask)
        graph_global = _metrics(graph_prediction, target, mask)
        lag_global = _metrics(lag_prediction, target, mask)
        static_global = _metrics(static_prediction, target, mask)
        _assert_ledger_match(no_global, ledger[no_graph_run])
        _assert_ledger_match(raw_graph_global, ledger[graph_run])
        _assert_ledger_match(graph_global, ledger[horizon_run])
        _assert_ledger_match(lag_global, ledger[lag_run])
        _assert_ledger_match(static_global, ledger[static_gat_run])
        no_downstream = _subset_metrics(
            no_prediction, target, mask, downstream_indices
        )
        graph_downstream = _subset_metrics(
            graph_prediction, target, mask, downstream_indices
        )
        lag_downstream = _subset_metrics(
            lag_prediction, target, mask, downstream_indices
        )
        no_headwater = _subset_metrics(no_prediction, target, mask, headwater_indices)
        graph_headwater = _subset_metrics(
            graph_prediction, target, mask, headwater_indices
        )
        headwater_max_differences.append(
            float(
                (
                    graph_prediction.index_select(2, headwater_indices)
                    - no_prediction.index_select(2, headwater_indices)
                )
                .abs()
                .max()
            )
        )

        target_delta: dict[str, float] = {}
        lag_target_delta: dict[str, float] = {}
        for target_name in TARGET_NAMES:
            delta = (
                graph_downstream[f"nse_{target_name}"]
                - no_downstream[f"nse_{target_name}"]
            )
            target_delta[target_name] = delta
            downstream_target_deltas[target_name].append(delta)
            lag_delta = (
                lag_downstream[f"nse_{target_name}"]
                - graph_downstream[f"nse_{target_name}"]
            )
            lag_target_delta[target_name] = lag_delta
            lag_target_deltas[target_name].append(lag_delta)
        band_delta: dict[str, float] = {}
        lag_band_delta: dict[str, float] = {}
        for band, (start, stop) in HORIZON_BANDS.items():
            no_band = _subset_metrics(
                no_prediction[:, start:stop],
                target[:, start:stop],
                mask[:, start:stop],
                downstream_indices,
            )
            graph_band = _subset_metrics(
                graph_prediction[:, start:stop],
                target[:, start:stop],
                mask[:, start:stop],
                downstream_indices,
            )
            delta = graph_band["macro_nse"] - no_band["macro_nse"]
            band_delta[band] = delta
            horizon_deltas[band].append(delta)
            lag_band = _subset_metrics(
                lag_prediction[:, start:stop],
                target[:, start:stop],
                mask[:, start:stop],
                downstream_indices,
            )
            lag_delta = lag_band["macro_nse"] - graph_band["macro_nse"]
            lag_band_delta[band] = lag_delta
            lag_horizon_deltas[band].append(lag_delta)
        for node_index in all_indices.tolist():
            node_tensor = torch.tensor([node_index])
            no_node = _subset_metrics(no_prediction, target, mask, node_tensor)
            graph_node = _subset_metrics(graph_prediction, target, mask, node_tensor)
            node_deltas[node_index].append(
                graph_node["macro_nse"] - no_node["macro_nse"]
            )

        seed_rows.append(
            {
                "seed": seed,
                "no_graph_run": no_graph_run,
                "uncalibrated_directed_graph_run": graph_run,
                "directed_graph_run": horizon_run,
                "learned_lag_run": lag_run,
                "static_gat_run": static_gat_run,
                "no_graph": no_global,
                "uncalibrated_directed_graph": raw_graph_global,
                "directed_graph": graph_global,
                "learned_lag": lag_global,
                "static_gat": static_global,
                "global_delta": {
                    name: graph_global[name] - no_global[name]
                    for name in ("macro_nse", "macro_mae", "macro_rmse")
                },
                "horizon_calibration_delta": {
                    name: graph_global[name] - raw_graph_global[name]
                    for name in ("macro_nse", "macro_mae", "macro_rmse")
                },
                "lag_refinement_delta": {
                    name: lag_global[name] - graph_global[name]
                    for name in ("macro_nse", "macro_mae", "macro_rmse")
                },
                "selected_model_delta": {
                    name: lag_global[name] - no_global[name]
                    for name in ("macro_nse", "macro_mae", "macro_rmse")
                },
                "static_gat_delta": {
                    name: lag_global[name] - static_global[name]
                    for name in ("macro_nse", "macro_mae", "macro_rmse")
                },
                "horizon_gate_parameters": _horizon_gate_parameters(
                    horizon_checkpoint
                ),
                "lag_refinement_parameters": _lag_refinement_parameters(
                    lag_checkpoint
                ),
                "downstream_delta_macro_nse": (
                    graph_downstream["macro_nse"] - no_downstream["macro_nse"]
                ),
                "headwater_delta_macro_nse": (
                    graph_headwater["macro_nse"] - no_headwater["macro_nse"]
                ),
                "downstream_target_delta_nse": target_delta,
                "downstream_horizon_delta_macro_nse": band_delta,
                "lag_downstream_target_delta_nse": lag_target_delta,
                "lag_downstream_horizon_delta_macro_nse": lag_band_delta,
            }
        )

    global_nse_deltas = [
        float(row["global_delta"]["macro_nse"]) for row in seed_rows  # type: ignore[index]
    ]
    global_mae_deltas = [
        float(row["global_delta"]["macro_mae"]) for row in seed_rows  # type: ignore[index]
    ]
    global_rmse_deltas = [
        float(row["global_delta"]["macro_rmse"]) for row in seed_rows  # type: ignore[index]
    ]
    horizon_nse_deltas = [
        float(row["horizon_calibration_delta"]["macro_nse"])  # type: ignore[index]
        for row in seed_rows
    ]
    horizon_mae_deltas = [
        float(row["horizon_calibration_delta"]["macro_mae"])  # type: ignore[index]
        for row in seed_rows
    ]
    horizon_rmse_deltas = [
        float(row["horizon_calibration_delta"]["macro_rmse"])  # type: ignore[index]
        for row in seed_rows
    ]
    lag_nse_deltas = [
        float(row["lag_refinement_delta"]["macro_nse"])  # type: ignore[index]
        for row in seed_rows
    ]
    lag_mae_deltas = [
        float(row["lag_refinement_delta"]["macro_mae"])  # type: ignore[index]
        for row in seed_rows
    ]
    lag_rmse_deltas = [
        float(row["lag_refinement_delta"]["macro_rmse"])  # type: ignore[index]
        for row in seed_rows
    ]
    selected_nse_deltas = [
        float(row["selected_model_delta"]["macro_nse"])  # type: ignore[index]
        for row in seed_rows
    ]
    selected_mae_deltas = [
        float(row["selected_model_delta"]["macro_mae"])  # type: ignore[index]
        for row in seed_rows
    ]
    selected_rmse_deltas = [
        float(row["selected_model_delta"]["macro_rmse"])  # type: ignore[index]
        for row in seed_rows
    ]
    static_nse_deltas = [
        float(row["static_gat_delta"]["macro_nse"])  # type: ignore[index]
        for row in seed_rows
    ]
    static_mae_deltas = [
        float(row["static_gat_delta"]["macro_mae"])  # type: ignore[index]
        for row in seed_rows
    ]
    static_rmse_deltas = [
        float(row["static_gat_delta"]["macro_rmse"])  # type: ignore[index]
        for row in seed_rows
    ]
    downstream_deltas = [float(row["downstream_delta_macro_nse"]) for row in seed_rows]
    node_rows = [
        {
            "node_id": node_id,
            "node_index": index,
            "has_upstream": index in downstream_set,
            "mean_delta_macro_nse": statistics.fmean(node_deltas[index]),
            "seed_deltas": {
                str(seed): delta
                for seed, delta in zip(sorted(run_pairs), node_deltas[index], strict=True)
            },
        }
        for index, node_id in enumerate(node_ids)
    ]
    return {
        "analysis": "attributable_directed_upstream_graph_gain",
        "split": "validation",
        "test_split_used": False,
        "dataset_path": str(dataset_path).replace("\\", "/"),
        "seed_count": len(seed_rows),
        "node_count": len(node_ids),
        "edge_count": int(edge_index.shape[1]),
        "upstream_eligible_node_count": int(downstream_indices.numel()),
        "headwater_node_count": int(headwater_indices.numel()),
        "seeds": seed_rows,
        "paired_global_delta": {
            "macro_nse": _summary(global_nse_deltas),
            "macro_mae": _summary(global_mae_deltas),
            "macro_rmse": _summary(global_rmse_deltas),
        },
        "paired_horizon_calibration_delta": {
            "macro_nse": _summary(horizon_nse_deltas),
            "macro_mae": _summary(horizon_mae_deltas),
            "macro_rmse": _summary(horizon_rmse_deltas),
        },
        "paired_lag_refinement_delta": {
            "macro_nse": _summary(lag_nse_deltas),
            "macro_mae": _summary(lag_mae_deltas),
            "macro_rmse": _summary(lag_rmse_deltas),
        },
        "paired_selected_model_delta": {
            "macro_nse": _summary(selected_nse_deltas),
            "macro_mae": _summary(selected_mae_deltas),
            "macro_rmse": _summary(selected_rmse_deltas),
        },
        "paired_static_gat_delta": {
            "macro_nse": _summary(static_nse_deltas),
            "macro_mae": _summary(static_mae_deltas),
            "macro_rmse": _summary(static_rmse_deltas),
        },
        "downstream_delta_macro_nse": _summary(downstream_deltas),
        "downstream_target_delta_nse": {
            target: _summary(values)
            for target, values in downstream_target_deltas.items()
        },
        "downstream_horizon_delta_macro_nse": {
            band: _summary(values) for band, values in horizon_deltas.items()
        },
        "lag_downstream_target_delta_nse": {
            target: _summary(values) for target, values in lag_target_deltas.items()
        },
        "lag_downstream_horizon_delta_macro_nse": {
            band: _summary(values) for band, values in lag_horizon_deltas.items()
        },
        "headwater_invariance": {
            "max_abs_prediction_difference": max(headwater_max_differences),
            "all_seed_differences": headwater_max_differences,
        },
        "node_delta_macro_nse": node_rows,
    }


def write_graph_gain_outputs(
    summary: Mapping[str, object], json_path: Path, node_csv_path: Path
) -> tuple[Path, Path]:
    """Write the audited summary and node-level table."""
    json_path = Path(json_path)
    node_csv_path = Path(node_csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    node_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    rows = summary.get("node_delta_macro_nse")
    if not isinstance(rows, list):
        raise ValueError("summary is missing node-level graph gains")
    seed_names = sorted(
        {seed for row in rows for seed in row.get("seed_deltas", {})}, key=int
    )
    with node_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["node_id", "node_index", "has_upstream", "mean_delta_macro_nse"]
            + [f"seed_{seed}_delta_macro_nse" for seed in seed_names]
        )
        for row in rows:
            writer.writerow(
                [
                    row["node_id"],
                    row["node_index"],
                    row["has_upstream"],
                    row["mean_delta_macro_nse"],
                    *[row["seed_deltas"][seed] for seed in seed_names],
                ]
            )
    return json_path, node_csv_path


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB3B8")
    axis.tick_params(labelsize=8.5, colors=TEXT_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def _paired_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    rows = list(summary["seeds"])
    for row in rows:
        local = row["no_graph"]["macro_nse"]
        static = row["static_gat"]["macro_nse"]
        river_values = [
            local,
            row["uncalibrated_directed_graph"]["macro_nse"],
            row["directed_graph"]["macro_nse"],
            row["learned_lag"]["macro_nse"],
        ]
        axis.plot(
            [0, 1],
            [local, static],
            color="#C8A98E",
            linewidth=0.9,
            alpha=0.65,
            zorder=1,
        )
        axis.plot(
            [0, 2, 3, 4],
            river_values,
            color="#8DA5AE",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        axis.scatter(
            range(5),
            [local, static, *river_values[1:]],
            color=[
                NO_GRAPH_COLOR,
                STATIC_GAT_COLOR,
                RAW_GRAPH_COLOR,
                GRAPH_COLOR,
                LAG_COLOR,
            ],
            s=31,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        axis.text(4.07, river_values[3], str(row["seed"]), fontsize=7.5, va="center")
    no_values = [float(row["no_graph"]["macro_nse"]) for row in rows]
    static_values = [float(row["static_gat"]["macro_nse"]) for row in rows]
    raw_values = [
        float(row["uncalibrated_directed_graph"]["macro_nse"]) for row in rows
    ]
    graph_values = [float(row["directed_graph"]["macro_nse"]) for row in rows]
    lag_values = [float(row["learned_lag"]["macro_nse"]) for row in rows]
    axis.plot(
        [0, 1],
        [statistics.fmean(no_values), statistics.fmean(static_values)],
        color=STATIC_GAT_COLOR,
        linewidth=1.8,
        marker="s",
        markersize=5.2,
        zorder=3,
    )
    axis.plot(
        [0, 2, 3, 4],
        [
            statistics.fmean(no_values),
            statistics.fmean(raw_values),
            statistics.fmean(graph_values),
            statistics.fmean(lag_values),
        ],
        color=TEXT_COLOR,
        linewidth=2.2,
        marker="D",
        markersize=5.5,
        zorder=3,
    )
    selected_delta = summary["paired_selected_model_delta"]["macro_nse"]
    static_delta = summary["paired_static_gat_delta"]["macro_nse"]
    horizon_delta = summary["paired_horizon_calibration_delta"]["macro_nse"]
    lag_delta = summary["paired_lag_refinement_delta"]["macro_nse"]
    axis.text(
        0.03,
        0.97,
        (
            f"Best vs local   {float(selected_delta['mean']):+.6f}"
            f" ({int(selected_delta['positive_count'])}/{int(selected_delta['n'])})\n"
            f"Best vs Static {float(static_delta['mean']):+.6f}"
            f" ({int(static_delta['positive_count'])}/{int(static_delta['n'])})\n"
            f"Horizon gate   {float(horizon_delta['mean']):+.6f}"
            f" ({int(horizon_delta['positive_count'])}/{int(horizon_delta['n'])})\n"
            f"Learned lag    {float(lag_delta['mean']):+.6f}"
            f" ({int(lag_delta['positive_count'])}/{int(lag_delta['n'])})"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": GRID_COLOR, "pad": 4},
    )
    axis.set_xticks(
        range(5),
        [
            "Local\n(no graph)",
            "Static\nGAT",
            "River\nno lag",
            "+ Horizon\ngate",
            "+ Learned\nlag",
        ],
    )
    axis.set_ylabel("Validation macro NSE", fontsize=9.5)
    axis.set_title("Validation-best across seeds", fontsize=10, fontweight="bold", pad=10)
    axis.set_xlim(-0.25, 4.30)
    _style_axis(axis)
    _panel_label(axis, "a")


def _mechanism_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    target = summary["downstream_target_delta_nse"]
    horizon = summary["downstream_horizon_delta_macro_nse"]
    labels = ["NH3N", "CODMn", "TP", "Days 1–7", "Days 8–14", "Days 15–30"]
    payloads = [
        target["NH3N"],
        target["CODMn"],
        target["TP"],
        horizon["days_1_7"],
        horizon["days_8_14"],
        horizon["days_15_30"],
    ]
    means = [float(payload["mean"]) for payload in payloads]
    errors = [float(payload["sample_sd"]) for payload in payloads]
    positions = np.arange(len(labels))
    colors = ["#4E8192", "#6B8A79", "#8B7191", "#A8B5BB", "#789AAA", "#356F82"]
    axis.axvline(0.0, color="#6D7478", linewidth=0.8, linestyle="--", zorder=1)
    for mean, position, error, color in zip(
        means, positions, errors, colors, strict=True
    ):
        axis.errorbar(
            [mean],
            [position],
            xerr=[error],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            zorder=2,
        )
    axis.scatter(means, positions, c=colors, s=38, edgecolor="white", zorder=3)
    axis.axhline(2.5, color=GRID_COLOR, linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("ΔNSE on 112 nodes with upstream inputs", fontsize=9.5)
    axis.set_title("Concentrated at longer leads", fontsize=10, fontweight="bold", pad=10)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    axis.grid(axis="y", visible=False)
    _style_axis(axis)
    _panel_label(axis, "b")


def _network_panel(
    axis: plt.Axes,
    summary: Mapping[str, Any],
    graph_summary: Mapping[str, Any],
    figure: plt.Figure,
) -> None:
    nodes = list(graph_summary["nodes"])
    edges = list(graph_summary["edges"])
    node_rows = {row["node_id"]: row for row in summary["node_delta_macro_nse"]}
    node_ids = [str(node["node_id"]) for node in nodes]
    edge_pairs = [
        (str(edge["src_station_id"]), str(edge["dst_station_id"])) for edge in edges
    ]
    positions = _large_graph_positions(node_ids, edge_pairs)
    for source, destination in edge_pairs:
        axis.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[destination],
                arrowstyle="-|>",
                mutation_scale=4.2,
                linewidth=0.48,
                color="#B8C8CF",
                alpha=0.68,
                shrinkA=0.4,
                shrinkB=0.7,
                zorder=1,
            )
        )
    downstream_ids = [
        node_id for node_id in node_ids if bool(node_rows[node_id]["has_upstream"])
    ]
    values = np.asarray(
        [
            sum(float(delta) > 0.0 for delta in node_rows[node_id]["seed_deltas"].values())
            for node_id in downstream_ids
        ]
    )
    color_map = ListedColormap(
        ["#C57C45", "#D6A57F", "#DCCABB", "#C8D5D7", "#86AEB7", "#2F7183"]
    )
    normalization = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), color_map.N)
    headwater_ids = [node_id for node_id in node_ids if node_id not in downstream_ids]
    axis.scatter(
        [positions[node_id][0] for node_id in headwater_ids],
        [positions[node_id][1] for node_id in headwater_ids],
        s=9,
        facecolor="white",
        edgecolor="#A8B4BA",
        linewidth=0.45,
        zorder=2,
    )
    scatter = axis.scatter(
        [positions[node_id][0] for node_id in downstream_ids],
        [positions[node_id][1] for node_id in downstream_ids],
        c=values,
        cmap=color_map,
        norm=normalization,
        s=16,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    outlets = [str(node["node_id"]) for node in nodes if node["role"] == "outlet"]
    for outlet in outlets:
        axis.scatter(
            [positions[outlet][0]],
            [positions[outlet][1]],
            c=[
                sum(
                    float(delta) > 0.0
                    for delta in node_rows[outlet]["seed_deltas"].values()
                )
            ],
            cmap=color_map,
            norm=normalization,
            s=45,
            edgecolor=TEXT_COLOR,
            linewidth=0.7,
            zorder=4,
        )
    axis.annotate(
        "",
        xy=(0.96, 1.015),
        xytext=(0.64, 1.015),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEXT_COLOR, "linewidth": 0.9},
        annotation_clip=False,
    )
    axis.text(0.61, 1.015, "UPSTREAM", transform=axis.transAxes, ha="right", va="center", fontsize=8.5, fontweight="bold")
    axis.text(0.98, 1.015, "DOWNSTREAM", transform=axis.transAxes, ha="left", va="center", fontsize=8.5, fontweight="bold")
    colorbar = figure.colorbar(scatter, ax=axis, orientation="horizontal", fraction=0.045, pad=0.055)
    colorbar.set_ticks(range(6))
    colorbar.set_label("Seeds with positive node-level ΔNSE (out of 5)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#A8B4BA", markersize=5, label="Headwater: unchanged"),
            Line2D([0], [0], color="#B8C8CF", linewidth=1.0, marker=">", markersize=4, label="Upstream → downstream edge"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=7.5,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(
        "Stable gains follow downstream paths", fontsize=10, fontweight="bold", pad=10
    )
    axis.axis("off")
    _panel_label(axis, "c")


def render_graph_gain_figure(
    summary: Mapping[str, Any],
    graph_summary: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> tuple[Path, Path]:
    """Render paired gains, mechanism slices, and topology-localized benefits."""
    if bool(summary.get("test_split_used")):
        raise ValueError("architecture-selection figure cannot use the test split")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    grid = figure.add_gridspec(1, 3, width_ratios=(1.48, 1.02, 2.15), wspace=0.42)
    paired_axis = figure.add_subplot(grid[0, 0])
    mechanism_axis = figure.add_subplot(grid[0, 1])
    network_axis = figure.add_subplot(grid[0, 2])
    _paired_panel(paired_axis, summary)
    _mechanism_panel(mechanism_axis, summary)
    _network_panel(network_axis, summary, graph_summary, figure)
    figure.suptitle(
        "RiverLagNet is validation-best; directed upstream gains persist across seeds",
        x=0.045,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.045,
        0.948,
        "Five paired seeds · Static GAT retrained independently · River refinements strictly nest their frozen predecessor",
        ha="left",
        fontsize=9,
        color=MUTED_COLOR,
    )
    figure.text(
        0.045,
        0.018,
        "Validation split only; the held-out test set remains unopened. Node colors show predictive increments, not causal effects or recovered travel times.",
        ha="left",
        fontsize=8,
        color=MUTED_COLOR,
    )
    figure.subplots_adjust(top=0.87, bottom=0.12, left=0.06, right=0.985)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a clear validation error."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
