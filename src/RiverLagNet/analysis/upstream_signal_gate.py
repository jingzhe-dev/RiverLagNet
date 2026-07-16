"""Leakage-safe travel-aligned upstream residual signal probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from RiverLagNet.data.feature_roles import FeatureRoles
from RiverLagNet.data.feature_roles import resolve_feature_roles
from RiverLagNet.data.schema import TARGET_NAMES
from RiverLagNet.data.splits import ChronologicalFold


CONTROLS = ("correct", "reverse_direction", "shuffled_graph", "shuffled_time")
PROBE_FAMILIES = ("ridge", "mlp", "hist_gradient_boosting")


@dataclass(frozen=True)
class UpstreamFeatureBatch:
    """Aligned feature values and masks for each origin, node, and target."""

    origins: np.ndarray
    features: np.ndarray
    feature_mask: np.ndarray
    available: np.ndarray
    feature_names: tuple[str, ...]
    edge_source_time_indices: np.ndarray
    control: str


@dataclass(frozen=True)
class ProbeTimePartition:
    """Chronological local-training, probe-fit, and validation boundaries."""

    local_train_end: int
    probe_fit_origin_mask: np.ndarray
    validation_origin_mask: np.ndarray


@dataclass(frozen=True)
class ProbeEvaluation:
    """One paired local-versus-residual-probe validation comparison."""

    fold: str
    control: str
    probe_family: str
    baseline_macro_nse: float
    corrected_macro_nse: float
    relative_gain_percent: float
    target_nse_gain: dict[str, float]
    lead_band_relative_gain_percent: dict[str, float]
    fit_samples: int
    evaluation_samples: int


@dataclass(frozen=True)
class FoldSignalReport:
    """Paired control evaluations for one rolling-origin fold."""

    fold: str
    selected_probe_family: str
    evaluations: tuple[ProbeEvaluation, ...]
    family_selection_nse: dict[str, float] = field(default_factory=dict)

    def evaluation(self, control: str) -> ProbeEvaluation:
        """Return the unique evaluation for a named counterfactual."""
        matches = [item for item in self.evaluations if item.control == control]
        if len(matches) != 1:
            raise ValueError(f"fold {self.fold} must contain one {control} evaluation")
        return matches[0]


@dataclass(frozen=True)
class SignalGateDecision:
    """Frozen readiness decision for beginning formal graph-model work."""

    advance: bool
    mean_relative_gain_percent: float
    positive_folds: int
    folds_above_three_percent: int
    correct_beats_controls: bool
    reasons: tuple[str, ...]


def build_probe_time_partition(
    fold: ChronologicalFold,
    origins: np.ndarray,
    *,
    local_train_fraction: float = 0.8,
) -> ProbeTimePartition:
    """Split fold-training origins before exposing fold validation to probes."""
    if not 0.0 < local_train_fraction < 1.0:
        raise ValueError("local_train_fraction must be in (0,1)")
    origins = np.asarray(origins, dtype=np.int64)
    local_train_end = fold.train[0] + int(
        (fold.train[1] - fold.train[0]) * local_train_fraction
    )
    probe_fit = (origins >= local_train_end) & (origins < fold.train[1])
    validation = (origins >= fold.validation[0]) & (origins < fold.validation[1])
    if np.any(probe_fit & validation):
        raise RuntimeError("probe fitting and fold validation overlap")
    return ProbeTimePartition(local_train_end, probe_fit, validation)


def fit_event_thresholds(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    train_end: int,
    target_indices: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Fit median absolute target-change thresholds from training data only."""
    if train_end < 2 or train_end > values.shape[0]:
        raise ValueError("train_end must leave at least two training timestamps")
    thresholds: list[float] = []
    for target in target_indices:
        valid = observed[1:train_end, :, target] & observed[: train_end - 1, :, target]
        changes = np.abs(
            values[1:train_end, :, target] - values[: train_end - 1, :, target]
        )
        selected = changes[valid]
        thresholds.append(float(np.median(selected)) if selected.size else 0.0)
    return np.asarray(thresholds, dtype=np.float64)


def build_aligned_upstream_features(
    *,
    values: np.ndarray,
    observed: np.ndarray,
    origins: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    edge_attr_names: Sequence[str],
    roles: FeatureRoles,
    control: str = "correct",
    lag_offsets: Sequence[int] = (-1, 0, 1),
    seed: int = 42,
    event_thresholds: np.ndarray | None = None,
) -> UpstreamFeatureBatch:
    """Build direct-upstream features using only ``origin - lag`` source data."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    origins = np.asarray(origins, dtype=np.int64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float64)
    offsets = tuple(int(offset) for offset in lag_offsets)
    if control not in CONTROLS:
        raise ValueError(f"unknown upstream feature control: {control}")
    if values.ndim != 3 or observed.shape != values.shape:
        raise ValueError("values and observed must have shape [T,N,V]")
    if len(roles.names) != values.shape[2]:
        raise ValueError("feature roles must match the values feature dimension")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have shape [E,A]")
    if len(edge_attr_names) != edge_attr.shape[1]:
        raise ValueError("edge attribute names must match edge_attr")
    if not offsets:
        raise ValueError("lag_offsets must not be empty")
    try:
        travel_time_index = tuple(edge_attr_names).index("travel_time_prior_days")
    except ValueError as error:
        raise ValueError("edge attributes must name travel_time_prior_days") from error
    if origins.ndim != 1 or not origins.size:
        raise ValueError("origins must be a non-empty vector")

    num_times, num_nodes, _ = values.shape
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains an invalid node")
    source, destination = _controlled_edges(edge_index, num_nodes, control, seed)
    priors = edge_attr[:, travel_time_index]
    lag_candidates = np.stack(
        [np.maximum(1, np.rint(priors).astype(np.int64) + offset) for offset in offsets],
        axis=-1,
    )
    edge_source_times = origins[:, None, None] - lag_candidates[None, :, :]
    if control == "shuffled_time":
        rng = np.random.default_rng(seed)
        shape = (len(origins), edge_index.shape[1], len(offsets))
        past_span = np.maximum(origins[:, None, None] - 1, 1)
        edge_source_times = 1 + np.floor(rng.random(shape) * past_span).astype(
            np.int64
        )
    if np.any(edge_source_times >= origins[:, None, None]):
        raise RuntimeError("upstream probe attempted to access a non-past source value")

    thresholds = (
        np.zeros(len(TARGET_NAMES), dtype=np.float64)
        if event_thresholds is None
        else np.asarray(event_thresholds, dtype=np.float64)
    )
    if thresholds.shape != (len(TARGET_NAMES),):
        raise ValueError("event_thresholds must contain three target values")

    feature_values: list[np.ndarray] = []
    feature_masks: list[np.ndarray] = []
    feature_names: list[str] = []
    primary_available = np.zeros(
        (len(origins), num_nodes, len(TARGET_NAMES)), dtype=bool
    )
    primary_offset = offsets.index(0) if 0 in offsets else 0
    safe_destination_time = np.clip(origins - 1, 0, num_times - 1)

    for lag_position, offset in enumerate(offsets):
        source_time = edge_source_times[:, :, lag_position]
        safe_source_time = np.clip(source_time, 0, num_times - 1)
        valid_source_time = (source_time >= 1) & (source_time < num_times)
        for target_position, target_index in enumerate(roles.target_indices):
            source_value = values[safe_source_time, source[None, :], target_index]
            source_valid = (
                valid_source_time
                & observed[safe_source_time, source[None, :], target_index]
            )
            destination_value = values[
                safe_destination_time[:, None], destination[None, :], target_index
            ]
            destination_valid = observed[
                safe_destination_time[:, None], destination[None, :], target_index
            ]
            previous_source = values[
                np.clip(safe_source_time - 1, 0, num_times - 1),
                source[None, :],
                target_index,
            ]
            previous_valid = (
                valid_source_time
                & observed[
                    np.clip(safe_source_time - 1, 0, num_times - 1),
                    source[None, :],
                    target_index,
                ]
            )

            absolute, absolute_mask = _aggregate_edges(
                source_value, source_valid, destination, num_nodes
            )
            innovation, innovation_mask = _aggregate_edges(
                source_value - destination_value,
                source_valid & destination_valid,
                destination,
                num_nodes,
            )
            difference, difference_mask = _aggregate_edges(
                source_value - previous_source,
                source_valid & previous_valid,
                destination,
                num_nodes,
            )
            _append_target_feature(
                feature_values,
                feature_masks,
                feature_names,
                absolute,
                absolute_mask,
                target_position,
                num_targets=len(TARGET_NAMES),
                name=f"source_absolute_lag_{offset}",
            )
            _append_target_feature(
                feature_values,
                feature_masks,
                feature_names,
                innovation,
                innovation_mask,
                target_position,
                num_targets=len(TARGET_NAMES),
                name=f"source_innovation_lag_{offset}",
                reuse_name=target_position > 0,
            )
            _append_target_feature(
                feature_values,
                feature_masks,
                feature_names,
                difference,
                difference_mask,
                target_position,
                num_targets=len(TARGET_NAMES),
                name=f"source_first_difference_lag_{offset}",
                reuse_name=target_position > 0,
            )
            if lag_position == primary_offset:
                primary_available[..., target_position] = absolute_mask

            if roles.flow_index is not None:
                flow_index = roles.flow_index
                source_flow = values[safe_source_time, source[None, :], flow_index]
                source_flow_valid = (
                    valid_source_time
                    & observed[safe_source_time, source[None, :], flow_index]
                )
                destination_flow = values[
                    safe_source_time, destination[None, :], flow_index
                ]
                destination_flow_valid = (
                    valid_source_time
                    & observed[safe_source_time, destination[None, :], flow_index]
                )
                load, load_mask = _aggregate_edges(
                    source_value * source_flow,
                    source_valid & source_flow_valid,
                    destination,
                    num_nodes,
                )
                flow_ratio, flow_ratio_mask = _aggregate_edges(
                    source_flow / np.where(
                        np.abs(destination_flow) > 1e-8, destination_flow, 1.0
                    ),
                    source_flow_valid
                    & destination_flow_valid
                    & (np.abs(destination_flow) > 1e-8),
                    destination,
                    num_nodes,
                )
                _append_target_feature(
                    feature_values,
                    feature_masks,
                    feature_names,
                    load,
                    load_mask,
                    target_position,
                    num_targets=len(TARGET_NAMES),
                    name=f"source_load_proxy_lag_{offset}",
                    reuse_name=target_position > 0,
                )
                _append_target_feature(
                    feature_values,
                    feature_masks,
                    feature_names,
                    flow_ratio,
                    flow_ratio_mask,
                    target_position,
                    num_targets=len(TARGET_NAMES),
                    name=f"source_destination_flow_ratio_lag_{offset}",
                    reuse_name=target_position > 0,
                )

    # _append_target_feature fills one target slice at a time. Merge repeated
    # per-target entries before adding shared edge/topology channels.
    feature_values, feature_masks, feature_names = _merge_target_features(
        feature_values, feature_masks, feature_names
    )
    origin_count = len(origins)
    for attribute_index, attribute_name in enumerate(edge_attr_names):
        repeated = np.broadcast_to(edge_attr[None, :, attribute_index], (origin_count, len(source)))
        aggregated, mask = _aggregate_edges(
            repeated,
            np.ones_like(repeated, dtype=bool),
            destination,
            num_nodes,
        )
        feature_values.append(np.repeat(aggregated[..., None], len(TARGET_NAMES), axis=-1))
        feature_masks.append(np.repeat(mask[..., None], len(TARGET_NAMES), axis=-1))
        feature_names.append(f"edge_{attribute_name}")

    destination_now = np.clip(origins - 1, 0, num_times - 1)
    destination_previous = np.clip(origins - 2, 0, num_times - 1)
    events = np.zeros((origin_count, num_nodes, len(TARGET_NAMES)), dtype=np.float64)
    event_mask = np.zeros_like(events, dtype=bool)
    for target_position, target_index in enumerate(roles.target_indices):
        valid = (
            observed[destination_now, :, target_index]
            & observed[destination_previous, :, target_index]
        )
        change = np.abs(
            values[destination_now, :, target_index]
            - values[destination_previous, :, target_index]
        )
        events[..., target_position] = change >= thresholds[target_position]
        event_mask[..., target_position] = valid
    feature_values.append(events)
    feature_masks.append(event_mask)
    feature_names.append("destination_event_indicator")

    upstream_degree = np.bincount(destination, minlength=num_nodes).astype(np.float64)
    depth = _topological_depth(source, destination, num_nodes).astype(np.float64)
    for name, vector in (
        ("upstream_degree", upstream_degree),
        ("downstream_depth", depth),
    ):
        shared = np.broadcast_to(
            vector[None, :, None], (origin_count, num_nodes, len(TARGET_NAMES))
        ).copy()
        feature_values.append(shared)
        feature_masks.append(np.ones_like(shared, dtype=bool))
        feature_names.append(name)

    features = np.stack(feature_values, axis=-1)
    masks = np.stack(feature_masks, axis=-1)
    features = np.where(masks, features, 0.0)
    return UpstreamFeatureBatch(
        origins=origins,
        features=features,
        feature_mask=masks,
        available=primary_available,
        feature_names=tuple(feature_names),
        edge_source_time_indices=edge_source_times,
        control=control,
    )


def evaluate_residual_probe(
    *,
    feature_batch: UpstreamFeatureBatch,
    local_prediction: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    fit_origin_mask: np.ndarray,
    evaluation_origin_mask: np.ndarray,
    probe_family: str,
    fold: str,
    seed: int,
    max_fit_samples: int = 100_000,
) -> ProbeEvaluation:
    """Fit residual probes on training origins and evaluate one paired control."""
    local_prediction = np.asarray(local_prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    fit_origin_mask = np.asarray(fit_origin_mask, dtype=bool)
    evaluation_origin_mask = np.asarray(evaluation_origin_mask, dtype=bool)
    if probe_family not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family: {probe_family}")
    if local_prediction.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError("local prediction, target, and mask must share [O,L,N,3]")
    origins, leads, nodes, targets = target.shape
    if targets != len(TARGET_NAMES):
        raise ValueError("probe targets must preserve NH3N,CODMn,TP")
    if feature_batch.features.shape[:3] != (origins, nodes, targets):
        raise ValueError("feature batch axes do not match prediction origins/nodes/targets")
    if fit_origin_mask.shape != (origins,) or evaluation_origin_mask.shape != (origins,):
        raise ValueError("origin masks must have shape [O]")
    if np.any(fit_origin_mask & evaluation_origin_mask):
        raise ValueError("probe-fit and evaluation origins must be disjoint")

    corrected = local_prediction.copy()
    residual = target - local_prediction
    design = np.concatenate(
        (feature_batch.features, feature_batch.feature_mask.astype(np.float64)), axis=-1
    )
    design = np.nan_to_num(design, copy=False)
    fit_sample_count = 0
    evaluation_sample_count = 0
    rng = np.random.default_rng(seed)
    for target_index in range(targets):
        target_design = design[:, :, target_index]
        target_available = feature_batch.available[:, :, target_index]
        for _, lead_indices in _lead_bands(leads):
            block_leads = len(lead_indices)
            repeated_design = np.broadcast_to(
                target_design[:, None],
                (origins, block_leads, nodes, target_design.shape[-1]),
            )
            normalized_lead = np.broadcast_to(
                ((lead_indices + 1) / leads)[None, :, None, None],
                (origins, block_leads, nodes, 1),
            )
            features = np.concatenate((repeated_design, normalized_lead), axis=-1)
            block_mask = np.take(
                target_mask[..., target_index], lead_indices, axis=1
            )
            availability = np.broadcast_to(
                target_available[:, None], (origins, block_leads, nodes)
            )
            fit_rows = (
                fit_origin_mask[:, None, None] & block_mask & availability
            ).reshape(-1)
            evaluation_rows = (
                evaluation_origin_mask[:, None, None] & block_mask & availability
            ).reshape(-1)
            if not fit_rows.any() or not evaluation_rows.any():
                continue
            flat_features = features.reshape(-1, features.shape[-1])
            flat_residual = np.take(
                residual[..., target_index], lead_indices, axis=1
            ).reshape(-1)
            fit_indices = np.flatnonzero(fit_rows)
            if len(fit_indices) > max_fit_samples:
                fit_indices = np.sort(
                    rng.choice(fit_indices, size=max_fit_samples, replace=False)
                )
            model = _probe_model(probe_family, seed)
            model.fit(flat_features[fit_indices], flat_residual[fit_indices])
            evaluation_indices = np.flatnonzero(evaluation_rows)
            correction = model.predict(flat_features[evaluation_indices])
            corrected_block = np.take(
                corrected[..., target_index], lead_indices, axis=1
            ).copy()
            flat_corrected = corrected_block.reshape(-1)
            flat_corrected[evaluation_indices] += correction
            for block_position, lead_index in enumerate(lead_indices):
                corrected[:, lead_index, :, target_index] = corrected_block[
                    :, block_position
                ]
            fit_sample_count += len(fit_indices)
            evaluation_sample_count += len(evaluation_indices)

    baseline_target_nse = _nse_by_target(
        local_prediction[evaluation_origin_mask],
        target[evaluation_origin_mask],
        target_mask[evaluation_origin_mask],
    )
    corrected_target_nse = _nse_by_target(
        corrected[evaluation_origin_mask],
        target[evaluation_origin_mask],
        target_mask[evaluation_origin_mask],
    )
    baseline_macro = _finite_mean(baseline_target_nse)
    corrected_macro = _finite_mean(corrected_target_nse)
    lead_gains: dict[str, float] = {}
    for band_name, lead_indices in _lead_bands(leads):
        local_band = np.take(
            local_prediction[evaluation_origin_mask], lead_indices, axis=1
        )
        corrected_band = np.take(
            corrected[evaluation_origin_mask], lead_indices, axis=1
        )
        target_band = np.take(target[evaluation_origin_mask], lead_indices, axis=1)
        mask_band = np.take(
            target_mask[evaluation_origin_mask], lead_indices, axis=1
        )
        local_nse = _finite_mean(_nse_by_target(local_band, target_band, mask_band))
        probe_nse = _finite_mean(_nse_by_target(corrected_band, target_band, mask_band))
        lead_gains[band_name] = _relative_gain_percent(local_nse, probe_nse)
    return ProbeEvaluation(
        fold=fold,
        control=feature_batch.control,
        probe_family=probe_family,
        baseline_macro_nse=baseline_macro,
        corrected_macro_nse=corrected_macro,
        relative_gain_percent=_relative_gain_percent(baseline_macro, corrected_macro),
        target_nse_gain={
            name: float(corrected_target_nse[index] - baseline_target_nse[index])
            for index, name in enumerate(TARGET_NAMES)
        },
        lead_band_relative_gain_percent=lead_gains,
        fit_samples=fit_sample_count,
        evaluation_samples=evaluation_sample_count,
    )


def evaluate_control_set(
    *,
    feature_batches: Mapping[str, UpstreamFeatureBatch],
    local_prediction: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    probe_fit_origin_mask: np.ndarray,
    evaluation_origin_mask: np.ndarray,
    fold: str,
    seed: int,
    families: Sequence[str] = PROBE_FAMILIES,
) -> FoldSignalReport:
    """Select a probe family inside training data, then pair all controls."""
    if "correct" not in feature_batches:
        raise ValueError("feature batches must include correct alignment")
    fit_indices = np.flatnonzero(probe_fit_origin_mask)
    if len(fit_indices) < 5:
        raise ValueError("probe fitting requires at least five chronological origins")
    selection_start = max(1, int(len(fit_indices) * 0.8))
    family_train = np.zeros_like(probe_fit_origin_mask, dtype=bool)
    family_validation = np.zeros_like(probe_fit_origin_mask, dtype=bool)
    family_train[fit_indices[:selection_start]] = True
    family_validation[fit_indices[selection_start:]] = True
    family_scores: dict[str, float] = {}
    for family in families:
        selection = evaluate_residual_probe(
            feature_batch=feature_batches["correct"],
            local_prediction=local_prediction,
            target=target,
            target_mask=target_mask,
            fit_origin_mask=family_train,
            evaluation_origin_mask=family_validation,
            probe_family=family,
            fold=fold,
            seed=seed,
        )
        family_scores[family] = selection.corrected_macro_nse
    selected_family = min(
        family_scores,
        key=lambda family: (-family_scores[family], PROBE_FAMILIES.index(family)),
    )
    evaluations = tuple(
        evaluate_residual_probe(
            feature_batch=feature_batches[control],
            local_prediction=local_prediction,
            target=target,
            target_mask=target_mask,
            fit_origin_mask=probe_fit_origin_mask,
            evaluation_origin_mask=evaluation_origin_mask,
            probe_family=selected_family,
            fold=fold,
            seed=seed,
        )
        for control in feature_batches
    )
    return FoldSignalReport(
        fold=fold,
        selected_probe_family=selected_family,
        evaluations=evaluations,
        family_selection_nse=family_scores,
    )


def decide_signal_gate(reports: Sequence[FoldSignalReport]) -> SignalGateDecision:
    """Apply the preregistered signal gate without adapting its thresholds."""
    if len(reports) != 3:
        raise ValueError("signal gate requires exactly three rolling folds")
    correct = [report.evaluation("correct") for report in reports]
    gains = np.asarray([item.relative_gain_percent for item in correct], dtype=np.float64)
    mean_gain = float(gains.mean())
    positive_folds = int((gains > 0.0).sum())
    above_three = int((gains >= 3.0).sum())
    control_names = ("reverse_direction", "shuffled_graph", "shuffled_time")
    correct_mean_nse = float(np.mean([item.corrected_macro_nse for item in correct]))
    control_mean_nse = {
        name: float(
            np.mean([report.evaluation(name).corrected_macro_nse for report in reports])
        )
        for name in control_names
    }
    beats_controls = all(correct_mean_nse > score for score in control_mean_nse.values())
    target_means = {
        name: float(np.mean([item.target_nse_gain[name] for item in correct]))
        for name in TARGET_NAMES
    }
    reasons: list[str] = []
    if mean_gain <= 0.0:
        reasons.append("mean correct-alignment relative NSE gain is not positive")
    if above_three < 2:
        reasons.append("fewer than two folds reach the fixed 3% relative NSE gain")
    if not beats_controls:
        reasons.append("correct alignment does not beat every paired counterfactual control")
    if max(target_means.values()) <= 0.0:
        reasons.append("no target benefits on average")
    collapsed = {name: gain for name, gain in target_means.items() if gain < -0.01}
    if collapsed:
        reasons.append(f"target average NSE collapses by more than 0.01: {collapsed}")
    return SignalGateDecision(
        advance=not reasons,
        mean_relative_gain_percent=mean_gain,
        positive_folds=positive_folds,
        folds_above_three_percent=above_three,
        correct_beats_controls=beats_controls,
        reasons=tuple(reasons),
    )


def signal_report_as_dict(
    reports: Sequence[FoldSignalReport], decision: SignalGateDecision
) -> dict[str, Any]:
    """Return a JSON-safe signal-gate report."""
    return {
        "folds": [asdict(report) for report in reports],
        "decision": asdict(decision),
        "target_order": list(TARGET_NAMES),
        "controls": list(CONTROLS),
        "probe_families": list(PROBE_FAMILIES),
        "causal_interpretation": False,
    }


def evaluate_probe_bundle(
    bundle_path: str,
    *,
    folds: Sequence[str],
    controls: Sequence[str] = CONTROLS,
    seed: int = 42,
) -> tuple[tuple[FoldSignalReport, ...], SignalGateDecision, dict[str, Any]]:
    """Evaluate a sealed-development probe bundle produced from local checkpoints."""
    requested_controls = tuple(controls)
    if set(requested_controls) != set(CONTROLS) or len(requested_controls) != len(
        CONTROLS
    ):
        raise ValueError(f"formal signal gate requires exactly controls {CONTROLS}")
    with np.load(bundle_path, allow_pickle=False) as archive:
        required_global = {
            "values",
            "observed",
            "edge_index",
            "edge_attr",
            "variable_names",
            "edge_attr_names",
            "final_test_start",
            "dataset_sha256",
        }
        missing = required_global.difference(archive.files)
        if missing:
            raise ValueError(f"probe bundle lacks global fields {sorted(missing)}")
        values = np.asarray(archive["values"], dtype=np.float64)
        observed = np.asarray(archive["observed"], dtype=bool)
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(archive["edge_attr"], dtype=np.float64)
        variable_names = tuple(archive["variable_names"].astype(str).tolist())
        edge_attr_names = tuple(archive["edge_attr_names"].astype(str).tolist())
        final_test_start = int(np.asarray(archive["final_test_start"]).item())
        dataset_sha256 = str(np.asarray(archive["dataset_sha256"]).item())
        roles = resolve_feature_roles(variable_names)
        reports: list[FoldSignalReport] = []
        for fold in folds:
            prefix = f"{fold}__"
            required_fold = {
                f"{prefix}origins",
                f"{prefix}local_prediction",
                f"{prefix}target",
                f"{prefix}target_mask",
                f"{prefix}probe_fit_origin_mask",
                f"{prefix}evaluation_origin_mask",
                f"{prefix}train_end",
            }
            missing_fold = required_fold.difference(archive.files)
            if missing_fold:
                raise ValueError(f"probe bundle lacks {fold} fields {sorted(missing_fold)}")
            origins = np.asarray(archive[f"{prefix}origins"], dtype=np.int64)
            local_prediction = np.asarray(
                archive[f"{prefix}local_prediction"], dtype=np.float64
            )
            target = np.asarray(archive[f"{prefix}target"], dtype=np.float64)
            target_mask = np.asarray(archive[f"{prefix}target_mask"], dtype=bool)
            probe_fit_mask = np.asarray(
                archive[f"{prefix}probe_fit_origin_mask"], dtype=bool
            )
            evaluation_mask = np.asarray(
                archive[f"{prefix}evaluation_origin_mask"], dtype=bool
            )
            train_end = int(np.asarray(archive[f"{prefix}train_end"]).item())
            output_window = int(target.shape[1])
            if np.any(origins + output_window > final_test_start):
                raise ValueError("probe bundle crosses the sealed final-test boundary")
            thresholds = fit_event_thresholds(
                values, observed, train_end=train_end, target_indices=roles.target_indices
            )
            feature_batches = {
                control: build_aligned_upstream_features(
                    values=values,
                    observed=observed,
                    origins=origins,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_attr_names=edge_attr_names,
                    roles=roles,
                    control=control,
                    seed=seed,
                    event_thresholds=thresholds,
                )
                for control in requested_controls
            }
            reports.append(
                evaluate_control_set(
                    feature_batches=feature_batches,
                    local_prediction=local_prediction,
                    target=target,
                    target_mask=target_mask,
                    probe_fit_origin_mask=probe_fit_mask,
                    evaluation_origin_mask=evaluation_mask,
                    fold=fold,
                    seed=seed,
                )
            )
    decision = decide_signal_gate(reports)
    metadata = {
        "bundle_path": str(bundle_path),
        "dataset_sha256": dataset_sha256,
        "final_test_start": final_test_start,
    }
    return tuple(reports), decision, metadata


def render_signal_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render the machine-readable gate report without changing its values."""
    decision = payload["decision"]
    lines = [
        "# RiverLagNet v0.2 upstream signal gate",
        "",
        f"- Advance: `{decision['advance']}`",
        f"- Mean relative NSE gain: `{decision['mean_relative_gain_percent']:.6f}%`",
        f"- Positive folds: `{decision['positive_folds']}`",
        f"- Folds at or above 3%: `{decision['folds_above_three_percent']}`",
        f"- Correct beats controls: `{decision['correct_beats_controls']}`",
        "- Interpretation: predictive residual association only; not causal attribution.",
        "",
        "| Fold | Probe | Control | Local NSE | Corrected NSE | Relative gain |",
        "|---|---|---|---:|---:|---:|",
    ]
    for report in payload["folds"]:
        for evaluation in report["evaluations"]:
            lines.append(
                "| {fold} | {probe} | {control} | {local:.6f} | {corrected:.6f} | {gain:.6f}% |".format(
                    fold=report["fold"],
                    probe=evaluation["probe_family"],
                    control=evaluation["control"],
                    local=evaluation["baseline_macro_nse"],
                    corrected=evaluation["corrected_macro_nse"],
                    gain=evaluation["relative_gain_percent"],
                )
            )
    if decision["reasons"]:
        lines.extend(("", "## Failed clauses", ""))
        lines.extend(f"- {reason}" for reason in decision["reasons"])
    return "\n".join(lines) + "\n"


def _controlled_edges(
    edge_index: np.ndarray, num_nodes: int, control: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    source = edge_index[0].copy()
    destination = edge_index[1].copy()
    if control == "reverse_direction":
        source, destination = destination, source
    elif control == "shuffled_graph":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(num_nodes)
        if np.array_equal(permutation, np.arange(num_nodes)) and num_nodes > 1:
            permutation = np.roll(permutation, 1)
        source = permutation[source]
        destination = permutation[destination]
    return source, destination


def _aggregate_edges(
    values: np.ndarray,
    mask: np.ndarray,
    destination: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    origins, edges = values.shape
    if mask.shape != values.shape or destination.shape != (edges,):
        raise ValueError("edge aggregation shapes are inconsistent")
    flat_index = (
        np.arange(origins, dtype=np.int64)[:, None] * num_nodes + destination[None, :]
    ).reshape(-1)
    totals = np.zeros(origins * num_nodes, dtype=np.float64)
    counts = np.zeros(origins * num_nodes, dtype=np.int64)
    np.add.at(totals, flat_index, np.where(mask, values, 0.0).reshape(-1))
    np.add.at(counts, flat_index, mask.reshape(-1).astype(np.int64))
    valid = counts.reshape(origins, num_nodes) > 0
    aggregated = totals.reshape(origins, num_nodes) / np.maximum(
        counts.reshape(origins, num_nodes), 1
    )
    return np.where(valid, aggregated, 0.0), valid


def _append_target_feature(
    values: list[np.ndarray],
    masks: list[np.ndarray],
    names: list[str],
    target_values: np.ndarray,
    target_mask: np.ndarray,
    target_position: int,
    *,
    num_targets: int,
    name: str,
    reuse_name: bool = False,
) -> None:
    expanded = np.zeros((*target_values.shape, num_targets), dtype=np.float64)
    expanded_mask = np.zeros_like(expanded, dtype=bool)
    expanded[..., target_position] = target_values
    expanded_mask[..., target_position] = target_mask
    values.append(expanded)
    masks.append(expanded_mask)
    names.append(name if not reuse_name else name)


def _merge_target_features(
    values: list[np.ndarray], masks: list[np.ndarray], names: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    merged_values: list[np.ndarray] = []
    merged_masks: list[np.ndarray] = []
    merged_names: list[str] = []
    positions: dict[str, int] = {}
    for value, mask, name in zip(values, masks, names, strict=True):
        if name not in positions:
            positions[name] = len(merged_names)
            merged_names.append(name)
            merged_values.append(value.copy())
            merged_masks.append(mask.copy())
        else:
            index = positions[name]
            merged_values[index] += value
            merged_masks[index] |= mask
    return merged_values, merged_masks, merged_names


def _topological_depth(
    source: np.ndarray, destination: np.ndarray, num_nodes: int
) -> np.ndarray:
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = np.zeros(num_nodes, dtype=np.int64)
    for upstream, downstream in zip(source.tolist(), destination.tolist(), strict=True):
        children[upstream].append(downstream)
        indegree[downstream] += 1
    queue = [node for node in range(num_nodes) if indegree[node] == 0]
    depth = np.zeros(num_nodes, dtype=np.int64)
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != num_nodes:
        raise ValueError("probe graph controls must remain acyclic")
    return depth


def _probe_model(family: str, seed: int):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=150,
                early_stopping=True,
                random_state=seed,
            ),
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_depth=3,
            l2_regularization=1e-3,
            random_state=seed,
        )
    raise ValueError(f"unknown probe family: {family}")


def _lead_bands(output_window: int) -> tuple[tuple[str, np.ndarray], ...]:
    bands: list[tuple[str, np.ndarray]] = []
    for start, end in ((1, 10), (11, 20), (21, 30)):
        indices = np.arange(start - 1, min(end, output_window), dtype=np.int64)
        if indices.size:
            bands.append((f"days_{start}_{end}", indices))
    return tuple(bands)


def _nse_by_target(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    scores = np.full(target.shape[-1], np.nan, dtype=np.float64)
    for target_index in range(target.shape[-1]):
        valid = mask[..., target_index]
        if not valid.any():
            continue
        truth = target[..., target_index][valid]
        estimate = prediction[..., target_index][valid]
        denominator = np.square(truth - truth.mean()).sum()
        if denominator <= np.finfo(np.float64).eps:
            continue
        scores[target_index] = 1.0 - np.square(estimate - truth).sum() / denominator
    return scores


def _finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else 0.0


def _relative_gain_percent(baseline: float, candidate: float) -> float:
    return 100.0 * (candidate - baseline) / max(abs(baseline), 1e-8)
