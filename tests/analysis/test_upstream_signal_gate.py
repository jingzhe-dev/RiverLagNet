from __future__ import annotations

import numpy as np

from RiverLagNet.analysis.upstream_signal_gate import (
    FoldSignalReport,
    ProbeEvaluation,
    build_aligned_upstream_features,
    build_probe_time_partition,
    decide_signal_gate,
    evaluate_control_set,
    evaluate_residual_probe,
)
from RiverLagNet.data.feature_roles import resolve_feature_roles
from RiverLagNet.data.splits import ChronologicalFold
from RiverLagNet.data.schema import TARGET_NAMES


def _chain(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(100, 4, 5)).astype(np.float64)
    values[..., 3] = rng.lognormal(mean=1.0, sigma=0.2, size=(100, 4))
    observed = np.ones_like(values, dtype=bool)
    edge_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_attr = np.asarray(
        [[10.0, 0.01, 2.0], [12.0, 0.02, 2.0], [14.0, 0.03, 2.0]],
        dtype=np.float64,
    )
    return values, observed, edge_index, edge_attr


def _features(
    *,
    control: str = "correct",
    observed_override: np.ndarray | None = None,
    seed: int = 11,
):
    values, observed, edge_index, edge_attr = _chain()
    if observed_override is not None:
        observed = observed_override
    origins = np.arange(12, 82, dtype=np.int64)
    roles = resolve_feature_roles((*TARGET_NAMES, "discharge", "rainfall"))
    return build_aligned_upstream_features(
        values=values,
        observed=observed,
        origins=origins,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_names=("distance_km", "slope", "travel_time_prior_days"),
        roles=roles,
        control=control,
        lag_offsets=(0,),
        seed=seed,
        event_thresholds=np.ones(3),
    )


def test_correct_alignment_uses_origin_minus_lag_and_masks_unavailable_sources() -> None:
    values, observed, _, _ = _chain()
    origins = np.arange(12, 82, dtype=np.int64)
    observed[origins[0] - 2, 0, 0] = False

    batch = _features(observed_override=observed)
    absolute = batch.feature_names.index("source_absolute_lag_0")

    assert np.array_equal(batch.edge_source_time_indices[:, 0, 0], origins - 2)
    assert not batch.feature_mask[0, 1, 0, absolute]
    assert not batch.available[0, 1, 0]
    assert batch.feature_mask[1, 1, 0, absolute]
    assert batch.features[1, 1, 0, absolute] == values[origins[1] - 2, 0, 0]
    assert not batch.available[:, 0].any()  # node 0 is a headwater
    assert np.all(batch.edge_source_time_indices < origins[:, None, None])


def test_counterfactual_feature_controls_are_seeded_and_distinct() -> None:
    correct = _features(control="correct")
    reverse = _features(control="reverse_direction")
    shuffled_a = _features(control="shuffled_graph", seed=31)
    shuffled_b = _features(control="shuffled_graph", seed=31)
    shuffled_time = _features(control="shuffled_time", seed=31)

    assert np.array_equal(shuffled_a.features, shuffled_b.features)
    assert not np.array_equal(correct.features, reverse.features)
    assert not np.array_equal(correct.features, shuffled_a.features)
    assert not np.array_equal(correct.features, shuffled_time.features)


def test_correct_probe_recovers_signal_and_beats_all_controls() -> None:
    values, _, edge_index, _ = _chain()
    origins = np.arange(12, 82, dtype=np.int64)
    leads = 30
    target = np.zeros((len(origins), leads, 4, 3), dtype=np.float64)
    target_mask = np.zeros_like(target, dtype=bool)
    for destination in range(1, 4):
        source = edge_index[0, destination - 1]
        source_signal = values[origins - 2, source, :3]
        target[:, :, destination, :] = source_signal[:, None, :] + (
            np.arange(leads, dtype=np.float64)[None, :, None] / 100.0
        )
        target_mask[:, :, destination, :] = True
    local_prediction = target * 0.2
    fit_origins = np.arange(len(origins)) < 45
    validation_origins = ~fit_origins

    evaluations = {
        control: evaluate_residual_probe(
            feature_batch=_features(control=control, seed=19),
            local_prediction=local_prediction,
            target=target,
            target_mask=target_mask,
            fit_origin_mask=fit_origins,
            evaluation_origin_mask=validation_origins,
            probe_family="ridge",
            fold="synthetic",
            seed=19,
        )
        for control in (
            "correct",
            "reverse_direction",
            "shuffled_graph",
            "shuffled_time",
        )
    }

    assert evaluations["correct"].relative_gain_percent > 3.0
    assert evaluations["correct"].corrected_macro_nse > 0.95
    assert all(
        evaluations["correct"].corrected_macro_nse
        > evaluations[control].corrected_macro_nse
        for control in ("reverse_direction", "shuffled_graph", "shuffled_time")
    )
    assert set(evaluations["correct"].target_nse_gain) == set(TARGET_NAMES)
    assert set(evaluations["correct"].lead_band_relative_gain_percent) == {
        "days_1_10",
        "days_11_20",
        "days_21_30",
    }


def test_probe_time_partition_never_uses_validation_for_fitting() -> None:
    fold = ChronologicalFold(
        "fold",
        train=(0, 80),
        validation=(80, 95),
        final_test=(95, 110),
    )
    origins = np.arange(20, 95)

    partition = build_probe_time_partition(fold, origins, local_train_fraction=0.8)

    assert partition.local_train_end == 64
    assert np.array_equal(origins[partition.probe_fit_origin_mask], np.arange(64, 80))
    assert np.array_equal(origins[partition.validation_origin_mask], np.arange(80, 95))
    assert not np.any(partition.probe_fit_origin_mask & partition.validation_origin_mask)


def test_probe_family_selection_compares_ridge_mlp_and_shallow_tree_inside_fit_data() -> None:
    values, _, edge_index, _ = _chain()
    origins = np.arange(12, 82, dtype=np.int64)
    target = np.zeros((len(origins), 30, 4, 3), dtype=np.float64)
    target_mask = np.zeros_like(target, dtype=bool)
    for destination in range(1, 4):
        source_signal = values[origins - 2, edge_index[0, destination - 1], :3]
        target[:, :, destination] = source_signal[:, None]
        target_mask[:, :, destination] = True
    fit_origins = np.arange(len(origins)) < 45
    validation_origins = ~fit_origins

    report = evaluate_control_set(
        feature_batches={
            control: _features(control=control, seed=29)
            for control in (
                "correct",
                "reverse_direction",
                "shuffled_graph",
                "shuffled_time",
            )
        },
        local_prediction=target * 0.2,
        target=target,
        target_mask=target_mask,
        probe_fit_origin_mask=fit_origins,
        evaluation_origin_mask=validation_origins,
        fold="synthetic",
        seed=29,
    )

    assert set(report.family_selection_nse) == {
        "ridge",
        "mlp",
        "hist_gradient_boosting",
    }
    assert report.selected_probe_family in report.family_selection_nse
    assert {item.control for item in report.evaluations} == {
        "correct",
        "reverse_direction",
        "shuffled_graph",
        "shuffled_time",
    }


def _evaluation(control: str, gain: float, target_drop: float = 0.0) -> ProbeEvaluation:
    return ProbeEvaluation(
        fold="fold",
        control=control,
        probe_family="ridge",
        baseline_macro_nse=0.5,
        corrected_macro_nse=0.5 + gain / 100.0,
        relative_gain_percent=gain,
        target_nse_gain={"NH3N": 0.02, "CODMn": 0.01, "TP": target_drop},
        lead_band_relative_gain_percent={
            "days_1_10": gain,
            "days_11_20": gain,
            "days_21_30": gain,
        },
        fit_samples=100,
        evaluation_samples=100,
    )


def test_signal_gate_applies_every_fixed_clause_without_lowering_thresholds() -> None:
    reports = []
    for gain in (4.0, 3.5, 1.0):
        reports.append(
            FoldSignalReport(
                fold=f"fold_{len(reports)}",
                selected_probe_family="ridge",
                evaluations=(
                    _evaluation("correct", gain),
                    _evaluation("reverse_direction", 0.5),
                    _evaluation("shuffled_graph", 0.2),
                    _evaluation("shuffled_time", -0.1),
                ),
            )
        )

    passed = decide_signal_gate(reports)
    collapsed = list(reports)
    collapsed[-1] = FoldSignalReport(
        fold="fold_2",
        selected_probe_family="ridge",
        evaluations=(
            _evaluation("correct", 1.0, target_drop=-0.05),
            _evaluation("reverse_direction", 0.5),
            _evaluation("shuffled_graph", 0.2),
            _evaluation("shuffled_time", -0.1),
        ),
    )

    failed = decide_signal_gate(collapsed)

    assert passed.advance
    assert passed.folds_above_three_percent == 2
    assert passed.correct_beats_controls
    assert not failed.advance
    assert any("target" in reason.lower() for reason in failed.reasons)
