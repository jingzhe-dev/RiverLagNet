from __future__ import annotations

import pytest

from RiverLagNet.data.feature_roles import resolve_feature_roles
from RiverLagNet.data.schema import LocalForecastShapeContract


def test_feature_roles_keep_targets_first_and_find_exogenous_by_name() -> None:
    roles = resolve_feature_roles(
        ("NH3N", "CODMn", "TP", "temperature_2m", "discharge", "rainfall")
    )

    assert roles.names == (
        "NH3N",
        "CODMn",
        "TP",
        "temperature_2m",
        "discharge",
        "rainfall",
    )
    assert roles.target_indices == (0, 1, 2)
    assert roles.exogenous_indices == (3, 4, 5)
    assert roles.flow_index == 4
    assert roles.rainfall_indices == (5,)


def test_feature_role_aliases_resolve_with_fixed_priority() -> None:
    first = resolve_feature_roles(
        ("NH3N", "CODMn", "TP", "flow", "dis24", "streamflow", "discharge")
    )
    reordered = resolve_feature_roles(
        ("NH3N", "CODMn", "TP", "discharge", "streamflow", "dis24", "flow")
    )

    assert first.names[first.flow_index] == "discharge"
    assert reordered.names[reordered.flow_index] == "discharge"


def test_rainfall_aliases_and_missing_flow_are_explicit() -> None:
    roles = resolve_feature_roles(
        (
            "NH3N",
            "CODMn",
            "TP",
            "precipitation",
            "total_precipitation",
            "temperature_2m",
        )
    )

    assert roles.flow_index is None
    assert roles.rainfall_indices == (3, 4)


@pytest.mark.parametrize(
    "names",
    [
        ("CODMn", "NH3N", "TP"),
        ("NH3N", "CODMn", "unknown_target"),
        ("NH3N", "CODMn", "TP", "TP"),
        ("NH3N", "CODMn", "TP", "temp", "TEMP"),
    ],
)
def test_invalid_or_duplicate_feature_names_fail(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        resolve_feature_roles(names)


def test_local_forecast_shape_contract_freezes_all_public_axes() -> None:
    contract = LocalForecastShapeContract.from_dimensions(
        batch_size=2,
        input_window=180,
        num_nodes=1068,
        hidden_dim=128,
        num_scales=4,
    )

    assert contract.history_states == (2, 180, 1068, 128)
    assert contract.scale_states == (2, 4, 1068, 128)
    assert contract.horizon_states == (2, 30, 1068, 128)
    assert contract.prediction == (2, 30, 1068, 3)
