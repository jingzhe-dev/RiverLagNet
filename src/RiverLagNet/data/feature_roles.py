"""Name-based target, forcing, and hydrological feature roles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import TARGET_NAMES


# The priority is independent of NPZ channel order. Prefer an explicitly named
# discharge series, then conventional streamflow aliases, and finally the
# dataset-specific daily discharge product used by the frozen v0.4 benchmark.
FLOW_ALIASES = (
    "discharge",
    "river_discharge",
    "streamflow",
    "flow",
    "dis24",
)
RAINFALL_ALIASES = frozenset(
    {
        "rain",
        "rainfall",
        "precip",
        "precipitation",
        "daily_precipitation",
        "total_precipitation",
    }
)


@dataclass(frozen=True)
class FeatureRoles:
    """Immutable channel roles resolved from persisted variable names."""

    names: tuple[str, ...]
    target_indices: tuple[int, int, int]
    exogenous_indices: tuple[int, ...]
    flow_index: int | None
    rainfall_indices: tuple[int, ...]


def resolve_feature_roles(variable_names: Sequence[str]) -> FeatureRoles:
    """Resolve feature roles without guessing semantics from channel position."""
    names = tuple(str(name).strip() for name in variable_names)
    if len(names) < len(TARGET_NAMES):
        raise ValueError(f"variable names must begin with targets {TARGET_NAMES}")
    if any(not name for name in names):
        raise ValueError("variable names must not be empty")
    normalized = tuple(name.casefold() for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("variable names must be unique (case-insensitive)")
    if names[: len(TARGET_NAMES)] != TARGET_NAMES:
        raise ValueError(f"target channels 0-2 must be {TARGET_NAMES}, found {names[:3]}")

    index_by_name = {name: index for index, name in enumerate(normalized)}
    flow_index = next(
        (index_by_name[alias] for alias in FLOW_ALIASES if alias in index_by_name),
        None,
    )
    rainfall_indices = tuple(
        index
        for index, name in enumerate(normalized)
        if index >= len(TARGET_NAMES) and name in RAINFALL_ALIASES
    )
    return FeatureRoles(
        names=names,
        target_indices=(0, 1, 2),
        exogenous_indices=tuple(range(len(TARGET_NAMES), len(names))),
        flow_index=flow_index,
        rainfall_indices=rainfall_indices,
    )
