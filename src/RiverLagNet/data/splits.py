"""Immutable chronological development folds for RiverLagNet v0.2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChronologicalFold:
    """Index intervals for one development fold and the sealed final test."""

    name: str
    train: tuple[int, int]
    validation: tuple[int, int]
    final_test: tuple[int, int]


def build_v02_folds(num_days: int) -> tuple[ChronologicalFold, ...]:
    """Build the three approved expanding folds without opening final-test data."""
    if num_days <= 0:
        raise ValueError("num_days must be positive")

    def cut(fraction: float) -> int:
        return int(num_days * fraction)

    final_test = (cut(0.85), num_days)
    return (
        ChronologicalFold(
            "v02_fold_a",
            (0, cut(0.55)),
            (cut(0.55), cut(0.65)),
            final_test,
        ),
        ChronologicalFold(
            "v02_fold_b",
            (0, cut(0.65)),
            (cut(0.65), cut(0.75)),
            final_test,
        ),
        ChronologicalFold(
            "v02_fold_c",
            (0, cut(0.75)),
            (cut(0.75), cut(0.85)),
            final_test,
        ),
    )


def select_v02_fold(name: str, num_days: int) -> ChronologicalFold:
    """Return one named v0.2 fold or reject non-protocol split names."""
    for fold in build_v02_folds(num_days):
        if fold.name == name:
            return fold
    raise ValueError(f"unknown v0.2 fold: {name}")
