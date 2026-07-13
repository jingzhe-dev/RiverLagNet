"""Named graph and lag ablation settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    """Graph structure and lag behavior for one controlled ablation."""

    graph_variant: str
    lag_mode: str


ABLATIONS = {
    "no_graph": AblationSpec("no_graph", "no_lag"),
    "undirected_graph": AblationSpec("undirected", "learned_lag"),
    "shuffled_graph": AblationSpec("shuffled", "learned_lag"),
    "no_lag": AblationSpec("directed", "no_lag"),
    "fixed_lag": AblationSpec("directed", "fixed_lag"),
    "learned_lag": AblationSpec("directed", "learned_lag"),
}


def resolve_ablation(name: str) -> AblationSpec:
    """Resolve a required v0.1 ablation name to model settings."""
    try:
        return ABLATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown ablation: {name}") from error
