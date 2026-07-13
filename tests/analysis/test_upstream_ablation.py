import pytest

from RiverLagNet.analysis.upstream_ablation import resolve_ablation


@pytest.mark.parametrize(
    ("name", "graph_variant", "lag_mode"),
    [
        ("no_graph", "no_graph", "no_lag"),
        ("undirected_graph", "undirected", "learned_lag"),
        ("shuffled_graph", "shuffled", "learned_lag"),
        ("no_lag", "directed", "no_lag"),
        ("fixed_lag", "directed", "fixed_lag"),
        ("learned_lag", "directed", "learned_lag"),
    ],
)
def test_named_ablation_resolves_graph_and_lag_modes(
    name: str, graph_variant: str, lag_mode: str
) -> None:
    resolved = resolve_ablation(name)
    assert resolved.graph_variant == graph_variant
    assert resolved.lag_mode == lag_mode
