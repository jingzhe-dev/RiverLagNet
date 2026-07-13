import torch

from RiverLagNet.data.graph_builder import build_graph_variant
from RiverLagNet.data.synthetic_identifiable import generate_identifiable_synthetic_scenario


def test_graph_variants_keep_or_change_direction_deterministically() -> None:
    edges = torch.tensor([[0, 1], [1, 2]])
    attrs = torch.tensor([[1.0], [2.0]])
    directed, directed_attr = build_graph_variant(edges, attrs, "directed", seed=7)
    assert torch.equal(directed, edges)
    assert torch.equal(directed_attr, attrs)
    undirected, undirected_attr = build_graph_variant(edges, attrs, "undirected", seed=7)
    assert torch.equal(undirected, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]))
    assert undirected_attr.shape == (4, 1)
    shuffled_a, _ = build_graph_variant(edges, attrs, "shuffled", seed=7)
    shuffled_b, _ = build_graph_variant(edges, attrs, "shuffled", seed=7)
    assert torch.equal(shuffled_a, shuffled_b)
    assert shuffled_a.shape == edges.shape
    assert not any(source == destination for source, destination in shuffled_a.t().tolist())
    assert not set(map(tuple, shuffled_a.t().tolist())) & set(map(tuple, edges.t().tolist()))


def test_identifiable_shuffled_graphs_have_no_true_edges_or_self_loops() -> None:
    for seed in range(42, 47):
        scenario = generate_identifiable_synthetic_scenario(seed=seed)
        edges = scenario.data.graph.edge_index
        shuffled, shuffled_attr = build_graph_variant(
            edges, scenario.data.graph.edge_attr, "shuffled", seed=seed
        )
        true_pairs = set(map(tuple, edges.t().tolist()))
        shuffled_pairs = set(map(tuple, shuffled.t().tolist()))

        assert shuffled.shape == edges.shape
        assert shuffled_attr.shape == scenario.data.graph.edge_attr.shape
        assert len(shuffled_pairs) == edges.shape[1]
        assert not true_pairs & shuffled_pairs
        assert all(source != destination for source, destination in shuffled_pairs)
