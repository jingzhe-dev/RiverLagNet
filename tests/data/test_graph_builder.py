import torch

from RiverLagNet.data.graph_builder import build_graph_variant


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
    assert torch.equal(shuffled_a[0], edges[0])
    assert not torch.equal(shuffled_a[1], edges[1])
