import pytest
import torch

from RiverLagNet.data.schema import RiverGraph, TARGET_NAMES, TimeSeriesData


def test_schema_preserves_target_order_and_validates_shapes() -> None:
    assert TARGET_NAMES == ("NH3N", "CODMn", "TP")
    graph = RiverGraph(torch.tensor([[0], [1]]), torch.ones(1, 3), torch.ones(2, 2))
    data = TimeSeriesData(
        values=torch.ones(10, 2, 3),
        observed=torch.ones(10, 2, 3, dtype=torch.bool),
        quality=torch.ones(10, 2, 3),
        graph=graph,
    )
    data.validate()
    with pytest.raises(ValueError, match="observed"):
        TimeSeriesData(data.values, torch.ones(9, 2, 3, dtype=torch.bool), data.quality, graph).validate()
