from __future__ import annotations

from pathlib import Path

import torch

from RiverLagNet.data.hydrowq_import import (
    HydroWQChinaCatalog,
    import_hydrowq_china,
)
from RiverLagNet.data.schema import TARGET_NAMES


def _catalog(hydrowq_source: Path, tmp_path: Path) -> HydroWQChinaCatalog:
    destination = tmp_path / "imported"
    import_hydrowq_china(hydrowq_source, destination)
    return HydroWQChinaCatalog(destination)


def test_catalog_loads_directed_graph_without_reversing_edges(
    hydrowq_source: Path, tmp_path: Path
) -> None:
    catalog = _catalog(hydrowq_source, tmp_path)

    graph = catalog.load_graph("basin-001")

    graph.validate()
    assert torch.equal(graph.edge_index, torch.tensor([[0], [1]]))
    assert graph.edge_attr.shape == (1, 2)
    assert graph.static.shape == (2, 2)


def test_catalog_reorders_required_targets_and_preserves_masks(
    hydrowq_source: Path, tmp_path: Path
) -> None:
    catalog = _catalog(hydrowq_source, tmp_path)

    sample = catalog.load_sample("sample-001")

    assert sample.target_names == TARGET_NAMES
    assert sample.x.shape == (2, 2, 3)
    assert sample.y.shape == (3, 2, 3)
    assert torch.equal(sample.x[0, 0], torch.tensor([20.0, 10.0, 30.0]))
    assert torch.equal(sample.y[0, 0], torch.tensor([21.0, 11.0, 31.0]))
    assert sample.x_mask.dtype == torch.bool
    assert sample.y_mask.dtype == torch.bool
    assert sample.source_normalized is True


def test_catalog_reports_window_contract_incompatibility(
    hydrowq_source: Path, tmp_path: Path
) -> None:
    catalog = _catalog(hydrowq_source, tmp_path)

    default_contract = catalog.compatibility(required_history=90, required_forecast=30)
    fixture_contract = catalog.compatibility(required_history=2, required_forecast=3)

    assert default_contract.compatible is False
    assert default_contract.source_history == 2
    assert default_contract.source_forecast == 3
    assert "history" in " ".join(default_contract.reasons)
    assert "forecast" in " ".join(default_contract.reasons)
    assert fixture_contract.compatible is True
    assert fixture_contract.reasons == ()
