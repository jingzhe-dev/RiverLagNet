from __future__ import annotations

import json
from pathlib import Path

import pytest

from RiverLagNet.data.hydrowq_import import import_hydrowq_china


def test_import_copies_only_manifest_assets_and_metadata(
    hydrowq_source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "imported"

    summary = import_hydrowq_china(hydrowq_source, destination)

    assert summary.asset_count == 3
    assert summary.file_count == 7
    assert (destination / "_manifests" / summary.manifest_name).is_file()
    assert (destination / "hydrowq-v0.1" / "datasets" / "china-multibasin-v0.1" / "samples" / "sample-001" / "daily.npz").is_file()
    assert not (destination / "unrelated-large-file.bin").exists()
    receipt = json.loads(summary.receipt_path.read_text(encoding="utf-8"))
    assert receipt["asset_count"] == 3
    assert receipt["source_normalized"] is True
    assert receipt["excluded_categories"] == ["raw", "cache", "logs", "runs"]


def test_import_rejects_a_checksum_mismatch_without_copying_assets(
    hydrowq_source: Path, tmp_path: Path
) -> None:
    manifest_path = (
        hydrowq_source
        / "_manifests"
        / "sample-bundles-china-multibasin-v0.1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundles"][0]["graph_ref"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = tmp_path / "imported"

    with pytest.raises(ValueError, match="checksum"):
        import_hydrowq_china(hydrowq_source, destination)

    assert not destination.exists()
