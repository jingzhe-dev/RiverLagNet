"""Prepare the real daily dataset on a contracted monitored HydroRIVERS graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracted_graph import (
    build_contracted_graph,
    profile_segment_observation_coverage,
    read_hydrorivers_attributes,
    write_contracted_graph_assets,
)
from .real_daily import RealDailyPreparationSummary, prepare_china_real_daily


CONTRACTED_DATASET_VERSION = "china-real-daily-contracted-v0.2"


@dataclass(frozen=True)
class ContractedRealDailyPreparationSummary:
    """Paths and counts produced by contracted-graph real-data preparation."""

    dataset: RealDailyPreparationSummary
    coverage_profile_path: Path
    graph_report_path: Path
    graph_root: Path
    graph_report: dict[str, Any]


def prepare_china_contracted_real_daily(
    dynamic_path: str | Path,
    flags_path: str | Path,
    mapping_path: str | Path,
    hydrorivers_zip: str | Path,
    output_dir: str | Path,
    *,
    min_target_coverage: float = 0.90,
    min_component_nodes: int = 3,
    max_component_nodes: int = 256,
    component_limit: int = 1,
    travel_speed_km_per_day: float = 30.0,
    max_lag_days: int = 14,
    hash_sources: bool = True,
    dataset_id: str = CONTRACTED_DATASET_VERSION,
) -> ContractedRealDailyPreparationSummary:
    """Build the audited contracted graph and its leakage-safe daily tensor."""
    hydrorivers_zip = Path(hydrorivers_zip).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not hydrorivers_zip.is_file():
        raise FileNotFoundError(f"HydroRIVERS ZIP not found: {hydrorivers_zip}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = profile_segment_observation_coverage(flags_path, mapping_path)
    segments = read_hydrorivers_attributes(hydrorivers_zip)
    assets = build_contracted_graph(
        segments,
        coverage,
        min_target_coverage=min_target_coverage,
        min_component_nodes=min_component_nodes,
        max_component_nodes=max_component_nodes,
        component_limit=component_limit,
        travel_speed_km_per_day=travel_speed_km_per_day,
        max_lag_days=max_lag_days,
    )

    coverage_profile_path = output_dir / "segment_observation_coverage.parquet"
    coverage.write_parquet(coverage_profile_path)
    graph_root = output_dir / "contracted_graph"
    graph_report_path = write_contracted_graph_assets(assets, graph_root)
    dataset = prepare_china_real_daily(
        dynamic_path,
        flags_path,
        mapping_path,
        graph_root,
        output_dir,
        travel_speed_km_per_day=travel_speed_km_per_day,
        hash_sources=hash_sources,
        dataset_id=dataset_id,
        graph_construction=assets.report,
    )

    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"][str(hydrorivers_zip)] = (
        _sha256(hydrorivers_zip) if hash_sources else None
    )
    manifest["artifact_sha256"].update(
        {
            coverage_profile_path.name: _sha256(coverage_profile_path),
            graph_report_path.name: _sha256(graph_report_path),
        }
    )
    manifest["coverage_profile_path"] = str(coverage_profile_path)
    manifest["graph_construction_report_path"] = str(graph_report_path)
    dataset.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ContractedRealDailyPreparationSummary(
        dataset=dataset,
        coverage_profile_path=coverage_profile_path,
        graph_report_path=graph_report_path,
        graph_root=graph_root,
        graph_report=assets.report,
    )


def contracted_summary_as_dict(
    summary: ContractedRealDailyPreparationSummary,
) -> dict[str, Any]:
    """Return a concise JSON-safe contracted preparation summary."""
    dataset = asdict(summary.dataset)
    for key, value in list(dataset.items()):
        if isinstance(value, Path):
            dataset[key] = str(value)
    return {
        "dataset": dataset,
        "coverage_profile_path": str(summary.coverage_profile_path),
        "graph_report_path": str(summary.graph_report_path),
        "graph_root": str(summary.graph_root),
        "graph_report": summary.graph_report,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
