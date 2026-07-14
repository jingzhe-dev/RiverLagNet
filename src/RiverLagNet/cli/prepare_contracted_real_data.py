"""Prepare real daily data on a monitored HydroRIVERS path-contracted graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RiverLagNet.data.contracted_real_daily import (
    contracted_summary_as_dict,
    prepare_china_contracted_real_daily,
)


def parse_args() -> argparse.Namespace:
    """Parse source paths and auditable graph-selection thresholds."""
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe real daily data on a contracted river graph"
    )
    parser.add_argument("--dynamic-path", type=Path, required=True)
    parser.add_argument("--flags-path", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--hydrorivers-zip", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/china-real-daily-contracted-v0.2"),
    )
    parser.add_argument("--min-target-coverage", type=float, default=0.90)
    parser.add_argument("--min-component-nodes", type=int, default=3)
    parser.add_argument("--max-component-nodes", type=int, default=256)
    parser.add_argument("--component-limit", type=int, default=1)
    parser.add_argument("--travel-speed-km-per-day", type=float, default=30.0)
    parser.add_argument("--max-lag-days", type=int, default=14)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip source SHA256 only for quick local diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the dataset and print the complete audit summary."""
    args = parse_args()
    summary = prepare_china_contracted_real_daily(
        args.dynamic_path,
        args.flags_path,
        args.mapping_path,
        args.hydrorivers_zip,
        args.output_dir,
        min_target_coverage=args.min_target_coverage,
        min_component_nodes=args.min_component_nodes,
        max_component_nodes=args.max_component_nodes,
        component_limit=args.component_limit,
        travel_speed_km_per_day=args.travel_speed_km_per_day,
        max_lag_days=args.max_lag_days,
        hash_sources=not args.skip_source_hashes,
    )
    print(json.dumps(contracted_summary_as_dict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
