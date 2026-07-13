"""Prepare the real daily NH3N/CODMn/TP training artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RiverLagNet.data.real_daily import prepare_china_real_daily, summary_as_dict


def parse_args() -> argparse.Namespace:
    """Parse explicit source and destination paths."""
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe real daily China water-quality data"
    )
    parser.add_argument("--dynamic-path", type=Path, required=True)
    parser.add_argument("--flags-path", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/china-real-daily-v0.1"),
    )
    parser.add_argument("--edge-normalization-path", type=Path)
    parser.add_argument("--travel-speed-km-per-day", type=float, default=30.0)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip SHA256 only for quick local diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare data and print its audit summary as JSON."""
    args = parse_args()
    summary = prepare_china_real_daily(
        args.dynamic_path,
        args.flags_path,
        args.mapping_path,
        args.graph_root,
        args.output_dir,
        edge_normalization_path=args.edge_normalization_path,
        travel_speed_km_per_day=args.travel_speed_km_per_day,
        hash_sources=not args.skip_source_hashes,
    )
    print(json.dumps(summary_as_dict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
