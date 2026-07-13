"""Import verified HydroWQ China processed assets into the local data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RiverLagNet.data.hydrowq_import import import_hydrowq_china, summary_as_dict


def main() -> None:
    """Run a manifest-driven, checksum-verified local import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-processed",
        type=Path,
        required=True,
        help="Path to the seventh-paper data/processed directory",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/processed/hydrowq-china-multibasin-v0.1"),
        help="Ignored local destination inside RiverLagNet",
    )
    args = parser.parse_args()
    summary = import_hydrowq_china(args.source_processed, args.destination)
    print(json.dumps(summary_as_dict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
