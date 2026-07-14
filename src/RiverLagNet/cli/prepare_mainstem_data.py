"""Prepare a topology-only mainstem dataset from an existing panel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from RiverLagNet.data.mainstem import prepare_mainstem_dataset


def main() -> None:
    """Run deterministic longest-mainstem selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    summary = prepare_mainstem_dataset(
        args.source_dataset, args.output_dir, dataset_id=args.dataset_id
    )
    payload = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(summary).items()
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
