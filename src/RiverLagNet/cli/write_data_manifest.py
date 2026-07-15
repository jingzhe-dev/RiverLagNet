"""Write an auditable RiverLagNet dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.data.manifest import inspect_dataset, write_manifest


def main() -> None:
    """Inspect one prepared NPZ and write its raw-value-free manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = inspect_dataset(args.dataset)
    write_manifest(manifest, args.output)
    print(f"Wrote {args.output} with SHA-256 {manifest.sha256}")


if __name__ == "__main__":
    main()
