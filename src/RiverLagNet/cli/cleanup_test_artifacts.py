"""Remove repository-local pytest and smoke-test artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.testing.cleanup import clean_test_artifacts


def main() -> None:
    """CLI wrapper for safe transient-test cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    removed = clean_test_artifacts(args.root)
    print(f"removed_test_artifacts={len(removed)}")


if __name__ == "__main__":
    main()
