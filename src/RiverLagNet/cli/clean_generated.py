"""Safely list or remove explicitly allowlisted generated repository artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from RiverLagNet.testing.cleanup import (
    clean_generated_artifacts,
    discover_generated_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    """Print the allowlisted set, deleting it only with ``--apply``."""
    args = _parser().parse_args()
    root = args.root.resolve()
    candidates = discover_generated_artifacts(root)
    removed = clean_generated_artifacts(root, dry_run=not args.apply)
    action = "REMOVE" if args.apply else "DRY-RUN"
    reasons = {candidate.path: candidate.reason for candidate in candidates}
    for path in removed:
        print(f"{action}\t{path.relative_to(root)}\t{reasons[path]}")
    print(f"{action}_COUNT={len(removed)}")


if __name__ == "__main__":
    main()
