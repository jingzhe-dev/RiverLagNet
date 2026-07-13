"""Safely remove repository-local artifacts produced by tests and smoke runs."""

from __future__ import annotations

import shutil
from pathlib import Path


TRANSIENT_DIRECTORIES = (
    Path(".pytest_cache"),
    Path("build") / "pytest",
    Path("build") / "smoke",
)
TRANSIENT_RUN_PREFIXES = ("smoke_", "test_")


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def clean_test_artifacts(root: Path) -> tuple[Path, ...]:
    """Delete only known transient outputs below ``root`` and return removed paths."""
    root = Path(root).resolve()
    candidates = {root / relative for relative in TRANSIENT_DIRECTORIES}
    candidates.update(root.rglob("__pycache__"))
    runs = root / "runs"
    if runs.is_dir():
        candidates.update(
            path
            for path in runs.iterdir()
            if path.is_dir() and path.name.startswith(TRANSIENT_RUN_PREFIXES)
        )

    removed: list[Path] = []
    for path in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
        resolved = path.resolve()
        if not _inside(root, resolved):
            raise ValueError(f"refusing to delete outside repository root: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
            removed.append(resolved)
    return tuple(removed)
