"""Discover and remove only explicitly allowlisted repository artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


TRANSIENT_DIRECTORIES = (
    Path(".pytest_cache"),
    Path("build") / "pytest",
    Path("build") / "smoke",
)
TRANSIENT_RUN_PREFIXES = ("smoke_", "test_")
PROTECTED_DIRECTORY_PREFIXES = (
    Path("data"),
    Path("runs") / "selected",
)
PROTECTED_FILES = (Path("experiments") / "results.tsv",)


@dataclass(frozen=True)
class CleanupCandidate:
    """One path that matched a named generated-artifact allowlist rule."""

    path: Path
    reason: str


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _is_below(relative: Path, prefix: Path) -> bool:
    return relative == prefix or prefix in relative.parents


def _is_protected(relative: Path) -> bool:
    return relative in PROTECTED_FILES or any(
        _is_below(relative, prefix) for prefix in PROTECTED_DIRECTORY_PREFIXES
    )


def _allowlist_reason(relative: Path) -> str | None:
    if relative in TRANSIENT_DIRECTORIES:
        return "known cache directory"
    if relative.name == "__pycache__":
        return "Python bytecode cache"
    if relative.suffix == ".pyc":
        return "orphan Python bytecode"
    if len(relative.parts) == 1 and relative.suffix == ".log":
        return "repository-root log"
    if relative.name.endswith(".egg-info"):
        return "editable-install metadata"
    if (
        len(relative.parts) == 2
        and relative.parts[0] == "runs"
        and relative.parts[1].startswith(TRANSIENT_RUN_PREFIXES)
    ):
        return "test or smoke run"
    if (
        len(relative.parts) >= 2
        and relative.parts[0] == "runs"
        and relative.name.endswith(".partial.ckpt")
    ):
        return "explicitly partial checkpoint"
    return None


def _collapse_descendants(
    candidates: dict[Path, str],
) -> dict[Path, str]:
    collapsed: dict[Path, str] = {}
    for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
        if any(parent in collapsed for parent in path.parents):
            continue
        collapsed[path] = candidates[path]
    return collapsed


def discover_generated_artifacts(root: Path) -> tuple[CleanupCandidate, ...]:
    """Return existing generated artifacts that match the explicit allowlist."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repository root does not exist: {root}")

    paths: set[Path] = {root / relative for relative in TRANSIENT_DIRECTORIES}
    paths.update(root.rglob("__pycache__"))
    paths.update(root.rglob("*.pyc"))
    paths.update(root.glob("*.log"))
    paths.update(root.rglob("*.egg-info"))
    runs = root / "runs"
    if runs.is_dir():
        paths.update(
            path
            for path in runs.iterdir()
            if path.is_dir() and path.name.startswith(TRANSIENT_RUN_PREFIXES)
        )
        paths.update(runs.rglob("*.partial.ckpt"))

    matched: dict[Path, str] = {}
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise ValueError(f"refusing to clean a symbolic link: {path}")
        resolved = path.resolve()
        if not _inside(root, resolved):
            raise ValueError(f"refusing to delete outside repository root: {resolved}")
        relative = resolved.relative_to(root)
        if _is_protected(relative):
            continue
        reason = _allowlist_reason(relative)
        if reason is None:
            raise ValueError(f"path did not match the cleanup allowlist: {relative}")
        matched[resolved] = reason

    collapsed = _collapse_descendants(matched)
    return tuple(
        CleanupCandidate(path=path, reason=collapsed[path])
        for path in sorted(collapsed, key=str)
    )


def clean_generated_artifacts(
    root: Path,
    *,
    dry_run: bool = True,
) -> tuple[Path, ...]:
    """List or delete the same allowlisted generated-artifact path set."""
    candidates = discover_generated_artifacts(root)
    paths = tuple(candidate.path for candidate in candidates)
    if dry_run:
        return paths

    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    return paths


def clean_test_artifacts(root: Path) -> tuple[Path, ...]:
    """Compatibility wrapper used by pytest teardown to apply the allowlist."""
    return clean_generated_artifacts(root, dry_run=False)
