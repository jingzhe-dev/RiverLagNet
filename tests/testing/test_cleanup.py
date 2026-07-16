from pathlib import Path

from RiverLagNet.testing.cleanup import (
    clean_generated_artifacts,
    clean_test_artifacts,
    discover_generated_artifacts,
)


def _write(path: Path, content: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_cleanup_dry_run_and_apply_share_the_same_allowlisted_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    removable = (
        _write(root / ".pytest_cache" / "cache.bin").parent,
        _write(root / "build" / "pytest" / "result.bin").parent,
        _write(root / "build" / "smoke" / "result.bin").parent,
        _write(root / "src" / "package" / "__pycache__" / "module.pyc").parent,
        _write(root / "orphan.pyc"),
        _write(root / "session.log"),
        _write(root / "RiverLagNet.egg-info" / "PKG-INFO").parent,
        _write(root / "runs" / "smoke_real_data" / "artifact.bin").parent,
        _write(root / "runs" / "test_checkpoint" / "artifact.bin").parent,
        _write(root / "runs" / "interrupted" / "model.partial.ckpt"),
    )
    protected = (
        _write(root / "data" / "processed" / "dataset.npz"),
        _write(root / "runs" / "selected" / "best.ckpt"),
        _write(root / "experiments" / "results.tsv"),
        _write(root / "tests" / "test_model.py"),
        _write(root / "src" / "package" / "model.py"),
        _write(root / "runs" / "formal" / "best.ckpt"),
        _write(root / "nested" / "keep.log"),
        _write(root / "build" / "production" / "artifact.bin"),
    )

    discovered = discover_generated_artifacts(root)
    dry_run = clean_generated_artifacts(root, dry_run=True)

    expected = tuple(sorted((path.resolve() for path in removable), key=str))
    assert tuple(candidate.path for candidate in discovered) == expected
    assert dry_run == expected
    assert all(path.exists() for path in removable)
    assert all(path.exists() for path in protected)

    removed = clean_generated_artifacts(root, dry_run=False)

    assert removed == dry_run
    assert all(not path.exists() for path in removable)
    assert all(path.exists() for path in protected)


def test_compatible_test_cleanup_wrapper_applies_the_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    cache = _write(root / "tests" / "__pycache__" / "test_model.pyc").parent
    source = _write(root / "tests" / "test_model.py")

    removed = clean_test_artifacts(root)

    assert removed == (cache.resolve(),)
    assert not cache.exists()
    assert source.is_file()
