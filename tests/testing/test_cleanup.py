from pathlib import Path

from RiverLagNet.testing.cleanup import clean_test_artifacts


def test_cleanup_removes_only_transient_test_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    transient = (
        root / ".pytest_cache",
        root / "build" / "pytest",
        root / "build" / "smoke",
        root / "src" / "package" / "__pycache__",
        root / "runs" / "smoke_real_data",
        root / "runs" / "test_checkpoint",
    )
    for path in transient:
        path.mkdir(parents=True)
        (path / "artifact.bin").write_bytes(b"temporary")
    formal_run = root / "runs" / "real_lag_v1_s42_learned_lag"
    formal_run.mkdir(parents=True)
    (formal_run / "best.ckpt").write_bytes(b"formal evidence")
    test_source = root / "tests" / "test_model.py"
    test_source.parent.mkdir(parents=True)
    test_source.write_text("def test_model(): pass\n", encoding="utf-8")

    removed = clean_test_artifacts(root)

    assert len(removed) == len(transient)
    assert all(not path.exists() for path in transient)
    assert formal_run.is_dir()
    assert test_source.is_file()
