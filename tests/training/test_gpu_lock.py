from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from RiverLagNet.training.gpu_lock import SingleGpuLock


def test_second_process_cannot_take_same_gpu_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    code = (
        "from pathlib import Path; "
        "from RiverLagNet.training.gpu_lock import SingleGpuLock; "
        "import sys; "
        "\ntry:\n SingleGpuLock(Path(sys.argv[1]), run_name='child').acquire()"
        "\nexcept RuntimeError as error:\n print(error)\nelse:\n raise SystemExit(2)"
    )

    with SingleGpuLock(lock_path, run_name="parent"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(lock_path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    assert result.returncode == 0
    assert "GPU is reserved" in result.stdout


def test_gpu_lock_reclaims_only_dead_pid_and_context_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")

    with SingleGpuLock(lock_path, run_name="replacement"):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["run_name"] == "replacement"
        assert set(payload) >= {"host", "timestamp", "commit", "command"}

    assert not lock_path.exists()
    lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="GPU is reserved"):
        SingleGpuLock(lock_path, run_name="must-not-reclaim").acquire()
