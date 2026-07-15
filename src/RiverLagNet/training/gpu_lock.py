"""Atomic single-GPU reservation for RiverLagNet training processes."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType


class SingleGpuLock:
    """Own one atomic JSON lock file for the lifetime of a CUDA run."""

    def __init__(self, lock_path: Path, *, run_name: str) -> None:
        self.lock_path = Path(lock_path)
        self.run_name = run_name
        self._lock_id = uuid.uuid4().hex
        self._acquired = False

    def acquire(self) -> "SingleGpuLock":
        """Atomically reserve the GPU, reclaiming only locks from dead PIDs."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                payload = self._existing_payload()
                pid = payload.get("pid")
                if not isinstance(pid, int) or _pid_is_alive(pid):
                    owner = payload.get("run_name", "unknown")
                    raise RuntimeError(
                        f"GPU is reserved by run {owner!r} (PID {pid!r})"
                    )
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue

            payload = {
                "pid": os.getpid(),
                "run_name": self.run_name,
                "host": socket.gethostname(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "commit": _git_commit(),
                "command": [sys.executable, *sys.argv],
                "lock_id": self._lock_id,
            }
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
            except BaseException:
                self.lock_path.unlink(missing_ok=True)
                raise
            self._acquired = True
            return self
        raise RuntimeError("GPU is reserved; lock ownership changed during acquisition")

    def release(self) -> None:
        """Release this instance's lock without deleting another owner's lock."""
        if not self._acquired:
            return
        try:
            payload = self._existing_payload()
            if payload.get("lock_id") == self._lock_id:
                self.lock_path.unlink(missing_ok=True)
        finally:
            self._acquired = False

    def __enter__(self) -> "SingleGpuLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _existing_payload(self) -> dict[str, object]:
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"GPU is reserved by an unreadable lock: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("GPU is reserved by an invalid lock payload")
        return payload


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return kernel32.GetLastError() == 5
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
