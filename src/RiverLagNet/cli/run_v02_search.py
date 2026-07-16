"""Run the preregistered v0.2 local successive-halving search."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from RiverLagNet.analysis.v02_search import (
    SearchObservation,
    is_completed_run,
    load_search_protocol,
    promote_configs,
    read_completed_observation,
    stage_run_specs,
)


DEFAULT_PROTOCOL = Path("experiments/v0.2_local_search.yaml")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _claim_master(pid_path: Path) -> None:
    if pid_path.is_file():
        try:
            existing_pid = int(pid_path.read_text(encoding="ascii").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid != os.getpid() and _process_is_alive(existing_pid):
            raise RuntimeError(f"v0.2 local search is already alive as PID {existing_pid}")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")


def _run_search(protocol_path: Path, python_executable: str, repo_root: Path) -> None:
    protocol = load_search_protocol(protocol_path)
    run_root = repo_root / protocol.run_root
    pid_path = run_root / "master.pid"
    state_path = run_root / "state.json"
    _claim_master(pid_path)
    observations: list[SearchObservation] = []
    candidates = protocol.configs
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    environment.pop("PYTHONUTF8", None)
    ledger_path = repo_root / protocol.ledger_path
    try:
        for stage_index, stage in enumerate(protocol.stages):
            specs = stage_run_specs(
                protocol,
                stage,
                candidates,
                repo_root=repo_root,
                python_executable=python_executable,
            )
            stage_observations: list[SearchObservation] = []
            for run_index, spec in enumerate(specs):
                _write_json_atomic(
                    state_path,
                    {
                        "protocol_version": protocol.protocol_version,
                        "status": "running",
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "run_index": run_index,
                        "run_count": len(specs),
                        "experiment": spec.experiment_name,
                    },
                )
                if not is_completed_run(spec, ledger_path):
                    subprocess.run(
                        spec.command,
                        check=True,
                        cwd=repo_root,
                        env=environment,
                    )
                stage_observations.append(
                    read_completed_observation(spec, ledger_path)
                )
            observations.extend(stage_observations)
            if stage_index + 1 < len(protocol.stages):
                next_count = protocol.stages[stage_index + 1].candidate_count
                candidates = promote_configs(
                    protocol, stage_observations, candidate_count=next_count
                )
        winner = promote_configs(protocol, stage_observations, candidate_count=1)[0]
        _write_json_atomic(
            state_path,
            {
                "protocol_version": protocol.protocol_version,
                "status": "complete",
                "winner_config_id": winner.config_id,
                "observation_count": len(observations),
            },
        )
    except Exception as error:
        _write_json_atomic(
            state_path,
            {
                "protocol_version": protocol.protocol_version,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("local",), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and either inspect or execute the frozen search."""
    args = _parser().parse_args(argv)
    repo_root = Path.cwd()
    protocol = load_search_protocol(args.protocol)
    if args.family != protocol.family:
        raise ValueError("requested family does not match the protocol")
    print(
        "stage_counts="
        + "->".join(str(stage.candidate_count) for stage in protocol.stages)
    )
    print(f"protocol_version={protocol.protocol_version}")
    if args.dry_run:
        specs = stage_run_specs(
            protocol,
            protocol.stages[0],
            protocol.configs,
            repo_root=repo_root,
            python_executable=args.python,
        )
        for spec in specs:
            print(subprocess.list2cmdline(spec.command))
        return 0
    _run_search(args.protocol, args.python, repo_root)
    return 0


def main() -> None:
    """Console entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
