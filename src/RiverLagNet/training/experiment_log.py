"""Validated append-only experiment ledger for reproducible training runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path


EXPERIMENT_FIELDS = (
    "timestamp",
    "commit",
    "branch",
    "experiment",
    "seed",
    "val_macro_nse",
    "val_macro_mae",
    "val_macro_rmse",
    "duration_s",
    "peak_vram_gb",
    "status",
    "description",
)
ALLOWED_STATUSES = frozenset({"baseline", "keep", "discard", "crash"})


@dataclass(frozen=True)
class ExperimentRecord:
    """One immutable row in ``experiments/results.tsv``."""

    timestamp: str
    commit: str
    branch: str
    experiment: str
    seed: int
    val_macro_nse: float | None
    val_macro_mae: float | None
    val_macro_rmse: float | None
    duration_s: float | None
    peak_vram_gb: float | None
    status: str
    description: str


def _serialize(record: ExperimentRecord) -> list[str]:
    if tuple(field.name for field in fields(record)) != EXPERIMENT_FIELDS:
        raise ValueError("experiment record fields do not match ledger schema")
    if record.status not in ALLOWED_STATUSES:
        raise ValueError(f"invalid experiment status: {record.status}")
    row: list[str] = []
    for name in EXPERIMENT_FIELDS:
        value = getattr(record, name)
        text = "" if value is None else str(value)
        if "\t" in text or "\n" in text or "\r" in text:
            raise ValueError(f"experiment field {name} contains a TSV line break")
        row.append(text)
    return row


def append_experiment_record(path: Path, record: ExperimentRecord) -> None:
    """Validate and append one UTF-8 row without duplicating the header."""
    row = _serialize(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_header = next(csv.reader(handle, delimiter="\t"), [])
        if tuple(existing_header) != EXPERIMENT_FIELDS:
            raise ValueError("existing experiment ledger header does not match schema")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if needs_header:
            writer.writerow(EXPERIMENT_FIELDS)
        writer.writerow(row)
