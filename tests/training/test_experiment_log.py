from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from RiverLagNet.training.experiment_log import (
    EXPERIMENT_FIELDS,
    ExperimentRecord,
    append_experiment_record,
)


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        timestamp="2026-07-13T00:00:00+00:00",
        commit="abc123",
        branch="main",
        experiment="synthetic_seed42_station_gru",
        seed=42,
        val_macro_nse=0.1,
        val_macro_mae=0.2,
        val_macro_rmse=0.3,
        duration_s=1.5,
        peak_vram_gb=0.25,
        status="baseline",
        description="synthetic comparison",
    )


def test_append_experiment_record_creates_exact_tsv_schema(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "results.tsv"

    append_experiment_record(path, _record())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == list(EXPERIMENT_FIELDS)
    values = lines[1].split("\t")
    assert values[3] == "synthetic_seed42_station_gru"
    assert values[4] == "42"
    assert values[10] == "baseline"


def test_append_experiment_record_appends_without_rewriting_header(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"

    append_experiment_record(path, _record())
    append_experiment_record(path, replace(_record(), experiment="second"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[2].split("\t")[3] == "second"


def test_append_experiment_record_rejects_invalid_status(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="status"):
        append_experiment_record(
            tmp_path / "results.tsv", replace(_record(), status="success")
        )


def test_append_experiment_record_rejects_an_unexpected_existing_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.tsv"
    path.write_text("wrong\theader\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        append_experiment_record(path, _record())
