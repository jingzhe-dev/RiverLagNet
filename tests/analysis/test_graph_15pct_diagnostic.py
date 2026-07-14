from __future__ import annotations

import json
from pathlib import Path

from RiverLagNet.analysis.plot_graph_15pct_diagnostic import (
    _load_and_validate,
    build_figure,
)


def test_graph_15pct_audit_is_consistent_and_renders(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    audit_path = root / "experiments" / "graph_15pct_diagnostic.json"

    audit = _load_and_validate(audit_path)
    png, pdf = build_figure(audit_path, tmp_path / "diagnostic")

    assert audit["held_out_test_opened"] is False
    assert audit["mainstem"]["target_relative_gain_percent"] == 15.0
    assert png.is_file() and png.stat().st_size > 10_000
    assert pdf.is_file() and pdf.stat().st_size > 1_000


def test_graph_15pct_audit_rejects_test_split_use(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = json.loads(
        (root / "experiments" / "graph_15pct_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    source["held_out_test_opened"] = True
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(json.dumps(source), encoding="utf-8")

    try:
        _load_and_validate(unsafe)
    except ValueError as error:
        assert "held-out test" in str(error)
    else:
        raise AssertionError("test-split diagnostic must be rejected")
