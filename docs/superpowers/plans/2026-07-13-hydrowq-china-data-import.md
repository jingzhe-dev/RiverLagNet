# HydroWQ China Data Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy the usable China multi-basin processed assets into ignored RiverLagNet storage with reproducible integrity checks and a safe read-only adapter.

**Architecture:** A manifest-driven importer whitelists referenced NPZ assets and three dataset metadata files, verifies SHA-256 before and after copying, and writes a local receipt. A separate catalog loads graphs and fixed target channels without treating the source's normalized 45→46 samples as RiverLagNet's default 90→30 training data.

**Tech Stack:** Python 3.10+, pathlib, hashlib, shutil, JSON, NumPy, PyTorch, pytest

## Global Constraints

- Source root is configurable; no absolute source path is hard-coded in package code.
- Destination defaults to ignored `data/processed/hydrowq-china-multibasin-v0.1`.
- Never copy source `raw/`, climate rasters, archives, caches, logs, or unrelated datasets.
- Verify all manifest asset SHA-256 digests before and after copying.
- Preserve target order exactly as `NH3N`, `CODMn`, `TP`.
- Source arrays remain marked as source-normalized and must not be normalized again.
- Source 45→46 windows are not declared compatible with default 90→30 training.
- Do not commit imported real data assets to Git.

---

### Task 1: Manifest-driven importer and CLI

**Files:**
- Create: `src/RiverLagNet/data/hydrowq_import.py`
- Create: `src/RiverLagNet/cli/import_hydrowq.py`
- Test: `tests/data/test_hydrowq_import.py`

**Interfaces:**
- Produces: `import_hydrowq_china(source_processed: Path, destination: Path) -> HydroWQImportSummary` and CLI arguments `--source-processed`, `--destination`.

- [ ] Write a temporary miniature source manifest with one sample, graph, static asset, dataset metadata, and an unrelated raw sentinel; assert only whitelisted files are copied and the summary reports exact files and bytes.
- [ ] Run `python -m pytest tests/data/test_hydrowq_import.py -q`; expect import failure because `RiverLagNet.data.hydrowq_import` does not exist.
- [ ] Implement manifest validation, reference collection, source and destination digest checks, metadata copying, and receipt writing using `Path`, `hashlib.sha256`, and `shutil.copy2`.
- [ ] Add a corrupt-digest test that expects `ValueError` before any asset is accepted.
- [ ] Run the importer tests; expect all tests to pass.
- [ ] Commit with `feat: add manifest-driven HydroWQ import`.

### Task 2: Read-only graph and sample catalog

**Files:**
- Modify: `src/RiverLagNet/data/hydrowq_import.py`
- Modify: `src/RiverLagNet/data/__init__.py`
- Test: `tests/data/test_hydrowq_catalog.py`

**Interfaces:**
- Produces: `HydroWQChinaCatalog`, `HydroWQChinaSample`, `HydroWQCompatibility`, `load_graph`, `load_sample`, and `compatibility`.

- [ ] Write tests asserting graph source/destination indices are unchanged, graph/static shapes are valid, and source channels `[CODMn,NH3N,TP]` are returned as `[NH3N,CODMn,TP]`.
- [ ] Write a compatibility test asserting source `45→46` is false for required `90→30`, with both window dimensions in the reason.
- [ ] Run `python -m pytest tests/data/test_hydrowq_catalog.py -q`; expect missing catalog symbols.
- [ ] Implement indexed manifest lookup, safe NPZ loading with `allow_pickle=False`, tensor conversion, channel reordering, explicit source-normalized metadata, and compatibility reporting.
- [ ] Run importer and catalog tests; expect pass.
- [ ] Commit with `feat: add HydroWQ China catalog adapter`.

### Task 3: Actual import, quality audit, and verification

**Files:**
- Modify: `README.md`, `docs/data_schema.md`, `docs/agent_errors.md`
- Create: `docs/data/hydrowq-china-import-audit-2026-07-13.md`
- Local ignored output: `data/processed/hydrowq-china-multibasin-v0.1/**`

**Interfaces:**
- Produces: reproducible import command, evidence-backed quality findings, local receipt, and an actual RiverLagNet forward smoke result.

- [ ] Run the importer CLI against `D:/05.Paper/07.第七篇论文/Code/data/processed`; verify 110 samples, 10 graphs, 10 static assets, metadata, and zero digest failures.
- [ ] Load one imported graph/sample through `HydroWQChinaCatalog`; run a small RiverLagNet forward with the source 45-day history and assert output `[1,46,N,3]` is finite.
- [ ] Document source size exclusions, imported byte count, sample/basin/date counts, mask/range findings, source normalization, split semantics, and 45→46 incompatibility.
- [ ] Run `conda run -n DeepWater python -m pytest -q`; expect all tests pass.
- [ ] Run `git status --short` and confirm imported data remains ignored.
- [ ] Commit with `docs: record HydroWQ China data import`, push `main`, and verify remote HEAD.
