# Complete Training Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate all four required synthetic baselines under one fair budget while automatically preserving validation metrics and runtime evidence in the version-controlled experiment ledger.

**Architecture:** A focused experiment-ledger module owns TSV validation and append behavior. The existing Lightning `run()` function validates the best checkpoint after fitting, gathers metrics plus `RuntimeStatsCallback`, and records one row for full runs only. Four separate Hydra CLI invocations use unique experiment names and ignored run directories.

**Tech Stack:** Python 3.10, PyTorch 2.10, Lightning 2.6, Hydra/OmegaConf, pytest, TSV, Git.

## Global Constraints

- Use Conda environment `DeepWater`; do not create a new environment.
- Use `data=synthetic`, seed 42, `T_in=90`, `T_out=30`, and the existing chronological 70/15/15 target split.
- Use identical default 50-epoch, patience-8, gradient-clipping, mixed-precision training settings for learned models.
- Select checkpoints only by validation macro NSE; do not use test metrics for selection or tuning.
- Do not commit `runs/`, checkpoints, raw data, or imported processed arrays.
- Result status must be one of `baseline`, `keep`, `discard`, or `crash`.

---

### Task 1: Validated experiment ledger

**Files:**
- Create: `src/RiverLagNet/training/experiment_log.py`
- Create: `tests/training/test_experiment_log.py`

**Interfaces:**
- Produces: `ExperimentRecord` frozen dataclass and `append_experiment_record(path: Path, record: ExperimentRecord) -> None`.
- TSV columns exactly match the header in `experiments/results.tsv`.

- [ ] **Step 1: Write failing schema and append tests**

```python
record = ExperimentRecord(timestamp="2026-07-13T00:00:00+00:00", commit="abc", branch="main", experiment="station_gru", seed=42, val_macro_nse=0.1, val_macro_mae=0.2, val_macro_rmse=0.3, duration_s=1.0, peak_vram_gb=0.1, status="baseline", description="synthetic")
append_experiment_record(path, record)
assert path.read_text(encoding="utf-8").splitlines()[1].split("\t")[3] == "station_gru"
with pytest.raises(ValueError, match="status"):
    append_experiment_record(path, replace(record, status="success"))
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/training/test_experiment_log.py -q`

Expected: collection fails because `RiverLagNet.training.experiment_log` does not exist.

- [ ] **Step 3: Implement strict serialization and append**

Implement the dataclass, exact field tuple, allowed status set, tab/newline rejection for text fields, header creation for a missing/empty file, existing-header validation, UTF-8 append, and stable numeric string conversion. Create parent directories when needed.

- [ ] **Step 4: Run focused tests**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/training/test_experiment_log.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/RiverLagNet/training/experiment_log.py tests/training/test_experiment_log.py
git commit -m "feat: add validated experiment ledger"
```

### Task 2: Best-checkpoint validation and automatic recording

**Files:**
- Modify: `src/RiverLagNet/cli/train.py`
- Modify: `configs/experiment/default.yaml`
- Create: `tests/integration/test_experiment_recording.py`

**Interfaces:**
- `run(cfg: DictConfig) -> dict[str, Any]` continues returning trainer/module/datamodule/checkpoint and adds `validation_metrics` plus `experiment_record`.
- Full runs call `trainer.validate(..., ckpt_path="best")`, then append `ExperimentRecord` to `experiments/results.tsv`.
- Fast-dev runs do not validate a checkpoint and do not write the ledger.

- [ ] **Step 1: Write a failing integration test**

Compose the existing Hydra config with `trainer.max_epochs=1`, a temporary `run_dir`, temporary `experiment.results_path`, `experiment.record_result=true`, and `trainer.enable_progress_bar=false`. Assert one ledger row, a non-empty checkpoint, and `val_macro_nse`, `val_macro_mae`, and `val_macro_rmse` in `result["validation_metrics"]`. Add a fast-dev case asserting no ledger file.

- [ ] **Step 2: Run the focused integration test and verify failure**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/integration/test_experiment_recording.py -q`

Expected: failure because `run()` does not record results.

- [ ] **Step 3: Integrate recording**

Add `status: baseline`, `record_result: true`, and `results_path: experiments/results.tsv` to the experiment config. In `run()`, locate `RuntimeStatsCallback`, validate the selected checkpoint, convert tensor metrics to floats, obtain Git commit/branch with non-interactive subprocess calls, construct the record, append it, and return it. Suppress recording for `fast_dev_run`.

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/integration/test_experiment_recording.py -q
C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest -q
```

Expected: all tests pass and test temporary directories contain the only test ledger rows.

- [ ] **Step 5: Commit**

```powershell
git add configs/experiment/default.yaml src/RiverLagNet/cli/train.py tests/integration/test_experiment_recording.py
git commit -m "feat: record validation-selected training results"
```

### Task 3: Four-model training and held-out evaluation

**Files:**
- Modify: `experiments/results.tsv` through the training CLI only
- Modify: `README.md`
- Modify: `docs/agent_errors.md` only if an agent-caused error occurs

**Interfaces:**
- Consumes the standard Hydra entrypoints and automatic ledger recording from Task 2.
- Produces four best checkpoints under ignored run directories and four `baseline` ledger rows.

- [ ] **Step 1: Commit all training code before experiments**

Run: `git status --short` and require a clean worktree.

- [ ] **Step 2: Run the four full training jobs sequentially**

For each model in `persistence`, `station_gru`, `static_gat`, `riverlagnet`, run:

```powershell
$env:PYTHONUTF8='1'
conda run -n DeepWater python -m RiverLagNet.cli.train model=<model> experiment.name=synthetic_seed42_<model> run_dir=runs/synthetic_seed42_<model> trainer.enable_progress_bar=false
```

Expected: exit 0, best checkpoint path printed, and one new ledger row.

- [ ] **Step 3: Evaluate every best checkpoint once on the held-out test split**

Run `python -m RiverLagNet.cli.evaluate model=<model> checkpoint_path=<best.ckpt>` for each model. Preserve outputs in the ignored run directory. Do not copy test metrics into selection logic.

- [ ] **Step 4: Verify artifacts and ledger**

Assert four ledger rows have non-empty validation metrics, allowed status, seed 42, unique experiment names, and the same committed code revision. Load each checkpoint and ensure evaluation exits 0. Confirm `git check-ignore runs/...` succeeds.

- [ ] **Step 5: Document exact commands and limitations**

Update README with the four-model training procedure, ledger behavior, and the synthetic-only interpretation. Record only agent errors that actually occurred.

- [ ] **Step 6: Final verification**

Run:

```powershell
conda run -n DeepWater python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass; only README, ledger, and any actual error-log update are tracked changes; all `runs/` outputs are ignored.

- [ ] **Step 7: Commit and push**

```powershell
git add README.md experiments/results.tsv docs/agent_errors.md
git commit -m "exp: complete synthetic model comparison"
git push origin main
```
