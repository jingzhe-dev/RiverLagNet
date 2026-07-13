# Multi-seed Robustness and Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a reproducible 45-job, five-seed baseline and RiverLagNet ablation suite, aggregate validation evidence automatically, and evaluate the five validation-selected full-model checkpoints once on held-out test data.

**Architecture:** A pure experiment-suite module defines the condition matrix and builds Hydra subprocess commands; the CLI runs only pending jobs based on the append-only ledger. A separate pure summary module validates the completed matrix, calculates condition statistics and paired ablation deltas, and renders JSON plus Markdown. The evaluation CLI returns machine-readable test metrics so only the final learned-lag checkpoints are aggregated after validation decisions are frozen.

**Tech Stack:** Python 3.10, PyTorch 2.10, Lightning 2.6, Hydra/OmegaConf, standard-library CSV/JSON/statistics/subprocess, pytest.

## Global Constraints

- Work on `research/20260713-multiseed-ablation` and push without force.
- Use seeds exactly `42, 43, 44, 45, 46`.
- Use the existing synthetic data configuration, 50-epoch maximum, patience 8, batch size 16, and `16-mixed` on GPU.
- Select checkpoints only by validation macro NSE.
- Do not evaluate held-out test data until all 45 validation runs are complete and summarized.
- Do not manually edit numeric results or remove crash rows.
- Do not commit `runs/`, checkpoints, TensorBoard logs, or raw Caravan archives.
- Status remains `baseline` for this first complete robustness matrix; mechanism decisions live in the generated report.

---

### Task 1: Exact resumable experiment matrix

**Files:**
- Create: `src/RiverLagNet/analysis/experiment_suite.py`
- Create: `src/RiverLagNet/cli/run_experiment_suite.py`
- Create: `tests/analysis/test_experiment_suite.py`

**Interfaces:**
- `SuiteCondition(name: str, model: str, model_overrides: tuple[str, ...])`
- `ExperimentSpec(seed: int, condition: SuiteCondition, experiment_name: str, run_dir: Path)`
- `build_experiment_specs(seeds: Sequence[int]) -> tuple[ExperimentSpec, ...]`
- `successful_experiment_names(ledger_path: Path) -> set[str]`
- `pending_experiment_specs(specs, successful_names) -> tuple[ExperimentSpec, ...]`
- `training_command(spec: ExperimentSpec, python_executable: str) -> list[str]`
- `run_experiment_specs(specs, python_executable, dry_run=False) -> list[list[str]]`

- [ ] **Step 1: Write failing matrix tests**

Test that five seeds create exactly 45 unique names, nine conditions per seed, and condition names exactly `persistence`, `station_gru`, `static_gat`, `no_graph`, `undirected_graph`, `shuffled_graph`, `no_lag`, `fixed_lag`, `learned_lag`. Assert representative commands include `model=riverlagnet`, `model.graph_variant=undirected`, or `model.lag_mode=fixed_lag` as appropriate.

- [ ] **Step 2: Write failing resume tests**

Create a miniature ledger containing one `baseline`, one `keep`, one `discard`, and one `crash` row. Assert the first three names are successful, the crash name remains pending, and dry-run returns commands without invoking subprocess.

- [ ] **Step 3: Run tests to verify the missing-module failure**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/analysis/test_experiment_suite.py -q`

Expected: collection fails because `RiverLagNet.analysis.experiment_suite` does not exist.

- [ ] **Step 4: Implement the immutable matrix and runner**

Define the nine conditions in one tuple. Construct names as `robust_s{seed}_{condition}` and run directories as `runs/<name>`. Commands must invoke `python -m RiverLagNet.cli.train`, include seed/model/name/run-dir/progress-bar/status overrides, and append condition-specific model overrides. Use `subprocess.run(..., check=True, env={**os.environ, "PYTHONUTF8": "1"})`. Dry-run must not call subprocess.

- [ ] **Step 5: Implement the CLI**

Use `argparse` with `--seeds` defaulting to all five seeds, `--ledger` defaulting to `experiments/results.tsv`, and `--dry-run`. Print total, successful, pending, and each command. Run only pending specs.

- [ ] **Step 6: Run focused tests and commit**

Run the focused tests and `git diff --check`; expect all to pass.

Commit:

```powershell
git add src/RiverLagNet/analysis/experiment_suite.py src/RiverLagNet/cli/run_experiment_suite.py tests/analysis/test_experiment_suite.py
git commit -m "feat: add resumable robustness experiment suite"
```

### Task 2: Ledger-derived robustness summary

**Files:**
- Create: `src/RiverLagNet/analysis/robustness_summary.py`
- Create: `tests/analysis/test_robustness_summary.py`
- Modify: `src/RiverLagNet/cli/run_experiment_suite.py`

**Interfaces:**
- `load_successful_suite_rows(path: Path, specs) -> list[dict[str, str]]`
- `summarize_validation(rows, specs) -> dict[str, object]`
- `render_validation_markdown(summary: Mapping[str, object]) -> str`
- `write_validation_summary(summary, json_path: Path, markdown_path: Path) -> None`

- [ ] **Step 1: Write failing summary tests**

Build a two-seed miniature ledger with all nine conditions. Assert per-condition count/mean/sample-standard-deviation/min/max for macro NSE/MAE/RMSE, and assert paired deltas use `learned_lag - ablation`. Verify the Markdown contains the condition table, paired-delta table, seed list, and synthetic-only caveat.

- [ ] **Step 2: Run tests to verify the missing-module failure**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/analysis/test_robustness_summary.py -q`

Expected: collection fails because the summary module does not exist.

- [ ] **Step 3: Implement strict matrix validation and statistics**

Ignore crash rows only when a later successful row with the same experiment exists. Reject missing experiment names, duplicate successful names, unexpected seeds, non-numeric metrics, or a matrix whose successful rows span more than one Git commit. Calculate sample standard deviation with zero for a one-row group.

- [ ] **Step 4: Add CLI summary outputs**

Add `--summarize`, `--summary-json` defaulting to `experiments/robustness_summary.json`, and `--summary-markdown` defaulting to `docs/robustness_report_2026-07-13.md`. Summary mode must require the exact successful matrix and must not launch training.

- [ ] **Step 5: Run focused and full tests, then commit**

Run both analysis test files and the full `pytest -q`; expect all tests to pass.

Commit:

```powershell
git add src/RiverLagNet/analysis/robustness_summary.py src/RiverLagNet/cli/run_experiment_suite.py tests/analysis/test_robustness_summary.py
git commit -m "feat: summarize paired robustness experiments"
```

### Task 3: Machine-readable final checkpoint evaluation

**Files:**
- Modify: `configs/config.yaml`
- Modify: `src/RiverLagNet/configs/config.yaml`
- Modify: `src/RiverLagNet/cli/evaluate.py`
- Create: `tests/integration/test_evaluate_output.py`
- Modify: `src/RiverLagNet/analysis/robustness_summary.py`
- Modify: `src/RiverLagNet/cli/run_experiment_suite.py`

**Interfaces:**
- `evaluate.run(cfg: DictConfig) -> dict[str, float]`
- `evaluation_output: null` in both root and packaged Hydra configs
- `aggregate_test_metrics(paths_by_seed: Mapping[int, Path]) -> dict[str, object]`

- [ ] **Step 1: Write a failing evaluation-output integration test**

Train a one-epoch Station GRU checkpoint in a temporary directory, compose the evaluation config with `evaluation_output=<temporary JSON>`, call `evaluate.run(cfg)`, and assert the returned mapping and JSON contain test macro NSE/MAE/RMSE plus the three target NSE values as finite floats.

- [ ] **Step 2: Run the test and verify failure**

Run: `C:\Program Files\ANACONDA\envs\DeepWater\python.exe -m pytest tests/integration/test_evaluate_output.py -q`

Expected: import or attribute failure because `evaluate.run` does not exist.

- [ ] **Step 3: Extract reusable evaluation and JSON serialization**

Move the current evaluation body into `run()`, call `trainer.test(..., verbose=False)`, convert tensor/numeric values to floats, write UTF-8 JSON only when `evaluation_output` is non-null, and return the mapping. Keep Hydra `main()` as a thin wrapper that prints the JSON mapping.

- [ ] **Step 4: Add final-evaluation suite mode**

Add `--evaluate-final` to the suite CLI. Require all 45 validation rows, locate exactly one learned-lag checkpoint per seed, and invoke the evaluation module with matching seed/model and a quoted checkpoint override. Write ignored JSON to `runs/robust_s<seed>_learned_lag/test_metrics.json`. Never evaluate baseline or ablation checkpoints in this mode.

- [ ] **Step 5: Aggregate final test metrics into the summary**

Read the five JSON files only after validation summary creation. Report mean/sample-standard-deviation/min/max for all seven test metrics and append a visible held-out-test section to the generated Markdown.

- [ ] **Step 6: Run focused/full tests and commit**

Run the integration test, both analysis test files, packaging consistency test, and full `pytest -q`; expect all tests to pass.

Commit:

```powershell
git add configs/config.yaml src/RiverLagNet/configs/config.yaml src/RiverLagNet/cli/evaluate.py src/RiverLagNet/analysis/robustness_summary.py src/RiverLagNet/cli/run_experiment_suite.py tests/integration/test_evaluate_output.py tests/analysis
git commit -m "feat: record final checkpoint test metrics"
```

### Task 4: Execute, verify, document, and push the robustness study

**Files:**
- Modify automatically: `experiments/results.tsv`
- Create automatically: `experiments/robustness_summary.json`
- Create automatically: `docs/robustness_report_2026-07-13.md`
- Modify: `README.md`
- Modify: `docs/agent_errors.md` only for actual agent-caused errors

**Interfaces:**
- Consumes the committed suite CLI and `DeepWater` environment.
- Produces 45 successful validation rows, five ignored test JSON files, one version-controlled JSON summary, and one version-controlled Markdown report.

- [ ] **Step 1: Verify a clean committed implementation**

Run `git status --short`, `git branch --show-current`, and `git rev-parse HEAD`. Require the research branch and a clean tree. Install the committed package with `conda run -n DeepWater python -m pip install .`.

- [ ] **Step 2: Dry-run the exact matrix**

Run:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --dry-run
```

Expected: 45 unique pending commands before the first run.

- [ ] **Step 3: Execute all pending training jobs**

Run:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite
```

Expected: every job exits zero or writes a crash row and stops. After fixing any agent-caused defect through systematic debugging and a new commit, rerun; successful names skip and crash names rerun.

- [ ] **Step 4: Freeze validation summary**

Run:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --summarize
```

Expected: exact 45-row/single-commit validation matrix and generated JSON/Markdown outputs.

- [ ] **Step 5: Evaluate only final learned-lag checkpoints**

Run:

```powershell
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --evaluate-final
conda run -n DeepWater python -m RiverLagNet.cli.run_experiment_suite --summarize
```

Expected: five ignored JSON test files and a report containing aggregate held-out metrics.

- [ ] **Step 6: Validate evidence and artifacts**

Programmatically assert 45 unique successful experiment names, five seeds per condition, one shared code commit, no test file before the freeze point in command history, all expected checkpoints present, and all run artifacts ignored.

- [ ] **Step 7: Update README and run final verification**

Document the suite command, key findings, and synthetic/Caravan limitations. Run `conda run -n DeepWater python -m pytest -q`, one GPU fast-dev run, `git diff --check`, and `git status --short`.

- [ ] **Step 8: Commit and push**

```powershell
git add README.md experiments/results.tsv experiments/robustness_summary.json docs/robustness_report_2026-07-13.md docs/agent_errors.md
git commit -m "exp: complete multi-seed ablation study"
git push -u origin research/20260713-multiseed-ablation
```
