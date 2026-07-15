# RiverLagNet v0.2 Session D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use paper-figure-style before creating the formal scientific figures. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用三折五种子成对实验、预注册统计和机制可视化验证 RiverLagNet v0.2 是否相对同预算强无图模型提升至少 15%；仅在验证门槛通过后执行一次最终测试，并完成安全 legacy 清理和 v0.2 发布。

**Architecture:** 本会话不再改变模型结构或超参数。它冻结 Session C 的唯一候选和两种无图控制，在 15 个 fold-seed 区组上进行成对确认，并对核心反事实图控制执行同样矩阵。统计层从 ledger/checkpoint/prediction artifacts 派生结果；可视化层读取冻结摘要。最终测试由显式 unlock manifest 保护，使用 `[0,85%)` 数据、预先确定的共同更新数重新训练后，只评估 `[85%,100%)` 一次。

**Tech Stack:** PyTorch/Lightning、NumPy/Polars、SciPy 或 NumPy bootstrap、Matplotlib、paper-figure-style、PyTest、Git。

## Global Constraints

- 仅在 Session C `complete`、D `ready`、唯一候选和全部 hashes 已冻结后开始。
- 本会话禁止 HPO、结构修改、变量修改、fold 修改和挑选种子。
- 确认种子固定为 `42,43,44,45,46`；确认折固定为 A/B/C。
- 所有条件共享数据、物理/effective batch、最大更新数、优化器、early-stop patience、验证频率和硬件配置。
- 正式无图基线是 exact graph-off 与 capacity-matched no-graph 中验证均值更高者；必须同时报告两者。
- 验证成功公式固定为 `100*(mean_graph-mean_no_graph)/abs(mean_no_graph)`。
- 测试集只能在验证成功报告和冻结 commit 产生后打开；不得重跑以选择更有利 checkpoint。
- 删除 legacy 代码/测试必须有替代覆盖和 before/after 全测试证据；禁止全局清理命令。
- routing 图必须标注为模型内部权重/与先验的一致性，不声称因果或真实旅行时间识别。

---

## Task 1: 接管 Session D 并生成不可变确认清单

**Files:**

- Modify: `docs/coordination/riverlagnet-v0.2-status.md`
- Create: `experiments/v0.2_confirmation_manifest.json`

- [ ] **Step 1: 验证前置证据和 GPU**

```powershell
git status --short
git pull --ff-only
Get-Content docs/coordination/riverlagnet-v0.2-status.md -Encoding utf8
Get-Content experiments/v0.2_graph_inputs.json -Encoding utf8
Get-Content experiments/reports/v0.2_graph_search_summary.json -Encoding utf8
Get-Content experiments/reports/v0.2_ablation_summary.json -Encoding utf8
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Expected: C complete, D ready, screening/runtime gates pass, no final-test artifact, no competing GPU process.

- [ ] **Step 2: 生成 hash-locked manifest**

Record dataset, protocol, selected graph config, selected local config, warm-start checkpoint, code commit, hardware profile, confirmation folds/seeds/conditions, common update budget, statistical formulas, success thresholds, and final-test prohibition. Hash the manifest itself after canonical JSON serialization.

Required conditions over all 15 blocks:

```text
directed learned-lag candidate
exact graph-off
capacity-matched no-graph
shuffled graph
undirected graph
static travel-prior lag
```

- [ ] **Step 3: 标记 D 为 `in_progress` 并提交**

```powershell
git add experiments/v0.2_confirmation_manifest.json docs/coordination/riverlagnet-v0.2-status.md
git diff --cached --check
git commit -m "exp: freeze the v0.2 confirmation matrix"
git push
```

No model/config file may change after this commit unless a verified implementation bug invalidates all affected conditions; such a fix requires a new manifest and complete rerun.

---

## Task 2: 增加确认矩阵和结果完整性测试

**Files:**

- Create: `src/RiverLagNet/analysis/v02_confirmation.py`
- Create: `src/RiverLagNet/cli/run_v02_confirmation.py`
- Create: `tests/analysis/test_v02_confirmation.py`

- [ ] **Step 1: 写失败测试冻结 90 个运行单元**

Six conditions × three folds × five seeds = 90 unique specs. Tests must assert equal budgets and hashes, deterministic names, one checkpoint/result per successful spec, resume only on valid ledger + checkpoint, and no test-loader/evaluation command.

```python
specs = build_confirmation_specs(manifest)
assert len(specs) == 90
assert {s.seed for s in specs} == {42, 43, 44, 45, 46}
assert {s.fold for s in specs} == {"v02_fold_a", "v02_fold_b", "v02_fold_c"}
assert all("test" not in " ".join(s.command).lower() for s in specs)
```

- [ ] **Step 2: 实现只执行冻结 manifest 的运行器**

The CLI rejects dirty model/config hashes, unknown conditions, extra Hydra overrides, missing GPU lock, or changed update budgets. It launches one GPU run at a time, writes PID/stdout/stderr, and records crashes append-only.

- [ ] **Step 3: 实现产物核对**

```python
def audit_confirmation_artifacts(
    specs: Sequence[ConfirmationSpec],
    ledger_path: Path,
) -> ConfirmationAudit: ...
```

Audit commit, branch, fold encoded in run metadata, seed, condition, data hash, config hash, checkpoint, finite metrics, duration and hardware JSON. A ledger-only or checkpoint-only spec is incomplete.

- [ ] **Step 4: 测试、dry-run 和提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_confirmation.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.run_v02_confirmation --manifest experiments/v0.2_confirmation_manifest.json --dry-run
git add src/RiverLagNet/analysis/v02_confirmation.py src/RiverLagNet/cli/run_v02_confirmation.py tests/analysis/test_v02_confirmation.py
git diff --cached --check
git commit -m "feat: run the frozen v0.2 confirmation matrix"
git push
```

---

## Task 3: 执行三折五种子确认实验

**Files:**

- Modify: `experiments/results.tsv` (append only through training code)
- Create: `experiments/reports/v0.2_confirmation_audit.json`

- [ ] **Step 1: 后台启动 90 单元矩阵**

Use one hidden process and one GPU lock. Poll the PID, tail log, ledger count and checkpoint count. Communicate progress at least once per hour in a long-running task, but do not start concurrent training.

```powershell
$stdout = 'runs/v02_confirmation/runner.stdout.log'
$stderr = 'runs/v02_confirmation/runner.stderr.log'
$env:PYTHONIOENCODING = 'utf-8'
$process = Start-Process -WindowStyle Hidden -PassThru `
  -FilePath 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' `
  -ArgumentList @('-X','utf8=0','-m','RiverLagNet.cli.run_v02_confirmation','--manifest','experiments/v0.2_confirmation_manifest.json') `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content runs/v02_confirmation/runner.pid
```

- [ ] **Step 2: 处理真实失败而不改变研究协议**

Retry a crashed unit once only after documenting and fixing a reproducible implementation/environment issue. Deterministic nonfinite metrics or repeated OOM invalidate the candidate/hardware contract; stop and mark blocked. Never reduce graph batch alone or drop an unfavorable seed.

- [ ] **Step 3: 审计完整性**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.run_v02_confirmation --manifest experiments/v0.2_confirmation_manifest.json --audit-only --output experiments/reports/v0.2_confirmation_audit.json
```

Expected: 90/90 valid units, no duplicates, all hashes/budgets match, no final-test access.

- [ ] **Step 4: 提交真实结果和审计**

```powershell
git add experiments/results.tsv experiments/reports/v0.2_confirmation_audit.json
git diff --cached --check
git commit -m "exp: complete the v0.2 five-seed confirmation runs"
git push
```

---

## Task 4: 实现成对统计和 15% 成功判定

**Files:**

- Create: `src/RiverLagNet/analysis/v02_statistics.py`
- Create: `tests/analysis/test_v02_statistics.py`
- Create: `src/RiverLagNet/cli/summarize_v02.py`
- Create: `experiments/reports/v0.2_confirmation_summary.json`
- Create: `experiments/reports/v0.2_confirmation_summary.md`

- [ ] **Step 1: 写失败测试固定公式与极端情况**

```python
gain = relative_nse_gain(graph=0.69, baseline=0.60)
assert gain == pytest.approx(15.0)
```

Cover positive/negative/near-zero baseline, paired ordering, deterministic stratified bootstrap, missing block rejection, stronger-baseline selection, win count, target-drop rule, and confidence interval.

- [ ] **Step 2: 实现统计 API**

```python
@dataclass(frozen=True)
class V02SuccessDecision:
    passed: bool
    relative_gain_percent: float
    ci95: tuple[float, float]
    paired_wins: int
    target_mean_deltas: dict[str, float]
    directed_beats_controls: bool
    reasons: tuple[str, ...]
```

Choose the stronger no-graph condition by its mean over 15 blocks, then keep that identity fixed for every paired statistic. Bootstrap paired fold-seed blocks, stratified by fold, with a fixed RNG seed and at least 10,000 draws.

- [ ] **Step 3: 固定成功规则**

All must hold:

```text
relative gain >= 15.0%
95% paired bootstrap CI lower bound > 0
graph wins >= 12 of 15 paired blocks
mean NSE delta for each target >= -0.01
directed mean NSE > shuffled, undirected, and static-lag controls
gain on upstream-eligible nodes > gain on headwaters
all 90 artifacts pass audit
```

- [ ] **Step 4: 生成摘要，禁止先看测试**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.summarize_v02 --confirmation-manifest experiments/v0.2_confirmation_manifest.json --output experiments/reports/v0.2_confirmation_summary.json
```

Summary must state `final_test_opened: false` at this stage.

- [ ] **Step 5: 测试并提交冻结验证结论**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_statistics.py tests/analysis/test_v02_confirmation.py -q
git add src/RiverLagNet/analysis/v02_statistics.py src/RiverLagNet/cli/summarize_v02.py tests/analysis/test_v02_statistics.py experiments/reports/v0.2_confirmation_summary.json experiments/reports/v0.2_confirmation_summary.md
git diff --cached --check
git commit -m "exp: freeze the v0.2 fifteen-percent validation decision"
git push
```

If `passed=false`, skip Tasks 7–8 (final test), continue with validation-only figures and safe cleanup, mark the program as not meeting the target, and do not use test data to revise the conclusion.

---

## Task 5: 生成可追溯的预测级诊断数据

**Files:**

- Create: `src/RiverLagNet/analysis/v02_diagnostics.py`
- Create: `tests/analysis/test_v02_diagnostics.py`
- Create: `experiments/reports/v0.2_diagnostics.json`
- Create: `experiments/reports/v0.2_figure_source.npz`

- [ ] **Step 1: 写失败测试固定切片**

Diagnostics must include target × lead day/lead band, node ΔNSE, headwater/upstream-eligible, downstream topological depth, high-event/normal-event, fold/seed pairs, representative forecast trajectories, routing/prior agreement, learning curves and hardware curves. Every aggregate carries observed counts and masks.

- [ ] **Step 2: 实现从 checkpoint 重新推理的校验**

Prediction artifacts are regenerated only on validation intervals, then reconciled against ledger macro metrics within `1e-6`. Event thresholds are computed from each fold's training data only. Representative nodes/events use a predeclared deterministic rule (median, 10th, 90th gain quantiles), not hand-picked attractive cases.

- [ ] **Step 3: 实现拓扑分组**

Headwater means indegree zero in the directed formal graph. Upstream-eligible means indegree >0 and at least one valid aligned upstream observation. Compute topological depth from direction only; do not use outcomes.

- [ ] **Step 4: 输出小型源数据并测试**

`v0.2_figure_source.npz` contains only aggregated/sampled values needed for figures, no full raw dataset. JSON stores provenance hashes and `final_test_opened=false`.

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_diagnostics.py -q
```

- [ ] **Step 5: 提交**

```powershell
git add src/RiverLagNet/analysis/v02_diagnostics.py tests/analysis/test_v02_diagnostics.py experiments/reports/v0.2_diagnostics.json experiments/reports/v0.2_figure_source.npz
git diff --cached --check
git commit -m "feat: derive validation-only v0.2 diagnostics"
git push
```

---

## Task 6: 按论文风格生成验证可视化

**Files:**

- Create: `src/RiverLagNet/analysis/v02_visualization.py`
- Create: `tests/analysis/test_v02_visualization.py`
- Create: `reports/v0.2/figures/figure1_paired_gain.png`
- Create: `reports/v0.2/figures/figure1_paired_gain.pdf`
- Create: `reports/v0.2/figures/figure2_target_lead.png`
- Create: `reports/v0.2/figures/figure2_target_lead.pdf`
- Create: `reports/v0.2/figures/figure3_network_mechanism.png`
- Create: `reports/v0.2/figures/figure3_network_mechanism.pdf`
- Create: `reports/v0.2/figures/figure4_examples_hardware.png`
- Create: `reports/v0.2/figures/figure4_examples_hardware.pdf`

- [ ] **Step 1: 读取并遵循 `paper-figure-style` skill**

Use its palette, typography, parameterization, render-and-review workflow. All plotting parameters must be editable from a style dataclass/config rather than scattered constants.

- [ ] **Step 2: 写渲染失败测试**

Tests verify PNG/PDF creation, finite dimensions, no missing panels, exact axis labels/units, 15% reference line on the relative-gain axis, consistent target colors, and a visible non-causal routing warning.

- [ ] **Step 3: 实现四张复合图**

1. Paired 15 fold-seed gains, mean/CI, controls, and 15% threshold.
2. Target × lead-day heatmap plus lead-band/event stratification.
3. Node ΔNSE directed network, headwater vs upstream-eligible, downstream depth, routing vs travel prior.
4. Deterministically selected forecast trajectories, learning curves, samples/s, VRAM and wall-time ratios.

Use colorblind-safe palettes, 300 dpi PNG, vector PDF, concise panel labels and no decorative 3-D effects.

- [ ] **Step 4: 渲染并视觉检查**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_visualization.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.analysis.v02_visualization --source experiments/reports/v0.2_figure_source.npz --output-dir reports/v0.2/figures
```

Inspect all PNGs at full resolution; fix clipped labels, overlapping legends, misleading axes or illegible network edges, then rerun tests.

- [ ] **Step 5: 提交图件**

```powershell
git add src/RiverLagNet/analysis/v02_visualization.py tests/analysis/test_v02_visualization.py reports/v0.2/figures
git diff --cached --check
git commit -m "feat: visualize v0.2 paired gains and river mechanisms"
git push
```

---

## Task 7: 仅在验证成功后建立一次性测试解锁

**Files:**

- Create: `src/RiverLagNet/data/final_test_guard.py`
- Create: `tests/data/test_final_test_guard.py`
- Modify: `src/RiverLagNet/cli/evaluate.py`
- Create: `experiments/v0.2_final_test_unlock.json` (only if validation passed)

- [ ] **Step 1: 写失败测试**

Evaluate must reject missing unlock, `passed=false`, changed commit/config/data/report hash, already-consumed unlock, wrong `[85%,100%)` boundary, extra model override, and any attempt before the validation report commit. A valid lock permits exactly the declared models/seeds once.

- [ ] **Step 2: 实现 guard**

```python
@dataclass(frozen=True)
class FinalTestUnlock:
    validation_report_sha256: str
    frozen_commit: str
    dataset_sha256: str
    graph_config_sha256: str
    baseline_config_sha256: str
    seeds: tuple[int, ...]
    final_fit_steps: int
    consumed: bool
```

Do not treat a plain CLI flag as sufficient. Write consumption evidence atomically before metrics are published; a crash leaves an auditable state and requires explanation, not silent rerun.

- [ ] **Step 3: 固定 final-fit 预算**

Derive one common optimizer-step count from the median best-step distribution of the 15 validation pairs before opening test. Train graph and stronger no-graph from the same local initialization on `[0,85%)`, with no validation-based early stopping and identical steps for seeds 42–46.

- [ ] **Step 4: 创建 unlock 并提交冻结点**

Only if `V02SuccessDecision.passed` is true. Include validation report hash and current clean commit. Commit/push the unlock before any final-test data loader is invoked.

- [ ] **Step 5: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_final_test_guard.py tests/integration/test_evaluate_output.py -q
git add src/RiverLagNet/data/final_test_guard.py src/RiverLagNet/cli/evaluate.py tests/data/test_final_test_guard.py experiments/v0.2_final_test_unlock.json
git diff --cached --check
git commit -m "chore: unlock the frozen v0.2 final evaluation"
git push
```

---

## Task 8: 执行一次最终测试并冻结描述性结果

**Files:**

- Create: `experiments/reports/v0.2_final_test.json`
- Create: `experiments/reports/v0.2_final_test.md`
- Modify: `experiments/v0.2_final_test_unlock.json` (mark consumed with artifact hashes)
- Modify: `experiments/reports/v0.2_confirmation_summary.json`
- Modify: `experiments/reports/v0.2_confirmation_summary.md`

- [ ] **Step 1: 训练固定 final-fit 模型**

Run graph and the preselected stronger no-graph baseline for seeds 42–46, full development interval `[0,85%)`, identical final-fit steps. No test metric is available during training.

- [ ] **Step 2: 一次评估 `[85%,100%)`**

Use the guard-protected evaluator once per declared seed/model. Write per-seed and aggregate NSE/MAE/RMSE, target and horizon metrics. The test result is descriptive confirmation and may not change model/config/threshold.

- [ ] **Step 3: 消费 unlock 并校验产物**

Record output hashes, timestamps and completed model-seed set. Ensure there are exactly 10 final evaluation records and no alternate checkpoints.

- [ ] **Step 4: 更新报告但保留验证主结论**

The headline 15% decision remains based on validation confirmation. Present test relative gain separately, including if it is smaller or negative. Do not remove unfavorable seeds.

- [ ] **Step 5: 提交**

```powershell
git add experiments/v0.2_final_test_unlock.json experiments/reports/v0.2_final_test.json experiments/reports/v0.2_final_test.md experiments/reports/v0.2_confirmation_summary.json experiments/reports/v0.2_confirmation_summary.md
git diff --cached --check
git commit -m "exp: report the one-time v0.2 final test"
git push
```

---

## Task 9: 安全退役 legacy 资产并清理目录

**Files:**

- Modify: `docs/legacy_inventory.md`
- Create: `docs/legacy_retirement.md`
- Delete conditionally: only files marked `retire-in-D` with replacement coverage
- Modify: `README.md`

- [ ] **Step 1: 记录删除前完整测试证据**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q | Tee-Object reports/v0.2/pytest-before-retirement.txt
```

- [ ] **Step 2: 对照 legacy 清单逐文件核验**

For each candidate production module/config/test, identify replacement module and replacement test. Keep historical analysis needed to reproduce ledger/report. Do not delete a test while the production behavior remains supported.

- [ ] **Step 3: 使用显式 `git rm` 删除已证明退役的文件**

No wildcard recursive deletion. Update imports/config package lists/docs in the same change. If any deletion causes reduced essential coverage, restore/rework it rather than lowering assertions.

- [ ] **Step 4: 运行白名单 generated cleanup**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.clean_generated --dry-run
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.clean_generated --apply
```

Preserve formal reports, selected checkpoints required for reproducibility, manifests, data and ledger.

- [ ] **Step 5: 删除后完整测试并比较**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q | Tee-Object reports/v0.2/pytest-after-retirement.txt
```

Expected: all remaining tests pass; required coverage categories remain documented. Test count may decrease only by tests coupled exclusively to removed production modules.

- [ ] **Step 6: 提交**

```powershell
git add -A
git diff --cached --check
git commit -m "refactor: retire superseded v0.1 experiment assets"
git push
```

---

## Task 10: 发布前总验证、自审和交付

**Files:**

- Modify: `pyproject.toml`
- Create: `CHANGELOG.md` or modify existing changelog
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/coordination/riverlagnet-v0.2-status.md`
- Create: `reports/v0.2/final_report.md`

- [ ] **Step 1: 写最终报告**

Include objective, protocol, data hash, architecture boundary vs TimeXer/TimeMixer, local baseline selection, signal gate, graph screening, 15-pair confirmation, confidence interval, target/horizon/topology/event results, controls, hardware efficiency, one-time test if authorized, limitations and non-causal warning. Link every table/figure to a JSON/NPZ source hash.

- [ ] **Step 2: 更新版本与使用说明**

Set project version to `0.2.0`; document selected configs, exact reproduction commands, GPU lock, final-test guard, reports and retained artifacts.

- [ ] **Step 3: 执行最终验证**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=synthetic model=riverlagnet_v02_selected trainer=blackwell_96gb trainer.fast_dev_run=true experiment.record_result=false
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=china_real_daily_contracted_1068_v02 model=riverlagnet_v02_selected trainer=blackwell_96gb data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
git status --short
```

- [ ] **Step 4: 最终自审**

Check all success clauses from source artifacts; reconcile ledger/checkpoints/reports; verify no hidden test-based selection; verify graph/no-graph budget and parameter ratios; open every final figure; verify cleanup preserved reproducibility; inspect `git diff` and `git diff --check`.

- [ ] **Step 5: 更新最终状态**

If validation passed: mark D `complete`, record whether test supported the result, final commit and evidence. If validation failed: mark D `blocked` with exact failed clauses and `final_test_opened=false`; do not claim the 15% goal. No new architecture work occurs in this session.

- [ ] **Step 6: 提交、推送和可选发布标签**

```powershell
git add pyproject.toml CHANGELOG.md README.md docs/architecture.md docs/coordination/riverlagnet-v0.2-status.md reports/v0.2/final_report.md reports/v0.2/pytest-before-retirement.txt reports/v0.2/pytest-after-retirement.txt
git diff --cached --check
git commit -m "docs: release RiverLagNet v0.2 validation package"
git push
git tag -a v0.2.0 -m "RiverLagNet v0.2" # only when validation passed and the tag does not exist
git push origin v0.2.0                 # same condition
git status --short
git log -1 --oneline
```

Report the validation decision first, then relative gain/CI/wins/target deltas/control ordering, test result if opened, hardware efficiency, figures, cleanup, test count, commit and tag. Then stop.

## Session D Success Gate

RiverLagNet v0.2 meets the user goal only if:

- 90/90 frozen confirmation runs are complete and auditable;
- the stronger no-graph comparator was selected before pairwise statistics;
- mean validation relative NSE gain is >=15% across 3 folds × 5 seeds;
- paired bootstrap 95% CI lower bound is >0;
- directed wins >=12/15 paired blocks;
- no target mean NSE delta is below -0.01;
- directed beats shuffled, undirected and static-lag controls;
- graph gain is concentrated in upstream-eligible nodes rather than headwaters;
- graph wall-time ratio remains <=3× and all hardware evidence is reported;
- figures and reports are source-grounded and explicitly non-causal;
- final test was either correctly kept closed after failure or opened exactly once after success;
- legacy cleanup has before/after passing full-test evidence;
- final branch is clean, committed and pushed.
