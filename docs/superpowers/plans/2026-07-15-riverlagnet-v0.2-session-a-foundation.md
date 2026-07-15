# RiverLagNet v0.2 Session A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可审计的 v0.2 工程与实验基础：关闭中断运行、保护现有结果、冻结 1068 站数据和三折协议、提高 96 GB Blackwell 的吞吐与显存利用率，并提供安全清理机制。

**Architecture:** 保留现有 Lightning/Hydra 主入口，在数据层增加版本化滚动折与数据指纹，在训练层增加单 GPU 锁、固定更新预算和硬件统计，在分析层提供可复现的批量基准。所有新能力默认向后兼容；旧 70/15/15 行为仅供历史实验读取，v0.2 正式配置必须显式选择 `v02_fold_a/b/c`。

**Tech Stack:** Python 3.10+、PyTorch 2.10、Lightning 2.6、Hydra/OmegaConf、NumPy、PyTest、PowerShell、Conda `DeepWater`、NVIDIA RTX PRO 6000 Blackwell。

## Global Constraints

- 仓库根目录：`D:\05.Paper\05.第五篇论文\Code\RiverLagNet`。
- 本会话只做工程基础和协议，不启动模型搜索或长时正式训练。
- 正式数据唯一入口是 `data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz`。
- 不读取、汇总或绘制 `[85%,100%)` 最终测试段的预测或指标。
- 保留 `experiments/results.tsv` 现有追加行；中断运行不得伪造 ledger 行。
- 禁止 `git clean -fdX`；删除前必须用白名单和绝对路径边界检查。
- 一个时刻只能有一个 GPU 训练/基准进程。
- 每完成一个任务，先运行列出的验证，再提交；不得把失败结果隐藏为成功。

---

## Task 1: 接管状态、关闭旧分支遗留并建立 v0.2 分支

**Files:**

- Modify: `docs/coordination/riverlagnet-v0.2-status.md`
- Preserve/commit: `experiments/results.tsv`
- Preserve: `runs/real1068ext_d3r_v37_s42_no_graph/` until audited

- [ ] **Step 1: 核对仓库、远程和 GPU，不修改任何文件**

Run:

```powershell
git rev-parse --show-toplevel
git status --short
git fetch origin
git log -3 --oneline --decorate
git branch -vv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Expected: repository root is `...\RiverLagNet`; current branch is `research/20260714-graph-15pct`; no active RiverLagNet Python GPU process exists. If a training process exists, stop and report instead of competing for the GPU.

- [ ] **Step 2: 审计中断目录并证明它不是正式结果**

Run:

```powershell
Get-ChildItem -LiteralPath runs/real1068ext_d3r_v37_s42_no_graph -Recurse -Force |
  Select-Object FullName,Length,LastWriteTime
Select-String -Path experiments/results.tsv -Pattern 'real1068ext_d3r_v37_s42_no_graph'
```

Expected: partial checkpoints/logs may exist; the ledger search returns no completed formal row.

- [ ] **Step 3: 仅删除已确认的中断目录**

```powershell
$root = (Resolve-Path .).Path
$target = (Resolve-Path 'runs/real1068ext_d3r_v37_s42_no_graph').Path
if (-not $target.StartsWith((Join-Path $root 'runs'), [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to remove path outside runs: $target"
}
Remove-Item -LiteralPath $target -Recurse -Force
```

Expected: only that partial run disappears; `data/`, other `runs/`, and `experiments/results.tsv` remain.

- [ ] **Step 4: 提交当前分支上的真实追加记录**

```powershell
git diff -- experiments/results.tsv docs/agent_errors.md
git add experiments/results.tsv docs/agent_errors.md
git diff --cached --check
git commit -m "exp: close interrupted v0.1 research records"
git push origin research/20260714-graph-15pct
```

Expected: only existing真实结果和错误记录进入提交；不新增 v37 指标行。

- [ ] **Step 5: 创建唯一 v0.2 集成分支并登记状态**

```powershell
git switch -c research/20260715-riverlagnet-v02
git push -u origin research/20260715-riverlagnet-v02
```

Update the Session A row to `in_progress`, owner to the current task identifier, and record the predecessor commit.

```powershell
git add docs/coordination/riverlagnet-v0.2-status.md
git commit -m "chore: start RiverLagNet v0.2 integration branch"
git push
```

---

## Task 2: 用测试冻结三折开发协议和最终测试边界

**Files:**

- Create: `src/RiverLagNet/data/splits.py`
- Create: `tests/data/test_splits.py`
- Modify: `src/RiverLagNet/data/datamodule.py`
- Modify: `tests/data/test_datamodule.py`
- Create: `configs/data/china_real_daily_contracted_1068_v02.yaml`
- Create: `src/RiverLagNet/configs/data/china_real_daily_contracted_1068_v02.yaml`
- Modify: `tests/integration/test_packaging.py`

- [ ] **Step 1: 先写失败测试，固定边界和无泄漏性质**

```python
from RiverLagNet.data.splits import build_v02_folds


def test_v02_folds_end_before_locked_final_test() -> None:
    folds = build_v02_folds(1000)
    assert [(f.train, f.validation) for f in folds] == [
        ((0, 550), (550, 650)),
        ((0, 650), (650, 750)),
        ((0, 750), (750, 850)),
    ]
    assert all(f.validation[1] <= 850 for f in folds)
    assert folds[-1].final_test == (850, 1000)
```

Add a datamodule test asserting every target timestamp in train/validation is disjoint and `< final_test_start` for each fold, and that scaler statistics use exactly the fold's training interval.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_splits.py tests/data/test_datamodule.py -q
```

Expected: FAIL because `splits.py` and fold-aware datamodule behavior do not exist.

- [ ] **Step 2: 实现最小不可变折对象**

```python
@dataclass(frozen=True)
class ChronologicalFold:
    name: str
    train: tuple[int, int]
    validation: tuple[int, int]
    final_test: tuple[int, int]


def build_v02_folds(num_days: int) -> tuple[ChronologicalFold, ...]:
    if num_days <= 0:
        raise ValueError("num_days must be positive")
    cut = lambda fraction: int(num_days * fraction)
    final_test = (cut(0.85), num_days)
    return (
        ChronologicalFold("v02_fold_a", (0, cut(0.55)), (cut(0.55), cut(0.65)), final_test),
        ChronologicalFold("v02_fold_b", (0, cut(0.65)), (cut(0.65), cut(0.75)), final_test),
        ChronologicalFold("v02_fold_c", (0, cut(0.75)), (cut(0.75), cut(0.85)), final_test),
    )
```

Extend `RiverDataModule` with `split_name: str = "legacy"`. For v0.2, construct train/validation windows from the selected fold; keep `test_dataset` locked behind a later explicit access guard. Fit `MaskedStandardScaler` on `data.values[train_start:train_end]` only.

- [ ] **Step 3: 添加正式配置且保持根配置与打包配置完全一致**

```yaml
scenario: real_daily
dataset_path: data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz
split_name: v02_fold_a
input_window: 180
output_window: 30
batch_size: 8
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
seed: ${seed}
```

The A/B/C fold is overridden per run; `input_window` is only an initial benchmark value, not a selected model result.

- [ ] **Step 4: 运行定向测试和完整测试**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_splits.py tests/data/test_datamodule.py tests/integration/test_packaging.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q
```

Expected: all tests pass; no test reads final-test predictions.

- [ ] **Step 5: 提交**

```powershell
git add src/RiverLagNet/data/splits.py src/RiverLagNet/data/datamodule.py tests/data/test_splits.py tests/data/test_datamodule.py configs/data/china_real_daily_contracted_1068_v02.yaml src/RiverLagNet/configs/data/china_real_daily_contracted_1068_v02.yaml tests/integration/test_packaging.py
git diff --cached --check
git commit -m "feat: freeze v0.2 chronological development folds"
git push
```

---

## Task 3: 固定数据指纹、字段角色和实验协议

**Files:**

- Create: `src/RiverLagNet/data/manifest.py`
- Create: `tests/data/test_manifest.py`
- Create: `experiments/v0.2_protocol.yaml`
- Create: `experiments/v0.2_data_manifest.json` (generated by the CLI, small and versioned)
- Create: `src/RiverLagNet/cli/write_data_manifest.py`
- Modify: `docs/data_schema.md`

- [ ] **Step 1: 写失败测试**

The test must verify SHA-256 stability, node/edge/day/variable dimensions, target order `NH3N,CODMn,TP`, feature names, date bounds, missingness, and graph direction metadata. It must reject a changed file hash.

```python
manifest = inspect_dataset(dataset_path)
assert manifest.sha256 == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
assert manifest.targets == ("NH3N", "CODMn", "TP")
assert manifest.num_nodes == 1068
```

Use a tiny fixture for the unit test; the formal 1068 assertions belong to the generated manifest verification command.

- [ ] **Step 2: 实现只读 manifest 生成器**

Expose:

```python
@dataclass(frozen=True)
class DatasetManifest:
    path: str
    sha256: str
    num_days: int
    num_nodes: int
    num_edges: int
    variables: tuple[str, ...]
    targets: tuple[str, ...]
    start_date: str
    end_date: str
    observed_fraction: tuple[float, ...]


def inspect_dataset(path: Path) -> DatasetManifest: ...
def write_manifest(manifest: DatasetManifest, path: Path) -> None: ...
```

Load with `allow_pickle=False`; sort JSON keys; write UTF-8 with LF. Do not include raw observations.

- [ ] **Step 3: 冻结协议文件**

`experiments/v0.2_protocol.yaml` must contain:

```yaml
protocol_version: riverlagnet-v0.2-20260715
dataset_manifest: experiments/v0.2_data_manifest.json
targets: [NH3N, CODMn, TP]
input_windows: [90, 180, 365]
output_window: 30
folds: [v02_fold_a, v02_fold_b, v02_fold_c]
final_test_fraction: [0.85, 1.00]
selection_metric: val_macro_nse
seeds: [42, 43, 44, 45, 46]
screening_epochs: [25, 50, 100]
confirmation_patience: 12
max_trials_per_family: 12
relative_gain_target_percent: 15.0
parameter_ratio_bounds: [0.90, 1.10]
graph_wall_time_soft_limit: 2.0
graph_wall_time_hard_limit: 3.0
```

- [ ] **Step 4: 生成正式 manifest 并验证**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.write_data_manifest `
  --dataset data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz `
  --output experiments/v0.2_data_manifest.json
Get-Content experiments/v0.2_data_manifest.json -Encoding utf8
```

Expected: 1068 nodes, finite nonzero edge count, 3 targets in fixed order, SHA-256 present, no raw values.

- [ ] **Step 5: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_manifest.py tests/data/test_real_daily.py -q
git add src/RiverLagNet/data/manifest.py src/RiverLagNet/cli/write_data_manifest.py tests/data/test_manifest.py experiments/v0.2_protocol.yaml experiments/v0.2_data_manifest.json docs/data_schema.md
git diff --cached --check
git commit -m "feat: fingerprint the v0.2 benchmark protocol"
git push
```

---

## Task 4: 增加单 GPU 锁和固定预算契约

**Files:**

- Create: `src/RiverLagNet/training/gpu_lock.py`
- Create: `tests/training/test_gpu_lock.py`
- Create: `src/RiverLagNet/training/budget.py`
- Create: `tests/training/test_budget.py`
- Modify: `src/RiverLagNet/cli/train.py`
- Modify: `configs/trainer/default.yaml`
- Modify: `src/RiverLagNet/configs/trainer/default.yaml`

- [ ] **Step 1: 写并运行失败测试**

Tests must prove: a second process cannot take the same lock; stale locks are reclaimed only if the recorded PID is dead; context exit releases the lock; equal-budget configs resolve to equal optimizer-update counts even when physical batch sizes differ.

```python
with SingleGpuLock(lock_path, run_name="first"):
    with pytest.raises(RuntimeError, match="GPU is reserved"):
        SingleGpuLock(lock_path, run_name="second").acquire()
```

Run the two new test files and observe FAIL before implementation.

- [ ] **Step 2: 实现锁和预算解析**

The lock JSON contains PID, run name, host, timestamp, commit, and command. Use atomic exclusive creation. Define:

```python
@dataclass(frozen=True)
class TrainingBudget:
    effective_batch_size: int
    max_optimizer_steps: int
    gradient_accumulation: int


def resolve_accumulation(physical_batch_size: int, effective_batch_size: int) -> int:
    if effective_batch_size % physical_batch_size:
        raise ValueError("effective batch size must be divisible by physical batch size")
    return effective_batch_size // physical_batch_size
```

Add config keys `effective_batch_size`, `max_steps`, `gpu_lock_path`, and `use_gpu_lock`. Pass `accumulate_grad_batches` and `max_steps` to `Trainer`. Acquire the lock only for CUDA runs and release it in `finally`, including failures.

- [ ] **Step 3: 使用合成数据验证互斥和向后兼容**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/training/test_gpu_lock.py tests/training/test_budget.py tests/integration/test_fast_dev_run.py -q
```

Expected: lock and budget tests pass; current StationGRU/RiverLagNet fast-dev still pass.

- [ ] **Step 4: 提交**

```powershell
git add src/RiverLagNet/training/gpu_lock.py src/RiverLagNet/training/budget.py src/RiverLagNet/cli/train.py tests/training/test_gpu_lock.py tests/training/test_budget.py configs/trainer/default.yaml src/RiverLagNet/configs/trainer/default.yaml
git diff --cached --check
git commit -m "feat: enforce single gpu and equal update budgets"
git push
```

---

## Task 5: 扩展 DataLoader、优化器和硬件统计

**Files:**

- Modify: `src/RiverLagNet/data/datamodule.py`
- Modify: `tests/data/test_datamodule.py`
- Modify: `src/RiverLagNet/training/lightning_module.py`
- Modify: `src/RiverLagNet/training/callbacks.py`
- Create: `tests/training/test_runtime_stats.py`
- Create: `configs/trainer/blackwell_96gb.yaml`
- Create: `src/RiverLagNet/configs/trainer/blackwell_96gb.yaml`

- [ ] **Step 1: 测试配置传播**

Add assertions for `persistent_workers`, `prefetch_factor`, `pin_memory`, fused AdamW fallback, BF16 precision, TF32 high matmul precision, and recorded samples/second plus peak reserved/allocated VRAM.

- [ ] **Step 2: 实现可配置加载与优化器**

Only pass `prefetch_factor` when `num_workers > 0`. Configure `torch.set_float32_matmul_precision("high")`. Use fused AdamW only when CUDA and the installed PyTorch supports it; otherwise fall back without changing hyperparameters.

Extend `RuntimeStatsCallback` fields:

```python
self.peak_allocated_vram_gb = 0.0
self.peak_reserved_vram_gb = 0.0
self.samples_per_second = 0.0
self.optimizer_steps_per_second = 0.0
```

Do not replace the existing `peak_vram_gb` ledger field; map it to allocated VRAM for backward compatibility and write extended hardware JSON under each run.

- [ ] **Step 3: 创建 Blackwell 候选配置**

```yaml
accelerator: gpu
devices: 1
precision: bf16-mixed
matmul_precision: high
fused_adamw: true
deterministic: true
gradient_clip_val: 1.0
effective_batch_size: 32
max_steps: -1
use_gpu_lock: true
gpu_lock_path: runs/.gpu0.lock
early_stopping_patience: 12
enable_progress_bar: false
```

- [ ] **Step 4: 运行测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_datamodule.py tests/training/test_runtime_stats.py tests/training/test_lightning_module.py tests/integration/test_packaging.py -q
git add src/RiverLagNet/data/datamodule.py src/RiverLagNet/training/lightning_module.py src/RiverLagNet/training/callbacks.py tests/data/test_datamodule.py tests/training/test_runtime_stats.py configs/trainer/blackwell_96gb.yaml src/RiverLagNet/configs/trainer/blackwell_96gb.yaml tests/integration/test_packaging.py
git diff --cached --check
git commit -m "perf: expose Blackwell data and optimizer controls"
git push
```

---

## Task 6: 构建真实数据短基准矩阵并选择硬件档案

**Files:**

- Create: `src/RiverLagNet/analysis/hardware_benchmark.py`
- Create: `src/RiverLagNet/cli/benchmark_hardware.py`
- Create: `tests/analysis/test_hardware_benchmark.py`
- Create: `docs/hardware/blackwell-96gb-profile.md`
- Create: `experiments/hardware/blackwell-96gb-benchmark.json`
- Modify: `configs/data/china_real_daily_contracted_1068_v02.yaml`

- [ ] **Step 1: 写失败测试，固定排名规则**

The selector must reject OOM/nonfinite runs, require at least 10 GiB free headroom, and rank acceptable candidates by median optimizer steps/s. It should prefer 70–82 GiB peak reserved VRAM and median SM utilization >=70%, but may document a lower-memory winner if throughput is higher.

```python
winner = select_hardware_profile(results, total_vram_gb=95.59)
assert winner.free_headroom_gb >= 10.0
assert winner.status == "ok"
```

- [ ] **Step 2: 实现基准入口**

Benchmark the existing trainable no-graph model on fold A training data for a fixed warm-up plus 100 optimizer steps. Candidate grid:

```python
physical_batches = (4, 8, 16, 24, 32)
worker_counts = (0, 4, 8, 12)
```

Use BF16, the same hidden size and `T_in=180`; catch OOM, clear cache, record batch/workers/prefetch/samples-s/steps-s/allocated/reserved/SM/power/duration/status. Do not append these short runs to `experiments/results.tsv`.

- [ ] **Step 3: 先跑单点 smoke，再运行完整矩阵**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.benchmark_hardware `
  --smoke --batch-size 8 --num-workers 4
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.benchmark_hardware `
  --batches 4 8 16 24 32 --workers 0 4 8 12 `
  --output experiments/hardware/blackwell-96gb-benchmark.json
```

Expected: one valid winner with >=10 GiB headroom; OOM candidates are recorded, not retried forever. Monitor through log files, not a fragile foreground wait.

- [ ] **Step 4: 固化实际赢家并记录编译决策**

Repeat the top two candidates three times. Test `torch.compile` only on the winner; keep it only if median speed improves by >=5%, outputs remain finite, and validation smoke differs by <1e-5 on identical weights/input. Update the v0.2 data/trainer config with the measured physical batch, workers, prefetch, accumulation, and compile flag.

`docs/hardware/blackwell-96gb-profile.md` must report all candidates, winner rationale, achieved utilization, VRAM headroom, and why rejected candidates failed.

- [ ] **Step 5: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_hardware_benchmark.py tests/integration/test_fast_dev_run.py -q
git add src/RiverLagNet/analysis/hardware_benchmark.py src/RiverLagNet/cli/benchmark_hardware.py tests/analysis/test_hardware_benchmark.py docs/hardware/blackwell-96gb-profile.md experiments/hardware/blackwell-96gb-benchmark.json configs/data/china_real_daily_contracted_1068_v02.yaml src/RiverLagNet/configs/data/china_real_daily_contracted_1068_v02.yaml configs/trainer/blackwell_96gb.yaml src/RiverLagNet/configs/trainer/blackwell_96gb.yaml
git diff --cached --check
git commit -m "perf: select the 96gb Blackwell training profile"
git push
```

---

## Task 7: 实现白名单清理和 legacy 资产清单

**Files:**

- Modify: `src/RiverLagNet/testing/cleanup.py`
- Modify: `tests/testing/test_cleanup.py`
- Create: `src/RiverLagNet/cli/clean_generated.py`
- Create: `docs/legacy_inventory.md`
- Modify: `.gitignore`

- [ ] **Step 1: 写失败测试，证明保留边界**

Create fixtures for removable caches (`.pytest_cache`, `build/pytest`, `build/smoke`, `__pycache__`, `*.pyc`, root `*.log`, `*.egg-info`) and protected assets (`data/**`, `runs/selected/**`, `experiments/results.tsv`, `tests/**`, `src/**`, checkpoints not explicitly marked partial). Test dry-run and real deletion return the same allowlisted path set.

- [ ] **Step 2: 实现安全 API**

```python
@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    reason: str


def discover_generated_artifacts(root: Path) -> tuple[CleanupCandidate, ...]: ...
def clean_generated_artifacts(root: Path, *, dry_run: bool = True) -> tuple[Path, ...]: ...
```

Every resolved candidate must remain below repository root and match an explicit allowlist. Keep `clean_test_artifacts` as a compatible wrapper used by `tests/conftest.py`.

- [ ] **Step 3: 生成 legacy 清单，不删除生产模块或测试**

`docs/legacy_inventory.md` classifies each old model/config/analysis file as `retain`, `retire-in-D`, or `historical-only`, with replacement coverage required before deletion. Session A deletes only generated artifacts.

- [ ] **Step 4: dry-run、人工核对、执行清理**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.clean_generated --dry-run
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.clean_generated --apply
git status --short --ignored
```

Expected: no source, versioned test, data, selected run, ledger, or current report is removed.

- [ ] **Step 5: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/testing/test_cleanup.py -q
git add src/RiverLagNet/testing/cleanup.py src/RiverLagNet/cli/clean_generated.py tests/testing/test_cleanup.py docs/legacy_inventory.md .gitignore
git diff --cached --check
git commit -m "chore: clean generated artifacts through an allowlist"
git push
```

---

## Task 8: Session A 总验证、状态交接和自审

**Files:**

- Modify: `README.md`
- Modify: `docs/coordination/riverlagnet-v0.2-status.md`

- [ ] **Step 1: 更新 README**

Document the v0.2 branch, data manifest, three-fold commands, final-test prohibition, GPU lock, selected Blackwell profile, cleanup dry-run, and where later session plans live.

- [ ] **Step 2: 运行完整验证**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train `
  data=synthetic model=station_gru trainer=blackwell_96gb `
  trainer.fast_dev_run=true experiment.record_result=false
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train `
  data=china_real_daily_contracted_1068_v02 model=station_gru trainer=blackwell_96gb `
  data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
git status --short
```

Expected: full pytest passes; synthetic and real fold-A fast-dev pass; no final-test metric file exists; no GPU lock remains.

- [ ] **Step 3: 进行计划要求自审**

Inspect `git diff`, confirm all protocol numbers match the approved design, no test assertion was weakened, packaged configs equal root configs, and benchmark JSON is derived from real measurements. If review finds issues, fix and rerun affected tests before proceeding.

- [ ] **Step 4: 完成状态并授权 Session B**

Set Session A to `complete`, record the final commit placeholder and evidence; set Session B to `ready`; do not change C/D.

- [ ] **Step 5: 最终提交与推送**

```powershell
git add README.md docs/coordination/riverlagnet-v0.2-status.md
git diff --cached --check
git commit -m "docs: complete the v0.2 engineering foundation"
git push
git status --short
git log -1 --oneline
```

Expected final state: clean working tree, pushed branch, Session A complete, Session B ready. Report full-test count, fast-dev results, data SHA-256, selected batch/workers, throughput, peak VRAM, free headroom, commit hash, and any measured limitation. Then stop; do not start Session B.

## Session A Exit Gate

Session A is complete only if all are true:

- interrupted v37 run is audited and not represented as a valid result;
- pending authentic ledger rows are preserved and pushed;
- `research/20260715-riverlagnet-v02` exists remotely;
- fold A/B/C boundaries and train-only normalization are covered by tests;
- final test remains unopened;
- data manifest and protocol are versioned;
- GPU lock and equal-update budget tests pass;
- the measured Blackwell profile has >=10 GiB headroom and is faster than the old batch-4/worker-0 profile;
- safe cleanup tests prove protected assets are retained;
- complete pytest and both fast-dev runs pass;
- status board authorizes only Session B.
