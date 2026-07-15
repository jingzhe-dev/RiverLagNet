# RiverLagNet v0.2 Session B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不使用图的前提下建立足够强的日尺度多站点局部预测骨干，并用严格的上游残差信号探针判断真实河网信息是否在正确传播时间上提供可泛化增量。

**Architecture:** 局部模型借鉴 TimeXer 的“目标序列与外生上下文分工”和 TimeMixer 的“多尺度历史表达”思想，但不复制其 token、patch、mixing block 或论文结构。新骨干采用 mask-aware 目标/外生双流编码、站点独立的因果多尺度时间块和直接 30 日多目标解码；图信号探针只读取冻结骨干残差，用正确河向与旅行时间对齐的上游特征和反事实控制进行比较。

**Tech Stack:** Session A 已冻结的 Lightning/Hydra/Blackwell 配置、PyTorch、NumPy、scikit-learn（仅探针）、PyTest、Matplotlib/JSON 报告。

## Global Constraints

- 仅在 Session A `complete` 且 Session B `ready` 后开始。
- 全部正式运行使用同一数据 manifest、fold A/B/C、目标顺序、30 日输出、有效 batch 和硬件档案。
- 最终测试段仍然禁止访问。
- 本会话不实现正式图模型；探针可以读取训练期与验证期上游特征，但不得把验证结果反向写入数据。
- 外生变量必须按 manifest 名称显式分组，不通过通道位置猜含义。
- 每个模型家族最多 12 个预注册配置；successive halving 规则不得在看到结果后改变。
- 每次长运行使用单 GPU 锁、隐藏后台进程和独立日志；不另开更多 Codex 会话。

---

## Task 1: 接管 Session B 并验证 Session A 门槛

**Files:**

- Modify: `docs/coordination/riverlagnet-v0.2-status.md`

- [ ] **Step 1: 验证分支、状态和产物**

```powershell
git status --short
git pull --ff-only
git log -3 --oneline --decorate
Get-Content docs/coordination/riverlagnet-v0.2-status.md -Encoding utf8
Get-Content experiments/v0.2_data_manifest.json -Encoding utf8
Get-Content experiments/hardware/blackwell-96gb-benchmark.json -Encoding utf8
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Expected: `research/20260715-riverlagnet-v02`, clean tree, Session A complete, B ready, formal dataset hash unchanged, no competing GPU process. Otherwise stop and report the violated prerequisite.

- [ ] **Step 2: 重跑 Session A 的关键门槛测试**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_splits.py tests/data/test_manifest.py tests/training/test_gpu_lock.py tests/analysis/test_hardware_benchmark.py -q
```

- [ ] **Step 3: 标记 B 为 `in_progress` 并提交**

Only update Session B owner/status; leave C/D unchanged.

---

## Task 2: 冻结动态变量角色与多尺度张量契约

**Files:**

- Create: `src/RiverLagNet/data/feature_roles.py`
- Create: `tests/data/test_feature_roles.py`
- Modify: `src/RiverLagNet/data/schema.py`
- Modify: `src/RiverLagNet/data/real_daily.py`
- Modify: `docs/data_schema.md`

- [ ] **Step 1: 写失败测试**

Tests must prove: `NH3N,CODMn,TP` are always target channels 0–2; exogenous covariates are found by name; flow/discharge/rainfall aliases resolve deterministically; missing optional flow produces `None` rather than silently using another variable; duplicate/unknown target names fail.

```python
roles = resolve_feature_roles(("NH3N", "CODMn", "TP", "discharge", "rainfall"))
assert roles.target_indices == (0, 1, 2)
assert roles.flow_index == 3
assert roles.exogenous_indices == (3, 4)
```

- [ ] **Step 2: 实现不可变角色对象**

```python
@dataclass(frozen=True)
class FeatureRoles:
    names: tuple[str, ...]
    target_indices: tuple[int, int, int]
    exogenous_indices: tuple[int, ...]
    flow_index: int | None
    rainfall_indices: tuple[int, ...]


def resolve_feature_roles(variable_names: Sequence[str]) -> FeatureRoles: ...
```

Persist variable names when loading the NPZ. Do not change the public batch shape.

- [ ] **Step 3: 定义局部模型中间契约测试**

Add a test fixture asserting later local context shapes:

```text
history_states [B,T,N,D]
scale_states   [B,K,N,D]
horizon_states [B,30,N,D]
prediction     [B,30,N,3]
```

This test may initially import a placeholder protocol/dataclass and fail until Task 3.

- [ ] **Step 4: 运行测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/data/test_feature_roles.py tests/data/test_real_daily.py -q
git add src/RiverLagNet/data/feature_roles.py src/RiverLagNet/data/schema.py src/RiverLagNet/data/real_daily.py tests/data/test_feature_roles.py docs/data_schema.md
git diff --cached --check
git commit -m "feat: make target and exogenous feature roles explicit"
git push
```

---

## Task 3: 用 TDD 实现强无图多尺度局部骨干

**Files:**

- Create: `src/RiverLagNet/models/local_multiscale.py`
- Create: `tests/models/test_local_multiscale.py`
- Modify: `src/RiverLagNet/models/input_encoder.py`
- Modify: `tests/models/test_components.py`

- [ ] **Step 1: 写形状、隔离和因果性失败测试**

Cover:

- common forward API returns `[B,30,N,3]`;
- changing station 0 history cannot alter station 1 output in graph-free mode;
- perturbing a future target not present in the input cannot alter output;
- fully missing values remain finite because masks travel with values;
- target and exogenous streams have separate parameters;
- `encode_context` returns the four exact shapes above;
- CPU float32 and CUDA BF16 autocast are finite.

Run and observe FAIL because the module does not exist.

- [ ] **Step 2: 实现目标/外生双流编码**

```python
class TargetExogenousInputEncoder(nn.Module):
    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_quality: Tensor | None,
        static: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        target = self.target_encoder(x[..., :3], x_mask[..., :3], quality[..., :3])
        external = self.external_encoder(
            x[..., 3:], x_mask[..., 3:], quality[..., 3:], static, time_features
        )
        return self.fusion(torch.cat((target, external), dim=-1))
```

For datasets with no exogenous channels, use a learned zero-context token rather than an empty linear layer. Reuse mask semantics from `InputMaskEncoder`.

- [ ] **Step 3: 实现站点独立的因果多尺度时间块**

Use scales `(1, 3, 7, 30)`. Downsample only within the observed window using mask-aware pooling; apply shared station-wise gated temporal convolutions/MLPs at each scale; never convolve across nodes. Align the last valid state of each scale and fuse with learned scale gates.

```python
@dataclass
class LocalForecastContext:
    history_states: Tensor
    scale_states: Tensor
    horizon_states: Tensor
    prediction: Tensor


class LocalMultiscaleForecaster(nn.Module):
    def encode_context(...) -> LocalForecastContext: ...
    def forward(...) -> Tensor:
        return self.encode_context(...).prediction
```

Generate 30 horizon queries from the fused local state plus learned lead embeddings. Decode directly with three target-specific heads; no autoregressive use of future truth.

- [ ] **Step 4: 证明局部隔离和 BF16**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/models/test_local_multiscale.py tests/models/test_components.py -q
```

Expected: all new tests pass, including exact station isolation and finite CUDA BF16 when CUDA is available.

- [ ] **Step 5: 提交**

```powershell
git add src/RiverLagNet/models/local_multiscale.py src/RiverLagNet/models/input_encoder.py tests/models/test_local_multiscale.py tests/models/test_components.py
git diff --cached --check
git commit -m "feat: add a multiscale graph-free forecasting backbone"
git push
```

---

## Task 4: 接入 Lightning/Hydra 并校验预算统计

**Files:**

- Modify: `src/RiverLagNet/training/lightning_module.py`
- Modify: `src/RiverLagNet/cli/train.py`
- Create: `configs/model/local_multiscale.yaml`
- Create: `src/RiverLagNet/configs/model/local_multiscale.yaml`
- Create: `configs/experiment/v02_local_search.yaml`
- Create: `src/RiverLagNet/configs/experiment/v02_local_search.yaml`
- Modify: `tests/training/test_lightning_module.py`
- Modify: `tests/integration/test_fast_dev_run.py`
- Modify: `tests/integration/test_packaging.py`

- [ ] **Step 1: 先扩展失败测试**

Assert `build_model("local_multiscale", ...)` works, fast-dev runs on synthetic and real fold A, parameter count/FLOPs/samples-s/steps-s are written to `run_dir/hardware.json`, and no graph variant is accepted by this model.

- [ ] **Step 2: 注册模型并创建默认配置**

```yaml
name: local_multiscale
hidden_dim: 128
target_dim: 3
scales: [1, 3, 7, 30]
num_layers: 2
dropout: 0.1
```

Do not silently map `no_graph` to an existing graph model; the local backbone must be a standalone model family.

- [ ] **Step 3: 添加模型复杂度统计**

Report trainable parameters and an input-shape-specific FLOP estimate with a clearly documented method. The same function must later be used for graph and capacity-matched controls.

- [ ] **Step 4: 运行 smoke 和提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/training/test_lightning_module.py tests/integration/test_fast_dev_run.py tests/integration/test_packaging.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=china_real_daily_contracted_1068_v02 model=local_multiscale trainer=blackwell_96gb data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
git add src/RiverLagNet/training/lightning_module.py src/RiverLagNet/cli/train.py configs/model/local_multiscale.yaml src/RiverLagNet/configs/model/local_multiscale.yaml configs/experiment/v02_local_search.yaml src/RiverLagNet/configs/experiment/v02_local_search.yaml tests/training/test_lightning_module.py tests/integration/test_fast_dev_run.py tests/integration/test_packaging.py
git diff --cached --check
git commit -m "feat: train the local multiscale model through Lightning"
git push
```

---

## Task 5: 实现正确对齐的上游残差信号探针

**Files:**

- Create: `src/RiverLagNet/analysis/upstream_signal_gate.py`
- Create: `src/RiverLagNet/cli/check_upstream_signal.py`
- Create: `tests/analysis/test_upstream_signal_gate.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 先写合成可识别性测试**

Build a directed chain with known two-day lag. Tests must show:

- correct direction/time alignment recovers positive validation gain;
- reversed direction, shuffled edges, and shuffled time do not;
- source value at `origin - lag` is used, never `origin + lag`;
- unavailable source observations are masked;
- headwaters produce no upstream probe feature;
- per-target and lead-band reports preserve all axes.

- [ ] **Step 2: 实现旅行时间对齐特征**

For direct upstream edges and candidate integer lags around the travel prior, build:

```text
absolute source target
source minus destination innovation
source first difference
source concentration × flow proxy (when available)
source/destination flow ratio (when available)
distance, slope, travel prior
event indicator, upstream degree, downstream depth
```

Expose explicit controls: `correct`, `reverse_direction`, `shuffled_graph`, `shuffled_time`. The shuffled variants must be deterministic per seed.

- [ ] **Step 3: 实现无验证泄漏的探针训练**

Within each fold, train a temporary local checkpoint on the first 80% of that fold's training interval. Generate residuals on the last 20% as probe-fit data; evaluate probes only on the fold validation interval. Compare ridge, a small MLP, and a shallow histogram gradient-boosted tree with frozen hyperparameters. Add `scikit-learn>=1.5` to `pyproject.toml` only if absent from `DeepWater`, then install the editable project in that environment.

```python
@dataclass(frozen=True)
class SignalGateDecision:
    advance: bool
    mean_relative_gain_percent: float
    positive_folds: int
    folds_above_three_percent: int
    correct_beats_controls: bool
    reasons: tuple[str, ...]
```

- [ ] **Step 4: 固定推进判据**

`advance=True` only if:

- mean gain is positive over all three folds;
- at least two folds show >=3% relative NSE gain;
- correct direction beats reverse, graph-shuffled, and time-shuffled controls;
- at least one of NH3N/CODMn/TP benefits without another target collapsing by >0.01 NSE.

- [ ] **Step 5: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_upstream_signal_gate.py -q
git add src/RiverLagNet/analysis/upstream_signal_gate.py src/RiverLagNet/cli/check_upstream_signal.py tests/analysis/test_upstream_signal_gate.py pyproject.toml
git diff --cached --check
git commit -m "feat: gate graph work on aligned upstream residual signal"
git push
```

---

## Task 6: 预注册 12 个局部配置和 successive-halving 运行器

**Files:**

- Create: `experiments/v0.2_local_search.yaml`
- Create: `src/RiverLagNet/analysis/v02_search.py`
- Create: `src/RiverLagNet/cli/run_v02_search.py`
- Create: `tests/analysis/test_v02_search.py`

- [ ] **Step 1: 写测试冻结搜索集合**

The test must assert exactly 12 unique local configurations, no later mutation, allowed values only, stage counts `12 -> 4 -> 2`, and commands include fold, seed, effective batch, max epochs/steps, selected hardware profile, and result description without unsafe Hydra punctuation.

- [ ] **Step 2: 写明 12 个配置**

Cover the approved ranges without a Cartesian explosion:

```text
T_in: 90, 180, 365
hidden: 128, 256
layers: 2, 4
dropout: 0.05, 0.10, 0.20
lr: 3e-4, 6e-4, 1e-3
weight decay: 1e-5, 1e-4, 1e-3
NSE auxiliary weight: 0.00, 0.05, 0.10
```

Each YAML item has an immutable ID and all values explicit.

- [ ] **Step 3: 实现恢复安全的 halving**

Stage 1: 12 configs, fold A, seed 42, 25 epochs. Stage 2: top 4 by fold-A macro NSE, folds A/B/C, seed 42, 50 epochs. Stage 3: top 2, folds A/B/C, seeds 42/43/44, 100 epochs and patience 12. Ties break by lower parameter count, then lower wall time, then config ID. Resume only skips a run with a valid ledger row and checkpoint.

- [ ] **Step 4: dry-run 并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_search.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.run_v02_search --family local --dry-run
git add experiments/v0.2_local_search.yaml src/RiverLagNet/analysis/v02_search.py src/RiverLagNet/cli/run_v02_search.py tests/analysis/test_v02_search.py
git diff --cached --check
git commit -m "feat: preregister the local successive-halving search"
git push
```

---

## Task 7: 运行局部搜索并冻结强基线

**Files:**

- Create: `experiments/reports/v0.2_local_search_summary.json`
- Create: `experiments/reports/v0.2_local_search_summary.md`
- Modify: `experiments/results.tsv` (append only through training code)
- Create: `configs/model/local_multiscale_selected.yaml`
- Create: `src/RiverLagNet/configs/model/local_multiscale_selected.yaml`

- [ ] **Step 1: 启动后台搜索并通过文件轮询**

Use `Start-Process -WindowStyle Hidden` with dedicated stdout/stderr paths under `runs/v02_local_search/`. The worker must write its PID and acquire the GPU lock. Poll process plus ledger/checkpoints; do not start a second run while the first is alive.

- [ ] **Step 2: 按预注册规则完成三个阶段**

Do not change the 12 configs or promotion counts after seeing outcomes. Crashes get real `crash` rows and may be retried once only after a documented code/environment fix. OOM is a hardware-profile defect and returns to Session A evidence, not a reason to shrink only one competitor's budget.

- [ ] **Step 3: 生成派生摘要并冻结赢家**

Report every configuration/stage/fold/seed, mean/std NSE, target NSE, duration, samples/s, VRAM, parameters and FLOPs. Select by mean three-fold/three-seed validation macro NSE with the fixed tie-break. Copy values, not code, into `local_multiscale_selected.yaml` and store the source config ID and commit.

- [ ] **Step 4: 核对 ledger 与 checkpoint**

Every summary row must match `experiments/results.tsv` and a checkpoint. No manual metric edits. Confirm no `test_metrics.json` was created for v0.2.

- [ ] **Step 5: 提交**

```powershell
git add experiments/results.tsv experiments/reports/v0.2_local_search_summary.json experiments/reports/v0.2_local_search_summary.md configs/model/local_multiscale_selected.yaml src/RiverLagNet/configs/model/local_multiscale_selected.yaml
git diff --cached --check
git commit -m "exp: select the strong v0.2 local baseline"
git push
```

---

## Task 8: 执行信号门槛，必要时做一次预注册的数据修复

**Files:**

- Create: `experiments/reports/v0.2_upstream_signal_gate.json`
- Create: `experiments/reports/v0.2_upstream_signal_gate.md`
- Conditional create: `data/processed/...-v0.5/` (ignored, only if manifest proves required flow variables absent)
- Conditional create: `experiments/v0.2_data_manifest_v05.json`
- Modify: `experiments/v0.2_protocol.yaml` only before the final gate rerun

- [ ] **Step 1: 对冻结局部模型运行三折信号探针**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.check_upstream_signal `
  --checkpoint-config configs/model/local_multiscale_selected.yaml `
  --folds v02_fold_a v02_fold_b v02_fold_c `
  --controls correct reverse_direction shuffled_graph shuffled_time `
  --output experiments/reports/v0.2_upstream_signal_gate.json
```

- [ ] **Step 2: 应用固定判据，不凭感觉解释**

If the gate passes, skip Step 3. If it fails specifically because the manifest has no usable flow/load/hydrometeorological covariate, permit exactly one data-enrichment iteration using already available source data and train-only transformations. Do not use validation/test outcomes to select stations, dates, or variables.

- [ ] **Step 3: 条件数据修复后重新生成 manifest 和重跑全部局部证据**

Any v0.5 dataset requires a new immutable directory/hash and rerunning local Stage 3 plus the signal gate. Never overwrite v0.4. If the failure is direction/time/control related rather than missing covariates, do not enrich opportunistically.

- [ ] **Step 4: 提交信号报告**

```powershell
git add experiments/reports/v0.2_upstream_signal_gate.json experiments/reports/v0.2_upstream_signal_gate.md experiments/v0.2_protocol.yaml experiments/v0.2_data_manifest*.json
git diff --cached --check
git commit -m "exp: evaluate aligned upstream residual signal"
git push
```

---

## Task 9: Session B 完整验证与交接

**Files:**

- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/coordination/riverlagnet-v0.2-status.md`

- [ ] **Step 1: 更新架构文档**

Explain which ideas were inspired by TimeXer/TimeMixer and which implementation choices are original to RiverLagNet. State explicitly that the local backbone is graph-free and not a reproduction.

- [ ] **Step 2: 运行完整测试和 selected-config fast-dev**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=china_real_daily_contracted_1068_v02 model=local_multiscale_selected trainer=blackwell_96gb data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
```

- [ ] **Step 3: 自审研究证据**

Confirm the local winner came from the frozen 12-config set, summaries match ledger/checkpoints, no test split was touched, all three folds and seeds 42/43/44 exist, and correct-alignment probe comparison is paired.

- [ ] **Step 4: 按门槛更新状态**

If the signal gate passes: mark B `complete`, record commit/evidence, set C `ready`. If it still fails after the single permitted enrichment: mark B `blocked`, leave C `waiting`, and report the exact failed clauses. Never lower the gate.

- [ ] **Step 5: 最终提交、推送并停止**

```powershell
git add README.md docs/architecture.md docs/coordination/riverlagnet-v0.2-status.md
git diff --cached --check
git commit -m "docs: complete v0.2 signal and local baseline stage"
git push
git status --short
git log -1 --oneline
```

Report: local winner config ID, three-fold/three-seed metrics, params/FLOPs/time/VRAM, signal gains by fold/target/lead band/control, gate decision, dataset hash, test count, commit hash. Then stop.

## Session B Exit Gate

Session B may authorize C only if:

- the graph-free local model is a standalone, tested family;
- the 12-config search and halving promotions match preregistration;
- its selected result covers three folds and seeds 42/43/44;
- feature roles and all probe alignments are name-based and mask-aware;
- the correct upstream probe has positive mean gain, >=3% gain in at least two folds, and beats all three controls;
- no target average drops by more than 0.01 NSE;
- final-test artifacts do not exist;
- full pytest and real selected-config fast-dev pass;
- all evidence is pushed and only Session C is made ready.
