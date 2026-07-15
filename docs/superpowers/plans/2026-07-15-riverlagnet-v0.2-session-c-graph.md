# RiverLagNet v0.2 Session C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结的强局部骨干上实现 RiverLagNet v0.2 有向条件传播，完成公平无图对照、三种子筛选、机制消融和运行时门槛，选出最多一个进入五种子确认的候选。

**Architecture:** 使用直接上游边和旅行时间先验构造有限滞后候选；每个预测日只能使用观测历史或上游局部预测，不读取未来真值。传播权重由下游状态、目标、预测提前期/lead band、多尺度状态、边属性和有界动态旅行时间共同决定，并在每个接收节点的“入边 × 候选滞后”范围内归一化。图分支输出零初始化的残差校正；无上游节点严格等于局部预测。堆叠 1–3 层获得多跳影响，而不是直接建立所有祖先边。

**Tech Stack:** Session B 的 `LocalMultiscaleForecaster` 与 `LocalForecastContext`、PyTorch scatter/index operations、Lightning/Hydra、PyTest、Blackwell BF16、现有 `build_graph_variant`。

## Global Constraints

- 仅在 Session B 的信号门槛通过、C 为 `ready` 后开始。
- 不改动 Session B 已选局部骨干的结构、输入变量或 HPO 结果。
- 图和无图正式对照从同一局部 checkpoint 初始化，使用相同折、种子、有效 batch、优化器更新数、早停信息和 HPO 试验数。
- 正式相对增益的无图分母取 `exact_graph_off` 与 `capacity_matched_no_graph` 中验证表现更强者，防止利用弱基线。
- 参数量匹配范围为图模型的 90%–110%；所有对照报告参数、FLOPs、样本/秒、时长、显存。
- 图模型墙钟时间目标 <= 强无图的 2 倍；任何候选 >3 倍不得进入 Session D。
- attention/routing 权重只能称为模型路由权重，不作因果贡献解释。
- 最终测试仍禁止访问。

---

## Task 1: 接管 Session C 并冻结输入证据

**Files:**

- Modify: `docs/coordination/riverlagnet-v0.2-status.md`
- Create: `experiments/v0.2_graph_inputs.json`

- [ ] **Step 1: 验证前置门槛**

```powershell
git status --short
git pull --ff-only
Get-Content docs/coordination/riverlagnet-v0.2-status.md -Encoding utf8
Get-Content experiments/reports/v0.2_upstream_signal_gate.json -Encoding utf8
Get-Content experiments/reports/v0.2_local_search_summary.json -Encoding utf8
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Expected: B complete, C ready, signal decision `advance=true`, local selected config/checkpoint hashes present, dataset hash matches protocol, no competing GPU process.

- [ ] **Step 2: 写冻结输入清单**

`experiments/v0.2_graph_inputs.json` records dataset/protocol/local-config/local-checkpoint SHA-256, source commit, folds, screening seeds 42/43/44, confirmation seeds 42–46, and forbidden final-test boundary.

- [ ] **Step 3: 标记 C 为 `in_progress` 并提交**

Do not change D.

---

## Task 2: 实现因果旅行时间候选和历史—预测桥接

**Files:**

- Create: `src/RiverLagNet/models/travel_alignment.py`
- Create: `tests/models/test_travel_alignment.py`
- Modify: `src/RiverLagNet/data/graph_builder.py`
- Modify: `tests/data/test_graph_builder.py`

- [ ] **Step 1: 写失败测试，固定索引语义**

For forecast origin `t=89`, lead `h=1..30`, and lag `tau`:

```text
source time = origin + h - tau
source time <= origin  -> observed masked history
source time > origin   -> source local forecast at lead (h - tau)
```

Tests must prove no future target enters, exact integer prior uses one candidate in `static_lag`, learned mode uses bounded offsets around the prior, negative/out-of-window indices are masked, and dynamic shift cannot exceed the configured bound.

```python
aligned, available = bridge_observed_and_local_forecast(
    target_history, target_mask, local_prediction, lag_days, lead_days
)
```

- [ ] **Step 2: 实现候选对象**

```python
@dataclass(frozen=True)
class LagCandidates:
    days: Tensor       # [E,L], integer >= 1
    valid: Tensor      # [E,L]
    prior_days: Tensor # [E]


def build_lag_candidates(
    edge_attr: Tensor,
    *,
    radius: int,
    max_lag: int,
    mode: str,
) -> LagCandidates: ...
```

Use the documented travel-time edge channel by name/index from the dataset schema. Candidate lags are at least 1 day; zero-lag future leakage is forbidden for day-ahead forecasts.

- [ ] **Step 3: 实现向量化桥接**

The implementation may loop over the small lag candidate axis, but not batch, nodes, edges, horizons, or targets. Use `gather` with clamped indices and a separate availability mask. For source times after origin, gather only from `local_prediction`, never `y`.

- [ ] **Step 4: 测试和提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/models/test_travel_alignment.py tests/data/test_graph_builder.py -q
git add src/RiverLagNet/models/travel_alignment.py src/RiverLagNet/data/graph_builder.py tests/models/test_travel_alignment.py tests/data/test_graph_builder.py
git diff --cached --check
git commit -m "feat: align upstream history and forecasts by travel time"
git push
```

---

## Task 3: 实现目标—提前期—尺度条件的有向传播算子

**Files:**

- Create: `src/RiverLagNet/models/directed_conditional_propagation.py`
- Create: `tests/models/test_directed_conditional_propagation.py`

- [ ] **Step 1: 写核心失败测试**

Cover:

- output correction `[B,30,N,3]`;
- routing tensor preserves layer/edge/lag/lead-band/target axes;
- weights sum to one over incoming edges × valid lags for each destination/lead-band/target/scale;
- only source-to-destination flow occurs;
- headwater correction is exactly zero bit-for-bit;
- correct candidate lag selects the planted synthetic signal;
- target, lead band and scale embeddings can produce different routing;
- empty graph and no-upstream graph are finite;
- CUDA BF16 aggregation has no dtype conflict;
- gradients reach routing, value and dynamic-shift parameters.

- [ ] **Step 2: 实现上游值特征**

For each edge/lag/lead/target build a compact value vector from:

```text
aligned source absolute target
source - destination local innovation
source first difference
source load proxy when flow exists
source/destination flow ratio when both exist
distance, slope, travel prior, dynamic shift
```

Mask unavailable components and include their availability bits. Do not impute missing upstream values as observed zero.

- [ ] **Step 3: 实现接收器条件和归一化**

```python
class DirectedConditionalPropagation(nn.Module):
    def forward(
        self,
        context: LocalForecastContext,
        x: Tensor,
        x_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        feature_roles: FeatureRoles,
    ) -> PropagationOutput: ...
```

Query features: destination horizon state, target embedding, lead-band embedding, scale state. Key features: source scale state, edge attributes, lag embedding, availability. Normalize logits by destination segment over incoming edges × valid lags. Store detached routing only when diagnostics are enabled to avoid retaining large graphs.

- [ ] **Step 4: 堆叠 1–3 层**

Each layer updates destination scale/horizon context with a gated residual derived only from upstream messages. Original direct edges are reused each layer; two-hop influence emerges after two layers. Initialize every residual output projection to zero.

- [ ] **Step 5: 测试和提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/models/test_directed_conditional_propagation.py tests/models/test_travel_alignment.py -q
git add src/RiverLagNet/models/directed_conditional_propagation.py tests/models/test_directed_conditional_propagation.py
git diff --cached --check
git commit -m "feat: add receiver-conditioned directed river propagation"
git push
```

---

## Task 4: 组装 RiverLagNet v0.2 并保证零起点嵌套

**Files:**

- Create: `src/RiverLagNet/models/riverlagnet_v02.py`
- Create: `tests/models/test_riverlagnet_v02.py`
- Modify: `src/RiverLagNet/training/lightning_module.py`
- Modify: `src/RiverLagNet/cli/train.py`

- [ ] **Step 1: 写严格嵌套失败测试**

With identical local weights:

- a new directed model before graph training equals the local model within `atol=0, rtol=0`;
- `graph_variant="no_graph"` equals local output bit-for-bit after arbitrary graph parameters change;
- headwater outputs remain bit-for-bit local under directed mode;
- upstream-eligible nodes may change;
- the public shape stays `[B,30,N,3]`;
- checkpoint warm start loads only matching local keys and verifies their hash.

- [ ] **Step 2: 实现包装器**

```python
class RiverLagNetV02(nn.Module):
    def forward(..., return_diagnostics: bool = False):
        context = self.local_backbone.encode_context(...)
        if self.graph_variant == "no_graph":
            return context.prediction
        propagated = self.propagation(context, ...)
        prediction = context.prediction + propagated.correction
        return prediction
```

Supported variants: `directed`, `undirected`, `shuffled`, `no_graph`; lag modes: `no_lag`, `static_lag`, `learned_lag`. `no_lag` means no learned lag distribution and uses the minimum causal one-day alignment, not a future zero-lag path.

- [ ] **Step 3: 注册新模型但保留历史模型可读性**

Add `riverlagnet_v02` to `MODEL_TYPES`. Do not rename or silently change historical `riverlagnet` and `river_crossformer` behavior; retirement is deferred to Session D.

- [ ] **Step 4: 测试、real fast-dev 和提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/models/test_riverlagnet_v02.py tests/training/test_lightning_module.py tests/integration/test_fast_dev_run.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=china_real_daily_contracted_1068_v02 model=riverlagnet_v02 trainer=blackwell_96gb data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
git add src/RiverLagNet/models/riverlagnet_v02.py src/RiverLagNet/training/lightning_module.py src/RiverLagNet/cli/train.py tests/models/test_riverlagnet_v02.py tests/training/test_lightning_module.py tests/integration/test_fast_dev_run.py
git diff --cached --check
git commit -m "feat: assemble RiverLagNet v0.2 as a zero-start residual"
git push
```

---

## Task 5: 实现精确 graph-off 与容量匹配无图控制

**Files:**

- Create: `src/RiverLagNet/models/v02_controls.py`
- Create: `tests/models/test_v02_controls.py`
- Create: `src/RiverLagNet/analysis/model_complexity.py`
- Create: `tests/analysis/test_model_complexity.py`

- [ ] **Step 1: 写失败测试**

Tests must assert:

- exact graph-off never indexes edges and equals the selected local model;
- capacity control uses only destination-local context;
- perturbing another node cannot affect capacity-control output;
- capacity-control trainable parameters are 90%–110% of the directed model;
- all models report FLOPs under the same counting implementation;
- each control returns the common output shape and supports BF16.

- [ ] **Step 2: 实现容量匹配器**

```python
@dataclass(frozen=True)
class CapacityMatch:
    hidden_dim: int
    num_layers: int
    parameter_ratio: float


def choose_local_capacity_control(
    local_context_dim: int,
    target_parameter_count: int,
    ratio_bounds: tuple[float, float] = (0.90, 1.10),
) -> CapacityMatch: ...
```

Use a destination-only residual MLP over horizon states. Do not add unused padding parameters solely to satisfy the ratio. If no candidate matches, fail rather than weaken bounds.

- [ ] **Step 3: 定义正式无图基线选择**

For each promoted graph config, train/evaluate both exact graph-off and capacity-matched no-graph under equal updates. The formal denominator is the higher mean macro NSE; retain both in reports.

- [ ] **Step 4: 测试并提交**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/models/test_v02_controls.py tests/analysis/test_model_complexity.py -q
git add src/RiverLagNet/models/v02_controls.py src/RiverLagNet/analysis/model_complexity.py tests/models/test_v02_controls.py tests/analysis/test_model_complexity.py
git diff --cached --check
git commit -m "feat: add exact and capacity-matched no-graph controls"
git push
```

---

## Task 6: 冻结 12 个图配置和公平筛选矩阵

**Files:**

- Create: `configs/model/riverlagnet_v02.yaml`
- Create: `src/RiverLagNet/configs/model/riverlagnet_v02.yaml`
- Create: `experiments/v0.2_graph_search.yaml`
- Modify: `src/RiverLagNet/analysis/v02_search.py`
- Modify: `src/RiverLagNet/cli/run_v02_search.py`
- Modify: `tests/analysis/test_v02_search.py`
- Modify: `tests/integration/test_packaging.py`

- [ ] **Step 1: 扩展测试冻结图搜索**

Assert exactly 12 graph configs and only approved values:

```text
graph layers: 1, 2, 3
lag radius: 1, 3, 5 days
dropout: 0.05, 0.10, 0.20
lr: 3e-4, 6e-4, 1e-3
weight decay: 1e-5, 1e-4, 1e-3
NSE auxiliary: 0.00, 0.05, 0.10
lead bands: 3 or 6
dynamic shift bound: 0, 2, or 4 days
```

Local config/checkpoint is identical for all. Every promoted directed config must generate paired exact and capacity controls.

- [ ] **Step 2: 定义 halving**

Stage 1: 12 directed configs, fold A, seed 42, 25 epochs. Stage 2: top 4 directed plus paired controls, folds A/B/C, seed 42, 50 epochs. Stage 3: top 2 directed plus paired controls, folds A/B/C, seeds 42/43/44, 100 epochs, patience 12. Rank directed candidates by mean relative gain over the stronger paired no-graph baseline, then absolute macro NSE, then runtime.

- [ ] **Step 3: 测试 dry-run 和配置打包**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest tests/analysis/test_v02_search.py tests/integration/test_packaging.py -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.run_v02_search --family graph --dry-run
```

- [ ] **Step 4: 提交**

```powershell
git add configs/model/riverlagnet_v02.yaml src/RiverLagNet/configs/model/riverlagnet_v02.yaml experiments/v0.2_graph_search.yaml src/RiverLagNet/analysis/v02_search.py src/RiverLagNet/cli/run_v02_search.py tests/analysis/test_v02_search.py tests/integration/test_packaging.py
git diff --cached --check
git commit -m "feat: preregister paired v0.2 graph screening"
git push
```

---

## Task 7: 运行三阶段图筛选和性能剖析

**Files:**

- Modify: `experiments/results.tsv` (append only through code)
- Create: `experiments/reports/v0.2_graph_search_summary.json`
- Create: `experiments/reports/v0.2_graph_search_summary.md`
- Create: `experiments/reports/v0.2_graph_runtime.json`
- Create: `configs/model/riverlagnet_v02_selected.yaml`
- Create: `src/RiverLagNet/configs/model/riverlagnet_v02_selected.yaml`

- [ ] **Step 1: 后台启动并监控固定筛选**

Use one hidden process with GPU lock, PID file, stdout/stderr, resumable ledger checks. Do not spawn parallel GPU workers.

- [ ] **Step 2: 执行 12→4→2 halving**

All competitors use the same effective batch and update budget. Any compile setting is shared only if Session A proved it faster and equivalent. Record real crashes/OOMs.

- [ ] **Step 3: 对前两名做 PyTorch profiler/硬件复测**

Profile representative forward/backward steps with real 1068-node batches. Report top kernels, CPU data wait, allocated/reserved VRAM, SM utilization, samples/s, steps/s, and graph/no-graph wall-time ratio. Optimize only vectorization/buffering without changing mathematical behavior; every optimization gets equivalence tests.

- [ ] **Step 4: 应用运行时红线**

Candidates >3× the stronger no-graph wall time are rejected. Candidates 2×–3× need a documented optimization attempt and explicit rationale; only <=3× may remain, with <=2× preferred.

- [ ] **Step 5: 选择一个候选并提交**

The promoted candidate must have positive gain on all three folds, win at least 7/9 fold-seed pairs in screening, no target mean drop >0.01, and pass runtime. If none passes, Session C does not authorize D.

```powershell
git add experiments/results.tsv experiments/reports/v0.2_graph_search_summary.json experiments/reports/v0.2_graph_search_summary.md experiments/reports/v0.2_graph_runtime.json configs/model/riverlagnet_v02_selected.yaml src/RiverLagNet/configs/model/riverlagnet_v02_selected.yaml
git diff --cached --check
git commit -m "exp: select the v0.2 directed propagation candidate"
git push
```

---

## Task 8: 运行反事实结构与机制消融

**Files:**

- Create: `experiments/v0.2_ablation_matrix.yaml`
- Create: `experiments/reports/v0.2_ablation_summary.json`
- Create: `experiments/reports/v0.2_ablation_summary.md`
- Modify: `src/RiverLagNet/analysis/v02_search.py`
- Modify: `tests/analysis/test_v02_search.py`

- [ ] **Step 1: 冻结消融矩阵测试**

Required conditions:

```text
directed learned lag (selected)
exact graph off
capacity-matched no graph
shuffled graph
undirected graph
minimum causal lag / no learned lag
static travel-prior lag
no innovation
no load/flow features
no dynamic travel shift
no receiver conditioning
one graph layer
three graph layers (if selected is not three)
```

Use folds A/B/C and seed 42 for all; use seeds 43/44 for directed, stronger no-graph, shuffled, undirected, and static-lag core controls.

- [ ] **Step 2: 运行消融**

Each condition changes one mechanism from the selected config. Same data/budget/checkpoint source. Store complete metrics and runtime, not only winners.

- [ ] **Step 3: 生成机制结论**

The directed learned-lag candidate must beat shuffled, undirected, and static/no-lag core controls. Attribute claims only to predictive comparisons. If routing peaks near priors, label this agreement, not recovered physical truth.

- [ ] **Step 4: 提交**

```powershell
git add experiments/v0.2_ablation_matrix.yaml experiments/reports/v0.2_ablation_summary.json experiments/reports/v0.2_ablation_summary.md experiments/results.tsv src/RiverLagNet/analysis/v02_search.py tests/analysis/test_v02_search.py
git diff --cached --check
git commit -m "exp: validate RiverLagNet v0.2 mechanisms and controls"
git push
```

---

## Task 9: Session C 完整验证与交接

**Files:**

- Modify: `docs/architecture.md`
- Modify: `README.md`
- Modify: `docs/coordination/riverlagnet-v0.2-status.md`

- [ ] **Step 1: 更新架构和张量契约文档**

Document equations, normalization domain, observed/local forecast bridge, graph layers, exact headwater behavior, controls, complexity, and non-causal interpretation warning.

- [ ] **Step 2: 运行完整测试、checkpoint 加载和真实 fast-dev**

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m pytest -q
$env:PYTHONIOENCODING='utf-8'; & 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -X utf8=0 -m RiverLagNet.cli.train data=china_real_daily_contracted_1068_v02 model=riverlagnet_v02_selected trainer=blackwell_96gb data.split_name=v02_fold_a trainer.fast_dev_run=true experiment.record_result=false
```

- [ ] **Step 3: 自审公平性和泄漏**

Trace every source value index for representative leads/lags; verify no `y` enters forward; compare update counts and early-stopping information; verify parameter ratio and stronger-baseline denominator; reconcile reports with ledger/checkpoints; confirm no final-test file.

- [ ] **Step 4: 按红灯更新状态**

Authorize D only if the promoted candidate passes all screening, control, target, and runtime gates. Otherwise mark C `blocked` and leave D `waiting`; do not lower thresholds or promote multiple candidates.

- [ ] **Step 5: 最终提交、推送并停止**

```powershell
git add docs/architecture.md README.md docs/coordination/riverlagnet-v0.2-status.md
git diff --cached --check
git commit -m "docs: complete v0.2 graph screening and ablation"
git push
git status --short
git log -1 --oneline
```

Report candidate ID, graph/no-graph screening table, 9 paired outcomes, target metrics, control ordering, params/FLOPs/time/VRAM/throughput, test count, final commit. Then stop.

## Session C Exit Gate

Session C may authorize D only if:

- causal travel alignment and normalization tests pass;
- zero-start, exact graph-off, headwater identity and station-isolated capacity control tests pass;
- capacity control is within ±10% trainable parameters;
- directed candidate uses the frozen local backbone and equal budget;
- mean gain is positive on all folds and wins >=7/9 screening pairs;
- no target mean NSE drops by >0.01;
- directed beats shuffled, undirected and static/no-lag core controls;
- graph wall-time ratio is <=3× and preferably <=2×;
- one and only one candidate is frozen;
- final test remains unopened;
- full pytest and real fast-dev pass;
- all evidence is pushed and only D becomes ready.
