# AGENTS.md — RiverLagNet

## 1. 项目身份

- 仓库：`jingzhe-dev/RiverLagNet`
- 默认分支：`main`
- Python 环境：Conda 环境 `DeepWater`
- 训练框架：PyTorch + Lightning
- 图学习框架：PyTorch Geometric
- 配置管理：Hydra / OmegaConf
- 当前模型：`RiverLagNet v0.1`

本文件适用于整个仓库。用户最新指令优先于本文件。

收到用户需求后直接检查、实现、测试、提交并推送，不重复询问已经能够合理决定的问题。只有涉及凭据、不可逆数据删除或无法推断的数据位置时才暂停。

------

## 2. 当前唯一核心目标

建立一个可运行、可训练、可扩展的有向时滞河网水质预测模型：

> 利用真实上下游河网结构和上游信息传播时滞，提高 NH3N、CODMn 和 TP 的多站点、多提前期预测性能。

当前先完成模型和训练系统，不写论文，不制作展示页面，不扩展为数字孪生或基础大模型。

模型创新集中在：

```text
Directed Lag-aware River Message Passing
```

不要把项目做成普通的：

```text
Transformer + GNN + Attention
```

时间编码器只是基础组件，核心必须是有向河网、传播时滞和上下游信息利用。

------

## 3. 已确定的 v0.1 设计

无需再次确认以下选择：

- 时间尺度：日尺度
- 图节点：水质监测断面
- 图边方向：上游断面 → 下游断面
- 主预测变量：`NH3N`、`CODMn`、`TP`
- 历史窗口：`T_in = 90`
- 预测窗口：`T_out = 30`
- 最大候选时滞：`max_lag = 14`
- 数据划分：按时间顺序 `70% / 15% / 15%`
- 时间编码器：GRU
- 图模块：离散时滞注意力有向消息传递
- 融合模块：门控融合
- 解码方式：直接多提前期预测
- 主损失：masked Huber loss
- 主选择指标：验证集 macro NSE
- 次级指标：MAE、RMSE、各变量 NSE
- 第一阶段不实现概率预测、复杂物理方程和极端事件编码
- 没有真实数据时，先用合成河网数据完成全部代码、测试和训练闭环

不得使用测试集选择模型或调整超参数。

------

## 4. 模型输入与输出

动态输入：

```text
x:          [B, T_in, N, V]
x_mask:     [B, T_in, N, V]
x_quality:  [B, T_in, N, V]   # 可选
static:     [N, S]
edge_index: [2, E]
edge_attr:  [E, A]
```

目标：

```text
y:      [B, T_out, N, 3]
y_mask: [B, T_out, N, 3]
```

输出：

```text
y_hat:  [B, T_out, N, 3]
```

三个输出通道固定依次为：

```text
NH3N, CODMn, TP
```

不得随意 flatten 时间维、节点维和变量维。

标准化参数只能由训练期真实观测计算。

------

## 5. RiverLagNet v0.1 架构

```text
InputMaskEncoder
    ↓
NodeTemporalGRU
    ↓
DirectedLagAwareMessagePassing
    ↓
LocalUpstreamGatedFusion
    ↓
MultiHorizonMultiTargetDecoder
```

### 5.1 InputMaskEncoder

输入数值、缺测 mask、静态属性和时间特征。

缺失值不能直接当作真实零值。数值填充值必须与 mask 同时输入。

### 5.2 NodeTemporalGRU

对每个节点独立编码历史序列，保留完整时间隐状态：

```text
h_seq:   [B, T_in, N, D]
h_local: [B, N, D]
```

### 5.3 DirectedLagAwareMessagePassing

对下游节点 `i` 聚合上游节点 `j` 在不同滞后 `τ` 的状态：

```text
m_i = Σ_j∈Up(i) Σ_τ α_ijτ · W h_j,t-τ
```

要求：

- 只沿上游到下游方向传播；
- `τ ∈ [0, max_lag]`；
- 联合学习上游节点权重和滞后权重；
- 支持边属性；
- 权重归一化范围必须明确并测试；
- 不得将 attention 权重直接解释为真实因果贡献。

### 5.4 LocalUpstreamGatedFusion

融合本站点状态和上游传播状态：

```text
z = gate * h_local + (1 - gate) * h_upstream
```

### 5.5 MultiHorizonMultiTargetDecoder

共享解码器后连接三个变量特异输出头，直接生成未来 30 天结果。

------

## 6. 数据接口

真实观测推荐使用长表 Parquet：

```text
date
station_id
NH3N
CODMn
TP
其他动态变量
*_observed
*_quality
```

河网边表：

```text
src_station_id
dst_station_id
distance_km
slope
travel_time_prior_days
其他可选边属性
```

其中：

```text
src_station_id = 上游
dst_station_id = 下游
```

原始数据、模型权重和大型训练结果不得提交到 Git。

仓库只保存：

- 数据格式定义；
- 小型合成测试数据；
- 预处理代码；
- 配置文件；
- 可复现实验记录。

------

## 7. Lightning 训练系统

所有训练必须通过：

- `LightningModule`
- `LightningDataModule`
- `lightning.pytorch.Trainer`

管理。

禁止另写独立手工训练循环。

Trainer 至少支持：

- CPU/GPU 自动选择；
- `16-mixed`，CPU 时自动回退；
- deterministic seed；
- checkpoint；
- early stopping；
- gradient clipping；
- CSVLogger；
- TensorBoardLogger；
- 学习率监控；
- 训练时间和显存统计；
- `fast_dev_run`；
- 单 GPU 首先可运行；
- 后续可扩展多 GPU。

推荐入口：

```bash
conda run -n DeepWater python -m RiverLagNet.cli.train
conda run -n DeepWater python -m RiverLagNet.cli.evaluate
```

快速验证：

```bash
conda run -n DeepWater python -m RiverLagNet.cli.train \
  trainer.fast_dev_run=true \
  data=synthetic
```

不要创建新的 Conda 环境。缺少依赖时更新 `pyproject.toml`，并在 `DeepWater` 中安装。

------

## 8. 仓库结构

```text
RiverLagNet/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   ├── trainer/
│   └── experiment/
├── src/RiverLagNet/
│   ├── data/
│   │   ├── datamodule.py
│   │   ├── dataset.py
│   │   ├── schema.py
│   │   ├── graph_builder.py
│   │   ├── normalization.py
│   │   └── synthetic.py
│   ├── models/
│   │   ├── input_encoder.py
│   │   ├── temporal_gru.py
│   │   ├── lag_message_passing.py
│   │   ├── fusion.py
│   │   ├── decoder.py
│   │   ├── riverlag_net.py
│   │   └── baselines.py
│   ├── training/
│   │   ├── lightning_module.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── callbacks.py
│   ├── analysis/
│   │   └── upstream_ablation.py
│   └── cli/
│       ├── train.py
│       └── evaluate.py
├── tests/
├── experiments/
│   └── results.tsv
└── docs/
    ├── data_schema.md
    ├── architecture.md
    └── agent_errors.md
```

所有模块必须具有清晰输入、输出和张量形状，并可独立测试。

------

## 9. 必须实现的基线

第一阶段至少实现：

1. Persistence
2. Station GRU，不使用图
3. Static Directed GAT，不使用时滞
4. RiverLagNet，完整模型

所有模型使用完全相同的数据划分、输入变量、训练预算和评价代码。

需要支持以下消融：

- `no_graph`
- `undirected_graph`
- `shuffled_graph`
- `no_lag`
- `fixed_lag`
- `learned_lag`

------

## 10. 测试要求

每次修改后运行相关测试。提交前至少运行：

```bash
conda run -n DeepWater python -m pytest -q
```

必须覆盖：

- 输入输出张量形状；
- masked loss 忽略缺测位置；
- 标准化只使用训练期；
- 时间窗口不存在未来泄漏；
- 河网方向正确；
- 滞后索引正确；
- 时滞权重归一化；
- 无上游节点时稳定运行；
- Lightning 单 batch 训练；
- synthetic `fast_dev_run`；
- checkpoint 保存和加载。

不得通过删除测试、降低断言或隐藏异常使测试通过。

------

## 11. 实验管理

参考 autoresearch 的方式，每次实验只验证一个明确假设。

实验顺序：

1. 先运行基线；
2. 修改一个机制；
3. 提交代码；
4. 运行训练；
5. 记录结果；
6. 决定 keep、discard 或 crash；
7. 再开始下一实验。

实验结果写入受版本控制的：

```text
experiments/results.tsv
```

字段：

```text
timestamp
commit
branch
experiment
seed
val_macro_nse
val_macro_mae
val_macro_rmse
duration_s
peak_vram_gb
status
description
```

`status` 只能是：

```text
baseline
keep
discard
crash
```

训练完整输出保存到忽略 Git 的：

```text
runs/<experiment_id>/
```

禁止伪造、手动修改或挑选有利结果。

------

## 12. Git 工作规则

每完成一个用户需求，必须：

1. 检查 `git status`；
2. 保留用户已有修改；
3. 实现需求；
4. 运行测试或 smoke run；
5. 更新必要文档和实验记录；
6. 创建 Git commit；
7. 推送到当前远程分支；
8. 报告 commit hash、测试结果和未解决问题。

提交信息格式：

```text
feat: ...
fix: ...
test: ...
refactor: ...
docs: ...
exp: ...
chore: ...
```

规则：

- 不使用 `git push --force`；
- 不使用破坏性 `git reset --hard`；
- 不擅自覆盖用户未提交修改；
- 不使用 `--amend` 改写已有提交；
- 失败实验使用新提交修复或 revert；
- 没有文件变化时不创建空提交；
- 推送失败时保留本地 commit 并明确报告。

普通功能修改在当前分支完成。连续自主实验使用：

```text
research/<YYYYMMDD>-<topic>
```

------

## 13. 错误记录

每次由代理自身造成的错误都必须简洁记录到：

```text
docs/agent_errors.md
```

格式：

```text
- 日期｜错误｜原因｜修复｜防止复发
```

只记录实际发生的错误，不预填模板错误，不粘贴完整堆栈。

训练堆栈保存在运行日志中，错误记录只保留结论。

错误修复后与代码一并提交。

------

## 14. 工程原则

- 使用类型标注；
- 公共模块写简短 docstring；
- 配置不得散落硬编码；
- 设置随机种子；
- 数值指标必须 mask-aware；
- 训练和评价使用同一指标实现；
- 不提交原始数据、密钥、checkpoint 和大型日志；
- 不为了“创新”增加无验证价值的复杂度；
- 不将 attention 权重宣称为因果关系；
- 不虚构数据、结果和运行成功状态；
- 没有真实数据时使用 synthetic 数据完成工程验证，并明确标注。

------

## 15. 当前立即执行的任务

读取本文件后，不再只输出架构建议，立即开始构建代码。

按以下顺序执行：

1. 克隆或进入 `jingzhe-dev/RiverLagNet`；
2. 检查 `DeepWater` 环境和 GPU；
3. 建立上述项目结构；
4. 创建 `pyproject.toml` 和 Hydra 配置；
5. 实现数据 schema、mask 和 synthetic 河网生成器；
6. 实现 LightningDataModule；
7. 实现 Persistence 和 Station GRU；
8. 实现 `DirectedLagAwareMessagePassing`；
9. 实现完整 `RiverLagNet`；
10. 实现 LightningModule、损失和指标；
11. 编写单元测试；
12. 运行 `pytest`；
13. 运行 synthetic `fast_dev_run`；
14. 修复所有自身引入的问题；
15. 更新 README、架构文档和错误日志；
16. 自动 commit 并 push；
17. 返回实际文件、命令、测试结果和 commit hash。

第一阶段完成标准：

```text
pytest 全部通过
Lightning fast_dev_run 成功
GRU baseline 可训练
RiverLagNet 可训练
模型输出形状正确
不存在已知时间泄漏
代码已提交并推送
```

------

## 16. 测试产物清理与结果可视化

- 测试源码和断言必须保留，不得删除或弱化；
- 每次 pytest 或 smoke test 结束后，删除 `.pytest_cache`、`__pycache__`、`build/pytest`、`build/smoke` 以及 `runs/smoke_*`、`runs/test_*`；
- 正式实验的 checkpoint、指标和运行目录不属于测试产物，不得由清理程序删除；
- 每次正式实验汇总必须同时生成受版本控制的 PNG 和 PDF 结果图；
- 图必须由机器可读汇总自动生成，保持指标、误差范围和结论一致，不得手工美化数值。
