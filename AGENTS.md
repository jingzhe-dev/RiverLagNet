# AGENTS.md — RiverLagNet v0.2

## 1. 项目身份与适用范围

- 仓库：`jingzhe-dev/RiverLagNet`
- 默认分支：`main`
- 研究分支：`research/<YYYYMMDD>-<topic>`
- Python 环境：Conda 环境 `DeepWater`
- 训练框架：PyTorch + Lightning
- 图学习框架：PyTorch Geometric
- 配置管理：Hydra / OmegaConf
- 当前研发版本：`RiverLagNet v0.2`

本文件适用于整个仓库。用户最新指令优先于本文件。

每个 Codex 会话必须先读取：

1. 本文件；
2. `docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md`；
3. `docs/coordination/riverlagnet-v0.2-status.md`；
4. 当前会话对应的实施计划任务。

不同会话不得自行越过阶段门槛。只执行协调状态文件中分配给本会话的阶段。

------

## 2. 当前唯一核心目标

在固定的 1068 节点真实日尺度收缩河网上，通过改进河网信息利用、模型架构和训练参数，使加入正确有向河网信息后的验证集 macro NSE 相对同预算强无图模型平均提升至少 15%，并通过 rolling-origin、多种子、反事实消融和可视化验证。

目标变量固定为：

```text
NH3N, CODMn, TP
```

预测窗口固定为：

```text
T_out = 30 days
```

15% 是待验证的研究目标，不得通过挑选种子、改动测试集、削弱无图基线或手工修改结果制造。若当前数据缺少足够的条件上游增量信息，必须如实报告并转向水文、旅行时间、事件或负荷数据增强。

当前不写论文、不制作展示网站、不扩展为数字孪生或基础大模型。

------

## 3. 正式数据基准

正式主基准是：

```text
data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz
```

该数据在阶段 S3 完成 hash、节点、边、变量、时间覆盖和测试期封存审计后冻结。任何数据内容变化必须生成新版本并重新训练全部配对基线。

其他数据的用途：

- 61 节点主干河网：快速开发和机制排错；
- 238 节点收缩树：历史结果和敏感性分析；
- 36 节点早期河网：历史结果；
- synthetic：工程 smoke test 和可识别性单元测试。

以上数据不得替代 1068 节点正式成功判定。

原始数据、正式数据、checkpoint 和大型运行日志不得提交到 Git。

------

## 4. 时间划分与测试集隔离

最后 15% 时间段是最终封存测试集，不得用于：

- 架构选择；
- 超参数搜索；
- early stopping；
- 数据版本选择；
- 图机制选择；
- 阈值拟合；
- 可视化挑选。

前 85% 开发区固定使用三个 expanding rolling-origin folds：

```text
Fold A: train [0%, 55%), validate [55%, 65%)
Fold B: train [0%, 65%), validate [65%, 75%)
Fold C: train [0%, 75%), validate [75%, 85%)
Final test: [85%, 100%)
```

窗口构造必须保证历史和目标不跨越各 fold 边界。标准化参数只能由对应 fold 的训练期真实观测计算。

------

## 5. 模型输入与输出合同

动态输入：

```text
x:          [B, T_in, N, V]
x_mask:     [B, T_in, N, V]
x_quality:  [B, T_in, N, V]   # optional
static:     [N, S]
edge_index: [2, E]
edge_attr:  [E, A]
```

目标和输出：

```text
y:      [B, 30, N, 3]
y_mask: [B, 30, N, 3]
y_hat:  [B, 30, N, 3]
```

三个输出通道固定依次为 `NH3N, CODMn, TP`。公共接口不得 flatten 时间、节点或目标维。

缺失值不能作为真实零值。填充值必须与 mask 同时输入。指标、损失和标准化都必须 mask-aware。

------

## 6. v0.2 架构研究边界

TimeXer 和 TimeMixer 仅作为“清晰组织不同信息、建立简单核心机制”的设计参考，不复制它们的具体架构、模块、命名或论文贡献。

允许使用 GRU、Transformer、MLP、Mixer、patch、多尺度分解、状态空间或混合骨干，但创新必须从河网问题本身推导，并集中于：

```text
Directed topology-constrained conditional upstream propagation
```

v0.2 主路线包含：

1. 强本站多尺度时间骨干；
2. 旅行时间对齐的直接上游编码；
3. 接收端、目标、尺度和预测提前期条件化的有向传播；
4. 直接多提前期、多目标解码。

上游候选应优先包含绝对状态、上下游相对变化、一阶变化、负荷代理、流量条件和边属性。不得因纯 innovation 消融失败而只保留差值，也不得把 attention 权重解释为真实因果贡献。

默认不枚举所有祖先 × 所有时滞 × 所有未来步。主模型应使用直接边、物理旅行时间附近的有界候选时滞和少量传播层获得多跳感受野。昂贵递归仅作为有明确假设的消融。

历史窗口、时间尺度、隐藏维度、层数、学习率和损失权重不再永久固定；它们必须在预注册范围和统一搜索预算内选择。

------

## 7. 强基线与同预算公平性

保留 Persistence、Station GRU 和 Static Directed GAT 作为传统参考，但正式 15% 比较必须使用强无图基线。

每个候选图模型必须同时提供：

1. `exact_graph_off`：相同本站骨干、解码器、数据和训练配置，只关闭上游输入；
2. `capacity_matched_no_graph`：使用本站模块补足近似参数量和计算量，不读取其他节点；
3. 正确 `directed_graph`；
4. `shuffled_graph`；
5. `undirected_graph`；
6. `no_lag` 或静态旅行时间对照。

同预算至少要求：

- 相同输入变量和 fold；
- 相同有效 batch size；
- 相同最大优化器更新次数；
- 相同 early-stopping 信息；
- 相同种子；
- 每个模型族相同超参数试验次数；
- capacity-matched 对照参数量尽量控制在 ±10%；
- 同时报告参数量、FLOPs、吞吐率、时长和峰值显存。

图模型正式训练时长目标不超过 exact graph-off 的 2 倍；超过 3 倍不得进入五种子确认。

------

## 8. 硬件利用规则

正式硬件为单张 NVIDIA RTX PRO 6000 Blackwell 96 GB、24 CPU 核和约 253 GiB RAM。

任何多种子正式训练前必须完成无指标硬件基准。目标：

- 稳态峰值显存约 70–82 GiB，保留系统余量；
- 稳态 GPU SM 利用率中位数至少 70%；
- DataLoader、物理 batch、图预计算和算子设置已通过吞吐比较；
- 只保留数值一致且可复现的加速选项。

必须比较物理 batch `4/8/16/24/32` 和 DataLoader workers `0/4/8/12`。允许使用 BF16、TF32 high、fused AdamW、Flash SDPA 和 `torch.compile`，但 `torch.compile` 只有在无关键 graph break、数值一致且实际更快时才能进入正式配置。

图结构、边属性和时滞索引应预计算并复用，禁止在每个 batch 中重复构图。应优先消除 Python 边循环和无必要的 30 步递归。

同一时刻只允许一个正式 GPU 训练进程。新会话必须先检查 GPU 和 `runs/` 中的训练锁或现有进程，不得与正在运行的正式实验争抢 GPU。

不要创建新的 Conda 环境。缺少依赖时更新 `pyproject.toml` 并在 `DeepWater` 中安装。

------

## 9. 训练与超参数预算

所有训练必须通过 `LightningModule`、`LightningDataModule` 和 `lightning.pytorch.Trainer`。

候选模型采用 successive halving：

```text
screen:  maximum 25 epochs
promote: maximum 50 epochs
confirm: maximum 100 epochs, early-stopping patience 12
```

每个模型族最多 12 个预注册超参数组合。搜索优先覆盖：

- `T_in ∈ {90, 180, 365}`；
- `hidden_dim ∈ {128, 256}`；
- 本站层数 `∈ {2, 4}`；
- 图传播层数 `∈ {1, 2, 3}`；
- dropout `∈ {0.05, 0.10, 0.20}`；
- learning rate `∈ {3e-4, 6e-4, 1e-3}`；
- weight decay `∈ {1e-5, 1e-4, 1e-3}`；
- NSE auxiliary weight `∈ {0.0, 0.05, 0.10}`。

不得未经预注册遍历所有笛卡尔积。每次实验只验证一个明确假设，结果全部追加到 `experiments/results.tsv`。

------

## 10. 数据可识别性门槛

在实现大型图模型前，必须冻结一个强无图预测器并对其训练期和 rolling-validation 残差做无泄漏上游探针。

探针至少比较：

- 正确旅行时间对齐的上游绝对状态、相对变化、变化率和负荷代理；
- 错误方向；
- shuffled graph；
- 打乱时间；
- 不同提前期、目标、普通期和快速变化期。

只有当三个 folds 的平均图增量为正，且至少两个 folds 的简单上游残差探针达到 3% 相对 NSE 增益时，才进入大型图模型实现。该门槛不是理论上界；未通过时转向动态流量、旅行时间、事件或负荷数据增强。

------

## 11. 最终成功判定

相对增益定义为：

```text
G_rel = 100 * (mean_NSE_graph - mean_NSE_no_graph) / abs(mean_NSE_no_graph)
```

其中均值覆盖 3 rolling folds × 5 配对种子，macro NSE 由统一 mask-aware 指标实现计算。

正式通过必须同时满足：

- `G_rel >= 15%`；
- 配对绝对 ΔNSE 的 95% 置信区间下界大于 0；
- 15 个 fold-seed 配对中至少胜出 12 个；
- 任一目标的平均 NSE 不下降超过 0.01；
- directed graph 优于 shuffled、undirected 和 no-lag 对照；
- 增益主要出现在具有真实上游节点的站点；
- 所有结构和超参数冻结后才打开最终测试集一次。

测试集结果无论是否有利都必须完整报告。

------

## 12. 测试要求

每次修改后运行相关测试。提交前至少运行：

```powershell
& 'C:\Program Files\ANACONDA\envs\DeepWater\python.exe' -m pytest -q
```

必须覆盖：

- 公共输入输出张量形状；
- masked loss 和指标；
- fold 和窗口不存在未来泄漏；
- 标准化只使用训练期；
- 河网方向、时滞索引和权重归一化；
- headwater 精确退化；
- correct、shuffled、undirected 和 graph-off 对照；
- BF16 混合精度；
- Lightning 单 batch、fast dev run、checkpoint 保存和加载；
- 1068 数据 manifest 与 split hash。

不得通过删除仍有效测试、降低断言或隐藏异常使测试通过。

------

## 13. 目录清理与退役规则

允许安全删除的生成物：

```text
.hydra/
build/
*.egg-info/
__pycache__/
.pytest_cache/
root-level train.log and evaluate.log
runs/smoke_*
runs/test_*
```

禁止使用无差别的 `git clean -fdX`，因为 `data/` 和 `runs/` 也被 Git 忽略。

旧测试只有在以下条件全部满足时才能删除：

1. 对应生产模块已退役；
2. 当前配置、CLI 和文档不再引用；
3. 新测试覆盖仍需保留的行为；
4. 删除源模块、配置和测试在同一提交完成；
5. 删除前后完整 pytest 均通过。

旧模型、旧配置、旧绘图脚本和只验证旧报告文本的测试，应在 v0.2 跑通后统一退役。不得仅为了缩短测试时间删除有效回归测试。

正式 `data/` 不属于临时文件。`runs/` 只保留当前正式基线、最终候选、必要 warm-start checkpoint 和被关键诊断引用的产物；crash、重复 smoke、重复 checkpoint 和已确认 discard 的运行目录在审计后删除。

------

## 14. 实验记录和可视化

`experiments/results.tsv` 是 append-only 账本，字段和 status 约束沿用现有实现。禁止伪造、手动挑选或覆盖结果。

正式多种子总结必须自动生成机器可读 JSON/TSV，以及 PNG 和 PDF：

- fold-seed 配对增益分布；
- 各目标和提前期 heatmap；
- 节点 ΔNSE 河网图；
- headwater 与 downstream 对比；
- 不同上游深度的增益；
- 学习曲线、吞吐率、GPU 利用率和峰值显存；
- 旅行时间先验与学习路由的关系，明确非因果解释；
- 代表性快速变化事件案例。

图必须由机器可读汇总生成，不得手工修改数值。

------

## 15. 跨 Codex 会话协调

协调状态文件是不同会话之间的唯一事实来源：

```text
docs/coordination/riverlagnet-v0.2-status.md
```

每个会话必须：

1. 只执行一个标记为 ready 的阶段；
2. 开始前检查 git status、远程分支、现有 GPU 进程和上一阶段证据；
3. 完成后更新状态、commit、push；
4. 报告 commit hash、测试、实验结果和未解决问题；
5. 不自动开始下一个阶段。

顺序执行会话可使用同一集成分支。若多个会话并发修改代码，必须使用独立 worktree 和任务分支；GPU 正式训练仍只能串行。

------

## 16. Git 和错误记录

每个完成的用户需求都必须检查状态、保留用户修改、验证、提交并推送。不得使用 `git push --force`、`git reset --hard` 或 `--amend`。没有文件变化时不创建空提交。

提交信息使用：

```text
feat: ...
fix: ...
test: ...
refactor: ...
docs: ...
exp: ...
chore: ...
```

代理自身造成的实际错误记录到 `docs/agent_errors.md`：

```text
- 日期｜错误｜原因｜修复｜防止复发
```

只记录结论，不粘贴完整堆栈。

------

## 17. 当前阶段

不得继续执行旧 v0.1 的“从头构建项目”任务列表。当前工作严格按照协调状态文件中的 S0–S10 推进：

```text
S0 current-run closure
S1 repository hygiene
S2 hardware optimization
S3 protocol freeze
S4 upstream signal gate
S5 strong no-graph backbone
S6 RiverLagNet v0.2 graph mechanism
S7 mechanism screening
S8 controlled ablations
S9 five-seed confirmation and visualization
S10 final freeze and one-time test
```

任何阶段未通过门槛时，先修复或记录 discard，不得通过降低门槛直接进入下一阶段。
