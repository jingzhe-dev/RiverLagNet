# RiverLagNet v0.2 Coordination Status

## Source of truth

- Design: `docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md`
- Agent rules: `AGENTS.md`
- Current integration branch before Session A: `research/20260714-graph-15pct`
- Planned v0.2 integration branch after Session A: `research/20260715-riverlagnet-v02`
- Primary data: `data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz`
- Final test access: prohibited until Session D

## Status values

```text
waiting     prerequisite main session is incomplete
ready       the coordinator has authorized the main session
in_progress one Codex session owns the main session
complete    evidence, commit, and push are recorded
blocked     a documented external condition prevents progress
```

Only the coordinator changes a main session from `waiting` to `ready`. A worker changes only its assigned main session from `ready` to `in_progress` and then to `complete`, or reports a blocker without starting a later session.

## Current observation

Observed on 2026-07-15:

- Run `real1068ext_d3r_v37_s42_no_graph` used the 1,068-node extended data, D3R no-graph model, seed 42, and formal GPU trainer.
- Its configuration had `batch_size=4` and `num_workers=0`.
- The user explicitly requested termination before completion; process PID 225976 exited and GPU allocation returned to the desktop baseline.
- The interrupted run has no valid formal metric and must not be entered as a baseline, keep, discard, or model crash.
- Session A must inspect and remove or explicitly retain its partial `runs/` directory without inventing a ledger row.
- Existing uncommitted changes in `docs/agent_errors.md` and `experiments/results.tsv` belong to the ongoing research branch and must be preserved.
- No formal model training is authorized until Session A completes hardware optimization and protocol freezing.

## Main session board

| Session | Status | Owner | Combined scope | Completion evidence | Commit |
|---|---|---|---|---|---|
| A — engineering foundation and protocol | ready | unassigned | interrupted-run audit; pending-record closure; v0.2 branch; safe cleanup; legacy inventory; hardware optimization; immutable data/split/metric/budget protocol | clean pushed branch; cleanup audit; full pytest; throughput matrix; selected Blackwell profile; data hash; split/leakage tests | — |
| B — signal feasibility and strong local backbone | waiting | unassigned | upstream residual probes; event analysis; conditional data enrichment; strong no-graph backbone; frozen 12-config search set | signal-gate decision; frozen data; three-fold/three-seed local summary | — |
| C — v0.2 model, screening, and ablation | waiting | unassigned | vectorized propagation; graph-off and capacity controls; tests; three-seed screening; runtime profile; counterfactual ablations | promoted candidate; mechanism attribution; runtime gate | — |
| D — confirmation and release | waiting | unassigned | three-fold/five-seed confirmation; uncertainty; visualization; 15% decision; one-time test; final retirement cleanup; release | complete reports; full pytest; final commit and push | — |

## Mandatory session opening prompt

Replace `X` with the single ready main session assigned by the coordinator:

```text
读取 AGENTS.md、docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md、
docs/coordination/riverlagnet-v0.2-status.md 和 Session X 对应实施计划。
只执行主会话 X，但连续完成该会话内的所有相关子任务；不要为每个子任务另开会话。
先检查 git status、远程分支、上一主会话 commit、测试证据和现有 GPU 进程。
保留已有修改；正式 GPU 任务不得并发。
按 TDD 实现并运行规定验证。
完成后更新协调状态，commit 并 push，报告 commit hash、测试、实验结果和未解决问题，然后停止。
```

## Coordinator review checklist

Before authorizing the next main session, verify:

- the recorded commit exists on the remote branch;
- all required tests, benchmarks, or experiments actually ran;
- experiment records are append-only and contain the real commit;
- no final-test access occurred before Session D;
- hardware and optimization budgets match the design;
- generated files and checkpoints follow the retention policy;
- the main-session gate is satisfied without weakening it;
- failures are recorded as discard or crash rather than hidden.

## Decision log

| Date | Decision | Evidence |
|---|---|---|
| 2026-07-15 | Use the 1,068-node contracted network as the sole formal benchmark | User approval and v0.2 design |
| 2026-07-15 | Treat TimeXer/TimeMixer as design references only, not architectures to reproduce | User clarification and approval |
| 2026-07-15 | Optimize Blackwell utilization before multi-seed training | Hardware audit: 95.6 GiB VRAM, batch 4, workers 0, low utilization |
| 2026-07-15 | Remove temporary and retired assets through an allowlisted, tested process | User approval and ignored-file audit |
| 2026-07-15 | Consolidate execution into four main Codex sessions | User requested fewer session handoffs |
