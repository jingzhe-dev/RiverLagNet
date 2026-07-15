# RiverLagNet v0.2 Coordination Status

## Source of truth

- Design: `docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md`
- Agent rules: `AGENTS.md`
- Current integration branch before S0: `research/20260714-graph-15pct`
- Planned v0.2 integration branch after S0: `research/20260715-riverlagnet-v02`
- Primary data: `data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz`
- Final test access: prohibited until S10

## Status values

```text
waiting     prerequisite stage is incomplete
ready       the coordinator has authorized the stage
in_progress one Codex session owns the stage
complete    evidence, commit, and push are recorded
blocked     a documented external condition prevents progress
```

Only the coordinator changes a stage from `waiting` to `ready`. A worker changes only its assigned stage from `ready` to `in_progress` and then to `complete`, or reports a blocker without starting later stages.

## Current observation

Observed on 2026-07-15:

- Run `real1068ext_d3r_v37_s42_no_graph` used the 1,068-node extended data, D3R no-graph model, seed 42, and formal GPU trainer.
- Its configuration had `batch_size=4` and `num_workers=0`.
- The user explicitly requested termination before completion; process PID 225976 exited and GPU allocation returned to the desktop baseline.
- The interrupted run has no valid formal metric and must not be entered as a baseline, keep, discard, or model crash.
- S0 must inspect and remove or explicitly retain its partial `runs/` directory without inventing a ledger row.
- Existing uncommitted changes in `docs/agent_errors.md` and `experiments/results.tsv` belong to the ongoing research branch and must be preserved.
- No formal GPU training is currently authorized before S2 and S3 complete.

## Stage board

| Stage | Status | Owner | Required input | Completion evidence | Commit |
|---|---|---|---|---|---|
| S0 current-run closure | ready | unassigned | active v37 process outcome | audited ledger row, error record, clean pushed branch, v0.2 branch | — |
| S1 repository hygiene | waiting | unassigned | S0 complete | cleanup audit, before/after pytest, retained-run manifest | — |
| S2 hardware optimization | waiting | unassigned | S1 complete | throughput matrix, selected Blackwell profile, utilization report | — |
| S3 protocol freeze | waiting | unassigned | S2 complete | data hash manifest, rolling folds, metric/budget tests, GPU lock | — |
| S4 upstream signal gate | waiting | unassigned | S3 complete | residual-probe JSON/TSV, figures, gate decision | — |
| S5 strong no-graph backbone | waiting | unassigned | S4 passes or enriched data is frozen | preregistered 12-config set, three-fold/three-seed summary | — |
| S6 v0.2 graph mechanism | waiting | unassigned | S5 complete | vectorized implementation, graph-off/capacity controls, tests | — |
| S7 mechanism screening | waiting | unassigned | S6 complete | three-seed screening summary and runtime gate | — |
| S8 controlled ablations | waiting | unassigned | S7 candidate promoted | directed/shuffled/undirected/no-lag attribution report | — |
| S9 five-seed confirmation | waiting | unassigned | S8 passes | 15 paired runs, uncertainty, visual package, pass/fail decision | — |
| S10 final freeze and test | waiting | unassigned | S9 frozen decision | one-time test, final cleanup, release report | — |

## Mandatory session opening prompt

Replace `Sx` with the single ready stage assigned by the coordinator:

```text
读取 AGENTS.md、docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md、
docs/coordination/riverlagnet-v0.2-status.md 和 Sx 对应实施计划。
只执行 Session Sx，不提前执行后续阶段。
先检查 git status、远程分支、上一阶段 commit、测试证据和现有 GPU 进程。
保留已有修改；正式 GPU 任务不得并发。
按 TDD 实现并运行规定验证。
完成后更新协调状态，commit 并 push，报告 commit hash、测试、实验结果和未解决问题，然后停止。
```

## Coordinator review checklist

Before authorizing the next stage, verify:

- the recorded commit exists on the remote branch;
- all required tests or smoke checks actually ran;
- experiment records are append-only and contain the real commit;
- no final-test access occurred;
- hardware and optimization budgets match the design;
- generated files and checkpoints follow the retention policy;
- the stage gate is satisfied without weakening it;
- failures are recorded as discard or crash rather than hidden.

## Decision log

| Date | Decision | Evidence |
|---|---|---|
| 2026-07-15 | Use the 1,068-node contracted network as the sole formal benchmark | User approval and v0.2 design |
| 2026-07-15 | Treat TimeXer/TimeMixer as design references only, not architectures to reproduce | User clarification and approval |
| 2026-07-15 | Optimize Blackwell utilization before multi-seed training | Hardware audit: 95.6 GiB VRAM, batch 4, workers 0, low utilization |
| 2026-07-15 | Remove temporary and retired assets through an allowlisted, tested process | User approval and ignored-file audit |
