# RiverLagNet v0.2 Coordination Status

## Source of truth

- Design: `docs/superpowers/specs/2026-07-15-riverlagnet-v0.2-design.md`
- Session A plan: `docs/superpowers/plans/2026-07-15-riverlagnet-v0.2-session-a-foundation.md`
- Session B plan: `docs/superpowers/plans/2026-07-15-riverlagnet-v0.2-session-b-signal-local.md`
- Session C plan: `docs/superpowers/plans/2026-07-15-riverlagnet-v0.2-session-c-graph.md`
- Session D plan: `docs/superpowers/plans/2026-07-15-riverlagnet-v0.2-session-d-confirm-release.md`
- Agent rules: `AGENTS.md`
- Current integration branch before Session A: `research/20260714-graph-15pct`
- Planned v0.2 integration branch after Session A: `research/20260715-riverlagnet-v02`
- Active v0.2 integration branch after Session A: `research/20260715-riverlagnet-v02`
- Session A predecessor commit: `1de534e` on `research/20260714-graph-15pct`
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

The four plans are deliberately self-contained. One Codex task owns one main session and completes all of its internal tasks sequentially; internal tasks must not be split into extra user-facing Codex tasks.

## Current observation

Observed on 2026-07-15:

- Run `real1068ext_d3r_v37_s42_no_graph` used the 1,068-node extended data, D3R no-graph model, seed 42, and formal GPU trainer.
- Its configuration had `batch_size=4` and `num_workers=0`.
- The user explicitly requested termination before completion; process PID 225976 exited and GPU allocation returned to the desktop baseline.
- The interrupted run has no valid formal metric and must not be entered as a baseline, keep, discard, or model crash.
- Session A must inspect and remove or explicitly retain its partial `runs/` directory without inventing a ledger row.
- Existing uncommitted changes in `docs/agent_errors.md` and `experiments/results.tsv` belong to the ongoing research branch and must be preserved.
- No formal model training is authorized until Session A completes hardware optimization and protocol freezing.

Session A completed on 2026-07-16:

- The interrupted v37 run was removed without a ledger row; authentic pending research records were preserved and pushed.
- The frozen dataset has SHA-256 `7e49e836997d7aac68628f61634ed1f89c82e0706d30c6b721b67eac5d381710`, 3,972 days, 1,068 nodes, 1,067 directed edges, and 27 dynamic variables.
- Fold A/B/C chronology, train-only normalization, sealed final-test access, manifest fingerprinting, GPU locking, equal-update budgets, safe cleanup, and real CUDA teardown are covered by the 224-test suite.
- The selected measured profile is BF16 fused AdamW, physical batch 8, workers 0, effective batch 96, accumulation 12, and no compile: 0.7559 optimizer updates/s, 71.02 samples/s, 7.90 GiB peak reserved VRAM, and 87.69 GiB free headroom.
- The profile is 2.29% faster than the equal-exposure batch-4/worker-0 legacy stack. Its 44% median SM utilization is retained as a measured limitation; compile was rejected because Triton is unavailable.
- Full pytest and both required single-GPU fast-dev runs passed. The GPU lock was released and no v0.2 final-test metric was created.

## Main session board

| Session | Status | Owner | Combined scope | Completion evidence | Commit |
|---|---|---|---|---|---|
| A — engineering foundation and protocol | complete | Codex Session A (2026-07-15–16) | interrupted-run audit; pending-record closure; v0.2 branch; safe cleanup; legacy inventory; hardware optimization; immutable data/split/metric/budget protocol | 224 tests passed; synthetic and real fold-A GPU fast-dev passed; manifest/protocol/profile versioned; protected assets retained; no GPU lock or v0.2 final-test metric | `1de534e`, `063e2e1`, `3d38b6e`, `6fa62d2`, `543bb78`, `f7e3996`, `60b73f8`, `3b768da`, plus branch-tip handoff commit |
| B — signal feasibility and strong local backbone | in_progress | Codex Session B (2026-07-16) | upstream residual probes; event analysis; conditional data enrichment; strong no-graph backbone; frozen 12-config search set | takeover verified at `ba230ce`: clean synchronized branch, immutable dataset hash matched, no GPU lock or competing Python training process, and 13 Session A gate tests passed | branch-tip takeover commit |
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
| 2026-07-16 | Select batch 8, workers 0, effective batch 96, accumulation 12, BF16 fused AdamW, and no compile | Real fold-A equal-exposure benchmark: 0.7559 optimizer updates/s, 7.90 GiB reserved, 87.69 GiB headroom, 2.29% faster than legacy; compile failed without Triton |
