# RiverLagNet v0.2 Research and Engineering Design

## 1. Decision

RiverLagNet v0.2 resets the project around one fixed research question:

> On the 1,068-node real daily contracted river network, can correctly directed and lag-aligned upstream information improve validation macro NSE by at least 15% relative to a strong same-budget no-graph model?

TimeXer and TimeMixer are references for clear information organization and simple central mechanisms only. RiverLagNet v0.2 does not copy their modules, architecture, naming, or claimed contributions. Its mechanism is derived from directed river transport, dynamic travel time, confluences, upstream load, and conditional downstream relevance.

The project remains focused on model and training-system development. Manuscript writing, presentation websites, digital twins, foundation models, probabilistic prediction, and complex PDE solvers are outside this stage.

## 2. Evidence for the reset

The existing branch contains a long sequence of late graph corrections, trajectory propagation, output transport, expanded ancestor-lag attention, recursive graph decoding, innovation values, adaptive values, counterfactual fusion, dynamic travel-time shifts, and dual-stage recurrence.

The best paired 61-node recurrent result before this reset was:

| Condition | Validation macro NSE | Duration | Peak VRAM |
|---|---:|---:|---:|
| recurrent no-graph | 0.589238 | 254 s | 0.72 GiB |
| recurrent directed graph | 0.595144 | 3,562 s | 5.97 GiB |

The relative gain was about 1.00%, while the graph model was about fourteen times slower. The later D3R 61-node no-graph result reached 0.594073 and the directed version reached 0.593697, so the latest graph mechanism did not improve the latest local model.

The current 1,068-node configuration uses `batch_size=4` and `num_workers=0`. During the design audit, the active run occupied about 8.4 GiB of a 95.6 GiB RTX PRO 6000 and sampled at roughly 38–45% SM utilization. Hardware under-utilization must be fixed before multi-seed formal training.

These results support three decisions:

1. stop extending the existing recursive correction family as the main route;
2. establish a strong, efficient no-graph backbone first;
3. redesign upstream information as a compact, conditional, river-specific signal rather than an expensive generic graph correction.

## 3. Scope and fixed benchmark

### 3.1 Primary benchmark

The sole formal benchmark is the 1,068-node real daily contracted river network currently stored at:

```text
data/processed/china-real-daily-contracted-1068-extended-v0.4/dataset.npz
```

Session A will freeze its manifest, hashes, node order, edge order, variable order, date range, masks, and graph statistics. A content change creates a new version and invalidates prior paired comparisons.

### 3.2 Secondary datasets

- The 61-node mainstem is a development and debugging subset.
- The 238-node contracted tree is a historical sensitivity dataset.
- The 36-node graph is retained only for historical comparison.
- Synthetic data is used for engineering smoke tests and known-signal identifiability tests.

No result on a secondary dataset satisfies the v0.2 success criterion.

### 3.3 Forecast task

- Daily time scale.
- Targets: `NH3N`, `CODMn`, `TP` in this exact order.
- Forecast horizon: 30 days.
- Candidate input windows: 90, 180, and 365 days.
- Public output: `[B, 30, 1068, 3]`.
- All scaling, loss, and metrics are mask-aware.

## 4. Temporal evaluation protocol

The final 15% of dates remain sealed until all model and hyperparameter decisions are frozen. The first 85% is the development region and supplies three expanding rolling-origin folds:

| Fold | Train interval | Validation interval |
|---|---|---|
| A | `[0%, 55%)` | `[55%, 65%)` |
| B | `[0%, 65%)` | `[65%, 75%)` |
| C | `[0%, 75%)` | `[75%, 85%)` |

Window construction must not cross a fold boundary. Each fold fits normalization from its own training observations. No final-test target, metric, prediction, visualization, or statistic may be opened before Session D.

Screening uses three paired seeds. Confirmation uses seeds 42, 43, 44, 45, and 46 across all three folds. Seed 42 cannot serve as a gate by itself.

## 5. Approaches considered

### 5.1 Continue RCELA/D3R tuning

This route has the smallest engineering cost but the weakest evidence. It has already accumulated complex recursive mechanisms, small or negative graph gains, and excessive runtime. It is retained only as a historical baseline.

### 5.2 Strong local backbone plus conditional directed propagation

This is the selected route. It establishes the strongest same-budget local forecaster, freezes its role, then adds a compact river-specific upstream pathway with exact graph-off and capacity-matched controls. It supports clean attribution and efficient vectorization.

### 5.3 Data-first hydrological enrichment

This route adds dynamic flow, speed, event, regulation, and load information before further architecture work. It becomes the active route if the Session B upstream-signal gate fails. It is a contingency, not an excuse to continue architecture search on uninformative inputs.

## 6. Model design

### 6.1 Local multiscale backbone

The local backbone predicts every node independently with shared parameters. It separates:

- endogenous target histories: NH3N, CODMn, and TP;
- local exogenous forcings: meteorology, soil water, discharge, time features, and static attributes;
- missingness and quality indicators.

It builds representations at multiple temporal resolutions and produces a direct 30-day forecast. The implementation may use attention, MLP mixing, recurrent blocks, patching, learned pooling, or a combination, but must remain a strong graph-free model in its own right.

The local backbone is selected under the same rolling folds and fixed search budget used for graph models. It is not intentionally weakened to create graph gain.

### 6.2 Travel-aligned upstream features

For every direct upstream edge, the graph pathway may consume:

- upstream absolute latent state;
- upstream-minus-downstream latent innovation;
- temporal first differences and event magnitude;
- concentration × flow or another explicitly documented load proxy;
- upstream/downstream flow ratio;
- distance, slope, hop identity, and base travel-time prior;
- a bounded hydrology-conditioned travel-time adjustment.

Absolute and relative channels remain separately identifiable. A learned selector may combine them; a fixed innovation-only pathway is not the default because that ablation already degraded performance.

### 6.3 Conditional directed propagation

The graph pathway uses only upstream-to-downstream edges. The default candidate lag set is a bounded neighborhood around each physical travel-time prior rather than every possible lag. Multi-hop context is obtained by stacking one to three direct-edge propagation layers.

Routing is conditioned on:

- destination local context;
- target identity;
- forecast lead or lead band;
- temporal scale;
- edge attributes and hydrological state.

The implementation must be vectorized over batch, edge, lag candidate, scale, target, and lead dimensions. Static graph variants and lag indices are precomputed and reused. Python loops over edges are forbidden in the formal path; a 30-step Python future loop requires measured justification and remains an ablation by default.

### 6.4 Fusion and decoding

The graph pathway is residual and zero-started so graph-off behavior is exact at initialization. The receiver uses learned bounded modulation to combine local and upstream contexts. A headwater with no incoming edge must produce the exact local prediction, not a learned approximation.

The decoder remains direct multi-horizon and multi-target. Scale-specific predictors may be combined, but the public output shape and target order remain fixed.

## 7. Fair controls and budget

Every formal graph candidate has two principal no-graph controls:

1. `exact_graph_off`: identical local backbone, decoder, data, optimizer, and update budget with upstream input disabled;
2. `capacity_matched_no_graph`: replaces the graph pathway with local computation of approximately equal parameter count and compute without reading other nodes.

Additional counterfactuals are directed, shuffled, undirected, no-lag, and static-lag variants.

Fairness requires equal input variables, folds, seeds, effective batch size, maximum optimizer updates, early-stopping information, and hyperparameter-trial count. The capacity-matched control targets parameter count within ±10%. Parameters, FLOPs, samples per second, wall time, and peak VRAM are reported for every formal condition.

The graph candidate targets no more than 2× exact-graph-off wall time. More than 3× blocks five-seed confirmation.

## 8. Hardware design

The reference machine is:

- NVIDIA RTX PRO 6000 Blackwell, 95.6 GiB usable VRAM;
- Intel Core Ultra 9 285K, 24 physical/logical cores;
- approximately 253 GiB RAM;
- PyTorch 2.10.0 + CUDA 12.8 + Lightning 2.6.1;
- BF16, Flash SDPA, and memory-efficient SDPA available.

Session A performs 100–200-step throughput benchmarks without using validation metrics. It compares physical batches 4, 8, 16, 24, and 32; DataLoader workers 0, 4, 8, and 12; persistent workers; prefetch; pinned memory; BF16; TF32 high; fused AdamW; Flash SDPA; and optional `torch.compile`.

The selected profile targets:

- 70–82 GiB training peak VRAM;
- median steady-state SM utilization at least 70%;
- at least 10 GiB system headroom;
- one formal GPU process at a time;
- the fastest numerically equivalent configuration.

Graph and no-graph models use the same effective batch. When physical batch differs, gradient accumulation preserves the effective batch and sample/update exposure.

## 9. Training design

Training uses Lightning exclusively. Candidate evaluation uses successive halving:

| Stage | Maximum epochs | Purpose |
|---|---:|---|
| screen | 25 | reject weak or unstable configurations |
| promote | 50 | compare surviving mechanisms on three folds and three seeds |
| confirm | 100, patience 12 | five-seed formal confirmation |

Each model family receives at most 12 preregistered configurations. Search dimensions are limited to:

- input window: 90, 180, 365;
- hidden dimension: 128, 256;
- local layers: 2, 4;
- graph layers: 1, 2, 3;
- dropout: 0.05, 0.10, 0.20;
- learning rate: 3e-4, 6e-4, 1e-3;
- weight decay: 1e-5, 1e-4, 1e-3;
- NSE auxiliary weight: 0.0, 0.05, 0.10.

The search is a preregistered subset, not the Cartesian product. The exact 12 configurations are frozen in Session B before formal screening.

## 10. Upstream-signal gate

Session B freezes predictions from a strong local baseline and fits leak-free residual probes using training data only. Linear, small MLP, and tree probes compare correct travel-aligned upstream features against wrong direction, shuffled graph, and shuffled time.

The analysis is stratified by target, lead band, upstream availability, flow/event state, and downstream depth.

The graph-model route advances only when:

- mean graph increment is positive in all three folds; and
- at least two folds show at least 3% relative NSE gain from a simple upstream residual probe; and
- correct direction beats shuffled and wrong-direction controls.

This is a readiness gate, not a theoretical upper bound. Failure activates data enrichment for dynamic flow, travel time, events, regulation, or load.

## 11. Primary success criterion

Let the mean cover all 15 fold-seed pairs from three folds and five paired seeds:

```text
G_rel = 100 * (mean_NSE_graph - mean_NSE_no_graph) / abs(mean_NSE_no_graph)
```

Success requires all of the following:

1. `G_rel >= 15%` against `exact_graph_off` and a positive gain against `capacity_matched_no_graph`;
2. the paired absolute ΔNSE 95% confidence-interval lower bound is above zero;
3. the graph model wins at least 12 of 15 fold-seed pairs;
4. no target loses more than 0.01 mean NSE;
5. directed graph beats shuffled, undirected, and no-lag controls;
6. graph gain is concentrated in upstream-eligible nodes rather than headwaters;
7. the test split remains unopened until Session D.

The final test result is reported once and in full regardless of direction.

## 12. Visualization package

Formal confirmation automatically generates machine-readable data plus PNG and PDF figures for:

- paired seed-fold gains and uncertainty;
- target × lead-band ΔNSE heatmaps;
- node-level ΔNSE on the directed network;
- headwater versus upstream-eligible distributions;
- gain versus downstream depth and travel-time prior;
- event and non-event performance;
- representative forecast trajectories;
- routing weights versus travel-time priors with a non-causal interpretation warning;
- learning curves, throughput, SM utilization, duration, and peak VRAM.

No plot may contain manually edited metrics.

## 13. Repository hygiene

Session A removes safe generated artifacts: `.hydra/`, `build/`, egg-info, Python caches, pytest caches, root-level training logs, test runs, and smoke runs.

The repository must never use global `git clean -fdX`, because formal `data/` and `runs/` are ignored. Cleanup uses an explicit allowlist.

Tracked legacy tests are removed only with their retired production modules and configurations, after replacement coverage exists and full tests pass before and after deletion. The intended cleanup removes dead model paths, obsolete configs, obsolete report-only analyses, and their tests together; it never deletes live tests merely to shorten pytest.

Formal data remains in place. `runs/` retains current baselines, promoted/final candidates, required warm starts, and artifacts referenced by key diagnostics. Duplicate, crashed, smoke, and confirmed-discard runs are pruned after ledger and summary verification.

## 14. Cross-session program

The program uses four main Codex sessions. Related engineering and experiments stay in the same session; context compaction and long-running commands are handled inside that session rather than creating a new session for every subtask.

| Session | Combined deliverable | Gate |
|---|---|---|
| A — engineering foundation and protocol | close the interrupted v37 run; preserve and commit pending records; create the v0.2 branch; safely clean generated files; inventory legacy retirement; benchmark Blackwell throughput; freeze the data manifest, rolling folds, metric formula, GPU lock, and budget runner | clean pushed branch; full pytest before/after cleanup; hardware profile selected; hash/split/leakage tests pass |
| B — signal feasibility and strong local backbone | run upstream residual probes and event-stratified diagnostics; enrich data if the gate fails; implement and select the strong no-graph backbone; freeze the 12-configuration search set | signal gate passes on frozen data; stable three-fold, three-seed local baseline selected |
| C — v0.2 graph model, screening, and ablation | implement vectorized directed conditional propagation plus exact and capacity-matched controls; run three-seed mechanism screening; profile runtime; run directed, shuffled, undirected, no-lag, and value-channel ablations | direction/lag/headwater/mask/BF16 tests pass; promoted candidate is stable, attributable, and no more than 3× runtime |
| D — confirmation and release | run three-fold, five-seed confirmation; produce uncertainty and visualization package; make the formal 15% decision; freeze configuration; open the test split once; remove retired code and publish v0.2 | complete validation and test results, full pytest, final cleanup, documentation, commit, and push |

Only one formal GPU session runs at a time. Sequential sessions use the integration branch. Concurrent CPU/code work requires separate worktrees and task branches. Every main session updates `docs/coordination/riverlagnet-v0.2-status.md`, commits, pushes, and stops before the next main session.

## 15. Risks and stop rules

- **No conditional upstream signal:** activate data enrichment; do not increase model size.
- **Graph model exceeds 3× runtime:** profile and vectorize; do not run five seeds.
- **Gain appears only on seed 42:** discard or redesign; seed 42 is not a gate.
- **Gain appears on headwaters:** treat as implementation or fairness failure.
- **Shuffled graph matches directed graph:** do not claim topology benefit.
- **Repeated validation adaptation:** only preregistered configurations proceed; unplanned trials are recorded as exploratory and cannot support the final claim.
- **Test split opened early:** create a newly sealed future period before any final claim.

## 16. Approved deliverables

The next implementation-planning step must produce exact file-level, TDD-based tasks for Sessions A–D. Execution does not begin until this revised written design is reviewed and approved.
