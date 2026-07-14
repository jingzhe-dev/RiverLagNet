# RiverLagNet v0.1 architecture

## Forward path

```text
values + mask + quality + static + calendar features
                         │
                         ▼
                 InputMaskEncoder
                         │ [B,T_in,N,D]
                         ▼
                 NodeTemporalGRU
                    ┌────┴────┐
        h_seq [B,T,N,D]   h_local [B,N,D]
                    │          │
                    ▼          │
       DirectedLagAwareMessagePassing
                    │ h_upstream [B,N,D]
                    └────┬─────┘
                         ▼
             LocalUpstreamGatedFusion
                         │ [B,N,D]
                         ▼
         MultiHorizonMultiTargetDecoder
                         │
                         ▼
                 y_hat [B,30,N,3]
```

The GRU is a local temporal encoder, not the principal innovation. The graph module only propagates along the supplied upstream-to-downstream edges.

## Algorithmic innovation: Causal Multi-hop Lagged History Diffusion

The current 15% gain research branch introduces **Causal Multi-hop Lagged
History Diffusion (CMLHD)**. It targets a specific information bottleneck in
the original architecture: when every station first compresses its own 90-day,
27-variable history into one GRU state, short-lived upstream water-quality,
meteorological, soil-water, and discharge signals may be irreversibly lost
before river-network interaction begins.

CMLHD therefore operates between `InputMaskEncoder` and `NodeTemporalGRU`:

```text
27-variable values + masks + quality + static + calendar
                            │
                            ▼
                    InputMaskEncoder
                            │ e^(0) [B,90,N,D]
                            ▼
       Causal Multi-hop Lagged History Diffusion
          upstream only + travel-time alignment
                            │ e^(K) [B,90,N,D]
                            ▼
                    NodeTemporalGRU
                            ▼
             direct 30-day multi-target decoder
```

For edge `j → i`, rounded travel lag `l_ji`, history time `t`, and diffusion
step `k`, the upstream state is

```text
u_i,t^(k) = mean_(j in Up(i), t-l_ji>=0)
            sigmoid(g(edge_ji)) ⊙ W e_j,t-l_ji^(k-1)

e_i,t^(k) = e_i,t^(k-1)
            + 1[valid upstream] F([e_i,t^(0), e_i,t^(k-1), u_i,t^(k)])
```

Repeating the shared operator `K` times expands the receptive field to `K`
directed upstream hops while accumulating edge travel time. A source index
before the start of the 90-day input window is masked; it is not clamped to an
artificial boundary value. Reverse edges are never created in the directed
condition. The final layer of `F` is initialized to exactly zero, so a
warm-started CMLHD model is bitwise identical to its paired no-graph local
forecaster. Residual-stage training freezes the local input encoder, GRU, and
decoder and updates only CMLHD. Consequently, any paired validation change is
attributable to the new upstream-history pathway rather than a retrained local
backbone.

The method addresses four concrete problems:

1. **Post-encoding information loss:** upstream 27-variable histories enter
   before temporal compression rather than after it.
2. **Travel-time misalignment:** every edge reads `t-l_ji`, not the source and
   destination values from the same day.
3. **Insufficient graph depth:** repeated diffusion exposes multi-hop upstream
   histories without flattening time, node, or variable axes.
4. **Unattributable graph gains:** zero-start residual nesting preserves the
   exact no-graph prediction at initialization and keeps headwaters unchanged.

CMLHD remains a predictive association model. Its learned gates are routing
preferences and must not be interpreted as causal pollutant contributions.

## RiverGraph CrossFormer: attention and Transformer–GNN fusion innovations

`RiverGraphCrossFormer` is the v0.2 research architecture. It retains CMLHD
for an eight-hop upstream receptive field and adds two coupled algorithmic
innovations requested for the attention and fusion stages:

```text
mask-aware 27-variable history
              │
              ▼
  CMLHD directed lag diffusion ── multi-hop upstream history
              │
              ▼
 node-wise Temporal Transformer ── local temporal query q_i,h
              │
              ├─────────────────────────────────────┐
              ▼                                     ▼
       local 30-day decoder      Ancestor-Path–Lag–Horizon Sparse Attention
                                                    │ routing-head tokens
                                                    ▼
                                  Transformer–GNN Cross Fusion
                                                    │ zero-start residual
              └─────────────────────── + ───────────┘
                                      ▼
                              y_hat [B,30,N,3]
```

### Innovation 1: Multi-scale Ancestor-Path–Lag–Horizon Sparse Attention (MAP-LHSA)

**Problem.** Ordinary GAT attention selects immediate neighbors at one timestamp and
ordinary Transformer attention ignores river direction. Both can assign
weight to a downstream node or a lag that is inconsistent with the forecast
lead. Simply masking every source time after the forecast origin avoids
leakage but creates another error: for a 20-day forecast on a 2-day edge, it
removes the physically relevant 2-day route and forces attention onto lags of
20 days or longer. A dense softmax also spreads positive mass over every
candidate, making the learned routing hard to isolate. Stacking many GNN
layers is not a sufficient remedy: a distant upstream signal is repeatedly
mixed and attenuated at every intermediate station before reaching the
forecast target.

**Multi-scale directed path construction.** MAP-LHSA enumerates valid
upstream-to-downstream paths of one to `K=8` edges. A path `p:j→…→i` becomes
one attention candidate with hop embedding `P_|p|`, mean non-temporal edge
attributes, and cumulative travel time
`T_p = sum_(e in p) travel_time_e`. Direct and distant ancestors therefore
compete in one routing operation without reverse edges or repeated GNN
mixing. CMLHD still uses only original one-hop edges; path expansion is
restricted to the attention branch.

**Method.** The **Observed–Forecast Bridge** defines the source candidate as

```text
c_j,h,τ = e_j,t+h-τ                 if h-τ <= 0
          context_local(j,h-τ)      if h-τ > 0
```

The first branch is an observed-history Transformer state. The second is the
model's own upstream forecast context and never a future target. Thus a short
travel lag remains available at every forecast lead without leakage. For
destination `i`, directed ancestor path `p:j→…→i`, lag `τ`, and routing head
`r`, MAP-LHSA
scores:

```text
s_pihτr = <Q_r q_i,h,
            K_r c_j,h,τ + E_r(path_p) + P_|p|,r + L_τr> / sqrt(d_r)
           + b_r(path_p)
           - (τ - T_p)^2 / (2 sigma_r^2)

alpha_i,h,r = sparsemax_{p in PathsTo(i), 0≤τ≤max_lag}(s_pihτr)
```

The joint normalization domain is the Cartesian set of all incoming paths and
all leakage-free observed/forecast-bridge lags for one destination, horizon,
and head. Sparsemax can set unneeded edge-lag routes to exact zero.
`travel_time_prior_days` is a
soft Gaussian anchor with a learned head-specific scale, not a hard label; the
query-key term can move attention away from it when training evidence supports
another lag. Nodes without incoming edges receive exact zero graph context.

MAP-LHSA therefore solves direction leakage, future-target leakage, short-lag
loss at long horizons, multi-hop attenuation, horizon/lag misalignment, and
dense attention dilution in one normalized operator. Its
weights remain predictive routing preferences, not causal effect estimates.

### Innovation 2: Transformer–GNN Head Cross Fusion (TGCF)

**Problem.** A serial `Transformer → GNN` stack forces the GNN output to modify
all temporal representations in the same way; concatenation treats local and
upstream features as interchangeable. Both approaches obscure whether a
specific temporal state actually needs a specific river-routing mechanism.

**Method.** The Temporal Transformer produces the local destination query
`q_i,h`. MAP-LHSA produces one upstream token `g_i,h,r` per graph-routing head.
TGCF performs a second, head-level cross-attention:

```text
beta_i,h,r = softmax_r(<W_q q_i,h, W_g,r g_i,h,r> / sqrt(D))
g_i,h = sum_r beta_i,h,r W_g,r g_i,h,r
z_i,h = sigmoid(G[q_i,h, g_i,h]) ⊙ W_z g_i,h
y_hat_i,h = y_local_i,h + decoder_upstream(z_i,h)
```

This makes the Transformer state the query and the GNN routing heads the
key/value tokens: fusion is conditional on node, forecast lead, and temporal
state rather than a fixed addition. The upstream output heads start at zero,
so the complete graph model initially equals its no-graph Transformer. A
headwater has no valid GNN token and its fused residual remains exactly zero.

Together, CMLHD + MAP-LHSA + TGCF form a specific solution to the project
question: preserve transient upstream covariates before temporal compression,
select physically admissible edge-lag routes for each prediction lead, and
inject them only when the local Transformer query requests that routing head.

## Joint directed lag attention

For destination `i`, source `j`, forecast lead `h`, and discrete lag `τ`:

```text
m_(i,h) = sum_(j in Up(i)) sum_(τ=h..max_lag) α_(ijhτ) W h_(j,t+h-τ)
```

This horizon alignment never reads an unobserved future source. When
`t+h-τ <= t`, it selects the corresponding historical hidden state. When the
aligned source time lies after the forecast origin, it carries forward the
last observed source hidden state as a leakage-free latent forecast proxy.
Scores use the destination local state, aligned source state, encoded edge
attributes, and a learned lag embedding. Learned-lag scores add a configurable
Gaussian log-prior centered on `travel_time_prior_days`; the neural score
remains a trainable residual that can move probability away from the prior.
For `learned_lag`, the routed state is parameterized as the latest observable
upstream state plus a learned scalar multiple of the attention-weighted lag
difference. The scalar is zero-initialized and passed through `tanh`, so the
full model starts from the empirically stronger current-upstream path and must
earn any departure toward historical states on validation data. A configurable
`lag_residual_max_mix` bounds the magnitude of that departure between zero and
one; cap experiments must be recorded rather than silently changing the
default.
For `learned_lag`, incoming-edge weights are normalized from the neural lag-0
scores, while lag weights are normalized within each edge after adding the
travel-time prior. Their product is the reported joint edge-lag weight and
sums to one per destination and horizon. This anchored factorization makes a
zero lag-residual mix exactly equal to `no_lag`, including at multi-upstream
confluences. The single-candidate `no_lag` and `fixed_lag` modes retain their
direct incoming-candidate normalization. A node with no incoming edges
receives an exact zero upstream state.

`no_lag` is the static-graph ablation and repeats the latest source state at
every horizon. `fixed_lag` selects the rounded and clipped
`travel_time_prior_days` channel when that source time is observable.
`learned_lag` exposes every causally observable lag through `max_lag`. Dropout
affects the message path but not the reported normalized weights.

## Fusion and decoding

The fusion equation is implemented as an identity-safe upstream residual:

```text
z = h_local + sigmoid(gate([h_local, h_upstream])) * W_upstream h_upstream
```

The gate starts near zero, and `W_upstream` has no bias. A node with no
incoming message therefore returns `h_local` exactly instead of attenuating
its local representation. This protects the local forecast while allowing a
useful upstream correction to be learned.

Fusion receives horizon-specific local and upstream states, but the fused
increment is decoded through a separate bias-free upstream correction head.
The main decoder always sees the untouched local state. The final prediction
is therefore:

```text
y_hat = decoder_local(h_local) + decoder_upstream(z - h_local)
```

With no upstream state, both the hidden increment and output correction are
exactly zero, so the graph path cannot perturb the local forecast. The local
decoder adds a learned embedding for each future horizon, applies a shared
decoder, then uses three target-specific scalar heads. The output axes are
never collapsed at the public interface.

## Training

`RiverForecastModule` uses masked Huber as the primary loss. A configurable,
non-negative `nse_aux_weight` can add a target-balanced masked `1 - NSE`
auxiliary term for explicitly recorded selection-alignment experiments; its
default is zero. Training and evaluation share the same mask-aware MAE, RMSE,
and NSE definitions. Validation macro NSE is the primary checkpoint and
early-stopping metric. Channels with no observations or zero target variance
are excluded from macro NSE. `RiverDataModule` supplies all models with
identical split and batching behavior.

An optional two-stage attribution experiment warm-starts a full RiverLagNet
from a validation-selected `no_graph` checkpoint, freezes the input encoder,
temporal GRU, and local decoder, and zero-initializes the target-specific
upstream output heads. The directed model therefore starts with predictions
exactly equal to the local checkpoint. During the second stage only the
message-passing, fusion, and upstream residual decoder are optimized. This
does not make routing weights causal, but it prevents apparent graph gains
from being produced by a changed local backbone or by headwater nodes.

CSV and TensorBoard loggers, learning-rate monitoring, checkpointing, early stopping, gradient clipping, deterministic seeds, timing, peak CUDA memory, CPU fallback, and `fast_dev_run` are configured through Lightning and Hydra.

Attention weights indicate learned routing preference only. They are not causal effect estimates.

## Current validation-selected real-data candidate

The contracted real-data architecture screen currently selects the strictly
nested directed upstream residual with a linear horizon gate and bounded
learned-lag refinement. Across seeds 42--46 it improves validation macro NSE
over its paired frozen `no_graph` checkpoint by `0.001337 ± 0.000412` and over
an independently retrained Static Directed GAT by `0.002749 ± 0.002109`; both
comparisons are positive for all five seeds. The graph-and-horizon gain is
`0.002302 ± 0.000711` on the 112 nodes with incoming upstream edges, while the
126 headwater predictions remain exactly unchanged. The largest horizon-band
gain is at days 15--30 (`+0.003773`). These are validation increments, not
causal effects; the held-out test split remains unopened.

The horizon gate independently adds `0.000232 ± 0.000248` macro NSE with five
of five positive seed differences. The bounded learned-lag stage is technically
validation-best, but its independent increment is only
`0.00000322 ± 0.00000235`; global peak-bias lags are inconsistent across seeds.
The empirical claim is therefore a stable directed-graph benefit, not recovered
physical travel time. Exact evidence and limitations are in
`docs/core_result.md`.

The five-seed checkpoint audit also shows a predeclared horizon pattern:
days 1--7 originally lost `0.001089` NSE on average, whereas days 8--14 and
15--30 gained `0.001242` and `0.003537`. The selected `linear` horizon gate
reduces the final days 1--7 difference to `-0.000068` while the days 8--14 and
15--30 gains reach `0.001340` and `0.003773`. It applies
a two-parameter bounded scale to the already decoded upstream correction:

```text
scale(h) = 2 * sigmoid(offset + slope * normalized_lead(h))
```

Both parameters start at zero, making every scale exactly one and the new
model exactly equal to the selected checkpoint. Horizon-calibration training
freezes the entire graph forecaster, keeps its dropout modules in evaluation
mode, and updates only `offset` and `slope`. This is an explicitly registered
validation experiment, not a post-hoc modification of stored predictions.

The controlled lag-refinement stage then warm-starts the selected
horizon-calibrated `no_lag` checkpoint as `learned_lag`. It freezes the local
forecaster, incoming-edge routing, upstream residual decoder, and horizon
gate. Only the zero-started scalar lag mixture and 14 lag-0-anchored global
relative-lag biases are trainable. The mixture remains capped at 10 percent,
so the stage starts exactly at `no_lag` and any validation difference is
attributable to historical upstream-state mixing rather than changed edge
weights or model capacity elsewhere.

The `shuffled_graph` ablation samples a deterministic directed null graph with
the same node set and edge count, while excluding self-loops and every true
edge. Edge attributes retain their empirical distribution but are detached
from the true topology. This makes the structural control disjoint instead of
silently preserving part of the causal graph.
