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

The contracted real-data architecture screen currently selects the two-stage
directed `no_lag` upstream residual. Across seeds 42--46 it improves validation
macro NSE over its paired frozen `no_graph` checkpoint by
`0.001101 ± 0.000255` with five of five positive differences. The gain is
`0.001914 ± 0.000457` on the 112 nodes with incoming upstream edges, while the
126 headwater predictions remain exactly unchanged. The largest horizon-band
gain is at days 15--30 (`+0.003537`). These are validation increments, not
causal effects; the held-out test split remains unopened.

The anchored learned-lag screen has not exceeded this candidate, so learned
lag remains an unvalidated mechanism rather than part of the current empirical
claim. Exact evidence and limitations are in `docs/core_result.md`.

The five-seed checkpoint audit also shows a predeclared horizon pattern:
days 1--7 lose `0.001089` NSE on average, whereas days 8--14 and 15--30 gain
`0.001242` and `0.003537`. The optional `linear` horizon gate therefore applies
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
