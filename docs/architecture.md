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
Softmax is computed jointly over every incoming edge and available lag for
each destination and horizon. Consequently, each candidate set sums to one. A
node with no incoming edges receives an exact zero upstream state.

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

`RiverForecastModule` provides masked Huber loss and the same mask-aware MAE, RMSE, and NSE functions used by evaluation. Validation macro NSE is the primary checkpoint and early-stopping metric. Channels with no observations or zero target variance are excluded from macro NSE. `RiverDataModule` supplies all models with identical split and batching behavior.

CSV and TensorBoard loggers, learning-rate monitoring, checkpointing, early stopping, gradient clipping, deterministic seeds, timing, peak CUDA memory, CPU fallback, and `fast_dev_run` are configured through Lightning and Hydra.

Attention weights indicate learned routing preference only. They are not causal effect estimates.

The `shuffled_graph` ablation samples a deterministic directed null graph with
the same node set and edge count, while excluding self-loops and every true
edge. Edge attributes retain their empirical distribution but are detached
from the true topology. This makes the structural control disjoint instead of
silently preserving part of the causal graph.
