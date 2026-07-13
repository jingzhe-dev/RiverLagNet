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

For destination `i`, source `j`, and discrete lag `τ`:

```text
m_i = sum_(j in Up(i)) sum_(τ=0..max_lag) α_(ijτ) W h_(j,t-τ)
```

Scores use the destination local state, source lagged state, encoded edge attributes, and a learned lag embedding. Softmax is computed jointly over every incoming edge and available lag for each destination node. Consequently, that complete candidate set sums to one. A node with no incoming edges receives an exact zero upstream state.

`no_lag` makes only `τ=0` available. `fixed_lag` selects the rounded and clipped `travel_time_prior_days` channel on each edge while still normalizing over incoming edges. `learned_lag` exposes every available lag from zero through `max_lag`. Dropout affects the message path but not the reported normalized weights.

## Fusion and decoding

The fusion equation is implemented as an identity-safe upstream residual:

```text
z = h_local + sigmoid(gate([h_local, h_upstream])) * W_upstream h_upstream
```

The gate starts near zero, and `W_upstream` has no bias. A node with no
incoming message therefore returns `h_local` exactly instead of attenuating
its local representation. This protects the local forecast while allowing a
useful upstream correction to be learned.

The decoder adds a learned embedding for each future horizon to every node state, applies a shared decoder, then uses three target-specific scalar heads. The output axes are never collapsed at the public interface.

## Training

`RiverForecastModule` provides masked Huber loss and the same mask-aware MAE, RMSE, and NSE functions used by evaluation. Validation macro NSE is the primary checkpoint and early-stopping metric. Channels with no observations or zero target variance are excluded from macro NSE. `RiverDataModule` supplies all models with identical split and batching behavior.

CSV and TensorBoard loggers, learning-rate monitoring, checkpointing, early stopping, gradient clipping, deterministic seeds, timing, peak CUDA memory, CPU fallback, and `fast_dev_run` are configured through Lightning and Hydra.

Attention weights indicate learned routing preference only. They are not causal effect estimates.
