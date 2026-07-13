# RiverLagNet v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a tested Lightning training system for Persistence, Station GRU, Static Directed GAT, and the directed lag-aware RiverLagNet model on leakage-safe synthetic data.

**Architecture:** A chronological windowed data layer preserves `[time, node, variable]` axes and shares one sparse directed graph across each batch. Modular encoders feed baseline or directed lag-aware graph paths, while one LightningModule supplies common masked loss, metrics, logging, and checkpoint behavior.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.4+, Lightning 2.2+, Hydra 1.3+, OmegaConf 2.3+, pytest 8+

## Global Constraints

- Daily inputs use `T_in = 90`, outputs use `T_out = 30`, and learned lag candidates use `max_lag = 14`.
- Target channels are fixed as `NH3N`, `CODMn`, `TP`.
- Split target timestamps chronologically as 70%/15%/15%; fit normalization only on observed training-period data.
- Preserve public tensor axes: `x [B,T_in,N,V]`, `y [B,T_out,N,3]`, `y_hat [B,T_out,N,3]`.
- Training must use LightningModule, LightningDataModule, and Trainer; no manual training loop.
- Do not interpret attention weights as causal contributions.
- Use synthetic data for engineering validation and never fabricate experiment metrics.

---

### Task 1: Package, schema, normalization, and synthetic river data

**Files:**
- Create: `pyproject.toml`, `.gitignore`, package `__init__.py` files
- Create: `src/RiverLagNet/data/schema.py`, `normalization.py`, `graph_builder.py`, `synthetic.py`
- Test: `tests/data/test_schema.py`, `test_normalization.py`, `test_graph_builder.py`, `test_synthetic.py`

**Interfaces:**
- Produces: `TARGET_NAMES`, `RiverGraph`, `TimeSeriesData`, `MaskedStandardScaler.fit/transform/inverse_transform`, `build_graph_variant`, `generate_synthetic_river_data`.

- [ ] Write tests asserting exact shapes, upstream-to-downstream edge orientation, deterministic generation, mask-aware statistics, and graph variants.
- [ ] Run `python -m pytest tests/data -q`; expect collection/import failure because modules do not exist.
- [ ] Implement typed dataclasses with validation, masked train-only normalization, deterministic sparse river generation, and directed graph variants.
- [ ] Run `python -m pytest tests/data -q`; expect all Task 1 tests to pass.
- [ ] Commit with `feat: add synthetic river data foundation`.

### Task 2: Leakage-safe dataset and LightningDataModule

**Files:**
- Create: `src/RiverLagNet/data/dataset.py`, `datamodule.py`
- Create: `configs/data/synthetic.yaml`
- Test: `tests/data/test_dataset.py`, `test_datamodule.py`

**Interfaces:**
- Consumes: `TimeSeriesData`, `MaskedStandardScaler`.
- Produces: `RiverWindowDataset`, `river_collate`, `RiverDataModule.setup`, train/val/test dataloaders, and shared `data_spec` dimensions.

- [ ] Write tests proving each target window lies wholly in its time split, no target timestamp appears in an input, batch shapes are exact, and scaler statistics use only `[:train_end]` observations.
- [ ] Run `python -m pytest tests/data/test_dataset.py tests/data/test_datamodule.py -q`; expect missing-module failures.
- [ ] Implement target-bound split indices, window extraction, calendar features, shared graph collation, and deterministic data loaders.
- [ ] Run the two test files; expect pass.
- [ ] Commit with `feat: add leakage-safe river datamodule`.

### Task 3: Mask-aware objectives and metrics

**Files:**
- Create: `src/RiverLagNet/training/losses.py`, `metrics.py`
- Test: `tests/training/test_losses.py`, `test_metrics.py`

**Interfaces:**
- Produces: `masked_huber_loss`, `masked_mae`, `masked_rmse`, `masked_nse`, `masked_metric_dict`.

- [ ] Write numerical tests where changing masked predictions does not change results, empty masks remain finite, and constant-target NSE is excluded from macro aggregation.
- [ ] Run `python -m pytest tests/training/test_losses.py tests/training/test_metrics.py -q`; expect missing-module failures.
- [ ] Implement differentiable masked reductions and per-target/macro metrics.
- [ ] Run the tests; expect pass.
- [ ] Commit with `feat: add mask-aware training metrics`.

### Task 4: Baseline model components

**Files:**
- Create: `src/RiverLagNet/models/input_encoder.py`, `temporal_gru.py`, `decoder.py`, `baselines.py`
- Create: `configs/model/persistence.yaml`, `station_gru.yaml`, `static_gat.yaml`
- Test: `tests/models/test_baselines.py`, `test_components.py`

**Interfaces:**
- Produces: `InputMaskEncoder.forward`, `NodeTemporalGRU.forward -> (h_seq,h_local)`, `MultiHorizonMultiTargetDecoder.forward`, `PersistenceModel`, `StationGRU`, `StaticDirectedGAT`.

- [ ] Write shape and behavioral tests, including persistence repeating the last observed target and directed GAT accepting sparse graph tensors.
- [ ] Run model tests; expect missing-module failures.
- [ ] Implement the minimal component and baseline APIs without changing public dimensions.
- [ ] Run model tests; expect pass.
- [ ] Commit with `feat: add forecasting baselines`.

### Task 5: Directed lag-aware model and ablations

**Files:**
- Create: `src/RiverLagNet/models/lag_message_passing.py`, `fusion.py`, `riverlag_net.py`
- Create: `src/RiverLagNet/analysis/upstream_ablation.py`
- Create: `configs/model/riverlagnet.yaml`
- Test: `tests/models/test_lag_message_passing.py`, `test_riverlag_net.py`, `tests/analysis/test_upstream_ablation.py`

**Interfaces:**
- Produces: `DirectedLagAwareMessagePassing.forward -> (h_upstream, attention)`, `LocalUpstreamGatedFusion`, `RiverLagNet.forward`, `apply_ablation`.

- [ ] Write tests with hand-selected histories proving lag zero selects the latest state, lag `τ` selects `t-τ`, only source-to-destination messages flow, attention sums to one over incoming edge×lag candidates, and no-upstream nodes return finite zeros.
- [ ] Run lag/model tests; expect missing-module failures.
- [ ] Implement stable segmented softmax, learned/fixed/no-lag modes, edge-attribute scoring, gated fusion, decoder, and graph ablations.
- [ ] Run lag/model tests; expect pass.
- [ ] Commit with `feat: implement directed lag-aware message passing`.

### Task 6: Lightning system, Hydra CLI, callbacks, and checkpointing

**Files:**
- Create: `src/RiverLagNet/training/lightning_module.py`, `callbacks.py`
- Create: `src/RiverLagNet/cli/train.py`, `evaluate.py`
- Create: `configs/config.yaml`, `configs/trainer/default.yaml`, `configs/experiment/synthetic_smoke.yaml`
- Test: `tests/training/test_lightning_module.py`, `tests/integration/test_fast_dev_run.py`, `test_checkpoint.py`

**Interfaces:**
- Produces: `RiverForecastModule`, `RuntimeStatsCallback`, `build_model`, `train.main`, `evaluate.main`.

- [ ] Write tests running one batch, a CPU `fast_dev_run`, and a checkpoint round trip for Station GRU and RiverLagNet.
- [ ] Run integration tests; expect missing-module failures.
- [ ] Implement common Lightning steps, optimizer/scheduler, loggers, callbacks, precision fallback, Hydra entrypoints, and checkpoint evaluation.
- [ ] Run integration tests; expect pass.
- [ ] Commit with `feat: add Lightning training and evaluation`.

### Task 7: Documentation, full verification, and delivery

**Files:**
- Create: `README.md`, `docs/data_schema.md`, `docs/architecture.md`, `docs/agent_errors.md`, `experiments/results.tsv`
- Modify: implementation files only if verification exposes reproducible defects; add a failing regression test first.

**Interfaces:**
- Produces: reproducible commands, documented tensor contracts, actual error record, and experiment ledger header.

- [ ] Document installation, CLI use, synthetic-only validation status, model architecture, schemas, ablations, and non-causal attention caveat.
- [ ] Run `conda run -n DeepWater python -m pytest -q`; expect all tests pass.
- [ ] Run Station GRU and RiverLagNet synthetic fast-dev CLI commands; expect Trainer completion without exception.
- [ ] Run `git diff --check` and inspect `git status --short` for generated artifacts; remove only untracked generated artifacts covered by `.gitignore`.
- [ ] Commit with `docs: document RiverLagNet v0.1 workflow`.
- [ ] Push `main` with `git push -u origin main`; report actual commit hashes and verification results.
