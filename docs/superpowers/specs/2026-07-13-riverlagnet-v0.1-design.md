# RiverLagNet v0.1 Design

## Objective

Build a runnable, trainable, and extensible daily multi-station water-quality forecasting system whose central mechanism is directed, lag-aware upstream message passing. The system predicts `NH3N`, `CODMn`, and `TP` for 30 future days from 90 historical days.

## Chosen approach

The model uses sparse upstream-to-downstream `edge_index` tensors compatible with PyTorch Geometric. For each destination node, attention is normalized jointly over every incoming edge and every candidate lag in `[0, max_lag]`. This exactly implements the specified `edge × lag` competition while preserving directionality and allowing edge attributes and travel-time priors to affect scores.

Two alternatives were rejected. Dense adjacency-lag tensors are simple but scale as `N² × L`. Independent edge and lag softmaxes are cheaper but do not implement the required joint upstream-and-lag weighting. The chosen sparse joint normalization provides the clearest semantic contract and the most direct tests.

## Data flow

Synthetic data generation produces a directed acyclic river network, dynamic observations, observation masks, optional quality values, static station attributes, edge attributes, and calendar features. A chronological 70/15/15 split is defined by target timestamps. Validation and test inputs may use prior history, but every target belongs wholly to its split. Normalization statistics are fitted only from observed training-period values.

Each sample retains the dimensions `[T, N, V]`. Batching adds only the leading batch dimension. Graph tensors remain shared rather than being duplicated across batch items.

## Model components

`InputMaskEncoder` concatenates filled normalized values with observation masks, quality values, broadcast static attributes, and time features. `NodeTemporalGRU` processes each node independently while returning both the full hidden sequence and latest local state. `DirectedLagAwareMessagePassing` gathers upstream hidden states at discrete lag indices, scores them with destination local state and edge attributes, normalizes across incoming edge-lag candidates per destination, and sums projected messages. Nodes without upstream edges receive a stable zero upstream state. `LocalUpstreamGatedFusion` combines local and upstream states. `MultiHorizonMultiTargetDecoder` uses a shared horizon representation and three target-specific heads without flattening time, node, or target axes at the public interface.

## Baselines and ablations

The common model API supports Persistence, Station GRU, Static Directed GAT, and RiverLagNet. Graph construction supports directed, undirected, and deterministically shuffled edges. Model configuration supports no graph, no lag, fixed lag from travel-time priors, and learned lag attention.

## Training and evaluation

A single LightningModule owns loss, metrics, optimizer, and scheduler logic for trainable models. A LightningDataModule owns deterministic synthetic generation and data loaders. Hydra selects data, model, trainer, and experiment configuration. Checkpointing and early stopping monitor validation macro NSE through its minimized negative form where needed. CSV and TensorBoard loggers use the same run directory. A callback records duration and peak GPU memory.

Masked Huber loss, MAE, RMSE, and NSE ignore missing targets. Macro NSE averages valid target-specific NSE values. The test set is evaluated only after model selection.

## Failure handling and invariants

Schema and tensor-shape validation fail early with descriptive errors. Masked reductions return a differentiable zero when a batch contains no observed targets. NSE excludes channels with no observations or zero target variance. Attention normalization is explicitly tested per destination, and source nodes with no incoming edges remain numerically stable.

## Verification

Unit tests cover schemas, train-only normalization, leakage-free windows, directed graph transformations, lag indexing, joint attention normalization, isolated upstream cases, model shapes, losses and metrics. Integration tests run one Lightning batch, a synthetic fast development run for Station GRU and RiverLagNet, and checkpoint save/load. Final verification runs the complete pytest suite and both requested CLI smoke runs in `DeepWater`.
