# RiverLagNet v0.2 legacy inventory

Session A does not delete production modules, versioned tests, configs, reports, or formal evidence. This inventory controls later Session D retirement. Status meanings are:

- `retain`: part of the v0.2 protocol, a required control, or shared infrastructure;
- `retire-in-D`: removable only after the named v0.2 replacement and equivalence/regression coverage exist;
- `historical-only`: retained as v0.1 provenance through Session D, then archived or deleted only in a dedicated reviewed commit.

## Model modules

| File | Status | Required replacement coverage before deletion |
|---|---|---|
| `models/__init__.py` | retain | Package API remains required. |
| `models/input_encoder.py` | retain | Shared masked-input contract. |
| `models/temporal_gru.py` | retain | StationGRU control and local encoder tests. |
| `models/baselines.py` | retain | Persistence, StationGRU, and static-graph controls. |
| `models/decoder.py` | retire-in-D | Session B selected local decoder plus checkpoint migration test. |
| `models/fusion.py` | retire-in-D | Session C selected local/upstream fusion plus numerical regression. |
| `models/lag_message_passing.py` | retire-in-D | Session C v0.2 directed lag module, graph controls, and equivalence tests. |
| `models/riverlag_net.py` | retire-in-D | Session C integrated v0.2 model and Session D checkpoint/export coverage. |
| `models/autoregressive_graph_decoder.py` | historical-only | Selected v0.2 decoder covers its ablation role. |
| `models/forecast_transport_fusion.py` | historical-only | Selected Session C fusion covers transport controls. |
| `models/graph_cross_attention.py` | historical-only | Session C graph module and static directed control cover comparison. |
| `models/history_propagation.py` | historical-only | Session C lag/history controls cover comparison. |
| `models/output_transport.py` | historical-only | Session C/D output-control evidence is archived. |
| `models/river_crossformer.py` | historical-only | v0.2 local and graph controls replace every registered CrossFormer variant. |
| `models/temporal_transformer.py` | historical-only | Session B capacity-matched local control is selected and tested. |
| `models/topology_encoder.py` | historical-only | Session C topology ablation and replacement tests exist. |
| `models/trajectory_propagation.py` | historical-only | Session C lag-aware propagation covers trajectory variants. |

## Root Hydra configs

Packaged copies under `src/RiverLagNet/configs/` have the same status and must remain byte-identical until any retirement commit.

| File | Status | Required replacement coverage before deletion |
|---|---|---|
| `configs/config.yaml` | retain | Root Hydra composition. |
| `configs/data/china_real_daily_contracted_1068_v02.yaml` | retain | Frozen v0.2 formal dataset and folds. |
| `configs/data/synthetic.yaml` | retain | Fast-dev and shape regression data. |
| `configs/data/synthetic_identifiable_v1.yaml` | historical-only | v0.1 identifiability report remains archived. |
| `configs/data/china_real_daily.yaml` | historical-only | v0.1 dataset provenance. |
| `configs/data/china_real_daily_contracted.yaml` | historical-only | v0.1 dataset provenance. |
| `configs/data/china_real_daily_contracted_1068.yaml` | historical-only | v0.1 1,068-node provenance. |
| `configs/data/china_real_daily_contracted_1068_extended.yaml` | historical-only | Superseded by the v0.2 frozen config and manifest. |
| `configs/data/china_real_daily_mainstem_61_extended.yaml` | historical-only | v0.1 main-stem provenance. |
| `configs/trainer/blackwell_96gb.yaml` | retain | Measured Session A hardware profile. |
| `configs/trainer/default.yaml` | retain | CPU and generic fallback. |
| `configs/trainer/formal_gpu.yaml` | retire-in-D | All formal plans use the measured Blackwell profile. |
| `configs/experiment/default.yaml` | retain | Base experiment composition. |
| `configs/experiment/synthetic_smoke.yaml` | retain | Fast-dev verification. |
| `configs/experiment/china_real_daily.yaml` | historical-only | v0.1 experiment provenance. |
| `configs/experiment/china_real_daily_contracted.yaml` | historical-only | v0.1 experiment provenance. |
| `configs/experiment/china_real_daily_contracted_1068.yaml` | historical-only | v0.1 experiment provenance. |
| `configs/experiment/china_real_daily_contracted_1068_extended.yaml` | historical-only | Replaced by fold-aware v0.2 plans. |
| `configs/experiment/china_real_daily_mainstem_61_extended.yaml` | historical-only | v0.1 experiment provenance. |
| `configs/model/persistence.yaml` | retain | Required control. |
| `configs/model/station_gru.yaml` | retain | Required local control. |
| `configs/model/static_gat.yaml` | retain | Required static directed-graph control. |
| `configs/model/riverlagnet.yaml` | retire-in-D | Session C selected v0.2 graph config and migration coverage. |
| `configs/model/riverlagnet_history.yaml` | historical-only | v0.1 ablation provenance. |
| `configs/model/riverlagnet_output_transport.yaml` | historical-only | v0.1 ablation provenance. |
| `configs/model/riverlagnet_trajectory.yaml` | historical-only | v0.1 ablation provenance. |
| `configs/model/riverlagnet_trajectory_topology.yaml` | historical-only | v0.1 ablation provenance. |
| `configs/model/river_crossformer.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_dual_stage.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_recurrent.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_recurrent_adaptive.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_recurrent_counterfactual.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_recurrent_dynamic_travel.yaml` | historical-only | v0.1 design-reference provenance. |
| `configs/model/river_crossformer_recurrent_innovation.yaml` | historical-only | v0.1 design-reference provenance. |

## Analysis modules

| File | Status | Required replacement coverage before deletion |
|---|---|---|
| `analysis/__init__.py` | retain | Package API. |
| `analysis/hardware_benchmark.py` | retain | Session A hardware evidence and selector tests. |
| `analysis/experiment_suite.py` | retain | Shared controlled-experiment orchestration. |
| `analysis/graph_gain_analysis.py` | retain | Session C graph-gain diagnostics. |
| `analysis/result_visualization.py` | retain | Session D controlled-result visualization. |
| `analysis/river_graph_visualization.py` | retain | Graph audit and final figures. |
| `analysis/upstream_ablation.py` | retain | Required graph counterfactuals. |
| `analysis/contracted_training_summary.py` | historical-only | v0.1 report and ledger remain archived. |
| `analysis/mainstem_error_diagnostic.py` | historical-only | v0.1 main-stem report remains archived. |
| `analysis/plot_graph_15pct_diagnostic.py` | historical-only | v0.1 15% diagnostic figures remain archived. |
| `analysis/real_training_summary.py` | historical-only | v0.1 real-data report remains archived. |
| `analysis/robustness_summary.py` | historical-only | v0.1 robustness report remains archived. |
| `analysis/synthetic_identifiability.py` | historical-only | v0.1 identifiability report remains archived. |

Session D must verify the replacement coverage in this table, run the full suite before and after any retirement, preserve experiment evidence, and use a separate non-destructive Git commit. Until then, all listed assets remain versioned.
