# Identifiable synthetic v1 data gate

Overall status: **PASS**

All metrics use only the chronological training period of complete synthetic signals.

| Seed | Routed variance ratio | True correlation | Reverse correlation | Shifted correlation | Direction margin | Lag recovered |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.2471 | 0.5793 | -0.0468 | 0.1847 | 0.3946 | 21/21 |
| 43 | 0.2891 | 0.6438 | -0.0523 | 0.1519 | 0.4919 | 21/21 |
| 44 | 0.2888 | 0.4836 | -0.0584 | 0.2762 | 0.2074 | 21/21 |
| 45 | 0.4212 | 0.7249 | -0.0138 | 0.2351 | 0.4897 | 21/21 |
| 46 | 0.4915 | 0.6747 | -0.0576 | 0.1918 | 0.4829 | 21/21 |

Pooled lag recovery within one day: `1.0000`.

Thresholds: routed variance ratio 0.20--0.60 per seed; direction margin at least 0.15 per seed; pooled lag recovery at least 0.80.

These diagnostics establish synthetic benchmark identifiability, not real-world causality.
