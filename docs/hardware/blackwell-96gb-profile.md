# RTX PRO 6000 Blackwell 96 GB hardware profile

All rows are measured on the frozen 1,068-node fold-A training data with `T_in=180`, StationGRU hidden size 64, BF16, TF32 high, fused AdamW, and no validation metrics. The table records warm-up-excluded training throughput.

| Batch | Effective | Accum | Workers | Rep | Compile | Status | Steps/s | Samples/s | Allocated GiB | Reserved GiB | SM % | Power W | Reason |
|---:|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 96 | 24 | 0 | 0 | false | ok | 0.6955 | 65.35 | 3.78 | 3.97 | 40.0 | 188.8 | accepted |
| 4 | 96 | 24 | 4 | 0 | false | ok | 0.4042 | 37.98 | 3.78 | 3.97 | 25.0 | 125.3 | accepted |
| 4 | 96 | 24 | 8 | 0 | false | ok | 0.4178 | 39.25 | 3.78 | 3.97 | 25.0 | 126.3 | accepted |
| 4 | 96 | 24 | 12 | 0 | false | ok | 0.4145 | 38.94 | 3.78 | 3.97 | 8.0 | 157.1 | accepted |
| 8 | 96 | 12 | 0 | 0 | false | ok | 0.7593 | 71.34 | 7.49 | 7.90 | 45.0 | 201.7 | accepted |
| 8 | 96 | 12 | 4 | 0 | false | ok | 0.4170 | 39.18 | 7.49 | 7.90 | 22.0 | 126.2 | accepted |
| 8 | 96 | 12 | 8 | 0 | false | ok | 0.4093 | 38.45 | 7.49 | 7.90 | 18.0 | 146.1 | accepted |
| 8 | 96 | 12 | 12 | 0 | false | ok | 0.4041 | 37.96 | 7.49 | 7.90 | 0.0 | 150.8 | accepted |
| 16 | 96 | 6 | 0 | 0 | false | ok | 0.7585 | 71.26 | 14.90 | 15.76 | 42.0 | 202.2 | accepted |
| 16 | 96 | 6 | 4 | 0 | false | ok | 0.4080 | 38.33 | 14.90 | 15.76 | 0.0 | 158.6 | accepted |
| 16 | 96 | 6 | 8 | 0 | false | ok | 0.3847 | 36.14 | 14.90 | 15.76 | 0.0 | 153.9 | accepted |
| 16 | 96 | 6 | 12 | 0 | false | ok | 0.3649 | 34.28 | 14.90 | 15.76 | 0.0 | 105.1 | accepted |
| 24 | 96 | 4 | 0 | 0 | false | ok | 0.7581 | 71.22 | 22.32 | 23.61 | 43.0 | 200.9 | accepted |
| 24 | 96 | 4 | 4 | 0 | false | ok | 0.4051 | 38.06 | 22.32 | 23.61 | 0.0 | 161.9 | accepted |
| 24 | 96 | 4 | 8 | 0 | false | ok | 0.3762 | 35.34 | 22.32 | 23.61 | 0.0 | 160.9 | accepted |
| 24 | 96 | 4 | 12 | 0 | false | ok | 0.3560 | 33.44 | 22.32 | 23.61 | 0.0 | 109.7 | accepted |
| 32 | 96 | 3 | 0 | 0 | false | ok | 0.6525 | 61.31 | 29.73 | 31.47 | 15.5 | 185.7 | accepted |
| 32 | 96 | 3 | 4 | 0 | false | ok | 0.3990 | 37.48 | 29.73 | 31.47 | 0.0 | 145.5 | accepted |
| 32 | 96 | 3 | 8 | 0 | false | ok | 0.3629 | 34.10 | 29.73 | 31.47 | 0.0 | 144.0 | accepted |
| 32 | 96 | 3 | 12 | 0 | false | ok | 0.3394 | 31.89 | 29.73 | 31.47 | 0.0 | 102.0 | accepted |
| 8 | 96 | 12 | 0 | 1 | false | ok | 0.7559 | 71.02 | 7.49 | 7.90 | 44.0 | 200.2 | accepted |
| 8 | 96 | 12 | 0 | 2 | false | ok | 0.7493 | 70.39 | 7.49 | 7.90 | 44.0 | 201.3 | accepted |
| 16 | 96 | 6 | 0 | 1 | false | ok | 0.7474 | 70.21 | 14.90 | 15.76 | 42.0 | 202.0 | accepted |
| 16 | 96 | 6 | 0 | 2 | false | ok | 0.7463 | 70.11 | 14.90 | 15.76 | 40.0 | 202.4 | accepted |
| 8 | 96 | 12 | 0 | 0 | true | error | 0.0000 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton<br><br>Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"<br> |

## Selected profile

The median-throughput winner is batch 8 with 0 workers: effective batch 96 with accumulation 12, 0.7559 optimizer steps/s, 71.02 samples/s, 7.90 GiB peak reserved VRAM, 87.69 GiB headroom, and 44.0% median SM utilization.

Profiles with OOM, nonfinite output, errors, or less than 10 GiB free headroom were rejected. Remaining profiles lost on median optimizer throughput; the 70–82 GiB and ≥70% SM targets are reported as utilization goals rather than substituted for measured speed.

## Compile decision

`torch.compile` retained: **false**. Reason: compile candidate failed: TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old.

## Previous-profile comparison

The independently measured previous batch-4/worker-0 training stack (FP16, non-fused AdamW) reached 0.7390 optimizer steps/s. The selected profile is faster by 2.29%: **true**.

These are hardware measurements only and were not written to the experiment ledger.
