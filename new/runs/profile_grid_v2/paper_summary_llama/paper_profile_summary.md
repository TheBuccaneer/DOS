# Llama profile_grid_v2 paper summary

## Dataset

- Input file: `C:\projects\DOS\new\runs\profile_grid_v2\llama_audit_filename.csv`
- Rows/files included: 540
- Offloads: [0, 2, 4, 8, 12, 16]
- Concurrency levels: [1, 2, 4, 8, 12, 16]
- Input lengths: [256]
- Output lengths: [32, 64, 128]

Interpretation: fixed-input, variable-output decode-focused profiling matrix.

## Main table: by offload

| offload_gb | n_files | median TPOT ms | p95 file TPOT ms | TPOT ratio vs offload0 | median ITL ms | ITL ratio vs offload0 | median TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90 | 20.688 | 22.478 | 1.00 | 20.469 | 1.00 | 60.747 |
| 2 | 90 | 251.999 | 256.914 | 12.18 | 250.328 | 12.23 | 579.818 |
| 4 | 90 | 484.377 | 492.383 | 23.41 | 480.944 | 23.50 | 1084.382 |
| 8 | 90 | 944.128 | 957.455 | 45.64 | 937.794 | 45.81 | 2077.562 |
| 12 | 90 | 1407.697 | 1427.185 | 68.04 | 1398.364 | 68.32 | 3079.892 |
| 16 | 90 | 1505.703 | 1525.014 | 72.78 | 1496.181 | 73.09 | 3299.312 |

## TPOT/ITL by offload and output length

| offload_gb | output_len | n_files | median TPOT ms | p95 file TPOT ms | median ITL ms | median TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 30 | 20.687 | 22.356 | 20.341 | 59.892 |
| 0 | 64 | 30 | 20.675 | 22.356 | 20.503 | 63.075 |
| 0 | 128 | 30 | 20.723 | 22.493 | 20.629 | 63.159 |
| 2 | 32 | 30 | 251.999 | 256.694 | 248.057 | 577.623 |
| 2 | 64 | 30 | 252.088 | 256.823 | 250.093 | 580.450 |
| 2 | 128 | 30 | 252.165 | 257.009 | 251.154 | 579.767 |
| 4 | 32 | 30 | 484.294 | 492.139 | 476.702 | 1084.282 |
| 4 | 64 | 30 | 484.538 | 492.195 | 480.732 | 1101.516 |
| 4 | 128 | 30 | 484.665 | 492.416 | 482.758 | 1084.165 |
| 8 | 32 | 30 | 944.128 | 956.714 | 929.344 | 2134.943 |
| 8 | 64 | 30 | 944.542 | 957.245 | 937.095 | 2076.633 |
| 8 | 128 | 30 | 944.609 | 957.569 | 940.935 | 2074.681 |
| 12 | 32 | 30 | 1407.697 | 1426.220 | 1385.594 | 3081.678 |
| 12 | 64 | 30 | 1408.183 | 1426.879 | 1397.111 | 3079.393 |
| 12 | 128 | 30 | 1408.235 | 1427.313 | 1402.703 | 3080.613 |
| 16 | 32 | 30 | 1505.320 | 1524.697 | 1481.746 | 3298.542 |
| 16 | 64 | 30 | 1506.211 | 1525.624 | 1494.363 | 3300.138 |
| 16 | 128 | 30 | 1506.564 | 1524.964 | 1500.652 | 3301.178 |

## TPOT/ITL by offload and concurrency

| offload_gb | concurrency | n_files | median TPOT ms | p95 file TPOT ms | median ITL ms | median TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 15 | 19.482 | 19.501 | 19.331 | 28.397 |
| 0 | 2 | 15 | 20.189 | 20.211 | 20.031 | 55.866 |
| 0 | 4 | 15 | 20.499 | 20.517 | 20.321 | 60.426 |
| 0 | 8 | 15 | 20.884 | 20.940 | 20.690 | 65.592 |
| 0 | 12 | 15 | 22.030 | 22.122 | 21.804 | 76.488 |
| 0 | 16 | 15 | 22.356 | 22.496 | 22.162 | 85.709 |
| 2 | 1 | 15 | 107.036 | 107.190 | 106.193 | 258.868 |
| 2 | 2 | 15 | 248.909 | 251.360 | 248.011 | 465.283 |
| 2 | 4 | 15 | 251.297 | 251.436 | 249.296 | 569.746 |
| 2 | 8 | 15 | 252.779 | 252.960 | 250.763 | 615.457 |
| 2 | 12 | 15 | 256.309 | 256.482 | 254.137 | 580.513 |
| 2 | 16 | 15 | 256.822 | 257.013 | 254.765 | 615.564 |
| 4 | 1 | 15 | 195.050 | 195.066 | 193.518 | 490.251 |
| 4 | 2 | 15 | 480.368 | 483.472 | 477.456 | 849.166 |
| 4 | 4 | 15 | 482.995 | 483.305 | 479.454 | 1153.930 |
| 4 | 8 | 15 | 485.694 | 486.017 | 481.834 | 1168.515 |
| 4 | 12 | 15 | 491.479 | 491.843 | 487.409 | 1084.037 |
| 4 | 16 | 15 | 492.195 | 492.458 | 488.267 | 1145.930 |
| 8 | 1 | 15 | 364.719 | 364.728 | 361.856 | 948.698 |
| 8 | 2 | 15 | 936.477 | 942.018 | 929.710 | 1662.017 |
| 8 | 4 | 15 | 941.816 | 942.334 | 933.896 | 2233.393 |
| 8 | 8 | 15 | 946.896 | 947.738 | 939.403 | 2258.157 |
| 8 | 12 | 15 | 956.408 | 956.907 | 948.530 | 2076.335 |
| 8 | 16 | 15 | 957.168 | 957.624 | 949.582 | 2179.647 |
| 12 | 1 | 15 | 539.474 | 539.521 | 535.236 | 1411.321 |
| 12 | 2 | 15 | 1391.051 | 1404.361 | 1385.955 | 2497.819 |
| 12 | 4 | 15 | 1403.369 | 1404.877 | 1392.375 | 3329.875 |
| 12 | 8 | 15 | 1411.782 | 1412.321 | 1400.523 | 3361.981 |
| 12 | 12 | 15 | 1425.762 | 1426.389 | 1414.157 | 3079.411 |
| 12 | 16 | 15 | 1426.847 | 1427.390 | 1415.621 | 3241.060 |
| 16 | 1 | 15 | 580.126 | 580.134 | 575.570 | 1510.115 |
| 16 | 2 | 15 | 1494.870 | 1503.066 | 1481.314 | 2728.019 |
| 16 | 4 | 15 | 1501.920 | 1502.876 | 1490.203 | 3564.137 |
| 16 | 8 | 15 | 1509.602 | 1510.731 | 1497.491 | 3599.125 |
| 16 | 12 | 15 | 1524.352 | 1525.014 | 1511.965 | 3299.203 |
| 16 | 16 | 15 | 1524.522 | 1525.838 | 1513.504 | 3480.662 |

## Output-length stability

Within each offload level, median TPOT is highly stable across output lengths. This supports interpreting the measurement as a decode-regime effect rather than an output-length artifact.

| offload_gb | min output-median TPOT ms | max output-median TPOT ms | spread ms | spread % |
|---:|---:|---:|---:|---:|
| 0 | 20.675 | 20.723 | 0.048 | 0.23% |
| 2 | 251.999 | 252.165 | 0.165 | 0.07% |
| 4 | 484.294 | 484.665 | 0.371 | 0.08% |
| 8 | 944.128 | 944.609 | 0.481 | 0.05% |
| 12 | 1407.697 | 1408.235 | 0.538 | 0.04% |
| 16 | 1505.320 | 1506.564 | 1.244 | 0.08% |

## Paper-ready interpretation

The profiling run shows a clear monotonic increase in decode-time metrics as CPU-offload increases. TPOT and ITL form the strongest runtime-regime signals; TTFT also increases but is less suitable as the primary signal.

The fixed input length controls prefill cost, while varying output length tests the decode phase where CPU-offload effects dominate.

## Phase-A recommendation

- Recommended Phase-A state pair: low=offload0, high=offload12.
- Rationale: offload12 is already a strong high-state, while offload16 adds comparatively little extra TPOT over offload12.
- Observed median TPOT: offload12=1407.697 ms, offload16=1505.703 ms.
- offload16 is only 1.07× offload12 (6.96% higher TPOT), so offload12 is a cleaner high-state for Phase A.

Do not claim availability degradation from this profiling dataset alone. It supports state selection and regime characterization for the later Victim/Burst Phase-A experiment.