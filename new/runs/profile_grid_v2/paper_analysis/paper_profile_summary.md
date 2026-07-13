# Profile Grid v2 — Paper Analysis Summary

## 1. Data scope and validation

- Total runs loaded: 1080
- llama: 540 runs
- qwen: 540 runs
- **Validation: PASS** (grid complete, no NaN/Inf, all metadata as expected)

## 2. Central offload metrics — llama

| Offload GB | Runs | TPOT median (ms) | TPOT ratio vs 0 | ITL median (ms) | TTFT median (ms) | E2EL median (ms) |
|---|---|---|---|---|---|---|
| 0 | 90 | 20.707 | 1.00x | 20.659 | 62.423 | 1367.724 |
| 2 | 90 | 252.198 | 12.18x | 252.187 | 578.252 | 16228.542 |
| 4 | 90 | 484.850 | 23.41x | 484.835 | 1074.594 | 31117.069 |
| 8 | 90 | 945.093 | 45.64x | 945.018 | 2051.516 | 60741.884 |
| 12 | 90 | 1409.250 | 68.06x | 1408.969 | 3039.501 | 90281.882 |
| 16 | 90 | 1507.188 | 72.79x | 1507.042 | 3255.447 | 96833.129 |

## 3. Central offload metrics — qwen

| Offload GB | Runs | TPOT median (ms) | TPOT ratio vs 0 | ITL median (ms) | TTFT median (ms) | E2EL median (ms) |
|---|---|---|---|---|---|---|
| 0 | 90 | 19.636 | 1.00x | 19.579 | 65.231 | 1292.449 |
| 2 | 90 | 279.672 | 14.24x | 279.619 | 844.219 | 18412.969 |
| 4 | 90 | 546.011 | 27.81x | 545.516 | 1646.095 | 35947.163 |
| 8 | 90 | 1020.080 | 51.95x | 1019.410 | 3076.797 | 66862.738 |
| 12 | 90 | 1499.234 | 76.35x | 1498.293 | 4519.748 | 98245.416 |
| 16 | 90 | 1490.493 | 75.91x | 1489.473 | 4510.888 | 98249.404 |

## 4. Output-length stability

- Maximum observed relative output-length span across all model/offload cells: 0.344%
- llama: max relative span 0.344%, 0 cell(s) > 1%, 0 cell(s) > 5%
- qwen: max relative span 0.266%, 0 cell(s) > 1%, 0 cell(s) > 5%

TPOT and ITL remain highly stable across output lengths 32/64/128 within a given offload level.

## 5. Concurrency 1-to-2 transition

| Model | Offload GB | TPOT conc1 (ms) | TPOT conc2 (ms) | conc2/conc1 ratio |
|---|---|---|---|---|
| llama | 0 | 19.480 | 20.188 | 1.04x |
| llama | 2 | 107.032 | 249.073 | 2.33x |
| llama | 4 | 195.049 | 481.109 | 2.47x |
| llama | 8 | 364.715 | 937.350 | 2.57x |
| llama | 12 | 539.468 | 1390.720 | 2.58x |
| llama | 16 | 580.124 | 1495.740 | 2.58x |
| qwen | 0 | 19.538 | 19.193 | 0.98x |
| qwen | 2 | 280.282 | 276.987 | 0.99x |
| qwen | 4 | 548.239 | 540.954 | 0.99x |
| qwen | 8 | 1026.810 | 1012.443 | 0.99x |
| qwen | 12 | 1508.496 | 1487.510 | 0.99x |
| qwen | 16 | 1499.799 | 1479.438 | 0.99x |

Largest conc2/conc1 TPOT transition: 2.58x (llama, offload16 GB).
This transition is reported descriptively only; no mechanism (scheduler, batching, memory) is inferred from this script.

## 6. Saturation: offload12 vs offload16

| Model | TPOT@12 | TPOT@16 | ratio 16/12 | % diff | 16 higher than 12 |
|---|---|---|---|---|---|
| llama | 1409.250 | 1507.188 | 1.069 | 6.95% | True |
| qwen | 1499.234 | 1490.493 | 0.994 | -0.58% | False |

## 7. Cross-model consistency (Llama vs Qwen)

- Spearman correlation (TPOT ratio-vs-offload0 curves): 0.9429 (source: scipy)
- Spearman correlation (ITL ratio-vs-offload0 curves): 0.9429 (source: scipy)

## 8. Data-based recommendation: offload0 vs offload12

- Recommended low state: offload0 GB
- Recommended high state: offload12 GB
- Rationale (data-based only, no security claim):
    - Clear separation from baseline at offload12 across models.
    - offload12 already lies in the extreme regime of the profiled grid.
    - offload16 adds only a small additional effect or plateaus relative to offload12.

## 9. Scope

These profiling results establish runtime-regime separation only.
They do not establish a State x Burst availability interaction.

_Bootstrap settings: n=10000, seed=20260711._
