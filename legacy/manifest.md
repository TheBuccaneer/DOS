# Legacy Artifact Manifest

Stand: 2026-06-18

Source repository: https://github.com/TheBuccaneer/porto  
Source commit: TO BE FILLED AT NEXT COMMIT

## Rules

- Legacy artifacts are used only for state selection, estimator calibration, probe-budget checks, code reuse, and hypothesis generation.
- They are not treated as evidence from the new availability pilot.
- Files are copied from the source repository and must not be edited in place.
- New pilot results must be stored outside `legacy/`.

## Estimator inputs

| Local file | Original source | Purpose | Evidence status |
|---|---|---|---|
| `estimator_inputs/llama_runs_summary_conc2plus.csv` | `runs/2026-04-26_lama/base_runs/extracted_rerun_main/runs_summary_rerun_conc2plus.csv` | Llama state-estimator calibration and freeze | Calibration only |
| `estimator_inputs/llama_requests_summary.csv` | `runs/2026-04-26_lama/base_runs/extracted_rerun/requests_summary_rerun.csv` | Llama probe-budget and request-level checks | Calibration only |
| `estimator_inputs/qwen_runs_summary_conc2plus.csv` | `runs/2026-06-01_Qwen/extracted_rerun_main/runs_summary_rerun_conc2plus.csv` | Later Qwen replication and robustness checks | Not used in initial pilot |
| `estimator_inputs/qwen_requests_summary.csv` | `runs/2026-06-01_Qwen/extracted_rerun/requests_summary_rerun.csv` | Later Qwen probe-budget checks | Not used in initial pilot |

## Reusable source scripts

| Local file | Original source | Purpose | Evidence status |
|---|---|---|---|
| `source_scripts/extract_vllm_rerun.py` | `paper1_workspace/step1_base_signal/extract_vllm_rerun.py` | Timing and request-metric extraction | Reusable implementation |
| `source_scripts/infer_offload_binary.py` | `paper1_workspace/step1_base_signal/infer_offload_binary.py` | Starting point for frozen state estimator | Must be adapted and frozen |
| `source_scripts/probe_budget_analysis.py` | `runs/probe_budget_analysis.py` | Probe-budget analysis reference | Legacy analysis only |

## Old availability pilot

### Step 2: sustained interference

The files under `old_attack_pilot/step2_fixed_burst/` describe the earlier 300-second
victim-plus-attacker experiment. Despite the historical directory name, this was
continuous interference rather than the new fixed four-request burst.

| Local artifact | Original source | Purpose | Evidence status |
|---|---|---|---|
| `step2_fixed_burst/*.py` | `paper1_workspace/step2_fixed_burst/*.py` | Reuse victim, attacker, and logging code | Legacy implementation |
| `step2_fixed_burst/offload_00/*.csv` | `paper1_workspace/step2_fixed_burst/00/*.csv` | Low-state hypothesis generation | Preliminary evidence only |
| `step2_fixed_burst/offload_04/*.csv` | `paper1_workspace/step2_fixed_burst/04/*.csv` | High-state hypothesis generation | Preliminary evidence only |

### Step 3: policy gating

| Local artifact | Original source | Purpose | Evidence status |
|---|---|---|---|
| `step3_policy_gating/*.py` | `paper1_workspace/step3_policy_gating/*.py` | Reference implementation for probe and policy logic | Legacy implementation |
| `step3_policy_gating/combined_table.csv` | `paper1_workspace/step3_policy_gating/combined_table.csv` | Preliminary comparison of random, threshold, and state-aware gating | Not new pilot evidence |

## Frozen initial pilot scope

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Low state: `cpu_offload_gb = 0`
- High state: `cpu_offload_gb = 12`
- Victim concurrency: `4` and `8`
- Victim profile: input 256, output 64, temperature 0
- Candidate burst: 4 parallel requests, input 256, output 256, temperature 0
- Policies: No Attack, Always Attack, Budget-Matched Random, Naive Threshold, State-Aware
- Repetitions: 5 per cell

The state estimator, features, threshold, and probe budget must be frozen before
the first new availability episode.
