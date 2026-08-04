# Server-WAITING Campaign Freeze

Freeze created: 2026-08-03T10:37:46+00:00

## Purpose

Immutable pre-campaign freeze for the Server-WAITING confirmation study.

No full campaign had been started when this freeze was created.

## Validated diagnostic

- Block: qwen_off12_k8_rep01
- Model: Qwen/Qwen2.5-7B-Instruct
- CPU offload: 12 GiB
- max_num_seqs: 8
- Conditions: no_burst, then prefill_burst
- Diagnostic directory: diagnostic_pair_20260803T094340Z
- Stored classification: B_NO_OUTPUT_LEVEL_OVERLAP_WITH_FIRST_COHORT
- Raw-auditor classification: B_RAW_OUTPUT_BETWEEN_ACTIVE_AND_ALL
- overall_audit_status: PASS
- scientifically_evaluable: true
- diagnostic_tree_read_only_verified: true
- errors: []
- technical_errors: []

## Scientific boundary

The validated result concerns client-visible output overlap only.

It does not identify internal scheduler admission time or internal
prefill-start time.

The paired TPOT, ITL, and completion differences come from one
diagnostic pair and must not be interpreted as a statistical null effect.

## Frozen content

- exact runner and auditor sources
- protected base runner
- generated schedule and schedule audit
- complete diagnostic evidence tree
- complete independent audit tree
- wrapper log
- Git state and source identities
- Python/package environment
- GPU, driver, OS, and idle-state information
- exact model snapshot identity and file hashes
- original Closure ZIP
- recursive bundle SHA-256 manifest

## Secret handling

The VLLM API key value is deliberately not stored.
Only VLLM_API_KEY_SET=true is recorded.

## Campaign rule

Do not edit the frozen files.

Do not place additional files inside individual diagnostic or episode
result directories. Those trees are protected by strict integrity
manifests.
