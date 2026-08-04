Prefill-Confirmation compact analysis export v2

Files:
- validation.json: campaign completeness, paths, counts, integrity result
- REPORT.txt: quick local overview
- episodes.csv: one row per regular episode (192 total)
- victim_requests.csv: one compact row per victim request (3840 total)
- paired_effects.csv: one no_burst/prefill_burst pair per block (96 total)
- cell_summaries.csv: paired burst effects by model/offload/concurrency
- interaction_contrasts_detail.csv: repeat-matched state/concurrency/model contrasts
- interaction_summaries.csv: repeat-matched within-model interaction summaries
- exposure_group_episode_summaries.csv: Schema-5 exposure-group metrics per episode
- exposure_group_paired_effects.csv: paired burst effects within exposure groups
- model_comparison_unpaired.csv: independent Llama/Qwen bootstrap comparisons
- stabilization.csv: first-half/second-half drift per server start
- block_summaries.csv: lifecycle/stop status per block
- metadata/: original official summaries/manifests/integrity manifests

Excluded deliberately:
- raw SSE event payloads
- prompt token IDs
- generated token IDs
- server logs
