# Evidence ledger — 17 October 2026

## What was newly performed

Only offline reanalysis, local source inspection, public literature retrieval, manuscript production and read-only GPU inventory. **No GPU inference, server start, attack replay, new serving request, submission or publication.** Existing data were not edited.

Run from repository root:

```bash
python3 new/research/icissp2027_rdna_20261017/tools/reanalyze.py --repo /home/rock/projects/DOS
python3 -m unittest discover -s new/research/icissp2027_rdna_20261017/tools -p 'test_*.py' -v
```

`evidence/reanalysis.json` records source-relative file paths and SHA-256 hashes. `pilot_summary.json` is the initial extraction of stored runner assessments; it is **not** the independent reconstruction. The new script imports no old runner and uses top-level SSE wire records, raw JSON usage and receive timestamps. Original dataset collection was external to this autonomous turn; it was not observed live by this auditor.

## Independent reconstruction scope

14 episodes / 25 requests / 1,152 output token IDs / 1,202 wire events. All request token sequences agree with the stored request-result copy, expected output length and usage counts. All streams include `[DONE]`. Prompts agree with the saved plan. All D token-bearing events have one token, so receipt times for token 8 and 28 are individually available. Trigger timestamps equal receipt of D token 8. D8–D28 endpoints agree with the stored values. The check produced **zero integrity discrepancies** and **11/11 artifact/counter tests passed (14/14 including the three separate analytical-observer tests)**.

This validates consistency of retained records, not authenticity against an independent external clock, durable `fsync` behavior during collection, completeness of all machine traffic, or an absence of unlogged requests. Absence of `cached_tokens` in a successful usage object is interpreted as zero according to this artifact schema and checked against the saved runner assessments; it is not a protocol-independent assumption about arbitrary APIs. Aggregate metrics do not authenticate tenant identity.

The preemption deltas below are recomputed from the first and last retained raw Prometheus samples. Per-episode server log context-token sums equal initial prompt tokens minus cached tokens even in the two preemption episodes. **There is no measured extra context-token work in those sums.** Do not infer a recomputation penalty merely from the counter increment.

## Exploratory observed endpoints (ms)

Each cell has one episode, not an independent-repetition average. D input/output = 192/32; B = 128/64. Effective budget Q, configured blocks C; pool reserves one block.

| Condition | C64 Q512 | C64 Q64 | C24 Q64 |
|---|---:|---:|---:|
| Victim only | 419.262 | 402.789 | 406.431 |
| Shared common prefix | 423.739 | 422.004 | 425.382 |
| Isolated common prefix | 454.940 | 428.328 | 431.388* |
| Shared disjoint prefix | 455.015 | 430.730 | 432.515* |
| Isolated disjoint prefix | 453.758 | 430.725 | missing |

`*` Exactly one whole-episode preemption increment each. The two episodes have complete SSE measurements; the original validator rejects mechanism attribution in them. It does not establish which request was preempted or whether that event caused delay inside D8–D28. Preemption and eviction of unused cached blocks are different events.

B cache hits: 112 in each common-prefix shared episode, 0 in every other concurrent episode. D cache hits: 0 in every episode. B initial uncached work is therefore 16 versus 128 tokens. At C64/Q64, the stored iteration reconstruction gives B chunks `[16]` shared and `[63,63,2]` cold; 63 prefill tokens plus one D decode token exhaust a 64-token iteration. At Q512, `[16]` versus `[128]` fits without budget binding. These are scheduling observations, not proportional GPU-time estimates. Uncached work alone does not equal actual compute cost.

All 12 episodes without preemption pass the old validator; both preemption episodes fail its attribution path. A naive “valid episodes only” table would eliminate both pressure observations from these files. This is a concrete risk of the selection rule; the causal size or direction of resulting bias is not estimated with one episode per cell. A missing C24 disjoint/isolated cell prevents a complete capacity interaction contrast. No confidence interval, p-value, equivalence conclusion, effect amplification or new security vulnerability is claimed.

## Sequential cache controls — stored results, narrower verification

Two six-request runs retain the order A1→A2→B1→B2→B3→A3. Their final stored assessments report A1=0, A2=112 and B1–B3=0 hits in both runs. A3 has 112 at 64 configured blocks and 0 at 24. The new script hashes these files and extracts their final assessments; **it does not independently reconstruct their metrics or server binding**. This is consistent with capacity-dependent loss of reusable state, not proof of overlapping decode interference. AAB smoke data likewise support lookup separation only. The paper does not use these controls as a new mechanism result.

## Source mechanisms independently checked

`evidence/source_manifest.json`: local and upstream SHA-256 match for all seven files. Upstream base: `https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/`.

| File / function / lines | Checked mechanism | Boundary |
|---|---|---|
| `v1/core/kv_cache_utils.py`, `generate_block_hash_extra_keys`, 494–529; `hash_block_tokens`, 532–558 | Salt enters first-block extra keys; parent hashes propagate the separation. | No proof of all timing confidentiality, tenant authentication or quotas. |
| `v1/core/kv_cache_manager.py`, `get_computed_blocks`, 164–204 | Full cached blocks, prompt-length-minus-one lookup bound. 128-token prompt with 16-token blocks can reuse 112. | This arithmetic alone predicts neither latency nor eviction. |
| `v1/core/block_pool.py`, constructor 150–176; `get_new_blocks`, 320–350; `_maybe_evict_cached_block`, 352–390 | Common free queue includes reclaimable cached blocks; allocation can erase cached hash metadata. | In-use block pressure and reclamation of unused cached state differ. |
| `v1/core/block_pool.py`, `get_usage`, 487–498 | Denominator excludes null block. C24 has 23 usable blocks; C64 has 63. | Sampled usage is not every instantaneous allocation. |
| `v1/core/kv_cache_utils.py`, `may_override_num_blocks`, 820–833 | Configured override actually changes pool count. | Startup text alone still needs binding to the server measured. |
| `v1/core/sched/scheduler.py`, `schedule`, 317–486 and 650–669 | Uses effective `max_num_scheduled_tokens`, clips work to remaining budget, allocates blocks, can preempt on allocation failure. | A small configured budget is not evidence that B actually hit it. |
| Same, `_preempt_request`, 914–934 | Frees state, resets computed-token count, increments preemption, prepends waiting request. | Counter does not prove total recomputation; subsequent cache reuse may reduce work. |

Excerpts with line numbers are in `source_excerpts.txt`. These findings establish implementation behavior at one version. They do not establish adversarial intent, SLO violation, novel mechanism, or a universal resource-isolation guarantee.

## Historical evidence deliberately not promoted

The prior 100-episode audit, NordSec claim freeze and Qwen reanalysis are inventory resources only in this paper. Reported offload interactions, medians, server/client waiting and missing post-trigger metrics are not counted as confirmation of the new hypothesis. The original manuscript and original reviews remain unavailable, so claim overlap and prior-review reasoning are unresolved.

## Reproducibility and limits

New tests mutate evidence to check missing `[DONE]`, missing token, malformed JSON, missing metric, changed endpoint and warm-victim detection, plus preservation of preemption episodes with abstention. They do not validate live GPU behavior or the whole pre-existing runner suite. An initial test path error was corrected; the final suite has 14 passing tests: 11 artifact/counter tests plus 3 analytical-observer tests. No previous 54/83/85-test report is presented as newly reproduced.

Potential confounders: nonrandom fixed run order, unmeasured startup/warm-up/clock state, shared server lifetime across conditions, timing overhead from full iteration logs and 50-ms scrapes, asynchronous output pipelines, window placement relative to pressure, missing cell, single runs and short synthetic prompts. No new experiment was used to hide these gaps.

## Additional direction: counter-semantics calibration

Following the user's explicit encouragement to pursue a new direction, the existing preemption traces were also compared against the logging implementation, rather than assuming that a generation-token label means a new output token. In both episodes, a single iteration reports one generation request with 17 generation tokens (iterations 149 and 233). Generation-labeled scheduled tokens sum to 110, while the complete new output streams contain 96 tokens, of which 94 follow the two first tokens. The residual is 16 per episode. All 12 non-preemption episodes have zero residual under this accounting.

`v1/utils.py:compute_iteration_details` (432–476) classifies scheduler token counts by new-request/context-phase status; `v1/core/sched/output.py:is_context_phase` (161–163) uses saved output-token count. The metric is scheduled work by classification, not a new-token emission counter. Additional work may be generation-labeled even when the context-token sum has no excess. This is consistent with resumption/recovery but request and window attribution remain unresolved. No 16-token GPU recomputation cost is asserted.

`tools/counter_semantics.py` extracts two pure AST definitions from the source and runs four synthetic cases without importing vLLM, CUDA, or any server/client. `evidence/counter_semantics.json` distinguishes these constructed cases from the retained live measurements. A 17-token positive-output case returns generation=17/context=0; a new or zero-output case returns context=17. Source hashes for utils, scheduler output and engine logging also match upstream v0.17.1. This is a completed offline semantic experiment and an additional concrete example in the position draft, not a newly discovered vulnerability.

## Separate analytical exercise and latest literature

The H7 calculation (`evidence/conditional_observer.json`) uses invented uniform/Gaussian parameters to show that marginal distribution overlap need not bound an observer with correlated reference information. It is an exact constructed example, not a simulation of measured server behavior and not evidence of a real defense bypass. Three tests cover the analytical witness and weak/independent reference controls. The other eleven tests address retained artifacts or source-counter semantics. Current total: **14 passing tests**, reproduced after making source calibration portable.

The seven unchanged source files are retained under `reference/vllm_v0_17_1` with their Apache-2.0 license. The counter script now reads this snapshot and checks the same source semantics without the original local vLLM installation. Reanalysis still requires the retained JSONL data; the review ZIP includes those dependencies.

The supplied HCache PDF and subsequently retrieved FairInference/Cascade preprints are audited in `HCACHE_AUDIT.md`. They are literature, not measurements in our evidence table. No new restoration or serving experiment was run. The HCache PDF's title and DOI do not match FairCache; a renewed search found no accessible second supplied PDF. FairCache remains a full-text priority gap.
