# Frozen Short-Paper Claim and Robustness Protocol

**Project:** Exposure-dependent availability damage in chunked-prefill LLM serving  
**Frozen on:** 2026-07-18  
**Status:** Frozen after completion and analysis of the main Llama/Qwen campaign, but before any robustness-extension measurements.

## Terminology and reporting discipline

The main analysis plan was internally specified and frozen before the official runs. Unless a separate public, timestamped registration exists, the paper must use **pre-specified** or **frozen before the official runs**, not **preregistered**.

The robustness experiments below test a claim that is already fixed. Their results may support, qualify, or fail to support that claim, but must not redefine it.

---

## 1. Fixed core claim from the main study

### RQ1 — Aggregate primary endpoint

**Question.** Does CPU-weight-offload state robustly modulate the bounded-prefill-burst effect on episode-level median victim TPOT across all 20 victim requests?

**Result.** No robust normalized State × Burst interaction was observed. The estimated interactions were small and all reported 95% intervals included zero.

**Required reporting.** This negative/unclear aggregate result must be reported explicitly and must not be hidden by the stronger decomposition results.

### RQ2 — Pre-specified exposure decomposition

**Question.** Does the same bounded prefill burst produce different damage modes for requests that are already decoding at the trigger and requests that are still waiting for a local concurrency slot?

**Result.**

- `running_at_trigger`: large and highly reproducible TPOT increase;
- `queued_at_trigger`: essentially unchanged later TPOT cadence;
- queued requests nevertheless inherit a large increase in local pre-dispatch waiting time and E2EL measured from task creation.

**Fixed interpretation.** The burst produces exposure-dependent availability damage: an immediate decode stall for already-running requests and a queue-delay cascade for later requests.

### RQ3 — Aggregate-metric failure mode

**Question.** Why does aggregate median TPOT show a stronger apparent burst effect at concurrency 8 than at concurrency 4 despite similar exposure-selective behavior?

**Result.** The difference is an order-statistic/composition effect.

At concurrency 8, the eight damaged active requests move above all twelve later requests. The median of all twenty requests therefore selects ranks 10 and 11 within the twelve-request later distribution rather than its own ranks 6 and 7. The later-wave median remains approximately unchanged, but the aggregate median selects its upper order statistics.

**Fixed implication.** Aggregate median TPOT can conceal or distort the same selective mechanism depending on the proportion of exposed requests.

### RQ4 — Runtime-state modulation

**Result structure.**

- Relative active-wave TPOT slowdown is larger in the no-offload state.
- Absolute queue and completion delay is much larger in the high-offload states.
- Eight and twelve gigabytes of offload behave largely as one relative “offload-on” regime, while their absolute delays remain ordered.

**Fixed wording.** Runtime state changes the relative and absolute composition of damage in opposite directions.

No causal claim about PCIe weight streaming or fixed-cost amortization is established. Such an explanation may only be described as **consistent with the observations** or as a mechanism hypothesis.

---

## 2. Fixed short-paper contribution

The paper contribution is not “prefill interferes with decode,” which is already known.

The contribution is the combination of:

1. trigger-based separation of `running_at_trigger` and `queued_at_trigger`;
2. empirical separation of immediate decode stall and inherited queue-delay cascade;
3. demonstration that aggregate median TPOT can misrepresent this selective mechanism through an order-statistic effect;
4. evidence that CPU-offload regimes alter relative and absolute damage differently;
5. replication across Llama and Qwen in the main study.

### Fixed one-sentence claim

> Bounded prefill load produces exposure-dependent availability damage in chunked-prefill LLM serving: requests already decoding at burst onset suffer an immediate token-latency stall, whereas queued requests preserve their later decode cadence but inherit a pre-dispatch delay cascade; aggregate median TPOT can distort this decomposition, and CPU-offload regimes alter relative and absolute damage in opposite directions.

---

## 3. Threat-model boundary

This is a controlled systems/reliability experiment using a protocol-defined burst trigger. It is not presented as a complete end-to-end adversarial attack.

The measured queue delay is the client-side local semaphore/admission delay before request dispatch. It must not be described as a directly observed internal vLLM server queue.

---

## 4. Robustness experiment A — Alternative burst shape on RTX 3090

### Purpose

Test whether the exposure-dependent decomposition is specific to the original `4 × 2048` burst shape and probe the fixed-cost/amortization hypothesis without claiming causal proof.

### Frozen design

- GPU: RTX 3090
- Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- Offload: 0 GB and 12 GB
- Victim concurrency: 4 and 8
- Trigger: unchanged, after token 16
- Baseline condition: `no_burst`
- Robustness burst: `2 parallel requests × 4096 input tokens`
- Total burst input budget: 8192 tokens, equal to the original `4 × 2048`
- Repeats: 4 paired repetitions
- Total: 32 paired blocks / 64 episodes

The main-study inference remains primary. This `n=4` extension is descriptive/confirmatory robustness evidence, not a new powered inferential study.

### Frozen success criteria

The alternative profile supports the fixed mechanism if, for each model and concurrency after pooling only the four planned repeats:

1. geometric active-wave TPOT change is greater than +10%;
2. geometric later-wave TPOT change remains within ±5%;
3. the active-wave TPOT change exceeds the later-wave change by at least 10 percentage points;
4. median local queue-wait delta for later requests is positive;
5. median E2EL-from-task-creation delta for later requests is positive.

The state-structure criterion is directional, not numerical:

6. relative active-wave slowdown is larger at 0 GB than at 12 GB;
7. absolute later-request queue/E2EL delay is larger at 12 GB than at 0 GB.

Failure of one exact magnitude does not license redefining the claim. Results are reported cell by cell, including any non-replication.

---

## 5. Robustness experiment B — RTX 5060 Ti feasibility and structural replication

### Purpose

Test whether the exposure-selective structure survives on a different GPU architecture and memory regime.

Because the RTX 5060 Ti has 16 GB VRAM and the RTX 3090 has 24 GB, cross-GPU conditions are not described as identical operating regimes. This experiment tests structural replication under a different hardware/offload regime.

### Gate 1 — Unmodified-stack smoke

Attempt the same Qwen model, precision, serving software, request schema, and instrumentation.

Initial smoke:

- Qwen2.5-7B-Instruct
- 0 GB offload
- concurrency 8
- original `4 × 2048` burst profile
- one paired smoke block

Pass conditions:

- model starts without quantization or precision changes;
- no OOM or invalid episode;
- all Schema-5 timing/exposure fields are produced;
- all victim requests complete;
- no hidden fallback changes the execution path.

### Gate 2 — Memory-regime fallback

If 0 GB fails solely because the unchanged model cannot fit in 16 GB VRAM, a second smoke may use the same model and stack with an explicit offload state already belonging to the experimental factor.

Permitted fallback contrast:

- 8 GB versus 12 GB offload.

This is reported as **structural replication across different offload regimes**, not as an identical 3090-versus-5060 comparison and not as a replication of the no-offload/high-offload inversion.

No quantization, different model, different numeric precision, or materially different serving path is permitted solely to make the GPU run.

If the same model/stack does not run under the permitted regime, the cross-GPU extension is abandoned without changing the main claim.

### Frozen campaign if a gate passes

- GPU: RTX 5060 Ti
- Model: Qwen2.5-7B-Instruct
- States:
  - preferred: 0 GB and 12 GB;
  - permitted memory fallback: 8 GB and 12 GB.
- Concurrency: 4 and 8
- Burst: original `4 × 2048`
- Repeats: 4 paired repetitions
- Total: 16 paired blocks / 32 episodes

### Frozen structural success criteria

Cross-GPU robustness concerns structure, not matching effect sizes.

For each tested state and concurrency:

1. geometric active-wave TPOT change is greater than +10%;
2. geometric later-wave TPOT change remains within ±5%;
3. active-wave change exceeds later-wave change by at least 10 percentage points;
4. later-request local queue-wait delta is positive;
5. later-request E2EL-from-task-creation delta is positive.

If the preferred 0/12 contrast runs, the following directional checks are also reported:

6. relative active-wave slowdown is larger at 0 GB than at 12 GB;
7. absolute later-request queue/E2EL delay is larger at 12 GB than at 0 GB.

If only 8/12 is feasible, criteria 6–7 are not treated as required replication of the main state inversion. The experiment then supports or fails only the exposure-selective decomposition and queue-cascade structure.

No numerical similarity to the RTX 3090 effect sizes is required.

---

## 6. Analysis and reporting rules for both extensions

- Report all planned cells, including failures and direction reversals.
- Do not add repetitions based on observed effect size or significance.
- Do not select one concurrency, model, or state after viewing results.
- Do not use significance testing as the pass/fail rule for `n=4`.
- Preserve paired analysis within each cell.
- Keep main-study and robustness results in separate tables/figures.
- Report the very narrow main-study offload confidence intervals with raw ratio range, coefficient of variation, and an independent bootstrap check.
- Present the PCIe/fixed-cost explanation only as a hypothesis consistent with the pattern.
- Preserve the negative aggregate primary result in the abstract/results narrative.

---

## 7. Execution order

1. Save and version this frozen protocol.
2. Generate and audit the alternative-burst schedule.
3. Run a hardest-cell smoke on RTX 3090.
4. Run the 64-episode alternative-profile campaign.
5. Run the RTX 5060 Ti feasibility gate.
6. If a gate passes, run the 32-episode cross-GPU campaign.
7. Export and analyze without modifying the criteria above.
8. Write the short paper around the already fixed main-study claim; robustness results only strengthen or qualify it.
