# CLAIM FREEZE — Exposure-Dependent Availability Damage in Chunked-Prefill LLM Serving

**Version:** 1.0  
**Freeze date:** 20 July 2026  
**Project:** vLLM Short Paper  
**Status:** Authoritative for manuscript drafting  

This document freezes the scientific identity, analysis hierarchy, permitted claims, prohibited claims, and interpretation rules for the paper. Any later change to a frozen claim requires a dated amendment with the new evidence and the reason for the change.

---

## 1. Paper identity

### 1.1 Authoritative framing

> **This is a systems and reliability characterization with availability implications.**

The paper characterizes how a bounded prefill event produces different latency-damage paths depending on whether a victim request is already decoding at burst onset or is admitted later, and how CPU-weight-offload regimes change the relative and absolute composition of that damage.

### 1.2 The paper is not

The paper must not be presented as:

- a new end-to-end denial-of-service attack;
- a complete security attack chain;
- the discovery that prefill can interfere with decode;
- proof of a specific undocumented vLLM scheduler mechanism;
- a study of internal server-queue latency in the main experiment;
- a causal demonstration of PCIe transfer or weight-streaming amortization.

### 1.3 Working title direction

Preferred:

> **Exposure-Dependent Availability Damage in Chunked-Prefill LLM Serving**

Acceptable narrower alternative:

> **When Aggregate TPOT Obscures Exposure-Dependent Damage in LLM Serving**

Avoid a DoS-centered title.

---

## 2. Frozen study facts

### 2.1 Main campaign

The completed official campaign contains:

```text
Models:                 Llama-3.1-8B-Instruct; Qwen2.5-7B-Instruct
CPU weight offload:     0, 8, 12 GB
Client concurrency:     4, 8
Conditions:             no_burst; prefill_burst
Victims per episode:    20
Victim profile:         256 input / 64 output tokens
Trigger:                token 16 of the active first wave
Burst:                  4 parallel requests, each 2048 input / 16 output tokens
Repeats:                8 paired repeats per cell
Total:                  192 episodes; 96 paired blocks; 3,840 victim requests
```

### 2.2 Main-setup admission semantics

The main campaign used a client-side semaphore. Consequently:

- the active first wave was dispatched and decoding when the burst was triggered;
- the later requests waited locally before HTTP dispatch;
- in the audited episodes, **0 of 16 later requests overlapped the burst window**;
- the later-group result therefore characterizes post-burst behavior after later admission/dispatch, not concurrent later traffic during the burst.

This distinction is mandatory throughout the paper.

### 2.3 Server-log audit

Across the 96 official server logs, the audit found:

- no logged preemption event;
- no logged KV-capacity failure;
- no out-of-memory condition;
- no traceback;
- no fatal engine failure.

This does not show that the KV cache was far from capacity, only that these conditions were not logged.

---

## 3. Frozen analysis hierarchy

### 3.1 Original aggregate primary endpoint

The original pre-specified aggregate primary endpoint was the paired log change in episode-level median victim TPOT across all 20 victim requests.

Frozen result:

> **The pre-specified aggregate primary endpoint did not show a robust normalized State × Burst interaction.**

The estimated interaction was small or uncertain across both models and both concurrency settings, and the relevant confidence intervals included zero.

This result must be reported transparently. It must not be redefined, hidden, or replaced retroactively.

### 3.2 Exposure decomposition

Before the official confirmation runs, the runner and analysis pipeline already contained a frozen decomposition based on observed exposure at trigger time, request streaming, and request timing.

Frozen status:

- it was **pre-specified and frozen before the official runs**;
- it was not the original aggregate primary endpoint;
- it explains why the aggregate endpoint can be weak or misleading for selective damage;
- it must not be described as preregistered unless an external public registration is produced.

### 3.3 Secondary and explanatory analyses

The following are legitimate pre-specified explanatory analyses:

- TPOT/ITL of requests decoding at burst onset;
- TPOT of requests admitted later;
- task-creation-to-dispatch and task-creation-to-completion delay;
- order-statistic reconstruction of the aggregate median;
- relative versus absolute effects across offload regimes;
- audit of logged preemption and fatal failures.

They do not convert the negative/unclear aggregate primary endpoint into a positive primary endpoint.

---

## 4. Frozen core claims

### C1 — Aggregate primary result

**Permitted manuscript wording:**

> The pre-specified aggregate primary endpoint, the paired log change in episode-level median victim TPOT, did not show a robust normalized State × Burst interaction.

**Meaning:** The paper does not claim a general state-dependent change in the median TPOT of all victim requests.

---

### C2 — Immediate active-wave damage

**Permitted manuscript wording:**

> A pre-specified exposure decomposition showed that requests already decoding at burst onset suffered a large and highly reproducible immediate token-latency stall.

The effect was replicated across:

- Llama and Qwen;
- concurrency 4 and 8;
- 0, 8, and 12 GB offload;
- all eight official paired repeats.

Numerically, the active-wave TPOT increase was approximately:

```text
0 GB:      +123% to +129%
8/12 GB:   +46% to +51%
```

These ranges describe the tested configurations and must not be generalized to arbitrary models, GPUs, or schedulers.

---

### C3 — Later-admitted requests preserve post-admission decode cadence

**Permitted manuscript wording:**

> Requests that were not executing at burst onset and were admitted later retained approximately the same decode cadence after admission and dispatch.

Required qualification:

> In the audited main-study episodes, these later requests did not overlap the burst window; the result therefore describes post-burst recovery after later admission, not the effect of the burst on concurrently executing later traffic.

The typical later-wave TPOT change was approximately within −1% to +1%; at concurrency 8, cell medians were roughly within ±0.3%.

**Forbidden inference:**

> Concurrent later traffic was unaffected by the burst.

The experiment did not establish that claim.

---

### C4 — Closed-loop pre-dispatch backpressure

**Permitted manuscript wording:**

> Under the closed-loop client admission setup, later requests inherited a substantial client-side pre-dispatch backpressure delay.

Approximate additional local waiting observed in the tested configurations:

```text
0 GB:      +1.5 to +1.7 s
8 GB:      +29 to +31 s
12 GB:     +44 to +45 s
```

Safe labels:

- `client-side pre-dispatch backpressure`;
- `closed-loop admission-delay cascade`.

Forbidden labels for the main study:

- `vLLM queue time`;
- `internal server waiting time`;
- `internal server queue delay`;
- `scheduler admission latency`.

---

### C5 — End-to-end delay from task creation

**Permitted manuscript wording:**

> Measuring from task creation rather than HTTP dispatch revealed substantial completion-delay amplification that dispatch-based E2EL omitted.

Approximate relative increases:

```text
0 GB:      +35% to +48%
8/12 GB:   +14% to +21%
```

Approximate absolute increases:

```text
0 GB:      +1.5 to +1.7 s
8 GB:      +29 to +30 s
12 GB:     +43 to +45 s
```

These numbers combine the consequences of the tested closed-loop admission setup and serving time. They are not direct measurements of an internal server queue.

---

### C6 — Runtime-state inversion

**Permitted manuscript wording:**

> CPU-weight-offload state altered relative and absolute damage in opposite directions: relative active-wave slowdown was larger without offload, whereas absolute completion and admission-delay costs were much larger in the high-offload regimes.

Equivalent concise wording:

> Runtime state changes the relative and absolute composition of damage in opposite directions.

This is a descriptive result. It does not prove why the inversion occurs.

---

### C7 — Exposure-dependent order-statistic shift

**Permitted manuscript wording:**

> The aggregate median changed through a specific exposure-dependent order-statistic shift, even though the later-wave median TPOT remained approximately unchanged.

At concurrency 8:

- 8 active requests moved above all 12 later requests in all 48 burst episodes;
- the aggregate median of 20 requests was therefore determined by ranks 10 and 11 of the later-request distribution rather than by its median ranks 6 and 7;
- the reconstructed rank-10/11 uplift closely matched the observed aggregate-median change.

Preferred explanatory wording:

> Selective active-wave damage changed which order statistics of the approximately unaffected later-wave distribution determined the aggregate median.

Forbidden broad novelty claim:

> Median metrics are blind to heterogeneous damage.

General metric blind spots are known; the contribution is the concrete, measured order-statistic reconstruction in this experimental setting.

---

### C8 — Distinction from preemption/KV-exhaustion attacks

**Permitted manuscript wording:**

> Across all 96 official server logs, we found no logged preemption event, KV-capacity failure, out-of-memory condition, traceback, or fatal engine failure.

Permitted comparison:

> Unlike preemption- or KV-exhaustion-centered mechanisms, the bounded prefill event in our official runs produced the measured exposure-dependent degradation without any logged preemption or KV-capacity failure.

Forbidden inference:

> The KV cache was never close to exhaustion.

---

## 5. Frozen integrated paper claim

### 5.1 Full safe version

> Bounded prefill load produces exposure-dependent availability damage in chunked-prefill LLM serving. Requests already decoding at burst onset suffer a large immediate token-latency stall. Requests admitted later preserve approximately the same decode cadence after admission, but under the main study's closed-loop admission setup they inherit a substantial client-side pre-dispatch delay. The exposed-request fraction induces a concrete order-statistic shift in aggregate median TPOT, and CPU-weight-offload regimes alter relative and absolute damage in opposite directions.

### 5.2 Required footnote or adjacent qualification

> The later requests in the audited main-study episodes did not overlap the burst window; their decode-cadence result characterizes post-burst recovery after later admission rather than concurrent exposure.

### 5.3 Short abstract-compatible version

> A bounded prefill event caused a large immediate token-latency stall for requests already decoding at trigger time, while later-admitted requests recovered their decode cadence but inherited substantial closed-loop admission delay. This selective damage generated a concrete order-statistic distortion in aggregate median TPOT, and CPU-weight-offload state reversed the relative-versus-absolute damage profile.

---

## 6. Contribution claims

The paper may claim the following combined contributions:

1. **Trigger-aligned exposure decomposition.** A controlled protocol separates requests already decoding at burst onset from requests admitted later.
2. **Damage-path decomposition.** The study distinguishes immediate token-latency stalls from subsequent admission/completion delay.
3. **Order-statistic reconstruction.** The study explains the aggregate-median movement through the exact ranks selected after exposed requests move to the upper tail.
4. **Runtime-state inversion.** CPU weight offload changes relative and absolute degradation in opposite directions.
5. **Cross-model replication.** The structural findings reproduce across Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.
6. **Mechanism boundary.** The official runs show no logged preemption, KV-capacity failure, OOM, traceback, or fatal engine failure.

Do not claim absolute firstness without a new, venue-specific literature review that supports it.

---

## 7. Mechanistic interpretation ladder

### Level 1 — Directly observed and safe

- active requests show a large TPOT/ITL stall after the bounded prefill trigger;
- later-admitted requests recover approximately their prior TPOT cadence;
- the client-side semaphore produces substantial pre-dispatch delay;
- offload regimes change relative and absolute effects differently;
- the aggregate median is explained by a measured order-statistic shift;
- no listed preemption/KV/fatal condition appeared in official logs.

### Level 2 — Supported but non-causal interpretation

Use only formulations such as:

- `consistent with a deterministic iteration-level interference cost`;
- `suggests that the additional work is amortized differently across runtime regimes`;
- `compatible with transfer-dominated or runtime-state-dependent iteration costs`.

### Level 3 — Prohibited without new direct evidence

Do not write:

- `PCIe transfer caused the stall`;
- `weight streaming caused the state inversion`;
- `vLLM's internal queue produced the measured later delay`;
- `the scheduler selected a specific request-level state`;
- `max_num_seqs protected active decodes`;
- `the burst prefill started at dispatch time`.

---

## 8. Related-work positioning

### 8.1 Established findings that are not novelty claims

The paper must acknowledge prior work establishing that:

- prefill can interfere with decode;
- long prefills can cause head-of-line blocking;
- chunked prefill has throughput/responsiveness trade-offs;
- queueing and admission delays occur;
- aggregate metrics can hide heterogeneous or tail damage;
- CPU/GPU offloading incurs transfer and execution costs;
- KV pressure and preemption can create severe latency degradation.

### 8.2 Defensible delta

The paper's defensible delta is the combination of:

1. a controlled token-aligned trigger;
2. observed exposure grouping at burst onset;
3. separation of immediate decode stall from later delay/recovery;
4. explicit reconstruction of the aggregate median's order-statistic shift;
5. relative-versus-absolute offload-state inversion;
6. replication across two models;
7. a bounded-load setting without logged preemption or KV-capacity failure.

### 8.3 Safe positioning sentence

> Prior work establishes prefill–decode interference, queueing pathologies, offloading costs, and metric blind spots. We characterize how a bounded prefill event decomposes into exposure-dependent damage, how the exposed fraction changes the order statistics determining aggregate TPOT, and how CPU-weight-offload state reverses the relative and absolute degradation profile.

---

## 9. Server-WAITING diagnostic: interpretation freeze

The planned stock-vLLM diagnostic is supplementary. It cannot retroactively alter the main-study claim or analysis hierarchy.

### 9.1 Common rules

- The diagnostic is not part of the original primary endpoint.
- It is not required for the paper's core contribution.
- It must use a fresh output directory and exactly one diagnostic pair initially.
- No full 32-episode campaign follows automatically.
- Client timestamps alone do not reveal the exact internal prefill-start time.
- An active-victim ITL change is an effect signal, not by itself proof of server admission.

### 9.2 Outcome A — Output-level overlap

Criterion:

```text
at least one burst_first_token_perf_ns < last_active_victim_completion_ns
```

Permitted interpretation:

> At least one burst request made output-level progress while an original active victim was still running.

Not yet permitted:

> The burst prefill definitely began at a particular internal timestamp.

Required consequence:

- validate the result independently;
- harden full resume/campaign integrity before replication;
- seek scheduler or server-trace corroboration if the result will support a server-side replication claim.

### 9.3 Outcome B — No output-level overlap with the original active cohort

Criterion:

```text
all burst_first_token_perf_ns >= last_active_victim_completion_ns
```

Permitted interpretation:

> No output-level overlap with the original active cohort was observed.

Not permitted from client timestamps alone:

> No part of burst prefill execution overlapped the cohort.

The result may be discussed as evidence compatible with admission limiting, but not as a completed server-side interference replication.

### 9.4 Outcome C — Burst output after all 20 victims complete

Criterion:

```text
min(burst_first_token_perf_ns) > max(all_victim_completion_ns)
```

Permitted configuration-specific wording:

> In this configuration, the later burst requests produced no output until the pre-existing victim set had completed, consistent with a fully occupied sequence-admission cap and FCFS ordering shifting the burst cost into server-side waiting and TTFT.

Do not generalize this to all vLLM configurations.

### 9.5 Outcome D — Ambiguous or invalid

If transport concurrency, cohort invariants, timestamps, or server execution are not adequately established:

- exclude the diagnostic from the paper's claims;
- do not run the full campaign;
- either perform one narrowly targeted correction or terminate the branch.

### 9.6 Non-retroactivity rule

Regardless of A/B/C/D:

- the aggregate primary result remains negative/unclear;
- the main exposure decomposition remains the paper's explanatory result;
- an admission-cap control cannot replace the main result;
- an unexpected overlap result cannot become a main claim without independent validation;
- an ambiguous result is omitted rather than rationalized.

---

## 10. `2 × 4096` robustness experiment: interpretation freeze

The alternative burst experiment preserves the total nominal prefill-token budget:

```text
4 × 2048 = 8192 input tokens
2 × 4096 = 8192 input tokens
```

Its purpose is robustness and limited mechanism probing, not causal proof.

### Permitted outcomes

- **Similar effect:** compatible with total prefill budget being an important driver.
- **Stronger effect for 2 × 4096:** compatible with longer individual prefills increasing interference.
- **Weaker effect for 2 × 4096:** compatible with burst parallelism or chunk distribution playing a larger role.
- **Different state dependence:** a useful mechanism hypothesis requiring cautious discussion.

### Analysis rule

The new campaign and the original `4 × 2048` campaign are not treated as paired unless a shared randomized block structure is explicitly constructed. Cross-profile comparisons are descriptive or independently bootstrapped.

---

## 11. Limitations that must remain visible

The manuscript must state at least:

- one main GPU platform in the completed campaign;
- one serving stack and pinned vLLM version;
- two model families, not broad model coverage;
- one main burst profile before the robustness experiment;
- closed-loop client admission in the main study;
- no direct per-request internal scheduler timestamps;
- no causal proof of PCIe transfer, weight streaming, or a specific scheduler mechanism;
- later main-study requests did not overlap the burst window;
- diagnostic and robustness additions may have only four paired repeats and are therefore primarily structural/descriptive.

---

## 12. Terminology freeze

### Use

- `bounded prefill event` or `bounded prefill load`;
- `victim requests`;
- `requests decoding at burst onset`;
- `active wave` when operationally defined;
- `later-admitted requests`;
- `client-side pre-dispatch backpressure`;
- `closed-loop admission-delay cascade`;
- `exposure-dependent damage`;
- `order-statistic shift` or `order-statistic reconstruction`;
- `CPU-weight-offload regime/state`;
- `systems and reliability characterization with availability implications`;
- `pre-specified and frozen before the official runs`.

### Avoid or qualify

- `attack`, unless referring to related work or a hypothetical threat model;
- `DoS`, except to explain that the study does not claim a complete DoS attack;
- `queue time`, unless explicitly prefixed by `client-side` in the main study;
- `unaffected`, unless narrowed to an estimated metric and time interval;
- `proves`, `causes`, `demonstrates the internal mechanism`;
- `preregistered`;
- `first`, `novel`, or `unprecedented` without literature support.

---

## 13. Manuscript consistency checklist

Before submission, every abstract, introduction, contribution list, figure caption, result section, discussion paragraph, and conclusion must satisfy:

- [ ] The aggregate primary endpoint is identified and reported as negative/unclear.
- [ ] The exposure decomposition is identified as pre-specified but not the original primary endpoint.
- [ ] Later requests are described as later-admitted and non-overlapping in the main-study audit.
- [ ] Main-study delay is labeled client-side pre-dispatch backpressure, not vLLM queue time.
- [ ] State inversion is descriptive, not causally attributed to PCIe or weight streaming.
- [ ] The order-statistic contribution is concrete rather than a generic “median is blind” claim.
- [ ] The no-preemption statement is limited to logged events and failures.
- [ ] The server-WAITING diagnostic is supplementary and outcome-dependent.
- [ ] No diagnostic result retroactively changes the primary endpoint.
- [ ] Optional scheduler interventions are labeled as interventions, not stock-vLLM behavior.
- [ ] `pre-specified` is used; `preregistered` is not.
- [ ] Limitations include the non-overlap of later main-study requests with the burst.

---

## 14. Change-control rule

A frozen claim may be changed only by adding an amendment containing:

```text
Date
Claim identifier
Previous wording
New wording
New evidence
Reason for change
Effect on abstract, contributions, figures, and limitations
Independent audit status
```

Optional experiments may add a new bounded claim. They do not erase or silently rewrite the main-study results.
