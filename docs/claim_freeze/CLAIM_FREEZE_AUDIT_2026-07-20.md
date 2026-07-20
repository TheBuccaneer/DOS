# INDEPENDENT AUDIT — CLAIM_FREEZE_2026-07-20.md

**Audit date:** 20 July 2026  
**Artifact audited:** `CLAIM_FREEZE_2026-07-20.md`  
**Verdict:** **PASS**  

---

## 1. Audit scope

The audit checked whether the freeze:

1. preserves the original aggregate primary endpoint;
2. distinguishes it from the pre-specified exposure decomposition;
3. prevents post hoc relabeling of explanatory findings as primary;
4. accurately limits the main-study later-request claim;
5. distinguishes client-side backpressure from internal vLLM queueing;
6. keeps mechanistic interpretations non-causal;
7. states the order-statistic contribution narrowly;
8. constrains the server-WAITING diagnostic to outcome-dependent supplementary claims;
9. prevents optional experiments from rewriting the main result;
10. supports consistent manuscript drafting.

---

## 2. Findings

### A1 — Primary/secondary hierarchy

**PASS.**

The freeze explicitly states that the paired log change in episode-level median victim TPOT was the original aggregate primary endpoint and that it did not show a robust normalized State × Burst interaction. The exposure decomposition is described as pre-specified and frozen before the official runs, but not as the original primary endpoint.

This prevents the central narrative error of replacing a negative/unclear primary result with a positive secondary analysis.

### A2 — Later-request construct validity

**PASS after incorporated precision fix.**

The freeze states that 0 of 16 audited later requests overlapped the burst window and therefore limits the claim to post-burst decode cadence after later admission/dispatch. It expressly prohibits the claim that concurrent later traffic was unaffected.

This is the most important construct-validity boundary in the main-study story.

### A3 — Backpressure semantics

**PASS.**

The freeze consistently labels the large later delay as client-side pre-dispatch backpressure under a closed-loop admission setup. It prohibits calling it internal vLLM queue time, server waiting, or scheduler latency.

### A4 — Active-wave claim

**PASS.**

The claim is limited to requests already decoding at burst onset and to the tested configurations. It is supported as a replicated structural result across two models, two concurrency settings, three offload states, and eight paired repeats.

### A5 — State inversion

**PASS.**

The freeze clearly separates relative slowdown from absolute completion/admission delay. It treats the inversion as descriptive and prohibits causal attribution to PCIe transfer or weight streaming.

### A6 — Order-statistic contribution

**PASS.**

The contribution is stated as a concrete measured change in which later-wave ranks determine the aggregate median. The freeze does not claim that median blind spots in general are new.

### A7 — Log-audit boundary

**PASS.**

The freeze reports only the absence of logged preemption, KV-capacity failure, OOM, traceback, and fatal engine failure. It explicitly prohibits inferring that the KV cache was far from capacity.

### A8 — Related-work boundary

**PASS.**

The freeze does not claim novelty for general prefill–decode interference, queueing, chunked-prefill trade-offs, metric blind spots, or offloading costs. The defensible delta is correctly framed as the combined exposure decomposition, order-statistic reconstruction, state inversion, cross-model replication, and no-logged-preemption boundary.

### A9 — Server-WAITING diagnostic

**PASS.**

The A/B/C/D outcome map is epistemically conservative:

- output-level overlap is not equated with exact internal prefill start;
- absence of a burst first token before active completion is not treated as proof of no internal overlap;
- an admission-cap result is configuration-specific;
- ambiguous results are excluded;
- no outcome retroactively changes the original primary result.

### A10 — Optional robustness experiments

**PASS.**

The `2 × 4096` experiment is frozen as a robustness/mechanism-probing addition with non-causal interpretations. Cross-campaign comparisons are not falsely treated as paired.

### A11 — Terminology and manuscript reuse

**PASS.**

The freeze provides usable English manuscript sentences, forbidden formulations, terminology rules, and a pre-submission consistency checklist. It is suitable as the authoritative source for drafting the abstract, introduction, contributions, results, limitations, and conclusion.

---

## 3. Contradiction check

No internal contradiction was found between:

- the negative/unclear aggregate primary endpoint;
- the strong active-wave result;
- the approximately stable later post-admission TPOT cadence;
- the client-side backpressure cascade;
- the order-statistic explanation;
- the relative-versus-absolute offload inversion.

These results address different estimands and time paths. The freeze now states those estimands explicitly.

---

## 4. Residual risks

The following are not freeze defects but remain empirical or editorial risks:

1. A reviewer may still view the exposure decomposition as exploratory despite being frozen before official runs; the paper must document the timeline and artifacts clearly.
2. “Availability damage” should be operationally defined through latency and completion-delay outcomes, not implied service unavailability.
3. The very narrow confidence interval in a representative active-wave cell may invite scrutiny; raw paired ratios and bootstrap code should remain auditable.
4. The main-study later-delay result is tightly coupled to closed-loop client admission and must not dominate the abstract without qualification.
5. The diagnostic stock-vLLM experiment may yield no usable paper claim; the frozen core remains valid regardless.
6. Venue-specific novelty language still requires a current literature and policy check before submission.

None of these risks requires changing the frozen core claims now.

---

## 5. Final decision

**CLAIM_FREEZE_2026-07-20.md is approved for manuscript drafting.**

No scientific blocker remains before beginning the paper. The Server-WAITING diagnostic and `2 × 4096` experiment may add bounded robustness or discussion claims, but they are not prerequisites for the main narrative.
