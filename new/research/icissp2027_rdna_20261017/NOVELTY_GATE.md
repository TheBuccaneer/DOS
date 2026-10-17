# Novelty gate — 17 October 2026

## Completed fallback candidate: evidence contract for cache-isolation availability claims

**Decision: conditional GO for the low-cost position-paper development completed here; NO-GO for claiming demonstrated mechanism novelty or submitting the present package.** No expensive experiment passes this gate. FairCache full-text access and comparison remain a material scientific dependency. This is an exploratory research decision, not a preregistered hypothesis or a promise of acceptance.

### Contribution in three sentences

We propose reporting cache-isolation availability results using separate estimands for total deployment-policy cost and a prefix-specific control contrast, alongside a claim-specific evidence profile. A technically complete episode with unresolved mechanism attribution must remain in the outcome inventory rather than becoming either a clean negative or silently disappearing. A reconstruction of 14 existing overlap episodes demonstrates why these distinctions matter in practice, without estimating a causal effect or asserting a new attack.

## Closest competition and residual difference

| Work | What it removes from our novelty claim | Residual proposed contribution |
|---|---|---|
| PrefixShield, arXiv:2608.01657v1, §II/§V | Namespace-scoped reuse with common physical contention and capacity-dependent replacement are already investigated. | Evidence requirements for interpreting a privacy-policy change and finite decode-window outcome, rather than a replacement defense. |
| PrefixWall, arXiv:2603.10726v2, §§4–5 | Privacy/performance costs and varying cache conditions are already evaluated. | Explicit distinction between total policy impact and mechanism attribution; a no-shared-prefix control and abstention semantics. |
| GRIEF, arXiv:2605.11202v1, §§3–4 | Concurrent traces, telemetry, replay and confirmation are existing methods. | Application-specific contract linking each reported estimand to its required evidence, with accounting for complete but unattributable episodes. |
| VTC, OSDI 2024, §§3–5/App.C.1 | Service accounting, fairness and cache-aware scheduling conflicts are established. | A reporting contract, not a fairness algorithm or service guarantee. |
| FairCache, DOI 10.1109/TPDS.2026.3729988 | Abstract already anticipates multidimensional cache/resource cost attribution. | **Unknown until full-text inspection.** Its evaluation may already contain essentially the same operational requirements. |

The proposed synthesis is not proven absent from every previous work. It is a bounded position worth presenting for internal scientific review. Calling it a completed new empirical result would overstate both novelty and evidence.

## Strongest alternative explanations

1. Lost reuse changes actual prefill work and simultaneously live KV blocks; ordinary contention explains timing changes.
2. Two decoding requests cost more than one even when no prefill budget binds.
3. A preemption counter increment may refer to B, or occur outside the D8–D28 receive window.
4. Instrumentation, asynchronous output and client delivery can affect observed timestamps.
5. Complete-case filtering on a no-preemption attribution gate can change the sampled operating regime.
6. A reporting checklist may be seen as standard experimental practice rather than a sufficiently distinct scientific position.

## Distinguishing argument and test

The manuscript's core argument is a constructive non-implication: a cache-reuse test, complete endpoint measurement, and whole-episode preemption count each permit incompatible mechanism explanations. The explicit contract requires both total-policy and mechanism evidence to be labeled, so it can abstain while retaining measured outcomes. The local reconstruction supports the existence of that distinction, not the prevalence of the underlying problem in other studies.

**One prioritized follow-up, only after literature clearance:** replay one fixed two-request workload as a paired, randomized five-condition block (victim only; shared/isolated with common prefix; shared/isolated with disjoint prefix), at the two capacities already used and one effective budget of 64. Start with three complete paired blocks, a feasibility check rather than a power claim. Use request-ID-level scheduler/preemption timestamps to establish whether the event occurs inside the predeclared victim window; preserve all outcomes and separately mark attribution unknown. Stop at this bounded size and decide whether a larger study is justified. Do not modify historical runners; reuse existing capture components and add only scientifically necessary tracing in a separately versioned path.

Predictions:
- Ordinary contention: disjoint-prefix shared and isolated have similar trajectories; cold B can trigger pressure without salt changing lookup. A shared-prefix total-policy difference may remain, fully compatible with known resource contention.
- A distinct interaction: a predeclared prefix-specific contrast changes with capacity beyond uncertainty and is accompanied by a time-aligned attributable transition not explained by the disjoint control. This would still require comparison with PrefixShield/FairCache and a mechanism explanation.
- Attribution failure: counters change but request/window mapping is missing; retain outcome, abstain on mechanism. Do not label a negative causal result.

**Kill criterion:** if the distinguishing transition lies outside the victim window, also appears under the disjoint control with the same resource trajectory, or cannot be attributed, reject the new-mechanism candidate. If the reporting contract offers no operational prediction or decision beyond those already supplied by the nearest work, reject C4 as a publication contribution. More arbitrary parameter sweeps do not rescue either candidate.

## Practical consequences

- Empirical regular/short track: NO-GO on present evidence.
- Position draft: completed, with explicit limitations and testable consequences.
- Actual submission: NO-GO until FairCache and human review clear the residual contribution, provenance overlap with the old manuscript is checked, and anonymity/AI disclosure are accepted.
- Current hardware: driver communication failed; no proposed experiment was executed.

## Gate update: counter calibration and literature-led extension

C5 now adds a completed, purely offline semantic calibration to the selected position: two saved episodes have 110 generation-labeled scheduler tokens versus 94 new post-first output tokens, explained by the pinned logger's classification rule. This is a concrete counterexample to a potential inference, not evidence that an existing publication made that inference. It does not justify a new GPU campaign or a general bug claim.

C6 follows KVGov's explicit open efficiency question using PrefixWall's performance comparison and PrefixShield's namespace/shared-capacity scope. Its additional knowledge would be a validated allocation of privacy-policy cost to victim service stages near a resource boundary. FairCache clearance, request-level event timing and repeated randomized blocks are necessary; the existing pilots alone cannot supply that contribution. This preserves a substantive empirical direction instead of discarding all extensions of known contention.

## Selection update after the user's synthesis steering and HCache

The complete C4 draft is a fallback under review, not a commitment to submit it. H7 (joint observations against a marginal adaptive privacy gate) has passed only a low-cost analytical gate; the non-implication is known statistical reasoning and supplies no empirical bypass. Its next decisive step is checking the final PrefixWall/CacheSolidarity design and a realistic API observer. H11 (restoration allowances under competing recovery) is an additional lead derived from HCache, FairInference and Cascade; its scope and kill criteria are in `HCACHE_AUDIT.md`. Those three papers already remove generic compute/I/O tradeoffs and joint cache/scheduler budgeting from the novelty claim.

**Current recommendation: NO-GO for an empirical regular/short submission and NO-GO for submitting the fallback position draft as it stands.** Continue only the focused literature/assumption clearance for H7, with H11 behind it in feasibility. No larger campaign or new restoration backend is scientifically justified by the present gate. A negative assessment is retained rather than presenting the completed draft as proof that a publishable contribution was found.
