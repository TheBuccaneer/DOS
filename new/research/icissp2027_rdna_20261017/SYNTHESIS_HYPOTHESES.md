# Literature-led synthesis hypotheses — 17 October 2026

User steering explicitly broadens the search beyond a next experiment or the old cache hypothesis. This document derives higher-level questions by composing results and assumptions from several papers. None is asserted as a discovered vulnerability. A lack of a search hit is not a novelty proof.

## H7. Shared-load observability can invalidate a marginal privacy gate

**Synthesis:** PrefixWall v2 §3.3 uses overlap of hit/miss latency distributions to decide whether selective isolation is needed. Its §4 explicitly acknowledges traffic manipulation and gives qualitative mitigations. Fill and Squeeze §3/§4 uses an attacker's own streaming latency to infer shared resource state. Classical concurrent/relative timing attacks already show how common noise can be reduced. Thus a new question is whether high *marginal* overlap remains a useful privacy criterion when an attacker has correlated observations of the load responsible for that overlap.

**Hypothesis:** A defense can regard pooled hit/miss timing as indistinguishable while an API-only observer with a correlated reference can still distinguish them. The overarching issue is that another shared subsystem can make the masking variable observable. This is stronger than simply asking whether one extra request adds latency.

**Closest comparison:** PrefixWall arXiv:2603.10726v2 §§3.3–4; Fill and Squeeze arXiv:2602.07878v2 §§3–4; SafeKV arXiv:2508.08438v2 §§3,4.2.3,6; Carlini/Nasr arXiv:2410.17175v1 §7; Van Goethem et al., USENIX Security 2020, *Timeless Timing Attacks*. The first four were inspected in full-text form in the relevant sections. The Timeless Timing primary PDF was subsequently inspected in §§3 and 6: concurrent timing/noise cancellation is established prior art, and actual correlated execution/transport is a prerequisite; two independent HTTP/1.1 connections alone need not cancel jitter. Existing timing-noise cancellation must be credited; it is not a new statistical discovery.

**Distinction to test:** conditional, observer-relative distinguishability as an obligation for an adaptive LLM cache-privacy decision. SafeKV is an important counterexample to overgeneralization: under heavy load it tightens promotion toward isolation, whereas PrefixWall's activator can disable protection when overlap is high. Do not treat all adaptive defenses as having the same direction of response.

**Gate before a bounded offline calculation:** GO only for an analytical counterexample to the implication “large marginal overlap ⇒ small conditional distinguishability.” NO-GO for claiming a live PrefixWall bypass. A synthetic Gaussian example is useful to establish the non-implication and exposes its own decisive assumption: the adversary must actually observe correlated runtime variation. It cannot establish that available API timings supply such a reference in vLLM.

**Kill criteria:** reject a system-level claim if the strongest attainable API reference is insufficient, protection remains active through the relevant transition, the final published design removes this decision rule, or direct prior work already tests the same joint observer. Reject a standalone paper based only on a textbook common-noise example. A real contribution would require a defensible observer model and direct validation, not invented inference accuracy from arbitrary simulation parameters.

## H8. Privacy identities and resource-responsibility identities need not coincide

**Synthesis:** KVGov binds reuse to a principal; VTC accounts service per client; PrefixShield charges persistent admission responsibility to an operator-defined accounting group and separates that group from service contracts. One billing principal can have many confidentiality domains. The question is whether changing only confidentiality granularity can multiply persistent state while preserving or changing the resource charge.

**Hypothesis:** A serving design needs an explicit mapping between reuse namespaces and stable resource-accounting groups; privacy-safe refinement of one identity dimension may not preserve an availability entitlement defined on another.

**Immediate prior work:** VTC; PrefixShield §III-B; KVGov §§2–4; FairCache. Namespace cardinality, duplication and Sybil fairness are familiar issues. PrefixShield already aggregates contracts within groups, so “rename a contract to erase debt” is not a new attack. SafeKV already considers coordinated accounts. A useful increment would need a concrete composition counterexample or invariant with a realistic identity hierarchy.

**Gate:** NO-GO for an empirical claim now; retain as a synthesis question. FairCache's exact multi-principal rules and broader identity/resource-allocation literature must be read before implementing a new mechanism. Old two-tenant fixed-salt data cannot validate it. Kill if the mapping and bound are already supplied by existing accounting or if the example merely adds independently entitled tenants.

## H9. Recovery and telemetry semantics may break a service-cost inference

**Synthesis:** VTC reasons about service units, Sarathi-Serve about scheduled work, Fill and Squeeze about preemption burden, and GRIEF about diagnostic oracles. These quantities are not interchangeable. A preemption transition can move recovery work across a phase-based logging label while emitted useful output remains fixed.

**Completed narrow finding:** two retained episodes have 110 generation-labeled scheduler tokens versus 94 post-first emitted tokens, with no excess in the context-token sum. The pinned source labels work by request phase, not useful new output. Four pure-function constructed cases confirm that semantic rule; they do not reproduce a serving run.

**Gate:** GO as supporting evidence in the current position draft. NO-GO for a broad counter-invalidity or new overhead claim. The implementation is internally consistent; to become an empirical short paper, this direction would need to show how an important practical inference changes across real workloads or versions and clear existing instrumentation literature.

## H10. Retention-mediated observation can survive lookup separation

**Synthesis:** KVGov establishes a namespace lookup boundary; PrefixShield shows that namespace-scoped groups still influence replacement; Fill and Squeeze uses shared resource state as an observable signal. A different hypothesis is that an attacker learns coarse victim activity through survival of its *own* cached state, without ever reusing the victim's entries.

**Boundary:** this would be an activity/occupancy side channel, not prompt reconstruction, and would use the old Prime+Probe principle. The principle itself is not new. No inference about private text follows from generic occupancy changes. Authentication, capacity partitioning and external traffic all matter.

**Gate:** NO-GO for a new leakage claim on current evidence. A scientific increment needs a carefully scoped real observer, nearest primary work on software-cache occupancy channels, and controlled negative cases. Existing sequential eviction controls are not privacy experiments. Do not revive them as leakage evidence.

## Decision after synthesis

The search is not restricted to C1 or its immediate next step. H7 is the strongest new cross-paper hypothesis to investigate further; H8 and H10 have substantial established-principle overlap and larger validation needs. H9 yields a completed narrow supporting result. The current complete position manuscript remains a fallback artifact, not an irreversible choice of final research direction. No additional full manuscript is being developed in parallel.

One bounded analytical calculation for H7 is authorized by its gate above. Its result may motivate a better observer model but will not be relabeled as a serving measurement or sufficient publication novelty. The missing final-version/full-text checks and unavailable GPU driver remain explicit dependencies for any stronger claim.

## Completed H7 calculation — constructed examples only

`tools/conditional_observer.py` and `evidence/conditional_observer.json` give exact analytical witnesses, not a serving experiment. With a 20-unit signal and uniform common variation over 1,000 units, marginal Bayes accuracy is 0.51; an assumed reference with at most two units of error gives disjoint residual supports. A Gaussian example gives marginal accuracy about 0.520 and conditional accuracy about 0.9998. These invented parameters demonstrate the non-implication only. Independent or uninformative reference controls recover the marginal result. Three offline tests check these properties; no classifier was trained and no real API reference was established. The high conditional number must never be reported as attack effectiveness.

## H11. Restoration allowances depend on the service available for recovery

The user-supplied HCache full text led to two additional primary neighbors, FairInference and Cascade. See `HCACHE_AUDIT.md` for the page-specific audit and the complete gate. They already cover complementary resource scheduling, cache-delay reservations and budget-aware restoration. The residual question is a concrete implementation's restoration allowance under competing recovery while its published preconditions and tenant entitlement still hold. Generic contention or replacing a load-aware bandwidth model with an unloaded rate does not constitute a finding.

H11 is retained behind H7 in feasibility. No existing artifact identifies the required offloaded-state recovery path; a new backend is not justified before clearing FairCache and target-code/assumption gaps. Both H7 and H11 remain open hypotheses, not submission-ready contributions.
