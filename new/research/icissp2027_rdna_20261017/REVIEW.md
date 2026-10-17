# Adversarial internal review — 17 October 2026

This is a critical self-review by the same assistant that developed the artifact. It is **not** an independent human review, a reproduction of the old NordSec reviews, or an acceptance prediction. The original rejected manuscript/reviews were not located and no quotations from them are invented.

## Round 1: evidence and inference

| Objection | Correction / retained limitation |
|---|---|
| A preemption increment does not identify the victim or the endpoint window. | Both affected episodes retain their complete outcome; mechanism attribution remains unresolved. No measured eviction/preemption penalty is claimed. |
| Matching initial context-token sums does not prove an absence of recovery work. | Compared generation-labeled scheduled work against emitted output; found a 16-token accounting residual in each of two pressure episodes. Pure-source calibration explains why the logging label is not an output counter. No request-specific GPU cost is inferred. |
| A single episode per cell cannot estimate an interaction or establish equivalence. | Published descriptive endpoints only, exposed the missing cell, omitted significance/equivalence claims, and specified a bounded prospective design. |
| Reusing runner verdicts would not be an independent reconstruction. | New parser uses raw SSE records, timestamps, usage and raw metric bodies without importing the runner. Sequential cache controls are explicitly only stored-result extractions. |
| Treating a missing metric as zero hides absence of evidence. | Parser raises for absent counters; negative offline test checks this. |

**Outcome:** factual case study support is adequate for the modest artifact-consistency claims. Causal mechanism and new attack claims remain NO-GO.

## Round 2: novelty and competing explanations

1. **The main empirical mechanism is already known.** PrefixShield studies namespace-scoped reuse over shared replacement capacity. Fill and Squeeze studies global KV pressure and preemption. Sarathi establishes the chunk/latency tradeoff. Merely changing salts, capacity, budget, hardware or vLLM version does not supply the contribution.
2. **The reporting position risks being standard experimental practice.** Controls, fixed populations, counter calibration and separating observations from explanations are familiar. The draft's possible increment is their operational connection to cache-policy estimands and claim-specific abstention. No external auditor study or independent application establishes that this increment changes decisions reliably. This is the strongest rejection argument against C4.
3. **H7 is not a novel statistical theorem.** Marginal versus conditional distinguishability and common-noise cancellation are established. Timeless Timing supplies a direct warning against presenting the analytical witness as an attack discovery. A real defense-specific increment requires the final adaptive rule and an attainable API reference. The present invented parameters establish neither.
4. **HCache is closer competition than a generic offload citation.** It already measures concurrent decoding overhead, profiles compute/I/O mixtures and studies GPU cache reuse. Its MHA scope and write-path backpressure are explicit. The new audit corrects any suggestion that these are unexamined general issues.
5. **FairInference and Cascade further narrow H11.** Restoration-delay-aware memory protection and jointly budgeted scheduling/retrieval are already designed. Their own preconditions must be preserved in any proposed counterexample. Unbounded competition in ordinary vLLM cannot falsify another scheduler's conditional guarantee.
6. **A closest full-text gap remains.** FairCache was checked through primary metadata/abstract and its code description, not full text. The supplied HCache PDF has a different title and DOI. Accessible supplied-file locations were rechecked after the user's correction. No unconditional novelty assertion is justified.

**Corrections:** added HCache/FairInference/Cascade to the manuscript's prior-work boundary; retained H7/H11 as hypothesis documents; retained a NO-GO submission recommendation. No unsupported vulnerability language or firstness claim is used to rescue the direction.

## Round 3: reproducibility, presentation and readiness

- Seven pinned vLLM source files match upstream v0.17.1 hashes and are now included with their license. The pure-counter exercise no longer depends on the original machine's vLLM installation.
- Fourteen offline tests pass: eleven artifact/counter checks and three analytical-observer checks. They do not reproduce the historical 54/83/85-test suites and do not verify live serving behavior.
- Review archive includes the JSONL inputs needed by the reanalysis. The historical runners and collection infrastructure remain unchanged.
- The editable English draft builds in the publisher's template as eight A4 pages. Template files are unchanged. Character counts, font checks, build output and visual inspection are recorded separately.
- Double-blind draft omits author identities; AI use is disclosed and cited by section. Human authors must resolve the venue's acknowledgment/disclosure placement requirements and verify all claims before submission.
- Overlap with the unavailable original NordSec manuscript cannot be cleared. Anonymity of a PDF does not resolve prior-publication or content-overlap questions.
- The local GPU driver query failed; no GPU inference, server launch or live restoration validation occurred. No third-party target was tested and nothing was submitted or published.

## Decision

**NO-GO: empirical regular/short submission.** No repeated identified treatment contrast, new validated boundary or cleared empirical contribution is available.

**NO-GO: submission of the current position draft.** It is complete and reviewable, but novelty/usefulness against nearest work and old-manuscript overlap remain unresolved. Completing a manuscript is not evidence that its contribution passed the gate.

**GO: focused next research on H7's final-design and observer assumptions.** H11 remains a second lead. Neither decision authorizes an unspecific measurement campaign or asserts that a live bypass exists. If the remaining difference reduces to known noise cancellation or ordinary restoration contention, stop that candidate.
