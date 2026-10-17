# RDNA research audit — 17 October 2026

## Decision reached

**NO-GO for an empirical regular submission asserting a new salt × capacity × budget availability mechanism.** Develop one **position-paper draft** about evidence requirements for isolation claims. The position has a concrete operational proposal and an independently reconstructed case study; submission readiness remains NO-GO pending the closest inaccessible full text, human scientific review, and disclosure checks. No further GPU campaign is justified at this gate.

This is an incremental evaluation contribution, not a firstness claim, new attack, scheduler, or cryptographic defense. The full draft is in `PAPER/PAPER.tex`; its claims are intentionally narrower than the original mechanism hypothesis.

## A. Inventory and provenance

Repository: `/home/rock/projects/DOS`; checked-out HEAD `d1e247ba22b3c505f3b36173655b937f89c2693f`. Later historical reports were also inspected through Git objects at `aee916b` (9 October) and `d7b2bce` (6 October), without changing checkout. Pre-existing untracked gate scripts/data and handover files were preserved. New work is confined to this research directory.

| Resource | What exists | Evidential status in this audit |
|---|---|---|
| Historical claim freeze | `docs/claim_freeze/CLAIM_FREEZE_2026-07-20.md` | Prior interpretation; main aggregate interaction was negative/unclear. Not evidence for the salt hypothesis. |
| 6 October reanalyses | `DOS_Bestandsdaten_Auswertung_2026-10-06.md` and `DOS_Spaete_Requests_Kritikbewertung_2026-10-06.md` in Git history | Reported results, not newly independently rerun. Useful for identifying cohort and capacity confounders. |
| 9 October inventory | `docs/icissp_2027_gate/BESTANDSAUFNAHME_2026-10-09.md` | Reports 100 historical episodes / 50 pairs / 2,200 requests / 131,200 output tokens, with post-trigger metric gaps and cross-budget capacity differences. These counts are historical report claims here. |
| Qwen server-waiting audit | `new/paper_artifacts/server_waiting_qwen_20260804/reaudit/REPORT.txt`, JSON and CSV | Prior audit distinguishes fixed offered identities from dynamically selected exposure groups. No new scheduler mechanism follows. |
| New cache data | `new/runs/cache_isolation_gate/` | Two stored sequential eviction controls, AAB smoke test, and 14 overlap episodes. Only overlap SSE reconstruction is newly independent in this work. |
| Instrumentation | `new/scripts/cache_isolation_gate/`, historical prefill SSE executor and budget server starter | Existing components retained; no runner changes. Their previous offline test counts are not re-certified by the new tests. |
| vLLM source | Installed source under `porto/.venv/lib/python3.13/site-packages/vllm` | Seven audited files exactly match upstream tag v0.17.1 by SHA-256; see evidence manifest. |
| Original submission/reviews | No NordSec manuscript or original reviews located in inspected project sources, documents, or accessible Git inventory | Missing. No reviewer quotation or claim about their precise reasoning is made. Overlap with the rejected manuscript cannot be cleared completely. |
| Hardware now | Historical artifacts describe an RTX 3090 24 GB | Read-only `nvidia-smi` fails to communicate with the driver. Current GPU execution availability is unconfirmed. No server start or GPU workload attempted. |

Searches covered repository file inventory, historical root reports, docs, new/paper_artifacts, and available project documents. This is not a claim that missing materials do not exist elsewhere. It is sufficient to prevent treating old summaries as original reviews.

## B. Serious candidates and adversarial comparison

Scores are ordinal judgments, 1 poor to 5 strong, not estimated acceptance probabilities. Columns: independence / literature separation / feasibility / evidence / time / venue fit; risk is qualitative.

| Candidate | Scores | Closest work (3–5) | Assessment |
|---|---|---|---|
| C1. Salt-isolated shared KV creates a new capacity-dependent availability mechanism | 1/1/3/2/2/5 | PrefixShield; FairCache; Fill and Squeeze; PrefixWall; VTC | **Reject as mechanism novelty.** Namespace separation plus common replacement is already explicit in PrefixShield. Existing pilots lack replication and near-capacity attribution. A vLLM version change alone is insufficient. |
| C2. Small prefill budgets reduce stalls without reducing total damage under offload | 2/1/4/3/4/3 | Sarathi-Serve; VTC; Fill and Squeeze | **Reject as standalone paper.** Trade-off is established, only three historical pairs per budget/offload cell, and measured KV capacity varied with budget. Could be replication, but lacks a defensible independent boundary in current evidence. |
| C3. Exposure cohort versus fixed-population reanalysis | 2/2/4/4/4/4 | GRIEF; VTC; Fill and Squeeze; Sarathi-Serve | **Fallback only.** Identity selection and aggregate-median dilution are useful methodological lessons, but original manuscript is missing and general selection effects are known. Do not rebrand descriptive replotting as a mechanism. |
| C4. Evidence contract for availability claims about cache-isolation policies | 3/3/5/3/5/5 | GRIEF; PrefixShield; PrefixWall; VTC; FairCache | **Selected for position draft.** Separate total policy cost, prefix-specific contrast, and mechanism attribution; retain complete measurements when attribution is unknown. Supply explicit abstention and falsification rules plus a raw-data case. Literature gap remains provisional until FairCache is read. |

No full empirical papers were pursued in parallel. C3 remains the documented fallback; it would still require a fresh novelty gate, not automatic continuation.

## C. Literature audit, primary sources and scope

PDFs and extracted texts are retained privately in `literature/`; URLs, hashes and failed access attempts are in `sources.json` and response files. “Full-text inspection” below means the paper itself was obtained and its question, relevant method, results, discussion/limitations inspected. It does not mean independent reproduction of its experiments, proof checking of every theorem, or verification of every cited reference. No abstract is labeled a full-text audit.

### Closest inspected papers

- **Wang and Buyya, PrefixShield**, *Preserving Admission Responsibility in Multi-Tenant Large Language Model Prefix Caches*, arXiv:2608.01657v1, 3 August 2026. [Full text](https://arxiv.org/pdf/2608.01657v1). Inspected §§I–VIII, especially §II namespaces/shared pool, §V matched 10-run comparisons and capacity sweep, §VI–VIII limitations. It already studies responsibility for persistent replacement pressure with namespace-scoped lookup. Cross-group content deduplication is out of scope. Thus C1's general mechanism is anticipated. Our proposed contract concerns estimands and abstention, and makes no claim to outperform its defense.

- **Pennas et al., PrefixWall**, arXiv:2603.10726v2, 20 May 2026. [Full text](https://arxiv.org/pdf/2603.10726v2). **Version correction:** search results and artifact pages still use the v1 name *CacheSolidarity*. The acquired v2 title is *PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems*. Inspected §§1–7, including §4 conditional guarantees and §5 baselines, workloads and sensitivity. It already compares isolation's performance cost and considers cache pressure across models. A generic privacy/performance trade-off is not our novelty. Our focus is attribution of a bounded victim decode window under a shared resource limit.

- **Zhao et al., GRIEF**, *Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing*, arXiv:2605.11202v1, 11 May 2026. [Full text](https://arxiv.org/pdf/2605.11202v1). Inspected main §§1–6 and relevant appendices A.6–A.9. Timed request traces, controlled replay, behavioral versus structural confirmation and concurrent interference are already central. We therefore claim neither concurrent testing nor anomaly/confirmation separation as new. The narrower proposal specifies which cache-policy contrast is estimated and requires preserving outcome evidence when a mechanism check abstains.

- **Sheng et al., VTC**, *Fairness in Serving Large Language Models*, OSDI 2024, pp. 965–988; inspected arXiv:2401.00588v2. [Publication](https://www.usenix.org/conference/osdi24/presentation/sheng), [full text](https://arxiv.org/pdf/2401.00588v2). Inspected §§2–5 and Appendix C.1, with emphasis on service definition, fairness bounds, memory-size ablations and cache-aware scheduling conflict. Distinguishing offered requests from actual service is established. Our position adds a specific reporting requirement for cache-isolation interventions and finite-window interference claims, not a fairness algorithm or stronger bound.

- **Wang et al., Fill and Squeeze**, *Rethinking Latency Denial-of-Service: Attack the LLM Serving Framework, Not the Model*, arXiv:2602.07878v2, 8 October 2026. [Full text](https://arxiv.org/pdf/2602.07878v2). Inspected main §§2–5 and appendix G.3/H, including state transitions, experimental units and boundaries. It already addresses global KV saturation, preemption, capacity changes and ordinary contention. No new exhaustion attack is established here. Differences in version or a 128-token prompt are not sufficient contribution claims.

### Additional bounded checks

- **Agrawal et al., Sarathi-Serve**, OSDI 2024, pp. 117–134; arXiv:2403.02310 full text acquired. [Publication](https://www.usenix.org/conference/osdi24/presentation/agrawal). Inspected §§3–5, especially token-budget selection and ablations. Chunking balances decode latency with prefill efficiency; budget-only observations do not supply new mechanism novelty.
- **Addagada, KVGov**, arXiv:2608.09225v1. [Full text](https://arxiv.org/pdf/2608.09225v1). Inspected §§2, 4, 5 and 7. It distinguishes simulation of the defense from hardware measurement of the channel and leaves workload-specific efficiency measurement open. We do not equate its audit scheduler with request scheduling or characterize simulation as a production defense result.
- **Liu et al., When Fancy Eviction Fails**, arXiv:2609.28870v2, 1 October 2026. [Full text](https://arxiv.org/pdf/2609.28870v2). Targeted inspection of §§3.2, 4.4 and 6–8: cache hit rate and even uncached-token count alone need not determine latency. This constrains our work-normalization proposal. This was a targeted full-text check, not an exhaustive independent audit of all policy comparisons.

### Open direct-competition gap

**Bai et al., FairCache: Demystifying Cache-Induced Unfairness in Multi-Tenant Large Language Model Serving**, TPDS 37(11), 2390–2406, published 2 September 2026. DOI [10.1109/TPDS.2026.3729988](https://doi.org/10.1109/TPDS.2026.3729988). Publisher abstract, author page and official [implementation](https://github.com/icloud-ecnu/FairCache) inspected. Its abstract already covers multidimensional compute/memory/I/O attribution and shared-prefix costs. **Full text unavailable:** publisher retrieval returned no PDF, author page links publisher and code but no paper; OpenAlex reports no OA copy. Repository README is not a substitute. No access restriction was bypassed. This unresolved nearest competitor prevents an unconditional novelty/submission GO.

The earlier labels SLOWeave and FairInference were not established as usable primary sources in this focused search. They are not cited as papers or used as evidence. This is not proof that no work with those names exists.

## D. Selected contribution and outcome

See `NOVELTY_GATE.md` for the bounded position and kill criteria. See `EVIDENCE.md` for the independent reconstruction. A complete English draft is produced rather than another experiment plan. It supplies a falsifiable reporting proposal and clearly labeled exploratory observations; it is not represented as accepted, fully novel, or ready to submit.

No new GPU experiment: the remaining obstacle is scientific identifiability and literature clearance, not lack of more samples at arbitrary parameter values. Current driver access also fails. Future acquisition must be justified by a specific discriminating test after the gate is cleared.

## E. New-direction check after user steering

Two further candidates were explicitly examined rather than treating the original salt hypothesis as mandatory.

**C5 — calibrating work counters across preemption.** Direct competition: GRIEF (oracles and confirmation), VTC (service accounting), Fill and Squeeze (preemption as a burden measure), Sarathi-Serve (scheduled work). The local records contain generation-labeled counts exceeding new emitted output while context totals have no excess. Source semantics and four pure-function offline cases explain the distinction; the paper now includes this concrete counterexample. It improves C4's evidential basis. As a standalone regular paper, the result is presently too narrow: implementation-defined labels are not themselves a bug, there is no cross-engine evaluation and no evidence of prevalence in published conclusions. **GO as completed supporting analysis; NO-GO as an independently validated new attack or broad measurement-failure claim.**

**C6 — directly extending an open literature question.** KVGov §5.9/§7 leaves workload-specific efficiency cost of namespace isolation open. PrefixWall §5.1 evaluates the main privacy/performance comparison at one request per second; PrefixShield §II explicitly separates namespaces while retaining common capacity. A defensible extension asks which victim service stage bears isolation's total cost near a capacity boundary, after separately measuring reuse loss, effective work and transition timing. This is not rejected merely because shared-resource contention is known. It is not yet cleared against FairCache; present data lack randomized repetition, one near-capacity cell and request/window attribution. Thus **conditional research candidate, NO-GO as a completed empirical submission today**. It is the preferred empirical follow-up if literature access and hardware recover, without claiming that a parameter/version change is enough.

C4 was refined with C5; no second full manuscript was started. The process follows explicit limitations of existing papers and their overlapping scope, rather than inventing an unrelated hypothesis to escape competition.

## Latest primary-paper and selection update

The user's HCache PDF has been read in the relevant full-text sections and retained with its hash. Its evaluation already covers restoration-induced decoding overhead and GPU cache reuse; it is neither a new interference result of ours nor FairCache. Follow-up searches found primary full texts for FairInference (arXiv:2609.18112v1) and Cascade (arXiv:2608.06557v1). Their assumptions, designs and limitations were inspected as documented in `HCACHE_AUDIT.md`. Earlier uncertainty about whether FairInference could be identified from a primary source is superseded by that inspection.

User steering broadened candidate generation to cross-paper synthesis. H7 and H11 are the two retained new leads; see `SYNTHESIS_HYPOTHESES.md`. H7 currently has only a constructed analytical witness. H11 has only an assumption audit and cannot be tested using the saved cold-prefill cases. Neither is an established vulnerability or a cleared submission contribution. The complete evidence-contract manuscript remains a fallback; current submission recommendation is NO-GO, with detailed reasons in `REVIEW.md` and `SUBMISSION_STATUS.md`.
