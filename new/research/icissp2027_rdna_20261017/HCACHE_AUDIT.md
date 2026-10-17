# HCache and the restoration-boundary candidate

## Document identification and inspection

User-supplied file: `/home/rock/Downloads/3689031.3696072.pdf`. Title: **Fast State Restoration in LLM Serving with HCache**, Shiwei Gao, Youmin Chen and Jiwu Shu, EuroSys 2025, proceedings pp. 128–143, DOI [10.1145/3689031.3696072](https://doi.org/10.1145/3689031.3696072). SHA-256: `0492cc094dfc401bec3a0727b56f70e144ccd3c8f19601498a565edd03107529`. A research copy is retained in `literature/hcache.pdf`.

This is HCache, not FairCache (DOI 10.1109/TPDS.2026.3729988). Following the user's statement that FairCache had already been supplied, the accessible Downloads, Codex attachments and Documents/Codex trees were checked for PDFs and ZIP-contained PDFs. Only the HCache PDF was found; the one available old review ZIP contains no PDF. This is an access/inventory limitation, not a statement that the user never supplied a different file.

Read the introduction, motivation, restoration equations, scheduler/storage design, implementation, evaluation and related-work limitations (§§1–8, proceedings pp. 128–140). The bibliography was used to identify neighbors, not treated as an audited collection of those papers. Cross-column extraction was checked with focused page reads.

## What the paper actually establishes

| Location | Published design or reported observation | Consequence for our claims |
|---|---|---|
| §3.1–3.2, pp. 131–132 | Retains layer inputs and reconstructs KV with projections; pipelines transfers and computation. The cost analysis explicitly uses MHA. | “Recovery trades I/O for compute” and this intermediate representation are established ideas. The 2× storage relation is scoped to that analysis. |
| §4.1, pp. 133–134 | Chooses layer-wise mixtures of hidden-state recovery, KV transfer and token recomputation from offline hardware profiles. | Generic complementary-resource scheduling is not our contribution. An offline profile is not itself an adversarial service bound. |
| §4.2–5, pp. 134–135 | Chunked layout, two-stage saving, separate streams and host workers; built on DeepSpeed-MII 0.2.0/SplitFuse, not our vLLM runner. | Our existing CPU **weight** offload results do not reproduce KV/activation restoration. |
| §6.1.1, pp. 136–137 | Concurrent multi-round workload, Poisson session arrivals and 30 s inter-round intervals. All methods evict history at round end in this comparison. Reports at most 4% TBT increase over its ideal baseline. | It already measures impact on decoding. Claiming that HCache never investigates interference would be false. This is not a worst-case per-tenant bound. |
| §6.1.2 and §6.2, pp. 137–138 | Long-context overall tests use batch size 1; sensitivity tests change devices, storage bandwidth and context length. | Hardware/resource balance is already investigated. Changing only the GPU or model is insufficient novelty. |
| §6.3.3, p. 139 | Direct SSD saving can stall decoding; two-stage saving avoids that in the evaluated cases. | Write-path backpressure and its mitigation are known here. A generic “saving state can stall decode” claim is killed. |
| §6.4, p. 140 | Adds an LRU GPU cache with synthetic Zipfian context popularity; benefit shrinks as GPU hit ratio increases. | Cache hit rate and miss-path restoration are already combined experimentally. |
| §7, p. 140 | Limits unchanged-model support to MHA; discusses projecting a smaller representation for MQA/GQA as beyond scope. | Directly applying the headline storage ratio to our Llama-3.1 GQA configuration is unjustified. The architecture limitation is acknowledged, not a discovered flaw. |

Reported speedups and TBT observations are the authors' results, not measurements reproduced here. No HCache code was run or ported.

## Additional primary competition found through this paper

**FairInference:** Bali et al., *Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving*, [arXiv:2609.18112v1](https://arxiv.org/abs/2609.18112v1), 16 September 2026. Downloaded and inspected §§3–5, the evaluation setup and §9 limitations. §4.1 explicitly assumes pooled decode duration does not exceed the isolated fair-share duration and bounds admitted prefill demand by accumulated headroom. §5.2/Algorithm 3 already derives a protected KV reservation from the amount of state restorable within a cache-delay budget. §9 calls hybrid batching and relaxing the latency assumption future work. Thus neither “protect KV according to restoration time” nor “jointly control scheduling and cache delay” is a new contribution. These are conditional design claims, not unconditional guarantees for every engine.

**Cascade:** Adnan et al., *Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving*, [arXiv:2608.06557v1](https://arxiv.org/abs/2608.06557v1), 6 August 2026. Inspected §§III–V and VI–VII. §IV-B uses load-aware latency estimates and a profiled guardband; §IV-C uses effective link bandwidth under load to limit prefix retrieval and falls back to recomputation. The evaluation extends Vidur with profiles from physical hardware; it must not be represented as our own live reproduction. This already links restoration admission, scheduling and finite SLO headroom. An attack claim cannot be obtained merely by replacing its effective operational bandwidth with an unloaded peak bandwidth it does not assume.

Neither targeted inspection constitutes an independent verification of all proofs, code or reported performance in those preprints. HCache, FairInference, Cascade, VTC and FairCache are the nearest comparison set for the new candidate below. The FairCache full-text gap remains material.

## H11: Does a restoration allowance remain feasible under competing recovery?

**Possible contribution (conditional):** identify a concrete operating boundary where an implementation's calibrated allowance for restoring a well-behaved tenant's KV state ceases to hold under competing recovery, despite the tenant remaining within its declared entitlement. Demonstrate which runtime service assumption fails and whether a small, explicit resource reservation repairs the boundary. This is a proposed empirical validation of an assumption, not a new general contention principle or a claimed bypass of these systems.

The synthesis is: HCache makes the compute/I/O recovery path a design choice; FairInference uses recoverability within a delay allowance to set memory reservations; Cascade makes restoration feasibility depend on live resource estimates. Consequently, the useful object to validate is an **end-to-end restoration allowance under its stated service conditions**, rather than only cache capacity or an unloaded tokens/s rate.

**Already known / strongest alternative:** ordinary queueing, multi-resource admission and worst-case service bounds explain such a failure. The systems may already reserve sufficient service, use an adequately conservative bound or document that the test workload is outside their assumptions. A large delay in unmodified vLLM would not contradict a guarantee made by a different scheduler. No statistical simulation of arbitrary contention can clear these alternatives.

**Gate:** GO for this bounded literature/assumption audit, now completed. **NO-GO for implementing or measuring a new HCache/FairInference extension on present evidence.** A live follow-up requires accessible target code or a faithful specified implementation, the FairCache comparison, a declared entitlement and an identifiable recovery event. The existing overlap files measure cold prefill/reuse and some resumption; they do not identify an offloaded-state recovery path and cannot test H11.

**One minimal discriminating design, if the gate later clears:** use a single fixed-size victim restoration, just below the target's declared allowance, with two paired conditions: alone versus a concurrent, entitled competing restoration at the same trigger. Keep request identities, representation, recovery method, protected victim state and service contract fixed. Record restore queueing, transferred bytes, scheduled compute, dispatch/completion and victim deadline. Both conditions need repeated randomized ordering and verified live bounds; no broad sweep is justified. A failed allowance matters only if all published preconditions still hold. Ordinary unbounded best-effort competition is the null explanation, not a surprising security result.

**Kill:** discard the candidate if the proposed bound explicitly excludes this competition, the implementation already enforces a sufficient service reservation, the event lies outside the stated deadline interval, the result only shows generic oversubscription, or direct prior work already validates the same boundary. Do not save it with unrelated parameter sweeps.

## Selection impact

H11 is a useful second synthesis lead alongside H7's conditional-observer question. It is less immediately feasible: our saved data contain no identified HCache restoration, and implementing a new backend would violate the reuse/novelty-first priority. H7 remains the first low-cost question to clear through the final published defense design and a realistic observer model. Neither lead currently supports an empirical submission. The complete evidence-contract position draft remains a reviewable fallback with substantial novelty risk.
