# ICISSP 2027 RDNA research review package

Start with `SUBMISSION_STATUS.md` and `REVIEW.md`. **Current scientific recommendation: NO-GO for submission.** A complete position draft is supplied for review; H7 and H11 are open research hypotheses, not demonstrated attacks.

## Contents

- `RESEARCH_AUDIT.md`: inventory, provenance, prior work and candidates.
- `NOVELTY_GATE.md`: contribution boundaries, alternatives and kill criteria.
- `EVIDENCE.md`: measured versus constructed results and confounders.
- `SYNTHESIS_HYPOTHESES.md`, `HCACHE_AUDIT.md`: literature-derived directions and the user-supplied HCache comparison.
- `REVIEW.md`, `SUBMISSION_STATUS.md`: criticism and readiness decision.
- `PAPER/`: editable LaTeX/BibTeX and unchanged publisher styles.
- `output/pdf/PAPER.pdf`: compiled English draft.
- `tools/`, `reference/`, `evidence/`: offline analysis, seven licensed pinned vLLM source files, hashes and outputs.

The archive preserves repository-relative paths. Extract it into an empty review directory. Its `new/runs/cache_isolation_gate` contains the exact JSONL inputs; they are included dependencies, not newly collected measurements.

## Offline reproduction

Python 3.11+ and its standard library suffice (checked here with Python 3.14). No installed vLLM, CUDA, GPU, server, network access or historical executor is required.

From the extracted review root:

```bash
python3 -m unittest discover -s new/research/icissp2027_rdna_20261017/tools -p 'test_*.py' -v
python3 new/research/icissp2027_rdna_20261017/tools/reanalyze.py --repo .
python3 new/research/icissp2027_rdna_20261017/tools/counter_semantics.py
python3 new/research/icissp2027_rdna_20261017/tools/conditional_observer.py
```

Expected: 14 tests pass; reanalysis totals 14 episodes / 25 requests / 1,152 output IDs / 1,202 wire events / zero integrity discrepancies. The scripts rewrite their own derived JSON outputs. Paths in old provenance records describe the audit machine; the included snapshot and preserved archive layout remove those runtime dependencies.

The manuscript was built with Tectonic 0.17.0 using the official SCITEPRESS template. A normal TeX distribution with the packages named in the source can also compile it. Tectonic's binary and TeX package cache are not included: the PDF is prebuilt, and first-time typesetting may need package access. Build logs, character/font checks and template provenance are in `evidence/`. Reproducing the offline evidence does not require typesetting.

No serving experiment was added. The archive contains private research artifacts and local paths; it has not been anonymized or cleared for public distribution. The included vLLM sources retain their Apache-2.0 license.
