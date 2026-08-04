# Llama Port Audit

## Source basis

The port was derived from the uploaded Qwen closure sources whose relevant
SHA-256 identities were:

- Qwen runner: `007e7dda17ad7f6356d391b33c0efed06ce1478e65aca67e7c18189a37b3a44e`
- protected base runner: `981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75`
- active-cohort module: `57c88a8410a16d7a85432b3a3684842e813d75b4f9388392279e9bacfbc26c8e`
- Qwen schedule generator: `71fb6f25d18559de9192b83fdb4c51fa44b8df7bab158da24990dcaa3837128b`
- Qwen raw-trace auditor: `1eb5f90966041a41106faea3f503d75a33fba9679a2d62ce529e351cce853ca7`

## Deliberate changes

Only model/campaign binding and the independent schedule identity were ported:

- model key `qwen` -> `llama`
- model ID -> `meta-llama/Llama-3.1-8B-Instruct`
- block and episode IDs -> `llama_*`
- separate design version, seed namespace, seed, and schedule fingerprint
- separate script, log, result, diagnostic, freeze, and final-capture paths
- Llama-only server-launcher acceptance and Qwen rejection
- distinct runner/auditor version strings

The data-driven cohort algorithm `_active_cohort.py` remains byte-identical.
The protected base runner is not included for replacement and must match its
known SHA-256 before installation and real execution.

## Offline validation performed

- generator contract: 44/44 checks passed
- trigger/cohort timing suite: 38/38 checks passed, in addition to the embedded
  19/19 active-cohort checks
- runner/fake-server/semantic fail-closed suite: 342/342 checks passed
- independent raw-trace auditor regression suite: 21/21 tests passed
- focused Llama-port isolation checks included as `test_llama_port_contract.py`
- Python compilation and Bash syntax checks passed

No GPU, network, or real vLLM request was used for these package-build tests.
A real Llama diagnostic pair is intentionally the first post-install runtime
step before the 32-episode campaign.
