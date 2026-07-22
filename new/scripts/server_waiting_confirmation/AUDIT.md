# Server-WAITING Raw-Trace Auditor v1 — Final Patch Audit

**Date:** 21 July 2026  
**Scope:** one diagnostic pair only; the 32-episode campaign remains prohibited.

## Final status

This patch closes the two independently reproduced remaining code gaps and the delivery gap:

1. The persisted provenance hashes are now compared against SHA-256 values recomputed from the actual runner, protected base runner, cohort module, launch scripts, and the three schedule files. A jointly rewritten manifest/fingerprint/integrity chain no longer passes.
2. Role-specific first/last-token fields are mandatory strict integers and must equal the raw-SSE reconstruction. Missing or malformed `last_token_receive_ns` on active victims, non-active victims, or bursts fails closed.
3. A permanent test now creates a complete diagnostic tree with the real runner and its fake adapters, audits it, proves read-only behavior, and rejects a jointly rewritten provenance chain.
4. The protected base runner is included byte-identically in the final repository bundle.
5. The package-wide `SHA256SUMS.txt` covers every delivered file except itself.

## Independent test results

```text
Python compilation:          PASS
Shell syntax:                PASS
_active_cohort.py:            19/19
Runner self-test:             20/20
Schedule:                     44/44
Trigger/timing:               38/38
Runner/fake integration:     342/342
Raw-trace auditor:            21/21
unittest discover:            24 tests, OK
```

## Protected identities

```text
57c88a8410a16d7a85432b3a3684842e813d75b4f9388392279e9bacfbc26c8e  _active_cohort.py
71fb6f25d18559de9192b83fdb4c51fa44b8df7bab158da24990dcaa3837128b  generate_server_waiting_schedule.py
981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75  run_prefill_confirmation.py
```

## Required final operational gate

The real pair may be accepted only after an independent audit of this exact ZIP and then a real auditor result with:

```text
exit code = 0
overall_audit_status = PASS
scientifically_evaluable = true
diagnostic_tree_read_only_verified = true
```

No real GPU, vLLM server, tokenizer download, or network run was performed during coding.
