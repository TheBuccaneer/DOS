# Patch Summary — Raw-Trace Auditor v1 Final Closure

## Current versions

```text
RUNNER_VERSION  = run_server_waiting_confirmation-v4
AUDITOR_VERSION = audit_server_waiting_diagnostic-v1
```

## Changed files

```text
audit_server_waiting_diagnostic.py
test_audit_server_waiting_diagnostic.py
AUDIT.md
PATCH_SUMMARY.md
RAW_TRACE_AUDITOR_V1_FINAL_TEST_RESULTS_2026-07-21.txt
RAW_TRACE_AUDITOR_V1_FINAL_FULL_TEST_LOG_2026-07-21.txt
SHA256SUMS.txt
PACKAGE_STATUS.md
```

## Code changes

### Actual source and schedule hash binding

The production auditor now hashes exactly:

```text
run_server_waiting_confirmation.py
run_server_waiting_confirmation.sh
run_server_waiting_server.sh
_active_cohort.py
run_prefill_confirmation.py
server_waiting_confirmation_schedule.json
server_waiting_confirmation_schedule.csv
server_waiting_confirmation_schedule_audit.txt
```

The recomputed set and digests must strictly equal `diagnostic_run_manifest.file_hashes`. The environment fingerprint is still reconstructed, but no longer from untrusted stored hashes alone.

### Mandatory token-time aliases

Victims require strict integer values for:

```text
first_token_receive_ns
first_token_perf_ns
last_token_receive_ns
```

Bursts require strict integer values for:

```text
first_token_receive_ns
burst_first_token_perf_ns
last_token_receive_ns
```

All must exactly equal the raw-SSE reconstruction. Optional additional aliases, when present, are also strictly validated.

### Permanent real-runner integration

The auditor suite now creates a full diagnostic pair using `run_diagnostic_pair()` with the real runner and official fake adapters. It proves PASS/read-only on the valid tree and FAIL on a jointly rewritten provenance chain.

## Test totals

```text
342/342 existing runner checks
21/21 auditor tests
24 unittest-discover tests
```

No scientific protocol, schedule, cohort, trigger, launcher, or official fingerprint was changed.
