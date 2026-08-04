# Qwen Server-WAITING Paper Artifacts

This directory contains the finalized derived artifacts for the Qwen
Server-WAITING campaign.

## Immutable measurement evidence

The original campaign evidence remains under:

`new/runs/server_waiting_confirmation/qwen/final_capture_20260803T183032Z`

It must not be edited.

## Contents

- `reaudit/`: complete read-only campaign re-audit
- `paper_results/`: generated tables, figures, and paper-safe claims
- `prior_checks/`: earlier compact post-run report
- `tools/`: launch, re-audit, and paper-export scripts
- `archives/`: portable result bundles and SHA-256 files
- `provenance/`: source paths and Git provenance

## Scientific boundary

A/B/C describe client-visible output timing only. They do not identify
internal scheduler admission or internal prefill-start time.

Paired episodes use dynamically identified active cohorts. Their concrete
request indices can differ, so effect values compare cohort medians rather
than identity-matched requests.
