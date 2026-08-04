# Llama Server-WAITING Confirmation Port

This is a separate Llama-only port of the audited Qwen Server-WAITING runner.
It does not overwrite or import the Qwen Server-WAITING directory.

## Frozen experimental grid

- model: `meta-llama/Llama-3.1-8B-Instruct`
- CPU offload: `0`, `12` GiB
- `max_num_seqs`: `4`, `8`
- conditions: `no_burst`, `prefill_burst`
- trigger: 16 received decode tokens for every member of the dynamically
  identified active cohort
- four paired repeats per cell
- 32 episodes / 16 blocks

Schedule fingerprint:

`sha256:eaf7d27c0e1d5d19cae5fa5ea7e4c5495cd1391692e382ecf4dd0c757c1754e0`

Seed and namespace:

- seed: `20260804`
- design: `server-waiting-confirmation-llama-v1`
- namespace: `server-waiting-confirmation-llama`

## Safety

No command starts a measurement unless `--diagnostic`, `--campaign`, or the
explicit `--run-and-shutdown` option is supplied. The Qwen campaign tree is
not modified.

## Normal workflow

```bash
bash tools/llama_server_waiting_control.sh --check
bash tools/llama_server_waiting_control.sh --diagnostic
bash tools/llama_server_waiting_control.sh --freeze
bash tools/llama_server_waiting_control.sh --campaign
bash tools/llama_server_waiting_control.sh --reaudit
```

The official full campaign requires a verified freeze created by `--freeze`.
The freeze captures byte-identical code and schedule copies, hashes, Git state,
and software/GPU versions.

## Scope boundary

A/B/C describe client-visible output timing. They do not reveal internal
scheduler admission or internal prefill-start time. Active cohorts are
identified from observed token timing and are not assumed from request index.
