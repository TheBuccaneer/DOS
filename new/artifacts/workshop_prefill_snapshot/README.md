# Workshop Prefill Snapshot

## Purpose

This directory freezes the implementation and design artifacts used for the
workshop study on selective decode interference under chunked prefill and CPU
weight offloading.

The snapshot provides a stable, auditable reference before the first real
chunk-budget smoke test and before further project cleanup.

## Scientific scope

The current study examines:

- selective harm to requests that are already decoding when a prefill burst
  arrives (`active wave`);
- later request waves as a negative control;
- the visibility of selective stalls in aggregate latency metrics;
- the scaling of the effect across CPU-offload runtime regimes;
- the cost-benefit behavior of static chunked-prefill token budgets.

The current chunk-budget screen evaluates:

- runtime states: low and high;
- `max_num_batched_tokens`: 512, 1024, and 2048;
- conditions: `no_burst` and `prefill_burst`;
- three repeats per state and budget;
- 18 paired blocks and 36 total episodes.

## Frozen chunk-budget schedule

Official schedule fingerprint:

```text
sha256:3c35a72ee08e289258c254be6be9566d299b95ba8a2b38817be36bd4c79459cd
```

The schedule uses:

- a deterministic seed;
- globally balanced condition-first ordering: 9 `no_burst` first and
  9 `prefill_burst` first;
- a 2/1 condition-first distribution inside each state-budget cell;
- Latin-square budget rotation across repeats;
- substantially interleaved low/high runtime-state blocks.

## Directory contents

### `runner/`

Frozen copies of the chunk-budget runner:

- `make_chunk_budget_schedule.py`
- `run_chunk_budget_screen.py`
- `run_chunk_budget_screen.sh`
- `run_server.sh`

The active development copies remain under:

```text
new/scripts/chunk_budget_screen/
```

### `schedules/`

Frozen schedules and schedule audits for:

- the original prefill screen;
- the chunk-budget screen.

### `audits/`

- chunked-prefill configuration audit;
- last official chunk-budget runner output.

### `examples/`

- example `prefill_burst` episode artifact.

### `SHA256SUMS.txt`

SHA-256 checksums for the frozen snapshot files. This manifest must be
regenerated whenever the snapshot contents intentionally change.

## Validation status

The chunk-budget implementation has passed:

- Python bytecode compilation for both Python files;
- `bash -n` validation for both shell scripts;
- deterministic schedule generation;
- structural schedule validation;
- dry-run validation against the frozen schedule.

The following properties were checked statically or using synthetic log data:

- budget propagation through `MAX_NUM_BATCHED_TOKENS`;
- use of chunked prefill;
- server-log budget verification;
- schedule dimensions and fingerprint agreement;
- storage of burst makespan and burst aggregates.

## Measurement status

At the time of this snapshot:

- no real vLLM server was started by the final validation procedure;
- no HTTP measurement traffic was sent;
- no real chunk-budget smoke test was completed;
- no new scientific measurement result was produced.

The next experimental step is a one-block smoke test before executing the
complete 36-episode schedule.

## Known limitations before the smoke test

- The chunked-prefill server-log parser has not yet been validated against the
  exact output of a real vLLM server start.
- The optional `chunk_budget_screen_tests/` package does not currently exist.
- The runner's `--self-test` therefore reports this absence instead of
  providing the broader test coverage available for some sister runners.
- Any intentional schedule change requires regeneration of the schedule,
  update of the runner's official fingerprint, and regeneration of this
  snapshot's checksum manifest.

## Snapshot policy

Files in this directory are reference copies, not active development files.

Do not edit them silently. For an intentional new snapshot:

1. update and validate the active source files;
2. copy the selected artifacts into this directory;
3. update this README where necessary;
4. regenerate `SHA256SUMS.txt`;
5. record the new manifest checksum in the project documentation.
