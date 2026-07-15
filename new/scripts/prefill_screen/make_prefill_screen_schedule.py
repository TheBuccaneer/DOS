#!/usr/bin/env python3
"""
make_prefill_screen_schedule.py

Generates the frozen, explorative Prefill-Screen episode schedule --
reproducibly, self-validating, and only publishing output once every
check has passed. Does NOT execute any requests -- output is a
schedule (CSV + JSON) plus a human-readable audit report.

This is a fully separate, exploratory screening design, derived
mechanically from the frozen Phase A generator (make_phase_a_schedule.py,
untouched, not imported here). It does NOT modify, extend, or share
output files with Phase A. Scientific purpose (see project prompt):
does a prefill-heavy, bounded burst (2048 input / 16 output tokens)
produce a markedly stronger state-dependent victim degradation than
the previously used mixed 256/256 burst? This is an explorative
screen, not a confirmatory study -- no policies, SLOs, mitigations,
additional models, or additional load points are part of this design.

Design (single model, "llama" only):
  - 2 states (offload0="low", offload12="high")
  - 2 victim concurrencies (4, 8)
  - 2 conditions (no_burst, prefill_burst)
  - 3 repeats (frozen)
  => 3 * 2 * 2 * 2 = 24 episodes total

Block structure:
  - Each repeat contributes two state-blocks, executed back to back.
  - State order alternates by repeat parity:
      repeat 1 (odd):  low  -> high
      repeat 2 (even): high -> low
      repeat 3 (odd):  low  -> high
    (avoids "all low episodes, then all high episodes" ordering, since a
    state switch requires a server restart.)
  - Within each repeat, the 4 (concurrency x condition) cells are
    randomized once and that same order is reused for both the
    low-state block and the high-state block of that repeat (using a
    per-model RNG seeded from (global_seed, model) so the order of
    --models on the CLI does not affect reproducibility). This means a
    matched low/high pair of a given (concurrency, condition) always has
    the same order_in_block.
  - `restart_server_before_block` flags the start of each state-block
    (order_in_block == 1), since a restart is required before *every*
    block boundary -- including consecutive same-state blocks in an
    ABBA sequence.

  The binding per-block runtime protocol, executed later by
  run_prefill_screen.py (NOT by this generator), is:
      server start
      -> API readiness check via polling (not a generation request)
      -> exactly one full stabilization run
      -> stability diagnostics saved
      -> drain/cooldown
      -> four regular episodes
  There is no separate short warmup/probe generation request in
  addition to the readiness poll and the stabilization run. This
  schedule therefore contains only the 24 regular episodes; the
  stabilization run is documented as top-level protocol metadata (see
  `STABILIZATION_CONFIGURATION` below) rather than as a schedule row,
  a repeat, or a CSV line, and this generator does not execute it.

Seeds (three distinct, deliberately separated) -- same derivation
formula and semantics as the frozen Phase A generator:
  - schedule_seed (field: random_seed): the single global seed that
    controls block/cell randomization. Identical for every row; kept as
    a reproducibility record of *how the schedule itself was generated*.
  - episode_seed: a unique, deterministic seed per episode, derived from
    (schedule_seed, episode_id).
  - victim_workload_seed: deterministic from (schedule_seed, model,
    concurrency, repeat) -- deliberately NOT from offload_gb/state_label
    NOR from condition. This makes the victim workload identical across
    both states AND both conditions of a given (concurrency, repeat)
    cell.
  - burst_workload_seed: same derivation but with an additional "burst"
    tag, so it is a distinct stream from victim_workload_seed while
    still being constant across states and conditions. A no_burst
    episode simply does not use this seed, but it is still computed for
    every episode for schedule symmetry/audit.

The Prefill-Screen design is frozen. The CLI intentionally accepts only
the one official invocation (defaults, or the same values passed
explicitly) plus a free --output-dir override; see `validate_frozen_cli`.

Usage:
    python3 make_prefill_screen_schedule.py
    python3 make_prefill_screen_schedule.py --models llama --repeats 3 \
        --seed 20260716 --output-dir /path/to/runs/prefill_screen
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Fixed experimental configuration (frozen -- do not change values here)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
DESIGN_VERSION = "prefill-screen-v1"

STATES: tuple[tuple[int, str], ...] = ((0, "low"), (12, "high"))
CONCURRENCIES: tuple[int, ...] = (4, 8)
BURST_CONDITION = "prefill_burst"
CONDITIONS: tuple[str, ...] = ("no_burst", BURST_CONDITION)

VICTIM_REQUEST_COUNT = 20
VICTIM_INPUT_LEN = 256
VICTIM_OUTPUT_LEN = 64
VICTIM_TEMPERATURE = 0.0

# Prefill-heavy, bounded burst: large input, small output -- deliberately
# different from Phase A's mixed 256/256 burst (see module docstring).
BURST_PARALLEL_REQUESTS = 4
BURST_INPUT_LEN = 2048
BURST_OUTPUT_LEN = 16
BURST_TEMPERATURE = 0.0

# Documents (but does not execute) the mandatory per-block runtime
# protocol that run_prefill_screen.py must follow:
#   server start -> API readiness check via polling (NOT a generation
#   request) -> exactly one full stabilization run -> stability
#   diagnostics saved -> drain/cooldown -> four regular episodes.
# There is deliberately no separate short warmup/probe generation
# request (generation_probe_requests is 0): the readiness poll is not
# a generation request, and it is followed directly by the one
# stabilization run. These runs are explicitly NOT part of this
# schedule: not a regular episode, not a repeat, not a CSV row. See
# module docstring. Stabilization itself is unchanged from Phase A
# (no_burst / concurrency 4 / 256 input / 64 output) -- only the
# regular-episode burst shape differs in this screen.
#
# abort_on_stability_drift is deliberately False for the pilot, so no
# block is discarded based on an arbitrarily-chosen performance
# threshold; the later runner must fully log the drift instead.
# Aborts here are reserved for functional failures only: missing
# requests, timeouts, HTTP/streaming errors, truncated outputs, or
# wrong prompt/output token counts.
STABILIZATION_CONFIGURATION: dict[str, object] = {
    "enabled": True,
    "api_readiness_check_required": True,
    "generation_probe_requests": 0,
    "stabilization_runs_per_block": 1,
    "stabilization_condition": "no_burst",
    "stabilization_concurrency": 4,
    "stabilization_request_count": 20,
    "stabilization_input_len": 256,
    "stabilization_output_len": 64,
    "stabilization_temperature": 0.0,
    "excluded_from_analysis": True,
    "counted_repeat": False,
    "separate_output_required": True,
    "block_must_abort_on_request_failure": True,
    "stability_diagnostics_required": True,
    "stability_windowing": "first_half_vs_second_half",
    "stability_primary_metric": "median_tpot_ms",
    "stability_secondary_metrics": [
        "median_ttft_ms",
        "median_e2el_ms",
    ],
    "record_relative_change": True,
    "abort_on_stability_drift": False,
}

# Frozen official invocation. Explorative Prefill-Screen: single model
# (llama only), 3 repeats, distinct schedule seed -- see module
# docstring for the full contract and its scientific rationale.
DEFAULT_MODELS = ("llama",)
DEFAULT_REPEATS = 3
DEFAULT_SEED = 20260716

# Portable default output directory, derived from this file's location.
# Expected location:
#   <PROJECT_ROOT>/new/scripts/prefill_screen/make_prefill_screen_schedule.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "new" / "runs" / "prefill_screen"


# ---------------------------------------------------------------------------
# Episode schema
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    episode_id: str
    model: str
    offload_gb: int
    state_label: str
    concurrency: int
    condition: str
    repeat: int
    random_seed: int
    episode_seed: int
    victim_workload_seed: int
    burst_workload_seed: int
    victim_request_count: int
    victim_input_len: int
    victim_output_len: int
    victim_temperature: float
    burst_parallel_requests: int
    burst_input_len: int
    burst_output_len: int
    burst_temperature: float
    restart_server_before_block: int
    block_id: str
    order_in_block: int


REQUIRED_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Episode))


# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------

def derive_seed(*parts: str) -> int:
    """
    Deterministic, process-independent derived seed (unlike Python's
    built-in hash(), which is randomized per process unless
    PYTHONHASHSEED is fixed). Returns a positive int suitable as a
    downstream RNG seed.
    """
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Schedule generation (unchanged generation logic / ordering / seeds)
# ---------------------------------------------------------------------------

def generate_schedule(model: str, repeats: int, seed: int) -> list[Episode]:
    rng = random.Random(f"{seed}:{model}")
    episodes: list[Episode] = []
    block_number = 0

    for repeat in range(1, repeats + 1):
        state_order = STATES if repeat % 2 == 1 else tuple(reversed(STATES))

        # Shuffled once per repeat and reused for both state-blocks of
        # this repeat, so a matched low/high pair of a given
        # (concurrency, condition) always lands at the same
        # order_in_block.
        cells = [
            (concurrency, condition)
            for concurrency in CONCURRENCIES
            for condition in CONDITIONS
        ]
        rng.shuffle(cells)

        for offload_gb, state_label in state_order:
            block_number += 1
            block_id = f"{model}_block{block_number:02d}_{state_label}"

            for order_in_block, (concurrency, condition) in enumerate(
                cells, start=1
            ):
                is_block_start = order_in_block == 1
                restart = 1 if is_block_start else 0

                episode_id = (
                    f"{model}_off{offload_gb}_conc{concurrency}_"
                    f"{condition}_rep{repeat}"
                )

                episode_seed = derive_seed(str(seed), episode_id)
                # Deliberately independent of offload_gb/state_label AND
                # of condition, so the victim sees identical prompt
                # content across both states and both conditions of the
                # same (concurrency, repeat) cell -- see module
                # docstring.
                victim_workload_seed = derive_seed(
                    str(seed), model, str(concurrency), str(repeat)
                )
                burst_workload_seed = derive_seed(
                    str(seed), model, str(concurrency), str(repeat), "burst"
                )

                episodes.append(
                    Episode(
                        episode_id=episode_id,
                        model=model,
                        offload_gb=offload_gb,
                        state_label=state_label,
                        concurrency=concurrency,
                        condition=condition,
                        repeat=repeat,
                        random_seed=seed,
                        episode_seed=episode_seed,
                        victim_workload_seed=victim_workload_seed,
                        burst_workload_seed=burst_workload_seed,
                        victim_request_count=VICTIM_REQUEST_COUNT,
                        victim_input_len=VICTIM_INPUT_LEN,
                        victim_output_len=VICTIM_OUTPUT_LEN,
                        victim_temperature=VICTIM_TEMPERATURE,
                        burst_parallel_requests=BURST_PARALLEL_REQUESTS,
                        burst_input_len=BURST_INPUT_LEN,
                        burst_output_len=BURST_OUTPUT_LEN,
                        burst_temperature=BURST_TEMPERATURE,
                        restart_server_before_block=restart,
                        block_id=block_id,
                        order_in_block=order_in_block,
                    )
                )

    return episodes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_schedule(
    episodes: list[Episode],
    model: str,
    repeats: int,
    seed: int,
) -> list[str]:
    errors: list[str] = []

    expected_total = len(STATES) * len(CONCURRENCIES) * len(CONDITIONS) * repeats
    if len(episodes) != expected_total:
        errors.append(
            f"model={model}: expected {expected_total} episodes, "
            f"found {len(episodes)}"
        )

    cell_counts: dict[tuple[int, int, str], int] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.concurrency, ep.condition)
        cell_counts[key] = cell_counts.get(key, 0) + 1

    expected_cells = {
        (offload_gb, concurrency, condition)
        for offload_gb, _ in STATES
        for concurrency in CONCURRENCIES
        for condition in CONDITIONS
    }

    for key in sorted(expected_cells):
        count = cell_counts.get(key, 0)
        if count != repeats:
            errors.append(
                f"model={model}: cell offload={key[0]}, concurrency={key[1]}, "
                f"condition={key[2]!r} occurs {count} time(s), expected {repeats}"
            )

    unexpected_keys = set(cell_counts) - expected_cells
    if unexpected_keys:
        errors.append(f"model={model}: unexpected cell(s): {sorted(unexpected_keys)}")

    # --- Exact repeat-set check per design cell -----------------------------
    # A correct episode COUNT per cell is not sufficient: the actual set of
    # repeat values in each cell must be exactly {1, ..., repeats}. This
    # catches e.g. every repeat-1 episode being consistently renumbered to
    # repeat 99 (with matching episode_id/seeds), which would still pass
    # the count-only check above.
    repeats_by_cell: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.concurrency, ep.condition)
        repeats_by_cell.setdefault(key, set()).add(ep.repeat)

    expected_repeat_set = set(range(1, repeats + 1))
    for key in sorted(expected_cells):
        actual_repeat_set = repeats_by_cell.get(key, set())
        if actual_repeat_set != expected_repeat_set:
            errors.append(
                f"model={model}: cell offload={key[0]}, concurrency={key[1]}, "
                f"condition={key[2]!r} has repeat value(s) "
                f"{sorted(actual_repeat_set)}, expected exactly "
                f"{sorted(expected_repeat_set)}"
            )

    design_keys = [
        (ep.offload_gb, ep.concurrency, ep.condition, ep.repeat) for ep in episodes
    ]
    if len(design_keys) != len(set(design_keys)):
        errors.append(f"model={model}: duplicate (state, concurrency, condition, repeat) combination(s) found")

    episode_ids = [ep.episode_id for ep in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append(f"model={model}: duplicate episode_id(s) found")

    episode_seeds = [ep.episode_seed for ep in episodes]
    if len(episode_seeds) != len(set(episode_seeds)):
        errors.append(f"model={model}: duplicate episode_seed(s) found")

    for ep in episodes:
        row = asdict(ep)
        missing = [field for field in REQUIRED_FIELDS if row.get(field) is None]
        if missing:
            errors.append(
                f"model={model}: episode {ep.episode_id} missing field(s): "
                f"{', '.join(missing)}"
            )

    # --- Exact frozen value checks ----------------------------------------
    valid_models = {"llama"}
    state_label_by_offload = {offload_gb: label for offload_gb, label in STATES}

    for ep in episodes:
        ctx = f"model={model}, episode={ep.episode_id}"

        if ep.model not in valid_models:
            errors.append(
                f"{ctx}: invalid model {ep.model!r}, expected one of "
                f"{sorted(valid_models)}"
            )

        expected_state_label = state_label_by_offload.get(ep.offload_gb)
        if expected_state_label is None:
            errors.append(
                f"{ctx}: invalid offload_gb {ep.offload_gb!r}, expected one "
                f"of {[o for o, _ in STATES]}"
            )
        elif ep.state_label != expected_state_label:
            errors.append(
                f"{ctx}: state_label {ep.state_label!r} does not match "
                f"offload_gb {ep.offload_gb} (expected "
                f"{expected_state_label!r})"
            )

        if ep.concurrency not in CONCURRENCIES:
            errors.append(
                f"{ctx}: invalid concurrency {ep.concurrency!r}, expected "
                f"one of {CONCURRENCIES}"
            )

        if ep.condition not in CONDITIONS:
            errors.append(
                f"{ctx}: invalid condition {ep.condition!r}, expected one "
                f"of {CONDITIONS}"
            )

        if ep.random_seed != seed:
            errors.append(
                f"{ctx}: random_seed {ep.random_seed!r} does not match the "
                f"official seed {seed!r}"
            )

        if ep.victim_request_count != VICTIM_REQUEST_COUNT:
            errors.append(
                f"{ctx}: victim_request_count {ep.victim_request_count!r} "
                f"!= {VICTIM_REQUEST_COUNT}"
            )
        if ep.victim_input_len != VICTIM_INPUT_LEN:
            errors.append(
                f"{ctx}: victim_input_len {ep.victim_input_len!r} != "
                f"{VICTIM_INPUT_LEN}"
            )
        if ep.victim_output_len != VICTIM_OUTPUT_LEN:
            errors.append(
                f"{ctx}: victim_output_len {ep.victim_output_len!r} != "
                f"{VICTIM_OUTPUT_LEN}"
            )
        if ep.victim_temperature != VICTIM_TEMPERATURE:
            errors.append(
                f"{ctx}: victim_temperature {ep.victim_temperature!r} != "
                f"{VICTIM_TEMPERATURE}"
            )

        if ep.burst_parallel_requests != BURST_PARALLEL_REQUESTS:
            errors.append(
                f"{ctx}: burst_parallel_requests "
                f"{ep.burst_parallel_requests!r} != {BURST_PARALLEL_REQUESTS}"
            )
        if ep.burst_input_len != BURST_INPUT_LEN:
            errors.append(
                f"{ctx}: burst_input_len {ep.burst_input_len!r} != "
                f"{BURST_INPUT_LEN}"
            )
        if ep.burst_output_len != BURST_OUTPUT_LEN:
            errors.append(
                f"{ctx}: burst_output_len {ep.burst_output_len!r} != "
                f"{BURST_OUTPUT_LEN}"
            )
        if ep.burst_temperature != BURST_TEMPERATURE:
            errors.append(
                f"{ctx}: burst_temperature {ep.burst_temperature!r} != "
                f"{BURST_TEMPERATURE}"
            )

        if ep.order_in_block == 1:
            if ep.restart_server_before_block != 1:
                errors.append(
                    f"{ctx}: order_in_block=1 requires "
                    f"restart_server_before_block==1, found "
                    f"{ep.restart_server_before_block!r}"
                )
        else:
            if ep.restart_server_before_block != 0:
                errors.append(
                    f"{ctx}: order_in_block={ep.order_in_block} requires "
                    f"restart_server_before_block==0, found "
                    f"{ep.restart_server_before_block!r}"
                )

        # --- Deterministic re-derivation checks --------------------------
        # A structurally-valid but wrong/arbitrary episode_id, episode_seed,
        # victim_workload_seed, or burst_workload_seed (e.g. still unique,
        # but not actually derived from the documented formula) must be
        # rejected, not just checked for uniqueness.
        if ep.model != model:
            errors.append(
                f"{ctx}: episode.model {ep.model!r} does not match the "
                f"model being validated {model!r}"
            )

        expected_episode_id = (
            f"{model}_off{ep.offload_gb}_conc{ep.concurrency}_"
            f"{ep.condition}_rep{ep.repeat}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(
                f"{ctx}: episode_id does not match the expected derivation "
                f"{expected_episode_id!r}"
            )

        expected_episode_seed = derive_seed(str(seed), ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(
                f"{ctx}: episode_seed {ep.episode_seed!r} does not match "
                f"derive_seed(seed, episode_id) = {expected_episode_seed!r}"
            )

        expected_victim_seed = derive_seed(
            str(seed), model, str(ep.concurrency), str(ep.repeat)
        )
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(
                f"{ctx}: victim_workload_seed {ep.victim_workload_seed!r} "
                f"does not match the expected derivation "
                f"{expected_victim_seed!r}"
            )

        expected_burst_seed = derive_seed(
            str(seed), model, str(ep.concurrency), str(ep.repeat), "burst"
        )
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(
                f"{ctx}: burst_workload_seed {ep.burst_workload_seed!r} "
                f"does not match the expected derivation "
                f"{expected_burst_seed!r}"
            )

    # --- Block-level checks ---------------------------------------------
    blocks: dict[str, list[Episode]] = {}
    block_ids_in_order: list[str] = []
    for ep in episodes:
        if ep.block_id not in blocks:
            blocks[ep.block_id] = []
            block_ids_in_order.append(ep.block_id)
        blocks[ep.block_id].append(ep)

    expected_blocks_per_model = 2 * repeats
    if len(block_ids_in_order) != expected_blocks_per_model:
        errors.append(
            f"model={model}: expected {expected_blocks_per_model} blocks, "
            f"found {len(block_ids_in_order)}"
        )

    expected_block_size = len(CONCURRENCIES) * len(CONDITIONS)

    for position, block_id in enumerate(block_ids_in_order, start=1):
        block_episodes = blocks[block_id]

        if len(block_episodes) != expected_block_size:
            errors.append(
                f"model={model}: block {block_id} has "
                f"{len(block_episodes)} episode(s), expected "
                f"{expected_block_size}"
            )

        order_positions = sorted(ep.order_in_block for ep in block_episodes)
        expected_positions = list(range(1, expected_block_size + 1))
        if order_positions != expected_positions:
            errors.append(
                f"model={model}: block {block_id} order_in_block values "
                f"{order_positions}, expected {expected_positions}"
            )

        restart_episodes = [
            ep for ep in block_episodes if ep.restart_server_before_block > 0
        ]
        if len(restart_episodes) != 1 or (
            restart_episodes and restart_episodes[0].order_in_block != 1
        ):
            errors.append(
                f"model={model}: block {block_id} does not have exactly "
                f"one restart_server_before_block flag at order_in_block=1"
            )

        state_labels = {ep.state_label for ep in block_episodes}
        offloads = {ep.offload_gb for ep in block_episodes}
        repeats_in_block = {ep.repeat for ep in block_episodes}

        if len(state_labels) != 1 or len(offloads) != 1:
            errors.append(
                f"model={model}: block {block_id} mixes state/offload "
                f"values: states={sorted(state_labels)}, "
                f"offloads={sorted(offloads)}"
            )

        # Block position -> expected repeat: blocks 1&2 are repeat 1,
        # blocks 3&4 are repeat 2, etc. (every repeat contributes exactly
        # two consecutive state-blocks). This checks the actual repeat
        # VALUE, not merely that the block doesn't mix repeat values.
        expected_repeat_for_block = ((position - 1) // 2) + 1
        if repeats_in_block != {expected_repeat_for_block}:
            errors.append(
                f"model={model}: block {block_id} (position {position}) "
                f"has repeat value(s) {sorted(repeats_in_block)}, "
                f"expected exactly {{{expected_repeat_for_block}}}"
            )

    # --- Contiguous block / execution-order checks --------------------------
    # The order of `episodes` (and therefore of the CSV/JSON rows) IS the
    # intended execution order. A block must be exactly 4 immediately
    # consecutive episodes in that list; a block_id must never reappear
    # once left; and order_in_block / restart_server_before_block must
    # match 1,2,3,4 / 1,0,0,0 in that exact list order -- not merely as a
    # set, and not merely within an unordered grouping by block_id (the
    # `blocks` dict above groups by block_id regardless of contiguity or
    # list position, so it cannot by itself catch swapped or interleaved
    # episodes).
    seen_block_ids_so_far: set[str] = set()
    idx = 0
    n_episodes = len(episodes)
    while idx < n_episodes:
        bid = episodes[idx].block_id
        if bid in seen_block_ids_so_far:
            errors.append(
                f"model={model}: block_id {bid!r} reappears at a later, "
                f"non-contiguous position in the episode list; a block "
                f"must not be revisited once left"
            )
        seen_block_ids_so_far.add(bid)

        run_end = idx
        while run_end < n_episodes and episodes[run_end].block_id == bid:
            run_end += 1
        run_episodes = episodes[idx:run_end]

        if len(run_episodes) != expected_block_size:
            errors.append(
                f"model={model}: contiguous block {bid!r} in the episode "
                f"list has {len(run_episodes)} immediately consecutive "
                f"episode(s), expected exactly {expected_block_size}"
            )
        else:
            order_sequence = [ep.order_in_block for ep in run_episodes]
            expected_order_sequence = list(range(1, expected_block_size + 1))
            if order_sequence != expected_order_sequence:
                errors.append(
                    f"model={model}: contiguous block {bid!r} has "
                    f"order_in_block sequence {order_sequence} in list "
                    f"order, expected exactly {expected_order_sequence} "
                    f"in that exact order"
                )

            restart_sequence = [
                ep.restart_server_before_block for ep in run_episodes
            ]
            expected_restart_sequence = [1] + [0] * (expected_block_size - 1)
            if restart_sequence != expected_restart_sequence:
                errors.append(
                    f"model={model}: contiguous block {bid!r} has "
                    f"restart_server_before_block sequence "
                    f"{restart_sequence} in list order, expected exactly "
                    f"{expected_restart_sequence} in that exact order"
                )

        idx = run_end

    # --- ABBA state-sequence + block_id format check -----------------------
    expected_state_sequence: list[str] = []
    for repeat in range(1, repeats + 1):
        order = STATES if repeat % 2 == 1 else tuple(reversed(STATES))
        expected_state_sequence.extend(label for _, label in order)

    actual_state_sequence = [
        blocks[block_id][0].state_label for block_id in block_ids_in_order
    ]

    if actual_state_sequence != expected_state_sequence:
        errors.append(
            f"model={model}: block state sequence {actual_state_sequence} "
            f"does not match expected alternating sequence "
            f"{expected_state_sequence}"
        )

    for position, block_id in enumerate(block_ids_in_order, start=1):
        if position - 1 < len(expected_state_sequence):
            expected_state = expected_state_sequence[position - 1]
            expected_block_id = f"{model}_block{position:02d}_{expected_state}"
            if block_id != expected_block_id:
                errors.append(
                    f"model={model}: block at position {position} has "
                    f"block_id {block_id!r}, expected {expected_block_id!r}"
                )

    # --- Low/High order_in_block matching check ---------------------------
    # The low-state and high-state episode of a given
    # (concurrency, condition, repeat) must share the same order_in_block,
    # since both state-blocks of a repeat reuse the same cell ordering.
    order_by_match_key: dict[tuple[int, str, int], int] = {}
    for ep in episodes:
        key = (ep.concurrency, ep.condition, ep.repeat)
        if key not in order_by_match_key:
            order_by_match_key[key] = ep.order_in_block
        elif order_by_match_key[key] != ep.order_in_block:
            errors.append(
                f"model={model}: order_in_block mismatch between matched "
                f"low/high episodes for concurrency={ep.concurrency}, "
                f"condition={ep.condition!r}, repeat={ep.repeat}: "
                f"{order_by_match_key[key]} vs {ep.order_in_block}"
            )

    # --- Workload seed pairing check --------------------------------------
    # victim_workload_seed and burst_workload_seed must each be constant
    # across all 4 matched episodes (both states x both conditions) of a
    # given (concurrency, repeat), so prompt content cannot confound the
    # state x burst comparison.
    victim_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
    burst_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
    episodes_per_cell: dict[tuple[int, int], int] = {}

    for ep in episodes:
        key = (ep.concurrency, ep.repeat)
        victim_seeds_by_cell.setdefault(key, set()).add(ep.victim_workload_seed)
        burst_seeds_by_cell.setdefault(key, set()).add(ep.burst_workload_seed)
        episodes_per_cell[key] = episodes_per_cell.get(key, 0) + 1

    expected_episodes_per_cell = len(STATES) * len(CONDITIONS)

    for key in sorted(victim_seeds_by_cell):
        concurrency, repeat = key

        if episodes_per_cell[key] != expected_episodes_per_cell:
            errors.append(
                f"model={model}: concurrency={concurrency}, repeat={repeat} "
                f"has {episodes_per_cell[key]} episode(s), expected "
                f"{expected_episodes_per_cell} (2 states x 2 conditions)"
            )

        victim_seeds = victim_seeds_by_cell[key]
        if len(victim_seeds) != 1:
            errors.append(
                f"model={model}: victim_workload_seed not constant across "
                f"states/conditions for concurrency={concurrency}, "
                f"repeat={repeat}: {sorted(victim_seeds)}"
            )

        burst_seeds = burst_seeds_by_cell[key]
        if len(burst_seeds) != 1:
            errors.append(
                f"model={model}: burst_workload_seed not constant across "
                f"states/conditions for concurrency={concurrency}, "
                f"repeat={repeat}: {sorted(burst_seeds)}"
            )

    for ep in episodes:
        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(
                f"model={model}: episode {ep.episode_id} has identical "
                f"victim_workload_seed and burst_workload_seed "
                f"({ep.victim_workload_seed}); these must be independent "
                f"seed streams"
            )

    return errors


def validate_global(
    all_episodes: list[Episode],
    models: Sequence[str],
    repeats: int,
) -> list[str]:
    """Cross-model checks that only make sense on the merged schedule."""
    errors: list[str] = []

    expected_total = (
        len(models) * len(STATES) * len(CONCURRENCIES) * len(CONDITIONS) * repeats
    )
    if len(all_episodes) != expected_total:
        errors.append(
            f"global: expected {expected_total} total episodes across "
            f"{len(models)} model(s), found {len(all_episodes)}"
        )

    all_ids = [ep.episode_id for ep in all_episodes]
    if len(all_ids) != len(set(all_ids)):
        duplicate_ids = sorted({eid for eid in all_ids if all_ids.count(eid) > 1})
        errors.append(
            f"global: duplicate episode_id(s) across merged models: "
            f"{duplicate_ids}"
        )

    all_seeds = [ep.episode_seed for ep in all_episodes]
    if len(all_seeds) != len(set(all_seeds)):
        duplicate_seeds = sorted(
            {s for s in all_seeds if all_seeds.count(s) > 1}
        )
        errors.append(
            f"global: duplicate episode_seed(s) across merged models: "
            f"{duplicate_seeds}"
        )

    # Note: block_id is intentionally shared by all episodes of the same
    # block (4 episodes per block_id), so a per-episode occurrence count
    # is not a meaningful duplicate check here. block_id already embeds
    # the model name, and per-model block_id uniqueness (one block_id
    # per distinct block) is already fully checked inside
    # validate_schedule.

    # Exactly one restart_server_before_block=1 marker per block, and one
    # block per (model, state) x repeat -- i.e. total restart markers
    # across the whole merged schedule must equal the total block count
    # (2 blocks per repeat x repeats x models; 20 at the frozen values).
    expected_restart_markers = len(models) * 2 * repeats
    actual_restart_markers = sum(
        1 for ep in all_episodes if ep.restart_server_before_block == 1
    )
    if actual_restart_markers != expected_restart_markers:
        errors.append(
            f"global: expected {expected_restart_markers} "
            f"restart_server_before_block markers, found "
            f"{actual_restart_markers}"
        )

    return errors


# ---------------------------------------------------------------------------
# Canonical payload / fingerprint
# ---------------------------------------------------------------------------

def build_canonical_payload(
    all_episodes: list[Episode],
    models: Sequence[str],
    repeats: int,
    seed: int,
) -> dict:
    """
    The single canonical payload used both to compute the schedule
    fingerprint and as the basis for the published JSON (with
    schedule_fingerprint added afterwards). Must NOT contain
    schedule_fingerprint itself, absolute paths, creation timestamps,
    or temporary file names.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "repeats": repeats,
        "models": list(models),
        "states": [{"offload_gb": o, "state_label": s} for o, s in STATES],
        "concurrencies": list(CONCURRENCIES),
        "conditions": list(CONDITIONS),
        "victim_configuration": {
            "victim_request_count": VICTIM_REQUEST_COUNT,
            "victim_input_len": VICTIM_INPUT_LEN,
            "victim_output_len": VICTIM_OUTPUT_LEN,
            "victim_temperature": VICTIM_TEMPERATURE,
        },
        "burst_configuration": {
            "burst_parallel_requests": BURST_PARALLEL_REQUESTS,
            "burst_input_len": BURST_INPUT_LEN,
            "burst_output_len": BURST_OUTPUT_LEN,
            "burst_temperature": BURST_TEMPERATURE,
        },
        "stabilization_configuration": dict(STABILIZATION_CONFIGURATION),
        "episode_count": len(all_episodes),
        "episodes": [asdict(ep) for ep in all_episodes],
    }


def compute_fingerprint(canonical_payload: dict) -> str:
    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# In-memory content builders (nothing is written to disk here)
# ---------------------------------------------------------------------------

def build_csv_text(all_episodes: list[Episode]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(REQUIRED_FIELDS))
    writer.writeheader()
    for ep in all_episodes:
        writer.writerow(asdict(ep))
    return buf.getvalue()


def build_json_text(canonical_payload: dict, fingerprint: str) -> str:
    payload = dict(canonical_payload)
    payload["schedule_fingerprint"] = fingerprint
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_audit_text(
    per_model_episodes: dict[str, list[Episode]],
    repeats: int,
    seed: int,
    all_episodes: list[Episode],
    fingerprint: str,
) -> str:
    lines: list[str] = []
    lines.append("Prefill-Screen Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"design_version: {DESIGN_VERSION}")
    lines.append(f"schedule_fingerprint: {fingerprint}")
    lines.append(f"seed: {seed}")
    lines.append(f"repeats per cell: {repeats}")
    lines.append(f"models: {', '.join(per_model_episodes.keys())}")
    lines.append(f"total episodes (all models): {len(all_episodes)}")
    lines.append("")

    for model, episodes in per_model_episodes.items():
        lines.append(f"--- {model}: PASS ---")
        lines.append(f"total episodes: {len(episodes)}")

        block_ids: list[str] = []
        for ep in episodes:
            if ep.block_id not in block_ids:
                block_ids.append(ep.block_id)
        lines.append(f"blocks: {len(block_ids)}")

        state_sequence = []
        for block_id in block_ids:
            state_sequence.append(
                next(ep.state_label for ep in episodes if ep.block_id == block_id)
            )
        lines.append(f"state sequence: {', '.join(state_sequence)}")

        cell_counts: dict[tuple[int, int, str], int] = {}
        for ep in episodes:
            key = (ep.offload_gb, ep.concurrency, ep.condition)
            cell_counts[key] = cell_counts.get(key, 0) + 1

        lines.append("cell counts (offload_gb, concurrency, condition):")
        for key in sorted(cell_counts):
            lines.append(f"  {key}: {cell_counts[key]}")

        lines.append(
            "block sequence (each entry implies a server restart before "
            "its first episode):"
        )
        for block_id in block_ids:
            state_label = next(
                ep.state_label for ep in episodes if ep.block_id == block_id
            )
            lines.append(f"  {block_id} (state={state_label})")

        lines.append("")

    lines.append("=" * 60)
    lines.append("--- global checks ---")
    lines.append("  episode_id uniqueness across merged models: PASS")
    lines.append("  episode_seed uniqueness across merged models: PASS")
    lines.append("  csv/json consistency: PASS")
    lines.append("")

    lines.append("--- stabilization protocol (documented for run_prefill_screen.py; "
                 "not executed by this generator) ---")
    lines.append(f"regular episode count: {len(all_episodes)}")
    lines.append("stabilization episodes included in regular schedule: 0")
    lines.append(
        "stabilization runs executed later per block: "
        f"{STABILIZATION_CONFIGURATION['stabilization_runs_per_block']}"
    )
    for key in sorted(STABILIZATION_CONFIGURATION):
        lines.append(f"  {key}: {STABILIZATION_CONFIGURATION[key]}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("--- publication note ---")
    lines.append(
        "The CSV, JSON, and this audit file are each replaced atomically "
        "on their own (one os.replace() per file), but the three "
        "os.replace() calls together are NOT one atomic transaction. If "
        "this process is interrupted or a later replace fails after an "
        "earlier one already succeeded, the output directory can end up "
        "with a mix of old and new files. Before use, run_prefill_screen.py "
        "must: (1) recompute the JSON's schedule_fingerprint from its own "
        "canonical payload and confirm it matches the stored "
        "schedule_fingerprint, (2) compare the CSV episodes against the "
        "JSON episodes, and (3) compare this audit report's "
        "schedule_fingerprint against the JSON's schedule_fingerprint. "
        "Any mismatch means the files were not produced by the same run "
        "and must be treated as invalid."
    )
    lines.append("")

    lines.append("=" * 60)
    lines.append("OVERALL: PASS")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV/JSON consistency check
# ---------------------------------------------------------------------------

_EPISODE_FIELD_TYPES: dict[str, type] = typing.get_type_hints(Episode)


def _normalize_csv_value(raw: str, field_name: str) -> object:
    """
    Converts a single CSV cell (always a str, as read back by
    csv.DictReader) to the type declared on the Episode dataclass for
    that field, so it can be compared against the corresponding
    json.loads()-parsed JSON value on equal footing.
    """
    field_type = _EPISODE_FIELD_TYPES[field_name]
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def check_csv_json_consistency(csv_text: str, json_text: str) -> list[str]:
    """
    Re-parses the ACTUAL serialized csv_text/json_text strings that are
    about to be published (not two independently-built in-memory
    dicts), so this check can catch a bug in the serialization itself
    -- e.g. a stray csv.writer quoting/escaping issue, or a json.dumps
    call fed the wrong object -- not just a mismatch between two
    parallel asdict() calls that would always agree by construction.
    """
    errors: list[str] = []

    try:
        csv_reader = csv.DictReader(io.StringIO(csv_text))
        csv_rows = list(csv_reader)
        csv_fieldnames = list(csv_reader.fieldnames or [])
    except csv.Error as exc:
        return [f"failed to parse csv_text as CSV: {exc}"]

    try:
        json_obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return [f"failed to parse json_text as JSON: {exc}"]

    if csv_fieldnames != list(REQUIRED_FIELDS):
        errors.append(
            f"csv header {csv_fieldnames} does not match the expected "
            f"field order {list(REQUIRED_FIELDS)}"
        )

    json_episodes = json_obj.get("episodes")
    if not isinstance(json_episodes, list):
        errors.append("json_text has no 'episodes' list")
        return errors

    if len(csv_rows) != len(json_episodes):
        errors.append(
            f"csv/json episode count mismatch: csv={len(csv_rows)}, "
            f"json={len(json_episodes)}"
        )
        return errors

    for idx, (csv_row, json_row) in enumerate(zip(csv_rows, json_episodes)):
        if not isinstance(json_row, dict) or set(json_row.keys()) != set(
            REQUIRED_FIELDS
        ):
            errors.append(
                f"episode index {idx}: json episode field names do not "
                f"match the expected schema"
            )
            continue

        normalized_csv: dict[str, object] = {}
        for field in REQUIRED_FIELDS:
            raw = csv_row.get(field)
            if raw is None:
                errors.append(
                    f"episode index {idx}: csv row missing field {field!r}"
                )
                continue
            try:
                normalized_csv[field] = _normalize_csv_value(raw, field)
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"episode index {idx} ({field}): could not parse csv "
                    f"value {raw!r} as {_EPISODE_FIELD_TYPES[field]}: {exc}"
                )

        for field, csv_value in normalized_csv.items():
            json_value = json_row[field]
            if csv_value != json_value:
                errors.append(
                    f"episode index {idx} ({field}): csv value "
                    f"{csv_value!r} != json value {json_value!r}"
                )

    return errors


# ---------------------------------------------------------------------------
# Output writing (per-file atomic replace; NOT a multi-file transaction)
# ---------------------------------------------------------------------------

def write_and_replace_output_files(
    files: list[tuple[Path, str]],
) -> None:
    """
    Writes every (path, content) pair to a temp file in the same
    directory first. Only once ALL temp files have been written
    successfully does this function start replacing the final files,
    one os.replace() call per file.

    IMPORTANT -- this is deliberately simple and does NOT provide
    multi-file atomicity: each individual os.replace() is atomic on
    POSIX filesystems, but replacing three separate files one after
    another is not a single transaction. If this process is killed,
    or a later os.replace() call fails after an earlier one already
    succeeded, the output directory can be left with a MIX of old and
    new prefill_screen_schedule.csv / .json / audit.txt files (e.g. a
    freshly-replaced CSV next to a stale JSON). This function does not
    attempt to roll back an earlier successful replace() if a later
    one fails -- doing so would require a much more complex
    transaction mechanism than this generator needs (see module
    docstring / write-up for the accepted trade-off).

    Consequently, callers (in particular run_prefill_screen.py) MUST NOT
    assume the three published files are mutually consistent just
    because a previous run of this generator exited 0. Before using
    them, run_prefill_screen.py must at minimum:
      1. recompute the JSON's schedule_fingerprint from its own
         canonical payload and confirm it matches the JSON's stored
         schedule_fingerprint,
      2. compare the CSV episodes against the JSON episodes,
      3. compare the audit report's schedule_fingerprint against the
         JSON's schedule_fingerprint.
    Any mismatch means the three files were not produced together by
    one successful run and must be treated as invalid.

    On any failure during this function, already-written temp files
    are removed on a best-effort basis and the exception is
    re-raised. No .tmp file is ever renamed into a final file unless
    the write of every temp file succeeded first, so a failure before
    any os.replace() call is reached is guaranteed not to touch the
    previously-published final files at all.
    """
    written_tmp: list[tuple[Path, Path]] = []
    try:
        for path, content in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            written_tmp.append((tmp_path, path))

        for tmp_path, final_path in written_tmp:
            os.replace(tmp_path, final_path)
    except OSError:
        for tmp_path, _ in written_tmp:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen, explorative Prefill-Screen episode "
        "schedule "
        "(no requests are executed). The design is frozen: --models, "
        "--repeats, and --seed must match the official values exactly."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Model identifiers (frozen: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Repeats per cell (frozen: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Global RNG seed (frozen: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated CSV/JSON/audit files "
        f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def validate_frozen_cli(args: argparse.Namespace) -> str | None:
    """
    The Prefill-Screen design is frozen: --models/--repeats/--seed may be
    omitted (using the frozen defaults) or passed explicitly, but only
    if they match the official values exactly -- same models, same
    order, same count, same repeats, same seed. Returns an error
    message, or None if the invocation is the official one.
    """
    if list(args.models) != list(DEFAULT_MODELS):
        return (
            f"--models must be exactly {list(DEFAULT_MODELS)} in this "
            f"order; got {list(args.models)}"
        )
    if args.repeats != DEFAULT_REPEATS:
        return f"--repeats must be exactly {DEFAULT_REPEATS}; got {args.repeats}"
    if args.seed != DEFAULT_SEED:
        return f"--seed must be exactly {DEFAULT_SEED}; got {args.seed}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    frozen_error = validate_frozen_cli(args)
    if frozen_error:
        print(f"ERROR: {frozen_error}", file=sys.stderr)
        print(
            "The Prefill-Screen design is frozen. Allowed invocation: "
            f"--models {' '.join(DEFAULT_MODELS)} --repeats {DEFAULT_REPEATS} "
            f"--seed {DEFAULT_SEED} [--output-dir DIR]",
            file=sys.stderr,
        )
        return 1

    per_model_episodes: dict[str, list[Episode]] = {}
    per_model_errors: dict[str, list[str]] = {}
    all_episodes: list[Episode] = []

    for model in args.models:
        episodes = generate_schedule(model, args.repeats, args.seed)
        errors = validate_schedule(episodes, model, args.repeats, args.seed)

        per_model_episodes[model] = episodes
        per_model_errors[model] = errors
        all_episodes.extend(episodes)

    global_errors = validate_global(all_episodes, args.models, args.repeats)

    any_errors = (
        any(errors for errors in per_model_errors.values())
        or bool(global_errors)
    )

    # --- Print a validation summary to the console regardless of outcome --
    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print("FAIL" if any_errors else "PASS")

    for model in args.models:
        print(f"{model}: {len(per_model_episodes[model])} episodes")
        for err in per_model_errors[model]:
            print(f"  - {err}")

    if global_errors:
        print("global:")
        for err in global_errors:
            print(f"  - {err}")

    if any_errors:
        print()
        print(
            "FAIL: schedule validation failed. No output files were "
            "written or replaced.",
            file=sys.stderr,
        )
        return 1

    # --- Build all output content fully in memory before touching disk ----
    canonical_payload = build_canonical_payload(
        all_episodes, args.models, args.repeats, args.seed
    )
    fingerprint = compute_fingerprint(canonical_payload)

    csv_text = build_csv_text(all_episodes)
    json_text = build_json_text(canonical_payload, fingerprint)
    audit_text = build_audit_text(
        per_model_episodes, args.repeats, args.seed, all_episodes, fingerprint
    )

    consistency_errors = check_csv_json_consistency(csv_text, json_text)
    if consistency_errors:
        print()
        print("FAIL: csv/json consistency check failed. No output files "
              "were written or replaced.", file=sys.stderr)
        for err in consistency_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    out_csv = args.output_dir / "prefill_screen_schedule.csv"
    out_json = args.output_dir / "prefill_screen_schedule.json"
    out_audit = args.output_dir / "prefill_screen_schedule_audit.txt"

    try:
        write_and_replace_output_files(
            [
                (out_csv, csv_text),
                (out_json, json_text),
                (out_audit, audit_text),
            ]
        )
    except OSError as exc:
        print(
            f"ERROR: failed to write/replace one or more output files: "
            f"{exc}. The three files are not a single atomic "
            f"transaction -- if this happened after an earlier "
            f"os.replace() already succeeded, the output directory may "
            f"now contain a mix of old and new files; verify with the "
            f"fingerprint before trusting any of them.",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"schedule_fingerprint: {fingerprint}")
    print()
    print("Generated files:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_audit}")
    print()
    print("PASS: schedule generation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
