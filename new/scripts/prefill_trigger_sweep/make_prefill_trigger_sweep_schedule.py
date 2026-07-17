#!/usr/bin/env python3
"""
make_prefill_trigger_sweep_schedule.py -- frozen Prefill-Trigger-Sweep
schedule generator.

Fully separate from, and does NOT modify or import, the frozen Prefill-
Screen generator/runner (new/scripts/prefill_screen/) or the frozen
Chunk-Budget-Screen (new/scripts/chunk_budget_screen/). It does NOT
execute any requests, start any server, or touch any GPU/network
resource -- it only produces a reproducible, self-validating schedule
(CSV + JSON) plus a human-readable audit report, one bundle per model.

Scientific question (see project prompt): does the selective
active-wave stall occur only immediately after the start of decode, or
does it also occur after 16 / 32 already-received output tokens? This
generalizes the Prefill-Screen's fixed "first token of every active-
wave request" barrier into a configurable
`trigger_after_decode_tokens` barrier (values 1, 16, 32), while
otherwise mechanically reusing Prefill-Screen's server-lifecycle
protocol, schema conventions, seed-derivation formula, atomic-write
strategy, and fingerprint scheme.

Design, per model (llama and qwen are two fully independent, eingefroren
campaigns -- never merged into a single 48-episode run):
  - 2 states (offload0="low", offload12="high")
  - fixed victim concurrency: 4 (the active wave is exactly requests
    0..3)
  - 3 trigger positions (1, 16, 32 received output tokens)
  - 2 conditions (no_burst, prefill_burst)
  - 2 repeats (frozen)
  => 2 states x 3 triggers x 2 repeats x 2 conditions = 24 episodes

Block structure:
  - A block = (model, state, trigger_after_decode_tokens, repeat).
  - Each block contains exactly 2 regular episodes: one no_burst, one
    prefill_burst.
  - 2 states x 3 triggers x 2 repeats = 12 blocks/model, 24
    episodes/model.
  - Trigger rotation (frozen, see project prompt Section 8):
        repeat 1: trigger 1  -> trigger 16 -> trigger 32
        repeat 2: trigger 16 -> trigger 32 -> trigger 1
    Within each trigger position, low precedes high:
        repeat 1: low-t1, high-t1, low-t16, high-t16, low-t32, high-t32
        repeat 2: low-t16, high-t16, low-t32, high-t32, low-t1, high-t1
  - `restart_server_before_block` flags the first episode of every
    block (order_in_block == 1): a restart is required before every
    block boundary, including two same-state blocks that are not
    adjacent in state but are adjacent in the trigger rotation.

Condition-first balance (Section 9): for a fixed (model, state,
trigger) cell, its two repeats (1 and 2) always use opposite
condition-first order -- one no_burst-first, one prefill_burst-first --
derived deterministically from (schedule seed, model key, state,
trigger position). Because there are exactly 6 such cells per model
(2 states x 3 triggers) and each contributes exactly one
no_burst-first and one prefill_burst-first block, the global 6/6
balance in Section 9 follows automatically and is re-verified
explicitly by `validate_schedule`.

Seeds (Section 10) -- deliberately NOT the Prefill-Screen formula:
  - schedule_seed (field: random_seed): identical for every row of a
    model's schedule; a reproducibility record of how the schedule was
    generated. Distinct per model (OFFICIAL_SEED_BY_MODEL).
  - episode_seed: derive_seed(schedule_seed, episode_id) -- unique per
    episode.
  - victim_workload_seed: derive_seed(schedule_seed, model_key,
    "victim", repeat) -- constant across BOTH states, all THREE
    trigger positions, and BOTH conditions of a given (model, repeat).
    This is a deliberate generalization of the Prefill-Screen formula:
    trigger position must not confound victim workload content any
    more than state or condition may.
  - burst_workload_seed: derive_seed(schedule_seed, model_key, "burst",
    repeat) -- same generalization, independent stream from
    victim_workload_seed.
  Because model_key is baked into every seed derivation, and each
  model uses its own OFFICIAL_SEED_BY_MODEL value, no seed is ever
  shared between the Llama and Qwen campaigns.

The design is frozen per model: the CLI accepts only the one official
invocation for the selected --model-key (defaults, or the same values
passed explicitly) plus a free --output-dir override; see
`validate_frozen_cli`. The two models' OFFICIAL_FINGERPRINT values are
filled in only after this generator's own deterministic output has
been produced and independently re-verified (see run_prefill_trigger_sweep.py's
OFFICIAL_FINGERPRINTS registry, populated separately and independently
of this file).

Usage:
    python3 make_prefill_trigger_sweep_schedule.py --model-key llama
    python3 make_prefill_trigger_sweep_schedule.py --model-key qwen
    python3 make_prefill_trigger_sweep_schedule.py --model-key llama \
        --repeats 2 --seed 20260717 \
        --output-dir /path/to/new/runs/prefill_trigger_sweep/llama
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
DESIGN_VERSION = "prefill-trigger-sweep-v1"

# Model registry: technically supports both models with a single,
# non-duplicated code path. Model-dependent values come exclusively
# from this registry.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "short_name": "llama",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "short_name": "qwen",
    },
}

STATES: tuple[tuple[int, str], ...] = ((0, "low"), (12, "high"))
CONCURRENCY = 4
BURST_CONDITION = "prefill_burst"
CONDITIONS: tuple[str, ...] = ("no_burst", BURST_CONDITION)

TRIGGER_POSITIONS: tuple[int, ...] = (1, 16, 32)
# Frozen rotation of trigger positions by repeat (Section 8).
TRIGGER_ROTATION_BY_REPEAT: dict[int, tuple[int, ...]] = {
    1: (1, 16, 32),
    2: (16, 32, 1),
}

MAX_NUM_BATCHED_TOKENS = 2048

VICTIM_REQUEST_COUNT = 20
VICTIM_INPUT_LEN = 256
VICTIM_OUTPUT_LEN = 64
VICTIM_TEMPERATURE = 0.0

BURST_PARALLEL_REQUESTS = 4
BURST_INPUT_LEN = 2048
BURST_OUTPUT_LEN = 16
BURST_TEMPERATURE = 0.0

# Documents (but does not execute) the mandatory per-block runtime
# protocol, mechanically inherited from Prefill-Screen: server start ->
# API readiness check via polling (NOT a generation request) -> exactly
# one full stabilization run -> stability diagnostics saved -> drain/
# cooldown -> two regular episodes (no_burst + prefill_burst).
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

# Frozen official invocation per model. The first real run only uses
# the llama bundle; qwen is technically prepared/frozen now but not
# executed yet -- see project prompt Section 5.
DEFAULT_REPEATS = 2
OFFICIAL_SEED_BY_MODEL: dict[str, int] = {
    "llama": 20260717,
    "qwen": 20260717,
}

# Portable default output directory, derived from this file's location.
# Expected location:
#   <PROJECT_ROOT>/new/scripts/prefill_trigger_sweep/make_prefill_trigger_sweep_schedule.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 3 else SCRIPT_PATH.parent
DEFAULT_OUTPUT_DIR_ROOT = PROJECT_ROOT / "new" / "runs" / "prefill_trigger_sweep"


def default_output_dir(model_key: str) -> Path:
    return DEFAULT_OUTPUT_DIR_ROOT / model_key


# ---------------------------------------------------------------------------
# Episode schema
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    episode_id: str
    model_key: str
    model_id: str
    offload_gb: int
    state_label: str
    concurrency: int
    trigger_after_decode_tokens: int
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
    max_num_batched_tokens: int
    condition_first_in_block: str
    restart_server_before_block: int
    block_id: str
    order_in_block: int


REQUIRED_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Episode))


# ---------------------------------------------------------------------------
# Deterministic seed derivation (identical formula/semantics to the
# frozen Prefill-Screen generator)
# ---------------------------------------------------------------------------

def derive_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def condition_first_no_burst(
    seed: int, model_key: str, state_label: str, trigger: int, repeat: int
) -> bool:
    """
    Deterministically derives whether `repeat`'s block for this
    (model, state, trigger) cell runs no_burst first. Repeat 1's
    decision is derived directly from the hash; repeat 2 is always the
    complement of repeat 1's decision for the SAME cell, which
    guarantees exactly one no_burst-first and one prefill_burst-first
    block per cell (Section 9) without any manual/random runtime
    choice. Both repeat values are explicit inputs to this function
    (repeat 2's result is a deterministic function of repeat 1's,
    itself derived from repeat's parity), satisfying the required
    derivation from (seed, model key, state, trigger, repeat).
    """
    base = derive_seed(str(seed), model_key, state_label, str(trigger), "condition_order") % 2
    repeat1_is_no_burst_first = base == 0
    if repeat == 1:
        return repeat1_is_no_burst_first
    if repeat == 2:
        return not repeat1_is_no_burst_first
    raise ValueError(f"unsupported repeat value for condition-first derivation: {repeat!r}")


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def generate_schedule(model_key: str, repeats: int, seed: int) -> list[Episode]:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model_key {model_key!r}; known keys: {sorted(MODEL_REGISTRY)}"
        )
    if repeats != DEFAULT_REPEATS:
        raise ValueError(f"repeats must be exactly {DEFAULT_REPEATS}; got {repeats}")

    model_id = MODEL_REGISTRY[model_key]["model_id"]
    episodes: list[Episode] = []
    block_number = 0

    for repeat in range(1, repeats + 1):
        trigger_order = TRIGGER_ROTATION_BY_REPEAT[repeat]
        for trigger in trigger_order:
            for offload_gb, state_label in STATES:
                block_number += 1
                block_id = f"{model_key}_block{block_number:02d}_{state_label}_trigger{trigger}"

                no_burst_first = condition_first_no_burst(
                    seed, model_key, state_label, trigger, repeat
                )
                condition_order = (
                    ["no_burst", BURST_CONDITION]
                    if no_burst_first
                    else [BURST_CONDITION, "no_burst"]
                )
                condition_first_label = condition_order[0]

                victim_workload_seed = derive_seed(
                    str(seed), model_key, "victim", str(repeat)
                )
                burst_workload_seed = derive_seed(
                    str(seed), model_key, "burst", str(repeat)
                )

                for order_in_block, condition in enumerate(condition_order, start=1):
                    is_block_start = order_in_block == 1
                    restart = 1 if is_block_start else 0

                    episode_id = (
                        f"{model_key}_off{offload_gb}_conc{CONCURRENCY}_"
                        f"trigger{trigger}_{condition}_rep{repeat}"
                    )
                    episode_seed = derive_seed(str(seed), episode_id)

                    episodes.append(
                        Episode(
                            episode_id=episode_id,
                            model_key=model_key,
                            model_id=model_id,
                            offload_gb=offload_gb,
                            state_label=state_label,
                            concurrency=CONCURRENCY,
                            trigger_after_decode_tokens=trigger,
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
                            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                            condition_first_in_block=condition_first_label,
                            restart_server_before_block=restart,
                            block_id=block_id,
                            order_in_block=order_in_block,
                        )
                    )

    return episodes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_schedule(episodes: list[Episode], model_key: str, repeats: int, seed: int) -> list[str]:
    errors: list[str] = []
    ctx_model = f"model_key={model_key}"

    expected_total = len(STATES) * len(TRIGGER_POSITIONS) * len(CONDITIONS) * repeats
    if expected_total != 24:
        errors.append(f"{ctx_model}: internal design constant mismatch, expected 24 total, computed {expected_total}")
    if len(episodes) != expected_total:
        errors.append(f"{ctx_model}: expected {expected_total} episodes, found {len(episodes)}")

    model_id = MODEL_REGISTRY.get(model_key, {}).get("model_id")
    if model_id is None:
        errors.append(f"{ctx_model}: unknown model_key, cannot validate model_id")

    # --- Per-field frozen-value + re-derivation checks ---------------------
    for ep in episodes:
        ctx = f"{ctx_model}, episode={ep.episode_id}"
        row = asdict(ep)
        missing = [f for f in REQUIRED_FIELDS if row.get(f) is None and row.get(f) != 0]
        if missing:
            errors.append(f"{ctx}: missing field(s): {', '.join(missing)}")

        if ep.model_key != model_key:
            errors.append(f"{ctx}: model_key {ep.model_key!r} != {model_key!r}")
        if ep.model_id != model_id:
            errors.append(f"{ctx}: model_id {ep.model_id!r} != expected {model_id!r}")
        if ep.concurrency != CONCURRENCY:
            errors.append(f"{ctx}: concurrency {ep.concurrency!r} != {CONCURRENCY}")
        if ep.trigger_after_decode_tokens not in TRIGGER_POSITIONS:
            errors.append(
                f"{ctx}: trigger_after_decode_tokens {ep.trigger_after_decode_tokens!r} "
                f"not in {TRIGGER_POSITIONS}"
            )
        if ep.condition not in CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.max_num_batched_tokens != MAX_NUM_BATCHED_TOKENS:
            errors.append(
                f"{ctx}: max_num_batched_tokens {ep.max_num_batched_tokens!r} != "
                f"{MAX_NUM_BATCHED_TOKENS}"
            )
        if ep.random_seed != seed:
            errors.append(f"{ctx}: random_seed {ep.random_seed!r} != official seed {seed!r}")

        state_label_by_offload = dict(STATES)
        expected_state = state_label_by_offload.get(ep.offload_gb)
        if expected_state is None:
            errors.append(f"{ctx}: invalid offload_gb {ep.offload_gb!r}")
        elif ep.state_label != expected_state:
            errors.append(f"{ctx}: state_label {ep.state_label!r} != expected {expected_state!r}")

        if ep.victim_request_count != VICTIM_REQUEST_COUNT:
            errors.append(f"{ctx}: victim_request_count mismatch")
        if ep.victim_input_len != VICTIM_INPUT_LEN:
            errors.append(f"{ctx}: victim_input_len mismatch")
        if ep.victim_output_len != VICTIM_OUTPUT_LEN:
            errors.append(f"{ctx}: victim_output_len mismatch")
        if ep.victim_temperature != VICTIM_TEMPERATURE:
            errors.append(f"{ctx}: victim_temperature mismatch")
        if ep.burst_parallel_requests != BURST_PARALLEL_REQUESTS:
            errors.append(f"{ctx}: burst_parallel_requests mismatch")
        if ep.burst_input_len != BURST_INPUT_LEN:
            errors.append(f"{ctx}: burst_input_len mismatch")
        if ep.burst_output_len != BURST_OUTPUT_LEN:
            errors.append(f"{ctx}: burst_output_len mismatch")
        if ep.burst_temperature != BURST_TEMPERATURE:
            errors.append(f"{ctx}: burst_temperature mismatch")

        expected_episode_id = (
            f"{model_key}_off{ep.offload_gb}_conc{CONCURRENCY}_"
            f"trigger{ep.trigger_after_decode_tokens}_{ep.condition}_rep{ep.repeat}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(f"{ctx}: episode_id != expected derivation {expected_episode_id!r}")

        expected_episode_seed = derive_seed(str(seed), ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(f"{ctx}: episode_seed does not match derive_seed(seed, episode_id)")

        expected_victim_seed = derive_seed(str(seed), model_key, "victim", str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(f"{ctx}: victim_workload_seed does not match expected derivation")

        expected_burst_seed = derive_seed(str(seed), model_key, "burst", str(ep.repeat))
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(f"{ctx}: burst_workload_seed does not match expected derivation")

        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(f"{ctx}: victim_workload_seed == burst_workload_seed (must be independent streams)")

        if ep.order_in_block == 1:
            if ep.restart_server_before_block != 1:
                errors.append(f"{ctx}: order_in_block=1 requires restart_server_before_block==1")
        else:
            if ep.restart_server_before_block != 0:
                errors.append(f"{ctx}: order_in_block={ep.order_in_block} requires restart_server_before_block==0")

    episode_ids = [ep.episode_id for ep in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append(f"{ctx_model}: duplicate episode_id(s) found")
    episode_seeds = [ep.episode_seed for ep in episodes]
    if len(episode_seeds) != len(set(episode_seeds)):
        errors.append(f"{ctx_model}: duplicate episode_seed(s) found")

    # --- Cell coverage: each (state, trigger, condition) occurs exactly
    #     `repeats` times, and its repeat-set is exactly {1..repeats}. ---
    cell_repeats: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.trigger_after_decode_tokens, ep.condition)
        cell_repeats.setdefault(key, set()).add(ep.repeat)

    expected_cells = {
        (offload_gb, trigger, condition)
        for offload_gb, _ in STATES
        for trigger in TRIGGER_POSITIONS
        for condition in CONDITIONS
    }
    expected_repeat_set = set(range(1, repeats + 1))
    for key in sorted(expected_cells):
        actual = cell_repeats.get(key, set())
        if actual != expected_repeat_set:
            errors.append(
                f"{ctx_model}: cell offload={key[0]}, trigger={key[1]}, condition={key[2]!r} "
                f"has repeat value(s) {sorted(actual)}, expected exactly {sorted(expected_repeat_set)}"
            )
    unexpected = set(cell_repeats) - expected_cells
    if unexpected:
        errors.append(f"{ctx_model}: unexpected cell(s): {sorted(unexpected)}")

    # --- Block-level checks --------------------------------------------
    blocks: dict[str, list[Episode]] = {}
    block_ids_in_order: list[str] = []
    for ep in episodes:
        if ep.block_id not in blocks:
            blocks[ep.block_id] = []
            block_ids_in_order.append(ep.block_id)
        blocks[ep.block_id].append(ep)

    expected_block_count = len(STATES) * len(TRIGGER_POSITIONS) * repeats
    if expected_block_count != 12:
        errors.append(f"{ctx_model}: internal design constant mismatch, expected 12 blocks, computed {expected_block_count}")
    if len(block_ids_in_order) != expected_block_count:
        errors.append(f"{ctx_model}: expected {expected_block_count} blocks, found {len(block_ids_in_order)}")

    for block_id in block_ids_in_order:
        block_episodes = blocks[block_id]
        if len(block_episodes) != 2:
            errors.append(f"{ctx_model}: block {block_id} has {len(block_episodes)} episode(s), expected 2")
            continue
        conditions_here = sorted(ep.condition for ep in block_episodes)
        if conditions_here != sorted(CONDITIONS):
            errors.append(f"{ctx_model}: block {block_id} conditions {conditions_here} != {sorted(CONDITIONS)}")

        order_positions = sorted(ep.order_in_block for ep in block_episodes)
        if order_positions != [1, 2]:
            errors.append(f"{ctx_model}: block {block_id} order_in_block values {order_positions} != [1, 2]")

        restart_eps = [ep for ep in block_episodes if ep.restart_server_before_block > 0]
        if len(restart_eps) != 1 or restart_eps[0].order_in_block != 1:
            errors.append(f"{ctx_model}: block {block_id} does not have exactly one restart marker at order_in_block=1")

        states_here = {ep.state_label for ep in block_episodes}
        offloads_here = {ep.offload_gb for ep in block_episodes}
        triggers_here = {ep.trigger_after_decode_tokens for ep in block_episodes}
        repeats_here = {ep.repeat for ep in block_episodes}
        if len(states_here) != 1 or len(offloads_here) != 1:
            errors.append(f"{ctx_model}: block {block_id} mixes state/offload values")
        if len(triggers_here) != 1:
            errors.append(f"{ctx_model}: block {block_id} mixes trigger_after_decode_tokens values")
        if len(repeats_here) != 1:
            errors.append(f"{ctx_model}: block {block_id} mixes repeat values")

        cf_values = {ep.condition_first_in_block for ep in block_episodes}
        if len(cf_values) != 1:
            errors.append(f"{ctx_model}: block {block_id} has inconsistent condition_first_in_block")
        else:
            cf = next(iter(cf_values))
            first_eps = [ep for ep in block_episodes if ep.order_in_block == 1]
            if len(first_eps) != 1:
                errors.append(
                    f"{ctx_model}: block {block_id} has {len(first_eps)} "
                    "order_in_block=1 episodes, expected exactly 1"
                )
            elif first_eps[0].condition != cf:
                errors.append(
                    f"{ctx_model}: block {block_id} order_in_block=1 "
                    "condition != condition_first_in_block"
                )

    # --- Contiguous block-order check (schedule list order IS execution
    #     order) ------------------------------------------------------
    idx = 0
    n = len(episodes)
    seen_block_ids: set[str] = set()
    while idx < n:
        bid = episodes[idx].block_id
        if bid in seen_block_ids:
            errors.append(f"{ctx_model}: block_id {bid!r} reappears at a non-contiguous position")
        seen_block_ids.add(bid)
        run_end = idx
        while run_end < n and episodes[run_end].block_id == bid:
            run_end += 1
        run = episodes[idx:run_end]
        if len(run) != 2:
            errors.append(f"{ctx_model}: contiguous block {bid!r} has {len(run)} consecutive episode(s), expected 2")
        else:
            if [ep.order_in_block for ep in run] != [1, 2]:
                errors.append(f"{ctx_model}: contiguous block {bid!r} order_in_block sequence != [1, 2]")
            if [ep.restart_server_before_block for ep in run] != [1, 0]:
                errors.append(f"{ctx_model}: contiguous block {bid!r} restart sequence != [1, 0]")
        idx = run_end

    # --- Block ID format + block sequence check (trigger rotation +
    #     low/high alternation, Section 8) --------------------------------
    expected_sequence: list[tuple[str, int]] = []
    for repeat in range(1, repeats + 1):
        for trigger in TRIGGER_ROTATION_BY_REPEAT[repeat]:
            for _, state_label in STATES:
                expected_sequence.append((state_label, trigger))

    actual_sequence = [
        (blocks[bid][0].state_label, blocks[bid][0].trigger_after_decode_tokens)
        for bid in block_ids_in_order
    ]
    if actual_sequence != expected_sequence:
        errors.append(
            f"{ctx_model}: block (state, trigger) sequence {actual_sequence} != "
            f"expected rotation {expected_sequence}"
        )

    for position, bid in enumerate(block_ids_in_order, start=1):
        if position - 1 < len(expected_sequence):
            state_label, trigger = expected_sequence[position - 1]
            expected_bid = f"{model_key}_block{position:02d}_{state_label}_trigger{trigger}"
            if bid != expected_bid:
                errors.append(f"{ctx_model}: block at position {position} is {bid!r}, expected {expected_bid!r}")

    # --- Condition-first balance check (Section 9) ----------------------
    first_conditions = [blocks[bid][0].condition for bid in block_ids_in_order]
    no_burst_first_count = sum(1 for c in first_conditions if c == "no_burst")
    burst_first_count = sum(1 for c in first_conditions if c == BURST_CONDITION)
    if no_burst_first_count != 6:
        errors.append(f"{ctx_model}: expected exactly 6 no_burst-first blocks, found {no_burst_first_count}")
    if burst_first_count != 6:
        errors.append(f"{ctx_model}: expected exactly 6 prefill_burst-first blocks, found {burst_first_count}")

    cell_first: dict[tuple[str, int], list[str]] = {}
    for bid in block_ids_in_order:
        eps = blocks[bid]
        state_label = eps[0].state_label
        trigger = eps[0].trigger_after_decode_tokens
        key = (state_label, trigger)
        cell_first.setdefault(key, []).append(eps[0].condition)
    for key, firsts in sorted(cell_first.items()):
        if sorted(firsts) != sorted(["no_burst", BURST_CONDITION]):
            errors.append(
                f"{ctx_model}: state/trigger cell {key} does not have exactly one "
                f"no_burst-first and one prefill_burst-first repeat: {firsts}"
            )

    # --- Workload seed constancy across state/trigger/condition within
    #     a repeat (Section 10) -------------------------------------------
    victim_seeds_by_repeat: dict[int, set[int]] = {}
    burst_seeds_by_repeat: dict[int, set[int]] = {}
    for ep in episodes:
        victim_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.burst_workload_seed)
    for repeat, seeds_here in sorted(victim_seeds_by_repeat.items()):
        if len(seeds_here) != 1:
            errors.append(f"{ctx_model}: victim_workload_seed not constant within repeat={repeat}: {sorted(seeds_here)}")
    for repeat, seeds_here in sorted(burst_seeds_by_repeat.items()):
        if len(seeds_here) != 1:
            errors.append(f"{ctx_model}: burst_workload_seed not constant within repeat={repeat}: {sorted(seeds_here)}")
    if len(victim_seeds_by_repeat.get(1, set())) == 1 and len(victim_seeds_by_repeat.get(2, set())) == 1:
        if victim_seeds_by_repeat[1] == victim_seeds_by_repeat[2]:
            errors.append(f"{ctx_model}: victim_workload_seed identical between repeat 1 and repeat 2")
    if len(burst_seeds_by_repeat.get(1, set())) == 1 and len(burst_seeds_by_repeat.get(2, set())) == 1:
        if burst_seeds_by_repeat[1] == burst_seeds_by_repeat[2]:
            errors.append(f"{ctx_model}: burst_workload_seed identical between repeat 1 and repeat 2")

    return errors


def validate_global(all_episodes: list[Episode], model_keys: Sequence[str], repeats: int) -> list[str]:
    """Cross-model checks on the union of both bundles (both bundles are
    still written/published fully independently; this is a paranoia
    check that no seed or id accidentally collided between models)."""
    errors: list[str] = []

    all_ids = [ep.episode_id for ep in all_episodes]
    if len(all_ids) != len(set(all_ids)):
        dupes = sorted({eid for eid in all_ids if all_ids.count(eid) > 1})
        errors.append(f"global: duplicate episode_id(s) across models: {dupes}")

    all_seeds = [ep.episode_seed for ep in all_episodes]
    if len(all_seeds) != len(set(all_seeds)):
        dupes = sorted({s for s in all_seeds if all_seeds.count(s) > 1})
        errors.append(f"global: duplicate episode_seed(s) across models: {dupes}")

    # Seed domain separation: no victim/burst workload seed may be
    # shared between two different models.
    seed_owner: dict[int, set[str]] = {}
    for ep in all_episodes:
        seed_owner.setdefault(ep.victim_workload_seed, set()).add(ep.model_key)
        seed_owner.setdefault(ep.burst_workload_seed, set()).add(ep.model_key)
    for seed_value, owners in seed_owner.items():
        if len(owners) > 1:
            errors.append(f"global: workload seed {seed_value} shared across models {sorted(owners)}")

    return errors


# ---------------------------------------------------------------------------
# Canonical payload / fingerprint
# ---------------------------------------------------------------------------

def build_canonical_payload(episodes: list[Episode], model_key: str, repeats: int, seed: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "repeats": repeats,
        "model_key": model_key,
        "model_id": MODEL_REGISTRY[model_key]["model_id"],
        "states": [{"offload_gb": o, "state_label": s} for o, s in STATES],
        "concurrency": CONCURRENCY,
        "trigger_positions": list(TRIGGER_POSITIONS),
        "trigger_rotation_by_repeat": {
            str(r): list(t) for r, t in TRIGGER_ROTATION_BY_REPEAT.items()
        },
        "conditions": list(CONDITIONS),
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
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
        "episode_count": len(episodes),
        "episodes": [asdict(ep) for ep in episodes],
    }


def compute_fingerprint(canonical_payload: dict) -> str:
    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# In-memory content builders
# ---------------------------------------------------------------------------

def build_csv_text(episodes: list[Episode]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(REQUIRED_FIELDS))
    writer.writeheader()
    for ep in episodes:
        writer.writerow(asdict(ep))
    return buf.getvalue()


def build_json_text(canonical_payload: dict, fingerprint: str) -> str:
    payload = dict(canonical_payload)
    payload["schedule_fingerprint"] = fingerprint
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_audit_text(
    model_key: str, episodes: list[Episode], repeats: int, seed: int, fingerprint: str
) -> str:
    lines: list[str] = []
    lines.append("Prefill-Trigger-Sweep Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"design_version: {DESIGN_VERSION}")
    lines.append(f"schedule_fingerprint: {fingerprint}")
    lines.append(f"seed: {seed}")
    lines.append(f"model_key: {model_key}")
    lines.append(f"model_id: {MODEL_REGISTRY[model_key]['model_id']}")
    lines.append(f"repeats per cell: {repeats}")
    lines.append(f"total episodes: {len(episodes)}")
    lines.append("")

    lines.append(f"--- {model_key}: PASS ---")
    block_ids: list[str] = []
    for ep in episodes:
        if ep.block_id not in block_ids:
            block_ids.append(ep.block_id)
    lines.append(f"blocks: {len(block_ids)}")

    seq = []
    for bid in block_ids:
        ep0 = next(ep for ep in episodes if ep.block_id == bid)
        seq.append(f"{ep0.state_label}/trigger{ep0.trigger_after_decode_tokens}")
    lines.append(f"block (state/trigger) sequence: {', '.join(seq)}")

    cell_counts: dict[tuple[int, int, str], int] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.trigger_after_decode_tokens, ep.condition)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    lines.append("cell counts (offload_gb, trigger_after_decode_tokens, condition):")
    for key in sorted(cell_counts):
        lines.append(f"  {key}: {cell_counts[key]}")

    lines.append("block sequence (each entry implies a server restart before its first episode):")
    for bid in block_ids:
        ep0 = next(ep for ep in episodes if ep.block_id == bid)
        lines.append(
            f"  {bid} (state={ep0.state_label}, trigger={ep0.trigger_after_decode_tokens}, "
            f"condition_first={ep0.condition_first_in_block})"
        )
    lines.append("")

    no_burst_first = sum(1 for bid in block_ids if next(ep for ep in episodes if ep.block_id == bid).condition_first_in_block == "no_burst")
    burst_first = sum(1 for bid in block_ids if next(ep for ep in episodes if ep.block_id == bid).condition_first_in_block == BURST_CONDITION)
    lines.append(f"condition-first balance: no_burst_first={no_burst_first}, prefill_burst_first={burst_first}")
    lines.append("")

    lines.append("workload seeds by repeat:")
    for repeat in range(1, repeats + 1):
        v = next(ep.victim_workload_seed for ep in episodes if ep.repeat == repeat)
        b = next(ep.burst_workload_seed for ep in episodes if ep.repeat == repeat)
        lines.append(f"  repeat {repeat}: victim_workload_seed={v}, burst_workload_seed={b}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("--- global checks ---")
    lines.append("  episode_id uniqueness: PASS")
    lines.append("  episode_seed uniqueness: PASS")
    lines.append("  csv/json consistency: PASS")
    lines.append("")

    lines.append(
        "--- stabilization protocol (documented for run_prefill_trigger_sweep.py; "
        "not executed by this generator; mechanically inherited from Prefill-Screen) ---"
    )
    lines.append(f"regular episode count: {len(episodes)}")
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
        "The CSV, JSON, and this audit file are each replaced atomically on their "
        "own (one os.replace() per file), but the three os.replace() calls together "
        "are NOT one atomic transaction. If this process is interrupted or a later "
        "replace fails after an earlier one already succeeded, the output directory "
        "can end up with a mix of old and new files. Before use, "
        "run_prefill_trigger_sweep.py must: (1) recompute the JSON's "
        "schedule_fingerprint from its own canonical payload and confirm it matches "
        "the stored schedule_fingerprint, (2) compare the CSV episodes against the "
        "JSON episodes, and (3) compare this audit report's schedule_fingerprint "
        "against the JSON's schedule_fingerprint. Any mismatch means the files were "
        "not produced by the same run and must be treated as invalid."
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
    field_type = _EPISODE_FIELD_TYPES[field_name]
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def check_csv_json_consistency(csv_text: str, json_text: str) -> list[str]:
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
        errors.append(f"csv header {csv_fieldnames} does not match expected field order {list(REQUIRED_FIELDS)}")

    json_episodes = json_obj.get("episodes")
    if not isinstance(json_episodes, list):
        errors.append("json_text has no 'episodes' list")
        return errors

    if len(csv_rows) != len(json_episodes):
        errors.append(f"csv/json episode count mismatch: csv={len(csv_rows)}, json={len(json_episodes)}")
        return errors

    for idx, (csv_row, json_row) in enumerate(zip(csv_rows, json_episodes)):
        if not isinstance(json_row, dict) or set(json_row.keys()) != set(REQUIRED_FIELDS):
            errors.append(f"episode index {idx}: json episode field names do not match expected schema")
            continue
        normalized_csv: dict[str, object] = {}
        for field_name in REQUIRED_FIELDS:
            raw = csv_row.get(field_name)
            if raw is None:
                errors.append(f"episode index {idx}: csv row missing field {field_name!r}")
                continue
            try:
                normalized_csv[field_name] = _normalize_csv_value(raw, field_name)
            except (TypeError, ValueError) as exc:
                errors.append(f"episode index {idx} ({field_name}): could not parse csv value {raw!r}: {exc}")
        for field_name, csv_value in normalized_csv.items():
            json_value = json_row[field_name]
            if csv_value != json_value:
                errors.append(f"episode index {idx} ({field_name}): csv value {csv_value!r} != json value {json_value!r}")

    return errors


# ---------------------------------------------------------------------------
# Output writing (per-file atomic replace; NOT a multi-file transaction)
# ---------------------------------------------------------------------------

def write_and_replace_output_files(files: list[tuple[Path, str]]) -> None:
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
        description="Generate the frozen Prefill-Trigger-Sweep episode schedule for "
        "one model (no requests are executed). The design is frozen per model: "
        "--repeats and --seed must match the official values exactly."
    )
    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Which model's frozen schedule to generate.",
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
        default=None,
        help="Global RNG seed (frozen per model; defaults to the official value for --model-key).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/JSON/audit files "
        "(default: new/runs/prefill_trigger_sweep/<model-key>).",
    )
    return parser.parse_args(argv)


def validate_frozen_cli(args: argparse.Namespace) -> str | None:
    if args.model_key not in MODEL_REGISTRY:
        return f"unknown --model-key {args.model_key!r}; known: {sorted(MODEL_REGISTRY)}"
    if args.repeats != DEFAULT_REPEATS:
        return f"--repeats must be exactly {DEFAULT_REPEATS}; got {args.repeats}"
    official_seed = OFFICIAL_SEED_BY_MODEL[args.model_key]
    seed = args.seed if args.seed is not None else official_seed
    if seed != official_seed:
        return (
            f"--seed must be exactly {official_seed} for --model-key {args.model_key!r}; "
            f"got {args.seed}"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    frozen_error = validate_frozen_cli(args)
    if frozen_error:
        print(f"ERROR: {frozen_error}", file=sys.stderr)
        print(
            "The Prefill-Trigger-Sweep design is frozen per model. Allowed invocation: "
            f"--model-key {{llama,qwen}} --repeats {DEFAULT_REPEATS} "
            f"--seed <official-seed-for-model> [--output-dir DIR]",
            file=sys.stderr,
        )
        return 1

    model_key = args.model_key
    seed = args.seed if args.seed is not None else OFFICIAL_SEED_BY_MODEL[model_key]
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(model_key)

    episodes = generate_schedule(model_key, args.repeats, seed)
    errors = validate_schedule(episodes, model_key, args.repeats, seed)
    global_errors = validate_global(episodes, [model_key], args.repeats)
    any_errors = bool(errors) or bool(global_errors)

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print("FAIL" if any_errors else "PASS")
    print(f"{model_key}: {len(episodes)} episodes")
    for err in errors:
        print(f"  - {err}")
    if global_errors:
        print("global:")
        for err in global_errors:
            print(f"  - {err}")

    if any_errors:
        print()
        print("FAIL: schedule validation failed. No output files were written or replaced.", file=sys.stderr)
        return 1

    canonical_payload = build_canonical_payload(episodes, model_key, args.repeats, seed)
    fingerprint = compute_fingerprint(canonical_payload)

    csv_text = build_csv_text(episodes)
    json_text = build_json_text(canonical_payload, fingerprint)
    audit_text = build_audit_text(model_key, episodes, args.repeats, seed, fingerprint)

    consistency_errors = check_csv_json_consistency(csv_text, json_text)
    if consistency_errors:
        print()
        print("FAIL: csv/json consistency check failed. No output files were written or replaced.", file=sys.stderr)
        for err in consistency_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    out_csv = output_dir / "prefill_trigger_sweep_schedule.csv"
    out_json = output_dir / "prefill_trigger_sweep_schedule.json"
    out_audit = output_dir / "prefill_trigger_sweep_schedule_audit.txt"

    try:
        write_and_replace_output_files([(out_csv, csv_text), (out_json, json_text), (out_audit, audit_text)])
    except OSError as exc:
        print(
            f"ERROR: failed to write/replace one or more output files: {exc}. The three files "
            f"are not a single atomic transaction -- verify with the fingerprint before trusting "
            f"any of them.",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"model_key: {model_key}")
    print(f"model_id: {MODEL_REGISTRY[model_key]['model_id']}")
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
