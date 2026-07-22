#!/usr/bin/env python3
"""
generate_server_waiting_schedule.py

Standalone schedule generator for the server-side WAITING replication of
the vLLM Prefill-Confirmation study (see
CLAUDE_SERVER_WAITING_IMPLEMENTATION_PROMPT_FINAL.md). This is a
completely separate, independently fingerprinted campaign from
generate_prefill_confirmation_schedule.py: different grid, different
seed namespace, different bundle filenames, different result schema.
It does not import, read, write, or in any way depend on the original
Prefill-Confirmation generator or its schedules.

Design (frozen, not configurable via CLI beyond --output-dir/--force):
  - model: Qwen/Qwen2.5-7B-Instruct only (no --model-key; this campaign
    is Qwen-only per the frozen extension design)
  - 2 offload regimes: 0 GB ("low"), 12 GB ("high")
  - 2 server_max_num_seqs values: 4, 8 (a server-side vLLM scheduler
    limit -- NOT the client concurrency of the original study; see
    run_server_waiting_confirmation.py for why the active cohort is
    never assumed to be request_index < server_max_num_seqs)
  - 1 trigger position: 16 actually-received output tokens per member
    of the DATA-DRIVEN active cohort (never a synthetic wave)
  - 2 conditions: no_burst, prefill_burst
  - 4 paired repeats per (offload, server_max_num_seqs) cell
  => 2 * 2 * 4 = 16 paired blocks, 16 * 2 = 32 regular episodes

Block structure: a block is exactly one (offload, server_max_num_seqs,
repeat) combination and contains exactly its 2 paired episodes
(no_burst, prefill_burst), immediately consecutive in schedule order
(order_in_block 1, 2), with a mandatory server restart before the
block's first episode only.

Block order: for each repeat (1..4), the four (offload,
server_max_num_seqs) cells are visited in a deterministically
randomized order, from a seed that is specific to this campaign, this
repeat, and the Qwen model -- never the fixed enumeration order and
never a shared global random.Random.

Condition-first balance: for each of the four cells, across its 4
repeats, condition_first_in_block is no_burst exactly 2 times and
prefill_burst exactly 2 times -- assigned once per cell, never
adjusted post-hoc.

Seed namespace: OFFICIAL_SEED and DESIGN_VERSION below are unique to
this campaign and distinct from the original Prefill-Confirmation
values (seed 20260718, design_version 'prefill-confirmation-v1').
Every derive_seed() call additionally mixes in the literal string
'server-waiting-confirmation', so even a coincidentally identical
integer seed could never reproduce the original study's request
sequence -- the two seed streams are independent by construction, not
merely by having different starting integers.

Usage:
    python3 generate_server_waiting_schedule.py --output-dir DIR [--force]
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
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# ============================================================================
# Frozen official contract (not configurable via CLI beyond --output-dir/--force)
# ============================================================================

SCHEMA_VERSION = 1
DESIGN_VERSION = "server-waiting-confirmation-v1"
SEED_NAMESPACE_TAG = "server-waiting-confirmation"

OFFICIAL_SEED = 20260720

MODEL_KEY = "qwen"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CAMPAIGN_NAME = "qwen-server-waiting-confirmation"

OFFLOAD_VALUES: tuple[int, ...] = (0, 12)
SERVER_MAX_NUM_SEQS_VALUES: tuple[int, ...] = (4, 8)
TRIGGER_AFTER_DECODE_TOKENS = 16
CONDITIONS: tuple[str, ...] = ("no_burst", "prefill_burst")
BURST_CONDITION = "prefill_burst"

STATE_LABEL_BY_OFFLOAD: dict[int, str] = {0: "low", 12: "high"}

INITIAL_REPEATS = 4
INCLUDED_REPEATS: tuple[int, ...] = tuple(range(1, INITIAL_REPEATS + 1))

VICTIM_REQUEST_COUNT = 20
VICTIM_INPUT_LEN = 256
VICTIM_OUTPUT_LEN = 64
VICTIM_TEMPERATURE = 0.0

BURST_PARALLEL_REQUESTS = 4
BURST_INPUT_LEN = 2048
BURST_OUTPUT_LEN = 16
BURST_TEMPERATURE = 0.0

MAX_NUM_BATCHED_TOKENS = 2048

VICTIM_CONFIGURATION: dict[str, object] = {
    "victim_request_count": VICTIM_REQUEST_COUNT,
    "victim_input_len": VICTIM_INPUT_LEN,
    "victim_output_len": VICTIM_OUTPUT_LEN,
    "victim_temperature": VICTIM_TEMPERATURE,
}

BURST_CONFIGURATION: dict[str, object] = {
    "burst_parallel_requests": BURST_PARALLEL_REQUESTS,
    "burst_input_len": BURST_INPUT_LEN,
    "burst_output_len": BURST_OUTPUT_LEN,
    "burst_temperature": BURST_TEMPERATURE,
}

STABILIZATION_CONFIGURATION: dict[str, object] = {
    "enabled": True,
    "api_readiness_check_required": True,
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
}

EXTENSION_POLICY: dict[str, object] = {
    "initial_bundle_repeats": "1-4",
    "extension_policy": "none -- this is a fixed four-repeat robustness study; no "
    "result-dependent extension is defined or permitted",
}

# Portable default output directory, derived from this file's location.
# Expected location:
#   <PROJECT_ROOT>/new/scripts/server_waiting_confirmation/generate_server_waiting_schedule.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else SCRIPT_PATH.parent


def default_output_dir() -> Path:
    return PROJECT_ROOT / "new" / "runs" / "server_waiting_confirmation" / MODEL_KEY / "schedule"


BUNDLE_FILENAMES = (
    "server_waiting_confirmation_schedule.json",
    "server_waiting_confirmation_schedule.csv",
    "server_waiting_confirmation_schedule_audit.txt",
)

# ============================================================================
# Frozen Episode schema (26 fields). Identical in spirit to the original
# Prefill-Confirmation Episode, with `concurrency` replaced by
# `server_max_num_seqs` -- the field this campaign actually controls
# (a server-side vLLM scheduler limit, not a client-side admission
# semaphore; see module docstring).
# ============================================================================

EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "model_key",
    "model_id",
    "offload_gb",
    "state_label",
    "server_max_num_seqs",
    "trigger_after_decode_tokens",
    "condition",
    "repeat",
    "random_seed",
    "episode_seed",
    "victim_workload_seed",
    "burst_workload_seed",
    "victim_request_count",
    "victim_input_len",
    "victim_output_len",
    "victim_temperature",
    "burst_parallel_requests",
    "burst_input_len",
    "burst_output_len",
    "burst_temperature",
    "max_num_batched_tokens",
    "condition_first_in_block",
    "restart_server_before_block",
    "block_id",
    "order_in_block",
)

EPISODE_FIELD_TYPES: dict[str, type] = {
    "episode_id": str,
    "model_key": str,
    "model_id": str,
    "offload_gb": int,
    "state_label": str,
    "server_max_num_seqs": int,
    "trigger_after_decode_tokens": int,
    "condition": str,
    "repeat": int,
    "random_seed": int,
    "episode_seed": int,
    "victim_workload_seed": int,
    "burst_workload_seed": int,
    "victim_request_count": int,
    "victim_input_len": int,
    "victim_output_len": int,
    "victim_temperature": float,
    "burst_parallel_requests": int,
    "burst_input_len": int,
    "burst_output_len": int,
    "burst_temperature": float,
    "max_num_batched_tokens": int,
    "condition_first_in_block": str,
    "restart_server_before_block": int,
    "block_id": str,
    "order_in_block": int,
}


@dataclass
class Episode:
    episode_id: str
    model_key: str
    model_id: str
    offload_gb: int
    state_label: str
    server_max_num_seqs: int
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


def _check_type_strict(value: object, expected_type: type) -> bool:
    """type(value) is expected_type, never isinstance -- rejects bool
    where int is expected (bool is a subclass of int) and int where
    float is expected."""
    return type(value) is expected_type


# ============================================================================
# Deterministic seed derivation (local, self-contained -- same recipe as
# the audited Prefill-Confirmation scripts, but every call site below
# additionally mixes in SEED_NAMESPACE_TAG; see module docstring).
# ============================================================================

def derive_seed(*parts: str) -> int:
    joined = ":".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _ns_seed(seed: int, *parts: str) -> int:
    """derive_seed(), with SEED_NAMESPACE_TAG always mixed in first --
    the single choke point every seed derivation in this module goes
    through, so the independent-seed-namespace guarantee in the module
    docstring cannot be silently broken by a future edit that forgets
    to include the tag at some call site."""
    return derive_seed(SEED_NAMESPACE_TAG, str(seed), *parts)


# ============================================================================
# Schedule construction
# ============================================================================

def all_cells() -> list[tuple[int, int]]:
    """Canonical (offload_gb, server_max_num_seqs) cell enumeration
    order -- used only as a stable base list to shuffle per repeat,
    never as the schedule's actual block order."""
    return [
        (offload, k)
        for offload in OFFLOAD_VALUES
        for k in SERVER_MAX_NUM_SEQS_VALUES
    ]


def build_condition_first_assignments(seed: int) -> dict[tuple[int, int], list[str]]:
    """Per (offload, server_max_num_seqs) cell: a length-4 list of
    condition_first_in_block values (repeats 1..4, in list order),
    built once from a cell-specific seeded RNG shuffling
    2x"no_burst" + 2x"prefill_burst" -- guarantees an exact 2/2 balance
    per cell across its 4 repeats, with no post-hoc correction."""
    assignments: dict[tuple[int, int], list[str]] = {}
    for offload_gb, k in all_cells():
        rng = random.Random(
            _ns_seed(seed, "condition-order", str(offload_gb), str(k))
        )
        values = ["no_burst"] * 2 + [BURST_CONDITION] * 2
        rng.shuffle(values)
        assignments[(offload_gb, k)] = values
    return assignments


def build_block_order(seed: int) -> dict[int, list[tuple[int, int]]]:
    """Per repeat (1..4): the four (offload, server_max_num_seqs) cells
    in a deterministically randomized order, from a repeat-specific
    seeded RNG (a fresh local random.Random per repeat -- never the
    shared global random module state)."""
    order_by_repeat: dict[int, list[tuple[int, int]]] = {}
    base_cells = all_cells()
    for repeat in range(1, INITIAL_REPEATS + 1):
        rng = random.Random(_ns_seed(seed, "block-order", str(repeat)))
        shuffled = list(base_cells)
        rng.shuffle(shuffled)
        order_by_repeat[repeat] = shuffled
    return order_by_repeat


def build_episodes(seed: int) -> list[Episode]:
    condition_first_assignments = build_condition_first_assignments(seed)
    block_order = build_block_order(seed)

    episodes: list[Episode] = []
    for repeat in range(1, INITIAL_REPEATS + 1):
        # Constant across all four cells of this repeat -- isolates the
        # offload/server_max_num_seqs/burst comparisons from
        # prompt-content confounds, exactly as in the audited
        # Prefill-Confirmation design.
        victim_workload_seed = _ns_seed(seed, "victim", str(repeat))
        burst_workload_seed = _ns_seed(seed, "burst", str(repeat))

        for offload_gb, k in block_order[repeat]:
            state_label = STATE_LABEL_BY_OFFLOAD[offload_gb]
            block_id = f"{MODEL_KEY}_off{offload_gb}_k{k}_rep{repeat:02d}"
            condition_first = condition_first_assignments[(offload_gb, k)][repeat - 1]
            condition_second = "no_burst" if condition_first == BURST_CONDITION else BURST_CONDITION

            for order_in_block, condition in enumerate((condition_first, condition_second), start=1):
                episode_id = (
                    f"{MODEL_KEY}_off{offload_gb}_k{k}_trigger"
                    f"{TRIGGER_AFTER_DECODE_TOKENS}_{condition}_rep{repeat:02d}"
                )
                episode_seed = _ns_seed(seed, "episode", episode_id)
                restart = 1 if order_in_block == 1 else 0

                episodes.append(
                    Episode(
                        episode_id=episode_id,
                        model_key=MODEL_KEY,
                        model_id=MODEL_ID,
                        offload_gb=offload_gb,
                        state_label=state_label,
                        server_max_num_seqs=k,
                        trigger_after_decode_tokens=TRIGGER_AFTER_DECODE_TOKENS,
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
                        condition_first_in_block=condition_first,
                        restart_server_before_block=restart,
                        block_id=block_id,
                        order_in_block=order_in_block,
                    )
                )
    return episodes


# ============================================================================
# Validation
# ============================================================================

_EPISODE_FIELD_TYPE_HINTS: dict[str, type] = typing.get_type_hints(Episode)


def validate_episode_schema(episode: Episode) -> list[str]:
    errors: list[str] = []
    row = asdict(episode)
    if set(row.keys()) != set(EPISODE_FIELDS):
        errors.append(
            f"episode {episode.episode_id}: field set {sorted(row.keys())} != "
            f"expected {sorted(EPISODE_FIELDS)}"
        )
    for field_name, expected_type in EPISODE_FIELD_TYPES.items():
        value = row.get(field_name, "<MISSING>")
        if not _check_type_strict(value, expected_type):
            errors.append(
                f"episode {episode.episode_id}: field {field_name!r} has type "
                f"{type(value).__name__}, expected {expected_type.__name__} "
                f"(value={value!r})"
            )
    return errors


def validate_schedule(episodes: list[Episode], seed: int) -> list[str]:
    """Full structural + deterministic-derivation validation of the
    complete 32-episode schedule. Returns a list of error strings; an
    empty list means the schedule is valid."""
    errors: list[str] = []

    for ep in episodes:
        errors.extend(validate_episode_schema(ep))

    expected_episode_count = (
        len(OFFLOAD_VALUES) * len(SERVER_MAX_NUM_SEQS_VALUES) * INITIAL_REPEATS * len(CONDITIONS)
    )
    if len(episodes) != expected_episode_count:
        errors.append(f"expected {expected_episode_count} episodes, found {len(episodes)}")

    for ep in episodes:
        ctx = f"episode {ep.episode_id}"
        if ep.model_key != MODEL_KEY:
            errors.append(f"{ctx}: model_key {ep.model_key!r} != {MODEL_KEY!r}")
        if ep.model_id != MODEL_ID:
            errors.append(f"{ctx}: model_id {ep.model_id!r} != expected {MODEL_ID!r}")
        if ep.offload_gb not in OFFLOAD_VALUES:
            errors.append(f"{ctx}: forbidden offload_gb {ep.offload_gb!r}, allowed {OFFLOAD_VALUES}")
        expected_state = STATE_LABEL_BY_OFFLOAD.get(ep.offload_gb)
        if expected_state is not None and ep.state_label != expected_state:
            errors.append(f"{ctx}: state_label {ep.state_label!r} != expected {expected_state!r}")
        if ep.server_max_num_seqs not in SERVER_MAX_NUM_SEQS_VALUES:
            errors.append(
                f"{ctx}: forbidden server_max_num_seqs {ep.server_max_num_seqs!r}, "
                f"allowed {SERVER_MAX_NUM_SEQS_VALUES}"
            )
        if ep.trigger_after_decode_tokens != TRIGGER_AFTER_DECODE_TOKENS:
            errors.append(
                f"{ctx}: forbidden trigger_after_decode_tokens "
                f"{ep.trigger_after_decode_tokens!r}, only {TRIGGER_AFTER_DECODE_TOKENS} allowed"
            )
        if ep.condition not in CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.random_seed != seed:
            errors.append(f"{ctx}: random_seed {ep.random_seed!r} != official seed {seed!r}")
        if ep.max_num_batched_tokens != MAX_NUM_BATCHED_TOKENS:
            errors.append(f"{ctx}: max_num_batched_tokens != {MAX_NUM_BATCHED_TOKENS}")

        for field_name, expected_value in VICTIM_CONFIGURATION.items():
            if getattr(ep, field_name) != expected_value:
                errors.append(f"{ctx}: {field_name} != {expected_value!r}")
        for field_name, expected_value in BURST_CONFIGURATION.items():
            if getattr(ep, field_name) != expected_value:
                errors.append(f"{ctx}: {field_name} != {expected_value!r}")

        if ep.condition_first_in_block not in CONDITIONS:
            errors.append(f"{ctx}: invalid condition_first_in_block {ep.condition_first_in_block!r}")

        if ep.order_in_block == 1:
            if ep.restart_server_before_block != 1:
                errors.append(f"{ctx}: order_in_block=1 requires restart_server_before_block==1")
        elif ep.order_in_block == 2:
            if ep.restart_server_before_block != 0:
                errors.append(f"{ctx}: order_in_block=2 requires restart_server_before_block==0")
        else:
            errors.append(f"{ctx}: invalid order_in_block {ep.order_in_block!r}, expected 1 or 2")

        expected_episode_id = (
            f"{MODEL_KEY}_off{ep.offload_gb}_k{ep.server_max_num_seqs}_trigger"
            f"{ep.trigger_after_decode_tokens}_{ep.condition}_rep{ep.repeat:02d}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(f"{ctx}: episode_id does not match expected derivation {expected_episode_id!r}")

        expected_block_id = f"{MODEL_KEY}_off{ep.offload_gb}_k{ep.server_max_num_seqs}_rep{ep.repeat:02d}"
        if ep.block_id != expected_block_id:
            errors.append(f"{ctx}: block_id {ep.block_id!r} != expected {expected_block_id!r}")

        expected_episode_seed = _ns_seed(seed, "episode", ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(f"{ctx}: episode_seed does not match the expected derivation")

        expected_victim_seed = _ns_seed(seed, "victim", str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(f"{ctx}: victim_workload_seed does not match the expected derivation")

        expected_burst_seed = _ns_seed(seed, "burst", str(ep.repeat))
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(f"{ctx}: burst_workload_seed does not match the expected derivation")

        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(f"{ctx}: victim_workload_seed and burst_workload_seed must never be identical")

        if not (1 <= ep.repeat <= INITIAL_REPEATS):
            errors.append(f"{ctx}: repeat {ep.repeat!r} outside 1..{INITIAL_REPEATS}")

    episode_ids = [ep.episode_id for ep in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        dupes = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
        errors.append(f"duplicate episode_id(s): {dupes}")

    episode_seeds = [ep.episode_seed for ep in episodes]
    if len(episode_seeds) != len(set(episode_seeds)):
        dupes = sorted({s for s in episode_seeds if episode_seeds.count(s) > 1})
        errors.append(f"duplicate episode_seed(s): {dupes}")

    victim_seeds_by_repeat: dict[int, set[int]] = {}
    burst_seeds_by_repeat: dict[int, set[int]] = {}
    for ep in episodes:
        victim_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.burst_workload_seed)
    for repeat, seeds_here in sorted(victim_seeds_by_repeat.items()):
        if len(seeds_here) != 1:
            errors.append(f"repeat={repeat}: victim_workload_seed not constant across cells: {sorted(seeds_here)}")
    for repeat, seeds_here in sorted(burst_seeds_by_repeat.items()):
        if len(seeds_here) != 1:
            errors.append(f"repeat={repeat}: burst_workload_seed not constant across cells: {sorted(seeds_here)}")

    all_victim_seeds_by_repeat = {r: next(iter(s)) for r, s in victim_seeds_by_repeat.items() if len(s) == 1}
    if len(set(all_victim_seeds_by_repeat.values())) != len(all_victim_seeds_by_repeat):
        errors.append("victim_workload_seed is not distinct across all repeats")
    all_burst_seeds_by_repeat = {r: next(iter(s)) for r, s in burst_seeds_by_repeat.items() if len(s) == 1}
    if len(set(all_burst_seeds_by_repeat.values())) != len(all_burst_seeds_by_repeat):
        errors.append("burst_workload_seed is not distinct across all repeats")

    blocks: dict[str, list[Episode]] = {}
    for ep in episodes:
        blocks.setdefault(ep.block_id, []).append(ep)

    expected_block_count = len(OFFLOAD_VALUES) * len(SERVER_MAX_NUM_SEQS_VALUES) * INITIAL_REPEATS
    if len(blocks) != expected_block_count:
        errors.append(f"expected {expected_block_count} blocks, found {len(blocks)}")

    identical_fields = (
        "model_key", "model_id", "offload_gb", "state_label", "server_max_num_seqs",
        "trigger_after_decode_tokens", "repeat", "victim_workload_seed",
        "burst_workload_seed", "block_id", "condition_first_in_block",
    )
    for block_id, block_episodes in blocks.items():
        if len(block_episodes) != 2:
            errors.append(f"block {block_id!r} has {len(block_episodes)} episode(s), expected exactly 2")
            continue
        conditions_here = sorted(ep.condition for ep in block_episodes)
        if conditions_here != sorted(CONDITIONS):
            errors.append(f"block {block_id!r} conditions {conditions_here} != expected {sorted(CONDITIONS)}")
        orders_here = sorted(ep.order_in_block for ep in block_episodes)
        if orders_here != [1, 2]:
            errors.append(f"block {block_id!r} order_in_block values {orders_here} != [1, 2]")
        restarts_by_order = {ep.order_in_block: ep.restart_server_before_block for ep in block_episodes}
        if restarts_by_order.get(1) != 1 or restarts_by_order.get(2) != 0:
            errors.append(f"block {block_id!r} restart sequence {restarts_by_order} != {{1: 1, 2: 0}}")
        for field_name in identical_fields:
            values_here = {getattr(ep, field_name) for ep in block_episodes}
            if len(values_here) != 1:
                errors.append(f"block {block_id!r}: {field_name} differs between its two episodes: {values_here}")

    idx = 0
    n = len(episodes)
    seen_block_ids: set[str] = set()
    while idx < n:
        bid = episodes[idx].block_id
        if bid in seen_block_ids:
            errors.append(f"block_id {bid!r} reappears at a later, non-contiguous position")
        seen_block_ids.add(bid)
        run_end = idx
        while run_end < n and episodes[run_end].block_id == bid:
            run_end += 1
        if run_end - idx != 2:
            errors.append(
                f"block {bid!r} does not consist of exactly 2 immediately "
                f"consecutive episodes in schedule order (found {run_end - idx})"
            )
        else:
            order_sequence = [episodes[idx].order_in_block, episodes[idx + 1].order_in_block]
            if order_sequence != [1, 2]:
                errors.append(f"block {bid!r} in-schedule order_in_block sequence {order_sequence} != [1, 2]")
        idx = run_end

    cells_by_repeat: dict[int, list[tuple[int, int]]] = {}
    for block_id, block_episodes in blocks.items():
        ep0 = block_episodes[0]
        cells_by_repeat.setdefault(ep0.repeat, []).append((ep0.offload_gb, ep0.server_max_num_seqs))

    expected_cell_set = set(all_cells())
    for repeat in range(1, INITIAL_REPEATS + 1):
        cells_here = cells_by_repeat.get(repeat, [])
        if set(cells_here) != expected_cell_set or len(cells_here) != len(expected_cell_set):
            errors.append(
                f"repeat={repeat}: cells {sorted(cells_here)} != expected exactly "
                f"{sorted(expected_cell_set)} (each once)"
            )

    first_condition_by_cell: dict[tuple[int, int], list[str]] = {}
    for block_id, block_episodes in blocks.items():
        first_ep = next(ep for ep in block_episodes if ep.order_in_block == 1)
        key = (first_ep.offload_gb, first_ep.server_max_num_seqs)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)

    for cell, conditions_seen in sorted(first_condition_by_cell.items()):
        no_burst_count = conditions_seen.count("no_burst")
        burst_count = conditions_seen.count(BURST_CONDITION)
        if len(conditions_seen) != INITIAL_REPEATS or no_burst_count != 2 or burst_count != 2:
            errors.append(
                f"cell offload={cell[0]}, server_max_num_seqs={cell[1]}: condition-first "
                f"balance is no_burst={no_burst_count}, prefill_burst={burst_count} "
                f"across {len(conditions_seen)} repeats, expected exactly 2/2 across 4"
            )

    return errors


# ============================================================================
# Canonical payload / fingerprint
# ============================================================================

def build_canonical_payload(episodes: list[Episode], seed: int) -> dict:
    block_count = len({ep.block_id for ep in episodes})
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "campaign_name": CAMPAIGN_NAME,
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "seed": seed,
        "seed_namespace_tag": SEED_NAMESPACE_TAG,
        "offload_values": list(OFFLOAD_VALUES),
        "server_max_num_seqs_values": list(SERVER_MAX_NUM_SEQS_VALUES),
        "trigger_positions": [TRIGGER_AFTER_DECODE_TOKENS],
        "conditions": list(CONDITIONS),
        "initial_repeats": INITIAL_REPEATS,
        "included_repeats": list(INCLUDED_REPEATS),
        "block_count": block_count,
        "episode_count": len(episodes),
        "victim_configuration": dict(VICTIM_CONFIGURATION),
        "burst_configuration": dict(BURST_CONFIGURATION),
        "stabilization_configuration": dict(STABILIZATION_CONFIGURATION),
        "extension_policy": dict(EXTENSION_POLICY),
        "episodes": [asdict(ep) for ep in episodes],
    }


def compute_schedule_fingerprint(canonical_payload: dict) -> str:
    payload_without_fingerprint = {k: v for k, v in canonical_payload.items() if k != "schedule_fingerprint"}
    serialized = json.dumps(
        payload_without_fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ============================================================================
# CSV / audit rendering
# ============================================================================

def render_csv(episodes: list[Episode]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(EPISODE_FIELDS))
    writer.writeheader()
    for ep in episodes:
        writer.writerow(asdict(ep))
    return buf.getvalue()


def _normalize_csv_value(raw: str, field_name: str) -> object:
    field_type = EPISODE_FIELD_TYPES[field_name]
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def check_csv_json_consistency(csv_text: str, episodes: list[Episode]) -> list[str]:
    errors: list[str] = []
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        csv_rows = list(reader)
        csv_fieldnames = list(reader.fieldnames or [])
    except csv.Error as exc:
        return [f"failed to parse csv_text as CSV: {exc}"]

    if csv_fieldnames != list(EPISODE_FIELDS):
        errors.append(f"csv header {csv_fieldnames} != expected {list(EPISODE_FIELDS)}")

    if len(csv_rows) != len(episodes):
        errors.append(f"csv row count {len(csv_rows)} != episode count {len(episodes)}")
        return errors

    for idx, (csv_row, ep) in enumerate(zip(csv_rows, episodes)):
        json_row = asdict(ep)
        for field_name in EPISODE_FIELDS:
            raw = csv_row.get(field_name)
            if raw is None:
                errors.append(f"episode index {idx}: csv row missing field {field_name!r}")
                continue
            try:
                normalized = _normalize_csv_value(raw, field_name)
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"episode index {idx} ({field_name}): could not parse csv value "
                    f"{raw!r} as {EPISODE_FIELD_TYPES[field_name]}: {exc}"
                )
                continue
            if normalized != json_row[field_name]:
                errors.append(
                    f"episode index {idx} ({field_name}): csv value {normalized!r} != "
                    f"json value {json_row[field_name]!r}"
                )
    return errors


def render_audit(
    episodes: list[Episode], seed: int, fingerprint: str, validation_errors: list[str],
) -> str:
    lines: list[str] = []
    lines.append("Server-Waiting-Confirmation Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"design_version: {DESIGN_VERSION}")
    lines.append(f"campaign_name: {CAMPAIGN_NAME}")
    lines.append(f"seed: {seed}")
    lines.append(f"seed_namespace_tag: {SEED_NAMESPACE_TAG}")
    lines.append(f"schedule_fingerprint: {fingerprint}")
    lines.append(f"model_key: {MODEL_KEY}")
    lines.append(f"model: {MODEL_KEY} ({MODEL_ID})")
    lines.append(f"offload values: {list(OFFLOAD_VALUES)}")
    lines.append(f"server_max_num_seqs values: {list(SERVER_MAX_NUM_SEQS_VALUES)}")
    lines.append(f"trigger positions: [{TRIGGER_AFTER_DECODE_TOKENS}]")
    lines.append(f"conditions: {list(CONDITIONS)}")
    lines.append(f"initial repeats: {INITIAL_REPEATS}")

    blocks: dict[str, list[Episode]] = {}
    for ep in episodes:
        blocks.setdefault(ep.block_id, []).append(ep)

    lines.append(f"block count: {len(blocks)}")
    lines.append(f"episode count: {len(episodes)}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("--- explicit checks ---")

    def audit_check(label: str, condition: bool) -> None:
        lines.append(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    audit_check("32 episodes", len(episodes) == 32)
    audit_check("16 blocks", len(blocks) == 16)
    audit_check("only Qwen model_key/model_id", all(
        e.model_key == MODEL_KEY and e.model_id == MODEL_ID for e in episodes
    ))

    cells_by_repeat: dict[int, set[tuple[int, int]]] = {}
    for block_id, block_episodes in blocks.items():
        ep0 = block_episodes[0]
        cells_by_repeat.setdefault(ep0.repeat, set()).add((ep0.offload_gb, ep0.server_max_num_seqs))
    audit_check(
        "4 repeats, each with all 4 cells exactly once",
        set(cells_by_repeat) == set(range(1, 5)) and all(len(c) == 4 for c in cells_by_repeat.values()),
    )

    audit_check(
        "every block has exactly 2 episodes with both conditions",
        all(
            len(be) == 2 and sorted(e.condition for e in be) == sorted(CONDITIONS)
            for be in blocks.values()
        ),
    )
    audit_check(
        "every block has order_in_block == [1, 2]",
        all(sorted(e.order_in_block for e in be) == [1, 2] for be in blocks.values()),
    )
    audit_check(
        "every block has restart sequence [1, 0]",
        all(
            {e.order_in_block: e.restart_server_before_block for e in be} == {1: 1, 2: 0}
            for be in blocks.values()
        ),
    )

    first_condition_by_cell: dict[tuple[int, int], list[str]] = {}
    for block_episodes in blocks.values():
        first_ep = next(e for e in block_episodes if e.order_in_block == 1)
        key = (first_ep.offload_gb, first_ep.server_max_num_seqs)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)
    audit_check(
        "every cell has exactly 2 no_burst-first blocks",
        all(v.count("no_burst") == 2 for v in first_condition_by_cell.values()),
    )
    audit_check(
        "every cell has exactly 2 prefill_burst-first blocks",
        all(v.count(BURST_CONDITION) == 2 for v in first_condition_by_cell.values()),
    )

    episode_ids = [e.episode_id for e in episodes]
    audit_check("no duplicate episode_ids", len(episode_ids) == len(set(episode_ids)))
    episode_seeds = [e.episode_seed for e in episodes]
    audit_check("no duplicate episode_seeds", len(episode_seeds) == len(set(episode_seeds)))

    audit_check("correct seed derivation (episode/victim/burst)", not validation_errors)

    csv_text = render_csv(episodes)
    csv_consistency_errors = check_csv_json_consistency(csv_text, episodes)
    audit_check("JSON/CSV consistency", not csv_consistency_errors)

    audit_check("fingerprint correct", bool(fingerprint) and fingerprint.startswith("sha256:") and len(fingerprint) == 71)
    audit_check("no disallowed offload values", all(e.offload_gb in OFFLOAD_VALUES for e in episodes))
    audit_check(
        "no disallowed server_max_num_seqs values",
        all(e.server_max_num_seqs in SERVER_MAX_NUM_SEQS_VALUES for e in episodes),
    )
    audit_check("only trigger 16", all(e.trigger_after_decode_tokens == 16 for e in episodes))

    lines.append("")
    lines.append("=" * 60)
    if validation_errors or csv_consistency_errors:
        lines.append(f"VALIDATION ERRORS ({len(validation_errors) + len(csv_consistency_errors)}):")
        for e in validation_errors + csv_consistency_errors:
            lines.append(f"  - {e}")
        lines.append("")
        lines.append("OVERALL: FAIL")
    else:
        lines.append("OVERALL: PASS")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Transactional bundle writer (identical contract/implementation to the
# audited generate_prefill_confirmation_schedule.write_bundle_atomic;
# duplicated here, not imported, so this generator has zero runtime
# dependency on the original Prefill-Confirmation scripts -- see module
# docstring. Behavior verified by test_generate_server_waiting_schedule.py.)
# ============================================================================

def write_bundle_atomic(output_dir: Path, files: list[tuple[str, str]], *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_dir / name for name, _ in files]

    existing = [path for path in final_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing bundle file(s) without --force: "
            f"{[path.name for path in existing]}"
        )

    unique_suffix = f"{os.getpid()}.{uuid.uuid4().hex}"
    tmp_paths = [path.with_name(f"{path.name}.tmp.{unique_suffix}") for path in final_paths]
    backup_paths = [path.with_name(f"{path.name}.bak.{unique_suffix}") for path in final_paths]

    backed_up: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    committed = False

    try:
        for tmp_path, (_name, content) in zip(tmp_paths, files):
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

        if force:
            for final_path, backup_path in zip(final_paths, backup_paths):
                if final_path.exists():
                    os.replace(final_path, backup_path)
                    backed_up.append((final_path, backup_path))

        for tmp_path, final_path in zip(tmp_paths, final_paths):
            os.replace(tmp_path, final_path)
            installed.append(final_path)

        committed = True

    except BaseException:
        if not committed:
            for final_path in reversed(installed):
                try:
                    if final_path.exists():
                        final_path.unlink()
                except OSError:
                    pass
            for final_path, backup_path in reversed(backed_up):
                try:
                    if backup_path.exists():
                        if final_path.exists():
                            final_path.unlink()
                        os.replace(backup_path, final_path)
                except OSError:
                    pass
            for tmp_path in tmp_paths:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
        raise

    for _final_path, backup_path in backed_up:
        if backup_path.exists():
            backup_path.unlink()


# ============================================================================
# CLI / main
# ============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Bundle output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing bundle")
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED, help="Override the schedule seed (advanced)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()

    episodes = build_episodes(args.seed)
    validation_errors = validate_schedule(episodes, args.seed)

    canonical_payload = build_canonical_payload(episodes, args.seed)
    fingerprint = compute_schedule_fingerprint(canonical_payload)
    json_payload = dict(canonical_payload)
    json_payload["schedule_fingerprint"] = fingerprint

    csv_text = render_csv(episodes)
    audit_text = render_audit(episodes, args.seed, fingerprint, validation_errors)

    if validation_errors:
        sys.stderr.write("Schedule FAILED validation; refusing to write a bundle:\n")
        for e in validation_errors:
            sys.stderr.write(f"  - {e}\n")
        return 1

    json_text = json.dumps(json_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    files = [
        (BUNDLE_FILENAMES[0], json_text),
        (BUNDLE_FILENAMES[1], csv_text),
        (BUNDLE_FILENAMES[2], audit_text),
    ]

    try:
        write_bundle_atomic(output_dir, files, force=args.force)
    except FileExistsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    print(f"Wrote server-waiting-confirmation schedule bundle to: {output_dir}")
    print(f"schedule_fingerprint: {fingerprint}")
    print(f"episodes: {len(episodes)}, blocks: {len({e.block_id for e in episodes})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
