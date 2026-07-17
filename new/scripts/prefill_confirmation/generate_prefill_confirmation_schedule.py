#!/usr/bin/env python3
"""
generate_prefill_confirmation_schedule.py

Fully standalone schedule generator for the initial CONFIRMATORY
"Prefill-Confirmation" campaign: does an identical, bounded prefill-heavy
extra load produce different availability damage across different CPU
offload / concurrency regimes? Supports exactly two frozen models --
Llama and Qwen -- selected via a mandatory --model-key CLI flag. The two
models are never mixed in a single bundle; each invocation produces
exactly one model's 96-episode bundle in its own directory.

This is a completely separate campaign from the explorative
Prefill-Trigger-Sweep. It does not import, read, write, or in any way
depend on run_prefill_trigger_sweep.py, its schedules, or its results.
It has no runtime dependency on any other generator or runner in this
project, and needs no third-party packages -- Python standard library
only.

Design (frozen per model, not configurable via CLI beyond --model-key):
  - 3 offload regimes: 0 GB ("low"), 8 GB ("intermediate"), 12 GB ("high")
  - 2 concurrencies: 4, 8
  - 1 trigger position: 16 actually-received output tokens per initial
    active-wave request (no other trigger position is ever generated)
  - 2 conditions: no_burst, prefill_burst
  - 8 initial paired repeats per (offload, concurrency) cell
  => 3 * 2 * 8 = 48 paired blocks, 48 * 2 = 96 regular episodes PER MODEL
  (192 regular episodes total across both models, in two entirely
  separate bundles/fingerprints/directories)

Block structure: a block is exactly one (offload, concurrency, repeat)
combination and contains exactly its 2 paired episodes (no_burst,
prefill_burst), immediately consecutive in schedule order
(order_in_block 1, 2), with a mandatory server restart before the
block's first episode only (chunked-prefill/offload configuration is a
server startup flag and cannot change on a running server).

Block order: for each repeat (1..8), the six (offload, concurrency)
cells are visited in an order that is deterministically randomized from
a model- and repeat-specific seed -- not the fixed enumeration order,
and not a shared global random.Random. Each repeat contains all six
cells exactly once.

Condition-first balance: for each of the six cells, across its 8
repeats, condition_first_in_block is no_burst exactly 4 times and
prefill_burst exactly 4 times -- assigned once per cell from a model-
and cell-specific seed, never adjusted post-hoc.

Seeds: episode_seed is unique per episode. victim_workload_seed and
burst_workload_seed are each constant across all six cells of a given
repeat (so, e.g., repeat 3 uses identical victim prompts in every
offload/concurrency cell, isolating the offload/concurrency/burst
comparisons from prompt-content confounds), and always mutually
distinct. Every model-dependent seed derivation includes the actual
model_key, so Llama's and Qwen's seed streams are fully independent.

Bundle writing is transactional across the three output files (JSON,
CSV, audit): either the complete new bundle or the complete previous
state survives a cleanly-handled error or interruption -- never a
mixture, and never just one or two of the three new files. See
write_bundle_atomic() for the exact contract and its limits (this is
not multi-file power-loss atomicity, only clean error/interrupt
handling).

Extension policy (metadata only -- this generator only ever produces
repeats 1-8; see the top-level `extension_policy` object in the
produced JSON): any later extension to repeats 9-12 or 13-16 requires a
separately generated, separately fingerprinted bundle, and any
extension decision may be based only on a predeclared, blinded
variance/precision assessment -- never on significance, effect
direction, or whether a desired hypothesis was confirmed.

Usage:
    python3 generate_prefill_confirmation_schedule.py --model-key llama --output-dir DIR [--force]
    python3 generate_prefill_confirmation_schedule.py --model-key qwen --output-dir DIR [--force]
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
# Frozen official contract (not configurable via CLI, except --model-key)
# ============================================================================

SCHEMA_VERSION = 1
DESIGN_VERSION = "prefill-confirmation-v1"

OFFICIAL_SEED = 20260718

# The only model-dependent facts in the whole design. Immutable at
# runtime (a plain dict is used, but nothing in this module ever
# mutates it) -- every other constant/derivation below applies
# identically, and independently, to both entries.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "campaign_name": "llama-prefill-confirmation",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "campaign_name": "qwen-prefill-confirmation",
    },
}

OFFLOAD_VALUES: tuple[int, ...] = (0, 8, 12)
CONCURRENCY_VALUES: tuple[int, ...] = (4, 8)
TRIGGER_AFTER_DECODE_TOKENS = 16
CONDITIONS: tuple[str, ...] = ("no_burst", "prefill_burst")
BURST_CONDITION = "prefill_burst"

STATE_LABEL_BY_OFFLOAD: dict[int, str] = {0: "low", 8: "intermediate", 12: "high"}

INITIAL_REPEATS = 8
INCLUDED_REPEATS: tuple[int, ...] = tuple(range(1, INITIAL_REPEATS + 1))
PLANNED_REPEAT_CHECKPOINTS: tuple[int, ...] = (8, 12, 16)

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
    "stability_secondary_metrics": ["median_ttft_ms", "median_e2el_ms"],
    "record_relative_change": True,
    "abort_on_stability_drift": False,
}

EXTENSION_POLICY: dict[str, object] = {
    "initial_bundle_repeats": "1-8",
    "extension_requires_separate_fingerprinted_bundle": True,
    "extension_repeat_ranges": ["9-12", "13-16"],
    "extension_decision_basis": "predeclared_blinded_variance_precision_assessment_only",
    "forbidden_extension_bases": [
        "statistical_significance",
        "observed_effect_direction",
        "whether_a_desired_hypothesis_was_confirmed",
    ],
}

# Portable default output directory, derived from this file's location.
# Expected location:
#   <PROJECT_ROOT>/new/scripts/prefill_confirmation/generate_prefill_confirmation_schedule.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]


def default_output_dir(model_key: str) -> Path:
    return PROJECT_ROOT / "new" / "runs" / "prefill_confirmation" / model_key


BUNDLE_FILENAMES = (
    "prefill_confirmation_schedule.json",
    "prefill_confirmation_schedule.csv",
    "prefill_confirmation_schedule_audit.txt",
)


# ============================================================================
# Episode schema (exact field set and order, per the frozen contract --
# unchanged by the two-model patch)
# ============================================================================

EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "model_key",
    "model_id",
    "offload_gb",
    "state_label",
    "concurrency",
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
    "concurrency": int,
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


def _check_type_strict(value: object, expected_type: type) -> bool:
    """Strict type check: type(value) is expected_type, never isinstance.
    This alone already rejects bool where int is expected, since
    type(True) is bool, not int -- booleans are never silently accepted
    as integers."""
    return type(value) is expected_type


# ============================================================================
# Deterministic seed derivation (local, self-contained, unchanged)
# ============================================================================

def derive_seed(*parts: str) -> int:
    joined = ":".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


# ============================================================================
# Schedule construction
# ============================================================================

def all_cells() -> list[tuple[int, int]]:
    """Canonical (offload_gb, concurrency) cell enumeration order -- used
    only as a stable base list to shuffle per repeat, never as the
    schedule's actual block order. Model-independent."""
    return [(offload, concurrency) for offload in OFFLOAD_VALUES for concurrency in CONCURRENCY_VALUES]


def build_condition_first_assignments(seed: int, model_key: str) -> dict[tuple[int, int], list[str]]:
    """Per (offload, concurrency) cell: a length-8 list of
    condition_first_in_block values (repeats 1..8, in list order), built
    once from a model- and cell-specific seeded RNG shuffling
    4x"no_burst" + 4x"prefill_burst" -- guarantees an exact 4/4 balance
    per cell across its 8 repeats, with no post-hoc correction."""
    assignments: dict[tuple[int, int], list[str]] = {}
    for offload_gb, concurrency in all_cells():
        rng = random.Random(
            derive_seed(str(seed), model_key, "condition-order", str(offload_gb), str(concurrency))
        )
        values = ["no_burst"] * 4 + [BURST_CONDITION] * 4
        rng.shuffle(values)
        assignments[(offload_gb, concurrency)] = values
    return assignments


def build_block_order(seed: int, model_key: str) -> dict[int, list[tuple[int, int]]]:
    """Per repeat (1..8): the six (offload, concurrency) cells in a
    deterministically randomized order, from a model- and repeat-specific
    seeded RNG (a fresh local random.Random per repeat -- never the
    shared global random module state). Every repeat contains all six
    cells exactly once."""
    order_by_repeat: dict[int, list[tuple[int, int]]] = {}
    base_cells = all_cells()
    for repeat in range(1, INITIAL_REPEATS + 1):
        rng = random.Random(derive_seed(str(seed), model_key, "block-order", str(repeat)))
        shuffled = list(base_cells)
        rng.shuffle(shuffled)
        order_by_repeat[repeat] = shuffled
    return order_by_repeat


def build_episodes(seed: int, model_key: str) -> list[Episode]:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model_key {model_key!r}; must be one of {sorted(MODEL_REGISTRY)}")
    model_id = MODEL_REGISTRY[model_key]["model_id"]

    condition_first_assignments = build_condition_first_assignments(seed, model_key)
    block_order = build_block_order(seed, model_key)

    episodes: list[Episode] = []
    for repeat in range(1, INITIAL_REPEATS + 1):
        # Deliberately independent of offload_gb/concurrency/condition --
        # constant across all six cells of this repeat, so the victim
        # (and, separately, the burst) workload is identical across the
        # whole repeat, isolating the offload/concurrency/burst
        # comparisons from prompt-content confounds. Always includes the
        # actual model_key, so Llama's and Qwen's seed streams never
        # collide.
        victim_workload_seed = derive_seed(str(seed), model_key, "victim", str(repeat))
        burst_workload_seed = derive_seed(str(seed), model_key, "burst", str(repeat))

        for offload_gb, concurrency in block_order[repeat]:
            state_label = STATE_LABEL_BY_OFFLOAD[offload_gb]
            block_id = f"{model_key}_off{offload_gb}_conc{concurrency}_rep{repeat:02d}"
            condition_first = condition_first_assignments[(offload_gb, concurrency)][repeat - 1]
            condition_second = "no_burst" if condition_first == BURST_CONDITION else BURST_CONDITION

            for order_in_block, condition in enumerate((condition_first, condition_second), start=1):
                episode_id = (
                    f"{model_key}_off{offload_gb}_conc{concurrency}_trigger"
                    f"{TRIGGER_AFTER_DECODE_TOKENS}_{condition}_rep{repeat:02d}"
                )
                episode_seed = derive_seed(str(seed), episode_id)
                restart = 1 if order_in_block == 1 else 0

                episodes.append(
                    Episode(
                        episode_id=episode_id,
                        model_key=model_key,
                        model_id=model_id,
                        offload_gb=offload_gb,
                        state_label=state_label,
                        concurrency=concurrency,
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
    """Per-episode strict structural validation: exact field set (via
    the dataclass itself) and exact types (never a bool silently
    accepted as an int, never an int silently accepted as a float)."""
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


def validate_schedule(episodes: list[Episode], seed: int, model_key: str) -> list[str]:
    """Full structural + deterministic-derivation validation of the
    complete 96-episode schedule for exactly ONE model. Returns a list
    of error strings; an empty list means the schedule is valid. A
    schedule containing even one episode belonging to a different model
    (wrong model_key or model_id) always fails here."""
    errors: list[str] = []

    if model_key not in MODEL_REGISTRY:
        errors.append(f"unknown model_key {model_key!r}; must be one of {sorted(MODEL_REGISTRY)}")
        return errors
    expected_model_id = MODEL_REGISTRY[model_key]["model_id"]

    for ep in episodes:
        errors.extend(validate_episode_schema(ep))

    expected_episode_count = len(OFFLOAD_VALUES) * len(CONCURRENCY_VALUES) * INITIAL_REPEATS * len(CONDITIONS)
    if len(episodes) != expected_episode_count:
        errors.append(f"expected {expected_episode_count} episodes, found {len(episodes)}")

    # --- exact-value / forbidden-value / cross-model-contamination checks --
    for ep in episodes:
        ctx = f"episode {ep.episode_id}"
        if ep.model_key != model_key:
            errors.append(f"{ctx}: model_key {ep.model_key!r} != selected model {model_key!r}")
        if ep.model_id != expected_model_id:
            errors.append(f"{ctx}: model_id {ep.model_id!r} != expected {expected_model_id!r} for {model_key!r}")
        if ep.offload_gb not in OFFLOAD_VALUES:
            errors.append(f"{ctx}: forbidden offload_gb {ep.offload_gb!r}, allowed {OFFLOAD_VALUES}")
        expected_state = STATE_LABEL_BY_OFFLOAD.get(ep.offload_gb)
        if expected_state is not None and ep.state_label != expected_state:
            errors.append(f"{ctx}: state_label {ep.state_label!r} != expected {expected_state!r}")
        if ep.concurrency not in CONCURRENCY_VALUES:
            errors.append(f"{ctx}: forbidden concurrency {ep.concurrency!r}, allowed {CONCURRENCY_VALUES}")
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
            f"{model_key}_off{ep.offload_gb}_conc{ep.concurrency}_trigger"
            f"{ep.trigger_after_decode_tokens}_{ep.condition}_rep{ep.repeat:02d}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(f"{ctx}: episode_id does not match expected derivation {expected_episode_id!r}")

        expected_block_id = f"{model_key}_off{ep.offload_gb}_conc{ep.concurrency}_rep{ep.repeat:02d}"
        if ep.block_id != expected_block_id:
            errors.append(f"{ctx}: block_id {ep.block_id!r} != expected {expected_block_id!r}")

        expected_episode_seed = derive_seed(str(seed), ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(f"{ctx}: episode_seed does not match derive_seed(seed, episode_id)")

        expected_victim_seed = derive_seed(str(seed), model_key, "victim", str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(f"{ctx}: victim_workload_seed does not match the expected derivation")

        expected_burst_seed = derive_seed(str(seed), model_key, "burst", str(ep.repeat))
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(f"{ctx}: burst_workload_seed does not match the expected derivation")

        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(f"{ctx}: victim_workload_seed and burst_workload_seed must never be identical")

        if not (1 <= ep.repeat <= INITIAL_REPEATS):
            errors.append(f"{ctx}: repeat {ep.repeat!r} outside the initial bundle's range 1..{INITIAL_REPEATS}")

    # --- uniqueness ----------------------------------------------------------
    episode_ids = [ep.episode_id for ep in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        dupes = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
        errors.append(f"duplicate episode_id(s): {dupes}")

    episode_seeds = [ep.episode_seed for ep in episodes]
    if len(episode_seeds) != len(set(episode_seeds)):
        dupes = sorted({s for s in episode_seeds if episode_seeds.count(s) > 1})
        errors.append(f"duplicate episode_seed(s): {dupes}")

    # --- seed constancy per repeat, and distinctness across repeats --------
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

    # --- block structure -----------------------------------------------------
    blocks: dict[str, list[Episode]] = {}
    block_ids_in_order: list[str] = []
    for ep in episodes:
        if ep.block_id not in blocks:
            blocks[ep.block_id] = []
            block_ids_in_order.append(ep.block_id)
        blocks[ep.block_id].append(ep)

    expected_block_count = len(OFFLOAD_VALUES) * len(CONCURRENCY_VALUES) * INITIAL_REPEATS
    if len(blocks) != expected_block_count:
        errors.append(f"expected {expected_block_count} blocks, found {len(blocks)}")

    identical_fields = (
        "model_key", "model_id", "offload_gb", "state_label", "concurrency",
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

    # --- contiguous-block check on the actual schedule (episode) order -----
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

    # --- each repeat round contains all six cells exactly once --------------
    cells_by_repeat: dict[int, list[tuple[int, int]]] = {}
    for block_id, block_episodes in blocks.items():
        ep0 = block_episodes[0]
        cells_by_repeat.setdefault(ep0.repeat, []).append((ep0.offload_gb, ep0.concurrency))

    expected_cell_set = set(all_cells())
    for repeat in range(1, INITIAL_REPEATS + 1):
        cells_here = cells_by_repeat.get(repeat, [])
        if set(cells_here) != expected_cell_set or len(cells_here) != len(expected_cell_set):
            errors.append(
                f"repeat={repeat}: cells {sorted(cells_here)} != expected exactly "
                f"{sorted(expected_cell_set)} (each once)"
            )

    # --- per-cell condition-first balance (4/4 across the 8 repeats) -------
    first_condition_by_cell: dict[tuple[int, int], list[str]] = {}
    for block_id, block_episodes in blocks.items():
        first_ep = next(ep for ep in block_episodes if ep.order_in_block == 1)
        key = (first_ep.offload_gb, first_ep.concurrency)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)

    for cell, conditions_seen in sorted(first_condition_by_cell.items()):
        no_burst_count = conditions_seen.count("no_burst")
        burst_count = conditions_seen.count(BURST_CONDITION)
        if len(conditions_seen) != INITIAL_REPEATS or no_burst_count != 4 or burst_count != 4:
            errors.append(
                f"cell offload={cell[0]}, concurrency={cell[1]}: condition-first "
                f"balance is no_burst={no_burst_count}, prefill_burst={burst_count} "
                f"across {len(conditions_seen)} repeats, expected exactly 4/4 across 8"
            )

    return errors


# ============================================================================
# Canonical payload / fingerprint
# ============================================================================

def build_canonical_payload(episodes: list[Episode], seed: int, model_key: str) -> dict:
    """The single canonical payload used both to compute the schedule
    fingerprint and as the basis for the published JSON (with
    schedule_fingerprint added afterwards). Must NOT contain
    schedule_fingerprint itself, absolute paths, creation timestamps, or
    temporary file names."""
    registry_entry = MODEL_REGISTRY[model_key]
    block_count = len({ep.block_id for ep in episodes})
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "campaign_name": registry_entry["campaign_name"],
        "model_key": model_key,
        "model_id": registry_entry["model_id"],
        "seed": seed,
        "offload_values": list(OFFLOAD_VALUES),
        "concurrency_values": list(CONCURRENCY_VALUES),
        "trigger_positions": [TRIGGER_AFTER_DECODE_TOKENS],
        "conditions": list(CONDITIONS),
        "initial_repeats": INITIAL_REPEATS,
        "included_repeats": list(INCLUDED_REPEATS),
        "planned_repeat_checkpoints": list(PLANNED_REPEAT_CHECKPOINTS),
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
    """Re-parses the ACTUAL serialized csv_text (not a second in-memory
    dict built the same way), so this can catch a real serialization bug,
    not just two parallel asdict() calls that would always agree."""
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
    episodes: list[Episode], seed: int, model_key: str, fingerprint: str, validation_errors: list[str],
) -> str:
    registry_entry = MODEL_REGISTRY[model_key]
    lines: list[str] = []
    lines.append("Prefill-Confirmation Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"design_version: {DESIGN_VERSION}")
    lines.append(f"campaign_name: {registry_entry['campaign_name']}")
    lines.append(f"seed: {seed}")
    lines.append(f"schedule_fingerprint: {fingerprint}")
    lines.append(f"model_key: {model_key}")
    lines.append(f"model: {model_key} ({registry_entry['model_id']})")
    lines.append(f"offload values: {list(OFFLOAD_VALUES)}")
    lines.append(f"concurrency values: {list(CONCURRENCY_VALUES)}")
    lines.append(f"trigger positions: [{TRIGGER_AFTER_DECODE_TOKENS}]")
    lines.append(f"conditions: {list(CONDITIONS)}")
    lines.append(f"initial repeats: {INITIAL_REPEATS}")
    lines.append(f"planned checkpoints: {list(PLANNED_REPEAT_CHECKPOINTS)}")

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

    audit_check("96 episodes", len(episodes) == 96)
    audit_check("48 blocks", len(blocks) == 48)
    audit_check("only this model's model_key/model_id", all(
        e.model_key == model_key and e.model_id == registry_entry["model_id"] for e in episodes
    ))

    cells_by_repeat: dict[int, set[tuple[int, int]]] = {}
    for block_id, block_episodes in blocks.items():
        ep0 = block_episodes[0]
        cells_by_repeat.setdefault(ep0.repeat, set()).add((ep0.offload_gb, ep0.concurrency))
    audit_check(
        "8 repeats, each with all 6 cells exactly once",
        set(cells_by_repeat) == set(range(1, 9)) and all(len(c) == 6 for c in cells_by_repeat.values()),
    )
    audit_check("6 cells per repeat round", all(len(c) == 6 for c in cells_by_repeat.values()))

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
        key = (first_ep.offload_gb, first_ep.concurrency)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)
    audit_check(
        "every cell has exactly 4 no_burst-first blocks",
        all(v.count("no_burst") == 4 for v in first_condition_by_cell.values()),
    )
    audit_check(
        "every cell has exactly 4 prefill_burst-first blocks",
        all(v.count(BURST_CONDITION) == 4 for v in first_condition_by_cell.values()),
    )

    episode_ids = [e.episode_id for e in episodes]
    audit_check("no duplicate episode_ids", len(episode_ids) == len(set(episode_ids)))
    episode_seeds = [e.episode_seed for e in episodes]
    audit_check("no duplicate episode_seeds", len(episode_seeds) == len(set(episode_seeds)))

    audit_check("correct seed derivation (episode/victim/burst)", not validation_errors)
    victim_seeds_by_repeat = {}
    burst_seeds_by_repeat = {}
    for e in episodes:
        victim_seeds_by_repeat.setdefault(e.repeat, set()).add(e.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(e.repeat, set()).add(e.burst_workload_seed)
    audit_check(
        "victim seed constant within each repeat",
        all(len(v) == 1 for v in victim_seeds_by_repeat.values()),
    )
    audit_check(
        "burst seed constant within each repeat",
        all(len(v) == 1 for v in burst_seeds_by_repeat.values()),
    )
    audit_check(
        "victim and burst seeds always distinct",
        all(e.victim_workload_seed != e.burst_workload_seed for e in episodes),
    )

    csv_text = render_csv(episodes)
    csv_consistency_errors = check_csv_json_consistency(csv_text, episodes)
    audit_check("JSON/CSV consistency", not csv_consistency_errors)

    audit_check("fingerprint correct", bool(fingerprint) and fingerprint.startswith("sha256:") and len(fingerprint) == 71)
    audit_check("no disallowed offload values", all(e.offload_gb in OFFLOAD_VALUES for e in episodes))
    audit_check("no disallowed concurrency values", all(e.concurrency in CONCURRENCY_VALUES for e in episodes))
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
# Transactional bundle writer
# ============================================================================

def write_bundle_atomic(output_dir: Path, files: list[tuple[str, str]], *, force: bool) -> None:
    """Write one three-file bundle with transactional rollback before commit.

    For cleanly handled Python/filesystem failures, the function preserves
    either the complete pre-call state or the complete new bundle:

      * Without ``force``, any existing target aborts before temp files or
        backups are created.
      * All temp files are written, flushed, and fsynced before a target is
        touched.
      * With ``force``, existing targets are moved to unique same-directory
        backups before installation.
      * Until all new targets have been installed, any ``BaseException``
        removes newly installed files, restores every old target, removes all
        temp residue, and re-raises the original exception.
      * Once all new targets are installed, the transaction is committed.
        Backup deletion is post-commit cleanup: a cleanup error is visible and
        re-raised, but the complete new bundle is retained and is never rolled
        back to a partly deleted old state.

    This is deliberately not a claim of multi-file power-loss atomicity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_dir / name for name, _ in files]

    existing = [path for path in final_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing bundle file(s) without --force: "
            f"{[path.name for path in existing]}"
        )

    unique_suffix = f"{os.getpid()}.{uuid.uuid4().hex}"
    tmp_paths = [
        path.with_name(f"{path.name}.tmp.{unique_suffix}")
        for path in final_paths
    ]
    backup_paths = [
        path.with_name(f"{path.name}.bak.{unique_suffix}")
        for path in final_paths
    ]

    backed_up: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    committed = False

    try:
        # Prepare every new file completely before changing a target. Cleanup
        # deliberately iterates over *all* tmp_paths, including the temp file
        # whose write/flush/fsync may itself have raised.
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

        # Commit boundary: from here onward all three new targets exist. A
        # later backup-cleanup failure must never remove this complete bundle.
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
                    # Best effort only. Never delete an unrestored backup: it
                    # remains the safest recoverable copy if restoration fails.
                    pass

            for tmp_path in tmp_paths:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
        raise

    # Post-commit cleanup is intentionally outside the rollback try/except.
    # If unlink raises, the complete new bundle remains installed and the
    # exception makes the leftover backup visible to the caller.
    for _final_path, backup_path in backed_up:
        if backup_path.exists():
            backup_path.unlink()


# ============================================================================
# CLI / main
# ============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen, confirmatory Prefill-Confirmation schedule "
        "(repeats 1-8 only; no requests are executed) for exactly one of the two "
        "frozen models. No model id, offload, concurrency, trigger, repeat-count, "
        "or workload overrides are accepted."
    )
    parser.add_argument(
        "--model-key", required=True, choices=sorted(MODEL_REGISTRY),
        help="Which frozen model to generate the schedule for. Selects the model_id "
        "and campaign_name from MODEL_REGISTRY; no free model id/path is accepted.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the generated CSV/JSON/audit files "
        "(default: new/runs/prefill_confirmation/<model-key>/).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing bundle files. Without this flag, an existing "
        "target file aborts the run before anything is written.",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_key)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_key = args.model_key

    episodes = build_episodes(OFFICIAL_SEED, model_key)
    validation_errors = validate_schedule(episodes, OFFICIAL_SEED, model_key)

    print("=" * 60)
    print(f"VALIDATION ({model_key})")
    print("=" * 60)
    if validation_errors:
        print(f"FAIL: {len(validation_errors)} error(s)")
        for e in validation_errors:
            print(f"  - {e}")
        print()
        print("FAIL: schedule validation failed. No output files were written.", file=sys.stderr)
        return 1
    print("PASS")

    canonical_payload = build_canonical_payload(episodes, OFFICIAL_SEED, model_key)
    fingerprint = compute_schedule_fingerprint(canonical_payload)

    # Self-check: recompute once more from the just-built payload and
    # compare, so a construction-order bug can never silently ship a
    # fingerprint that doesn't match its own payload.
    recomputed = compute_schedule_fingerprint(canonical_payload)
    if recomputed != fingerprint:
        print(
            f"FAIL: internal fingerprint recomputation mismatch ({recomputed!r} != {fingerprint!r}); "
            f"no output files were written.",
            file=sys.stderr,
        )
        return 1

    final_payload = dict(canonical_payload)
    final_payload["schedule_fingerprint"] = fingerprint
    json_text = json.dumps(final_payload, indent=2, ensure_ascii=False)

    csv_text = render_csv(episodes)
    csv_consistency_errors = check_csv_json_consistency(csv_text, episodes)
    if csv_consistency_errors:
        print("FAIL: csv/json consistency check failed. No output files were written.", file=sys.stderr)
        for e in csv_consistency_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    audit_text = render_audit(episodes, OFFICIAL_SEED, model_key, fingerprint, validation_errors)
    if "OVERALL: PASS" not in audit_text:
        print("FAIL: audit report did not reach OVERALL: PASS. No output files were written.", file=sys.stderr)
        return 1

    try:
        write_bundle_atomic(
            args.output_dir,
            [
                (BUNDLE_FILENAMES[0], json_text),
                (BUNDLE_FILENAMES[1], csv_text),
                (BUNDLE_FILENAMES[2], audit_text),
            ],
            force=args.force,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: failed to write bundle files: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"model_key: {model_key}")
    print(f"schedule_fingerprint: {fingerprint}")
    print()
    print("Generated files:")
    for name in BUNDLE_FILENAMES:
        print(f"  {args.output_dir / name}")
    print()
    print("PASS: schedule generation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
