#!/usr/bin/env python3
"""
make_chunk_budget_schedule.py

Generates the frozen, explorative Chunk-Budget-Screen episode schedule --
reproducibly, self-validating, and only publishing output once every
check has passed. Does NOT execute any requests -- output is a schedule
(CSV + JSON) plus a human-readable audit report.

This is a fully separate, exploratory screening design, mechanically
adapted from the frozen Prefill-Screen generator
(make_prefill_screen_schedule.py, untouched, not imported here). It does
NOT modify, extend, or share output files with Prefill-Screen or Phase A.

Scientific purpose (see project prompt): does the vLLM chunked-prefill
budget (--max-num-batched-tokens) modulate the state-dependent victim
degradation under a bounded prefill-heavy burst? This is an explorative
screen, not a confirmatory study -- no policies, SLOs, mitigations,
additional models, additional concurrencies, or additional load points
are part of this design.

Design (single model, "llama" only):
  - 2 states (offload0="low", offload12="high")
  - 3 chunked-prefill budgets (max_num_batched_tokens = 512, 1024, 2048)
  - victim concurrency: 4 (fixed -- no longer a schedule dimension)
  - 2 conditions (no_burst, prefill_burst)
  - 3 repeats (frozen)
  => 2 * 3 * 2 * 3 = 36 episodes total

Block structure:
  - A block is exactly one (state, budget, repeat) combination and
    contains exactly its 2 regular episodes (no_burst, prefill_burst).
  - 2 states * 3 budgets * 3 repeats = 18 blocks; 18 * 2 = 36 episodes.
  - Every block requires a server restart: chunked-prefill budget and
    cpu_offload_gb are both vLLM startup flags that cannot change on a
    running server, so `restart_server_before_block` is set at every
    block's first episode (within_block_order == 1), with no exception --
    including a same-state block immediately followed by a
    different-budget block of the same state.
  - Within each repeat, state alternates low/high by repeat parity
    (repeat 1 odd: low -> high, repeat 2 even: high -> low, repeat 3
    odd: low -> high), matching the Prefill-Screen convention. Within a
    given state, the 3 budgets are visited in a fixed, deterministic
    order (512, 1024, 2048) for simplicity and reproducibility -- there
    is no scientific reason in this design to vary budget order.
  - The order of the 2 conditions within EACH block is independently,
    deterministically randomized via a single per-model RNG seeded from
    (schedule_seed, model), advanced sequentially block by block in
    schedule order.

  The binding per-block runtime protocol, executed later by
  run_chunk_budget_screen.py (NOT by this generator), is:
      server start (with the block's max_num_batched_tokens)
      -> API readiness check via polling (not a generation request)
      -> verify the server log confirms chunked prefill is enabled with
         the expected budget
      -> exactly one full stabilization run
      -> stability diagnostics saved
      -> drain/cooldown
      -> two regular episodes (this block's no_burst/prefill_burst pair,
         in the randomized order recorded here)
  There is no separate short warmup/probe generation request in addition
  to the readiness poll and the stabilization run. This schedule
  therefore contains only the 36 regular episodes; the stabilization run
  is documented as top-level protocol metadata (see
  `STABILIZATION_CONFIGURATION` below) rather than as a schedule row, a
  repeat, or a CSV line, and this generator does not execute it.

Seeds (three distinct, deliberately separated) -- same derivation
formula and semantics as the frozen Phase A / Prefill-Screen generators:
  - schedule_seed: the single global seed that controls block/condition
    randomization. Identical for every row; kept as a reproducibility
    record of *how the schedule itself was generated*.
  - episode_seed: a unique, deterministic seed per episode, derived from
    (schedule_seed, episode_id).
  - victim_workload_seed: deterministic from (schedule_seed, model,
    repeat) -- deliberately NOT from offload_gb/state_label, NOT from
    max_num_batched_tokens/budget, and NOT from condition. Concurrency is
    fixed at 4 for every episode, so it is not part of this derivation
    either. This makes the victim workload identical across both
    states, all three budgets, AND both conditions of a given repeat --
    isolating the state/budget/burst comparisons from prompt-content
    confounds.
  - burst_workload_seed: same derivation but with an additional "burst"
    tag, so it is a distinct stream from victim_workload_seed while
    still being constant across states, budgets, and conditions. A
    no_burst episode simply does not use this seed, but it is still
    computed for every episode for schedule symmetry/audit.

The Chunk-Budget-Screen design is frozen. The CLI intentionally accepts
only the one official invocation (defaults, or the same values passed
explicitly) plus a free --output-dir override; see `validate_frozen_cli`.

Usage:
    python3 make_chunk_budget_schedule.py
    python3 make_chunk_budget_schedule.py --models llama --repeats 3 \
        --seed 20260716 --output-dir /path/to/runs/chunk_budget_screen
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
DESIGN_VERSION = "chunk-budget-screen-v1"

STATES: tuple[tuple[int, str], ...] = ((0, "low"), (12, "high"))
BUDGETS: tuple[int, ...] = (512, 1024, 2048)
CONCURRENCY = 4
BURST_CONDITION = "prefill_burst"
CONDITIONS: tuple[str, ...] = ("no_burst", BURST_CONDITION)

VICTIM_REQUEST_COUNT = 20
VICTIM_INPUT_LEN = 256
VICTIM_OUTPUT_LEN = 64
VICTIM_TEMPERATURE = 0.0

# Prefill-heavy, bounded burst -- unchanged from Prefill-Screen (see
# module docstring); only the server's chunked-prefill budget varies in
# this screen, not the burst shape itself.
BURST_PARALLEL_REQUESTS = 4
BURST_INPUT_LEN = 2048
BURST_OUTPUT_LEN = 16
BURST_TEMPERATURE = 0.0

# Documents (but does not execute) the mandatory per-block runtime
# protocol that run_chunk_budget_screen.py must follow:
#   server start -> API readiness check via polling (NOT a generation
#   request) -> chunked-prefill server-log verification -> exactly one
#   full stabilization run -> stability diagnostics saved ->
#   drain/cooldown -> two regular episodes.
# There is deliberately no separate short warmup/probe generation
# request (generation_probe_requests is 0). These runs are explicitly
# NOT part of this schedule: not a regular episode, not a repeat, not a
# CSV row. Stabilization itself is unchanged from Prefill-Screen/Phase A
# (no_burst / concurrency 4 / 256 input / 64 output) -- only the block
# dimension (budget) differs in this screen.
#
# abort_on_stability_drift is deliberately False for the pilot, so no
# block is discarded based on an arbitrarily-chosen performance
# threshold; the later runner must fully log the drift instead. Aborts
# here are reserved for functional failures only: missing requests,
# timeouts, HTTP/streaming errors, truncated outputs, wrong
# prompt/output token counts, or a missing/mismatched chunked-prefill
# budget confirmation in the server log.
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
    "chunked_prefill_log_verification_required": True,
}

# Frozen official invocation. Explorative Chunk-Budget-Screen: single
# model (llama only), 3 repeats, distinct schedule seed -- see module
# docstring for the full contract and its scientific rationale.
DEFAULT_MODELS = ("llama",)
DEFAULT_REPEATS = 3
DEFAULT_SEED = 20260716

# Portable default output directory, derived from this file's location.
# Expected location:
#   <PROJECT_ROOT>/new/scripts/chunk_budget_screen/make_chunk_budget_schedule.py
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "new" / "runs" / "chunk_budget_screen"


# ---------------------------------------------------------------------------
# Episode schema
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    design_version: str
    schedule_seed: int
    block_id: str
    block_index: int
    episode_id: str
    state_label: str
    offload_gb: int
    max_num_batched_tokens: int
    concurrency: int
    condition: str
    repeat: int
    within_block_order: int
    model: str
    victim_input_len: int
    victim_output_len: int
    victim_request_count: int
    victim_temperature: float
    burst_parallel: int
    burst_input_len: int
    burst_output_len: int
    burst_temperature: float
    episode_seed: int
    victim_workload_seed: int
    burst_workload_seed: int
    restart_server_before_block: int


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
# Schedule generation
# ---------------------------------------------------------------------------

def budget_order_for_repeat(repeat: int) -> tuple[int, ...]:
    """Latin-square rotation: repeat 1 -> (512,1024,2048), repeat 2 ->
    (1024,2048,512), repeat 3 -> (2048,512,1024). Guarantees that, within
    each state, every budget occupies each of the three within-repeat
    budget positions (1st, 2nd, 3rd) exactly once across the 3 repeats."""
    n = len(BUDGETS)
    shift = (repeat - 1) % n
    return tuple(BUDGETS[(i + shift) % n] for i in range(n))


def _cell_keys() -> list[tuple[str, int]]:
    """Canonical (state_label, budget) cell order: states in their fixed
    STATES order, budgets in their fixed BUDGETS order. Used only to fix
    a deterministic order for the balanced condition-first assignment
    below -- it is NOT the schedule's block execution order."""
    return [(state_label, budget) for _offload_gb, state_label in STATES for budget in BUDGETS]


def _build_condition_first_plan(
    rng: random.Random, repeats: int
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], int]]:
    """Deterministically (from the given, already-seeded rng) assigns,
    per (state, budget) cell, a majority first-condition and exactly one
    "minority" repeat where the other condition goes first instead --
    giving every cell a 2/1 split (never 3/0) across its `repeats`
    occurrences. The 6 cells are split exactly 3/3 between
    'no_burst majority' and 'prefill_burst majority', which is what
    yields exactly 9/9 over all 18 blocks (3*2 + 3*1 = 9 no_burst-first,
    and symmetrically 9 prefill_burst-first).
    """
    cell_keys = _cell_keys()
    half = len(cell_keys) // 2
    majority_labels = ["no_burst"] * half + [BURST_CONDITION] * (len(cell_keys) - half)
    rng.shuffle(majority_labels)
    cell_majority = dict(zip(cell_keys, majority_labels))

    cell_minority_repeat: dict[tuple[str, int], int] = {}
    for key in cell_keys:
        cell_minority_repeat[key] = rng.randint(1, repeats)

    return cell_majority, cell_minority_repeat


def _first_condition_for(
    cell_majority: dict[tuple[str, int], str],
    cell_minority_repeat: dict[tuple[str, int], int],
    state_label: str,
    budget: int,
    repeat: int,
) -> str:
    key = (state_label, budget)
    majority = cell_majority[key]
    minority = BURST_CONDITION if majority == "no_burst" else "no_burst"
    return minority if repeat == cell_minority_repeat[key] else majority


def generate_schedule(model: str, repeats: int, seed: int) -> list[Episode]:
    rng = random.Random(f"{seed}:{model}")

    # Both precomputed plans deliberately draw from the SAME rng object,
    # in this fixed order (condition-first plan first, then the main
    # per-block loop draws nothing further from rng), so the whole
    # schedule remains fully reproducible from (seed, model) alone.
    cell_majority, cell_minority_repeat = _build_condition_first_plan(rng, repeats)

    episodes: list[Episode] = []
    block_number = 0

    for repeat in range(1, repeats + 1):
        state_order = STATES if repeat % 2 == 1 else tuple(reversed(STATES))
        budget_order = budget_order_for_repeat(repeat)

        # Low/High blocks are interleaved budget-by-budget within each
        # repeat (instead of all of one state's budgets, then all of the
        # other's), so no six -- or even three -- consecutive blocks of
        # the same state occur mid-repeat. See module docstring.
        for budget in budget_order:
            for offload_gb, state_label in state_order:
                block_number += 1
                block_id = f"{model}_block{block_number:02d}_{state_label}_budget{budget}"

                first_condition = _first_condition_for(
                    cell_majority, cell_minority_repeat, state_label, budget, repeat
                )
                second_condition = (
                    BURST_CONDITION if first_condition == "no_burst" else "no_burst"
                )
                ordered_conditions = [first_condition, second_condition]

                for within_block_order, condition in enumerate(ordered_conditions, start=1):
                    is_block_start = within_block_order == 1
                    restart = 1 if is_block_start else 0

                    episode_id = (
                        f"{model}_off{offload_gb}_budget{budget}_conc{CONCURRENCY}_"
                        f"{condition}_rep{repeat}"
                    )

                    episode_seed = derive_seed(str(seed), episode_id)
                    # Deliberately independent of offload_gb/state_label,
                    # max_num_batched_tokens/budget, AND condition, so the
                    # victim sees identical prompt content across both
                    # states, all three budgets, and both conditions of
                    # the same repeat -- see module docstring. Concurrency
                    # is fixed at 4 for every episode and therefore not
                    # part of the derivation.
                    victim_workload_seed = derive_seed(str(seed), model, str(repeat))
                    burst_workload_seed = derive_seed(str(seed), model, str(repeat), "burst")

                    episodes.append(
                        Episode(
                            design_version=DESIGN_VERSION,
                            schedule_seed=seed,
                            block_id=block_id,
                            block_index=block_number,
                            episode_id=episode_id,
                            state_label=state_label,
                            offload_gb=offload_gb,
                            max_num_batched_tokens=budget,
                            concurrency=CONCURRENCY,
                            condition=condition,
                            repeat=repeat,
                            within_block_order=within_block_order,
                            model=model,
                            victim_input_len=VICTIM_INPUT_LEN,
                            victim_output_len=VICTIM_OUTPUT_LEN,
                            victim_request_count=VICTIM_REQUEST_COUNT,
                            victim_temperature=VICTIM_TEMPERATURE,
                            burst_parallel=BURST_PARALLEL_REQUESTS,
                            burst_input_len=BURST_INPUT_LEN,
                            burst_output_len=BURST_OUTPUT_LEN,
                            burst_temperature=BURST_TEMPERATURE,
                            episode_seed=episode_seed,
                            victim_workload_seed=victim_workload_seed,
                            burst_workload_seed=burst_workload_seed,
                            restart_server_before_block=restart,
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

    expected_total = len(STATES) * len(BUDGETS) * len(CONDITIONS) * repeats
    if len(episodes) != expected_total:
        errors.append(
            f"model={model}: expected {expected_total} episodes, "
            f"found {len(episodes)}"
        )

    cell_counts: dict[tuple[int, int, str], int] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.max_num_batched_tokens, ep.condition)
        cell_counts[key] = cell_counts.get(key, 0) + 1

    expected_cells = {
        (offload_gb, budget, condition)
        for offload_gb, _ in STATES
        for budget in BUDGETS
        for condition in CONDITIONS
    }

    for key in sorted(expected_cells):
        count = cell_counts.get(key, 0)
        if count != repeats:
            errors.append(
                f"model={model}: cell offload={key[0]}, budget={key[1]}, "
                f"condition={key[2]!r} occurs {count} time(s), expected {repeats}"
            )

    unexpected_keys = set(cell_counts) - expected_cells
    if unexpected_keys:
        errors.append(f"model={model}: unexpected cell(s): {sorted(unexpected_keys)}")

    # --- Exact repeat-set check per design cell -----------------------------
    repeats_by_cell: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.max_num_batched_tokens, ep.condition)
        repeats_by_cell.setdefault(key, set()).add(ep.repeat)

    expected_repeat_set = set(range(1, repeats + 1))
    for key in sorted(expected_cells):
        actual_repeat_set = repeats_by_cell.get(key, set())
        if actual_repeat_set != expected_repeat_set:
            errors.append(
                f"model={model}: cell offload={key[0]}, budget={key[1]}, "
                f"condition={key[2]!r} has repeat value(s) "
                f"{sorted(actual_repeat_set)}, expected exactly "
                f"{sorted(expected_repeat_set)}"
            )

    design_keys = [
        (ep.offload_gb, ep.max_num_batched_tokens, ep.condition, ep.repeat)
        for ep in episodes
    ]
    if len(design_keys) != len(set(design_keys)):
        errors.append(
            f"model={model}: duplicate (state, budget, condition, repeat) "
            f"combination(s) found"
        )

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
        if ep.design_version != DESIGN_VERSION:
            errors.append(
                f"{ctx}: design_version {ep.design_version!r} != "
                f"{DESIGN_VERSION!r}"
            )
        if ep.schedule_seed != seed:
            errors.append(
                f"{ctx}: schedule_seed {ep.schedule_seed!r} does not match "
                f"the official seed {seed!r}"
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

        if ep.max_num_batched_tokens not in BUDGETS:
            errors.append(
                f"{ctx}: invalid max_num_batched_tokens "
                f"{ep.max_num_batched_tokens!r}, expected one of {BUDGETS}"
            )

        if ep.concurrency != CONCURRENCY:
            errors.append(
                f"{ctx}: concurrency {ep.concurrency!r} != frozen value "
                f"{CONCURRENCY!r}"
            )

        if ep.condition not in CONDITIONS:
            errors.append(
                f"{ctx}: invalid condition {ep.condition!r}, expected one "
                f"of {CONDITIONS}"
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

        if ep.burst_parallel != BURST_PARALLEL_REQUESTS:
            errors.append(
                f"{ctx}: burst_parallel {ep.burst_parallel!r} != "
                f"{BURST_PARALLEL_REQUESTS}"
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

        if ep.within_block_order == 1:
            if ep.restart_server_before_block != 1:
                errors.append(
                    f"{ctx}: within_block_order=1 requires "
                    f"restart_server_before_block==1, found "
                    f"{ep.restart_server_before_block!r}"
                )
        else:
            if ep.restart_server_before_block != 0:
                errors.append(
                    f"{ctx}: within_block_order={ep.within_block_order} "
                    f"requires restart_server_before_block==0, found "
                    f"{ep.restart_server_before_block!r}"
                )

        # --- Deterministic re-derivation checks --------------------------
        if ep.model != model:
            errors.append(
                f"{ctx}: episode.model {ep.model!r} does not match the "
                f"model being validated {model!r}"
            )

        expected_episode_id = (
            f"{model}_off{ep.offload_gb}_budget{ep.max_num_batched_tokens}_"
            f"conc{ep.concurrency}_{ep.condition}_rep{ep.repeat}"
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

        expected_victim_seed = derive_seed(str(seed), model, str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(
                f"{ctx}: victim_workload_seed {ep.victim_workload_seed!r} "
                f"does not match the expected derivation "
                f"{expected_victim_seed!r}"
            )

        expected_burst_seed = derive_seed(str(seed), model, str(ep.repeat), "burst")
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

    expected_blocks_per_model = len(STATES) * len(BUDGETS) * repeats
    if len(block_ids_in_order) != expected_blocks_per_model:
        errors.append(
            f"model={model}: expected {expected_blocks_per_model} blocks, "
            f"found {len(block_ids_in_order)}"
        )

    expected_block_size = len(CONDITIONS)

    for position, block_id in enumerate(block_ids_in_order, start=1):
        block_episodes = blocks[block_id]

        if len(block_episodes) != expected_block_size:
            errors.append(
                f"model={model}: block {block_id} has "
                f"{len(block_episodes)} episode(s), expected "
                f"{expected_block_size}"
            )

        order_positions = sorted(ep.within_block_order for ep in block_episodes)
        expected_positions = list(range(1, expected_block_size + 1))
        if order_positions != expected_positions:
            errors.append(
                f"model={model}: block {block_id} within_block_order "
                f"values {order_positions}, expected {expected_positions}"
            )

        restart_episodes = [
            ep for ep in block_episodes if ep.restart_server_before_block > 0
        ]
        if len(restart_episodes) != 1 or (
            restart_episodes and restart_episodes[0].within_block_order != 1
        ):
            errors.append(
                f"model={model}: block {block_id} does not have exactly "
                f"one restart_server_before_block flag at "
                f"within_block_order=1"
            )

        state_labels = {ep.state_label for ep in block_episodes}
        offloads = {ep.offload_gb for ep in block_episodes}
        budgets_in_block = {ep.max_num_batched_tokens for ep in block_episodes}
        repeats_in_block = {ep.repeat for ep in block_episodes}
        block_indices = {ep.block_index for ep in block_episodes}

        if len(state_labels) != 1 or len(offloads) != 1 or len(budgets_in_block) != 1:
            errors.append(
                f"model={model}: block {block_id} mixes state/offload/"
                f"budget values: states={sorted(state_labels)}, "
                f"offloads={sorted(offloads)}, budgets={sorted(budgets_in_block)}"
            )
        if len(block_indices) != 1 or next(iter(block_indices)) != position:
            errors.append(
                f"model={model}: block {block_id} has block_index "
                f"{sorted(block_indices)}, expected exactly {{{position}}}"
            )

        # Block position -> expected repeat: blocks 1-3 are repeat 1 (one
        # per budget), blocks 4-6 are repeat 2, etc. (every repeat
        # contributes exactly len(STATES)*len(BUDGETS)=6 consecutive
        # blocks). This checks the actual repeat VALUE.
        blocks_per_repeat = len(STATES) * len(BUDGETS)
        expected_repeat_for_block = ((position - 1) // blocks_per_repeat) + 1
        if repeats_in_block != {expected_repeat_for_block}:
            errors.append(
                f"model={model}: block {block_id} (position {position}) "
                f"has repeat value(s) {sorted(repeats_in_block)}, "
                f"expected exactly {{{expected_repeat_for_block}}}"
            )

    # --- Contiguous block / execution-order checks --------------------------
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
            order_sequence = [ep.within_block_order for ep in run_episodes]
            expected_order_sequence = list(range(1, expected_block_size + 1))
            if order_sequence != expected_order_sequence:
                errors.append(
                    f"model={model}: contiguous block {bid!r} has "
                    f"within_block_order sequence {order_sequence} in list "
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

    # --- Expected (state, budget) block sequence, block_id format check ----
    # Independently re-derives the exact block sequence the generator
    # produces (Latin-square budget rotation per repeat, low/high blocks
    # interleaved budget-by-budget within each repeat) and compares it
    # against the actual data, position by position.
    expected_state_budget_sequence: list[tuple[str, int]] = []
    for repeat in range(1, repeats + 1):
        order = STATES if repeat % 2 == 1 else tuple(reversed(STATES))
        budget_order = budget_order_for_repeat(repeat)
        for budget in budget_order:
            for _offload_gb, label in order:
                expected_state_budget_sequence.append((label, budget))

    actual_state_budget_sequence = [
        (blocks[block_id][0].state_label, blocks[block_id][0].max_num_batched_tokens)
        for block_id in block_ids_in_order
    ]

    if actual_state_budget_sequence != expected_state_budget_sequence:
        errors.append(
            f"model={model}: block (state, budget) sequence "
            f"{actual_state_budget_sequence} does not match the expected "
            f"interleaved/rotated sequence {expected_state_budget_sequence}"
        )

    for position, block_id in enumerate(block_ids_in_order, start=1):
        if position - 1 < len(expected_state_budget_sequence):
            expected_state, expected_budget = expected_state_budget_sequence[position - 1]
            expected_block_id = (
                f"{model}_block{position:02d}_{expected_state}_budget{expected_budget}"
            )
            if block_id != expected_block_id:
                errors.append(
                    f"model={model}: block at position {position} has "
                    f"block_id {block_id!r}, expected {expected_block_id!r}"
                )

    # --- Budget-order-within-(repeat,state) check ---------------------------
    # Within each repeat, filtering to just one state's (non-contiguous,
    # interleaved) blocks, the budgets encountered in schedule order must
    # equal that repeat's Latin-square rotation -- not the fixed canonical
    # order, and not merely a set.
    for repeat in range(1, repeats + 1):
        expected_budget_order = list(budget_order_for_repeat(repeat))
        for _offload_gb, state_label in STATES:
            budgets_here = [
                blocks[bid][0].max_num_batched_tokens
                for bid in block_ids_in_order
                if blocks[bid][0].repeat == repeat and blocks[bid][0].state_label == state_label
            ]
            if budgets_here != expected_budget_order:
                errors.append(
                    f"model={model}: repeat={repeat} state={state_label!r} "
                    f"has budget order {budgets_here}, expected the "
                    f"Latin-square rotation {expected_budget_order}"
                )

    # --- Budget-position balance check ---------------------------------------
    # Each budget must occupy each of the 3 within-repeat budget positions
    # (1st/2nd/3rd) exactly once across the repeats, independently within
    # each state (a direct, data-driven check of the Latin-square property
    # -- not just trust in the rotation formula).
    budget_position_by_state: dict[str, dict[int, list[int]]] = {
        state_label: {budget: [] for budget in BUDGETS} for _offload_gb, state_label in STATES
    }
    for repeat in range(1, repeats + 1):
        for _offload_gb, state_label in STATES:
            budgets_here = [
                blocks[bid][0].max_num_batched_tokens
                for bid in block_ids_in_order
                if blocks[bid][0].repeat == repeat and blocks[bid][0].state_label == state_label
            ]
            for position, budget in enumerate(budgets_here, start=1):
                budget_position_by_state[state_label][budget].append(position)

    for state_label, per_budget in budget_position_by_state.items():
        for budget, positions in per_budget.items():
            if sorted(positions) != list(range(1, len(BUDGETS) + 1)):
                errors.append(
                    f"model={model}: state={state_label!r} budget={budget} "
                    f"occupies within-repeat position(s) {positions}, "
                    f"expected each of {list(range(1, len(BUDGETS) + 1))} "
                    f"exactly once across the {repeats} repeats"
                )

    # --- Condition-first balance checks --------------------------------------
    # Global: exactly repeats*len(STATES)*len(BUDGETS)/2 no_burst-first and
    # the same number of prefill_burst-first, across all blocks.
    first_conditions = [
        min(block_episodes, key=lambda ep: ep.within_block_order).condition
        for block_episodes in blocks.values()
    ]
    no_burst_first_count = sum(1 for c in first_conditions if c == "no_burst")
    burst_first_count = sum(1 for c in first_conditions if c == BURST_CONDITION)
    expected_half = (len(STATES) * len(BUDGETS) * repeats) // 2
    if no_burst_first_count != expected_half or burst_first_count != expected_half:
        errors.append(
            f"model={model}: condition-first balance is "
            f"no_burst={no_burst_first_count}/prefill_burst={burst_first_count}, "
            f"expected exactly {expected_half}/{expected_half}"
        )

    # Per (state, budget) cell across its `repeats` occurrences: never 3/0
    # (or generally repeats/0) -- every cell must have BOTH conditions
    # appearing first at least once (a 2/1 split when repeats == 3).
    first_condition_by_cell: dict[tuple[str, int], list[str]] = {}
    for block_id in block_ids_in_order:
        first_ep = min(blocks[block_id], key=lambda ep: ep.within_block_order)
        key = (first_ep.state_label, first_ep.max_num_batched_tokens)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)

    for key, conditions_seen in sorted(first_condition_by_cell.items()):
        distinct = set(conditions_seen)
        if len(conditions_seen) == repeats and len(distinct) < 2:
            errors.append(
                f"model={model}: cell state={key[0]!r} budget={key[1]} has "
                f"condition {conditions_seen[0]!r} first in all {repeats} "
                f"repeats; every cell must have a mixed (e.g. 2/1) split, "
                f"never all the same"
            )

    # --- Workload seed pairing check --------------------------------------
    # victim_workload_seed and burst_workload_seed must each be constant
    # across all 12 matched episodes (both states x all 3 budgets x both
    # conditions) of a given repeat, so prompt content cannot confound the
    # state/budget/burst comparisons.
    victim_seeds_by_repeat: dict[int, set[int]] = {}
    burst_seeds_by_repeat: dict[int, set[int]] = {}
    episodes_per_repeat: dict[int, int] = {}

    for ep in episodes:
        victim_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.burst_workload_seed)
        episodes_per_repeat[ep.repeat] = episodes_per_repeat.get(ep.repeat, 0) + 1

    expected_episodes_per_repeat = len(STATES) * len(BUDGETS) * len(CONDITIONS)

    for repeat in sorted(victim_seeds_by_repeat):
        if episodes_per_repeat[repeat] != expected_episodes_per_repeat:
            errors.append(
                f"model={model}: repeat={repeat} has "
                f"{episodes_per_repeat[repeat]} episode(s), expected "
                f"{expected_episodes_per_repeat} (2 states x 3 budgets x "
                f"2 conditions)"
            )

        victim_seeds = victim_seeds_by_repeat[repeat]
        if len(victim_seeds) != 1:
            errors.append(
                f"model={model}: victim_workload_seed not constant across "
                f"states/budgets/conditions for repeat={repeat}: "
                f"{sorted(victim_seeds)}"
            )

        burst_seeds = burst_seeds_by_repeat[repeat]
        if len(burst_seeds) != 1:
            errors.append(
                f"model={model}: burst_workload_seed not constant across "
                f"states/budgets/conditions for repeat={repeat}: "
                f"{sorted(burst_seeds)}"
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

    expected_total = len(models) * len(STATES) * len(BUDGETS) * len(CONDITIONS) * repeats
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
        duplicate_seeds = sorted({s for s in all_seeds if all_seeds.count(s) > 1})
        errors.append(
            f"global: duplicate episode_seed(s) across merged models: "
            f"{duplicate_seeds}"
        )

    # Exactly one restart_server_before_block=1 marker per block, and one
    # block per (model, state, budget) x repeat.
    expected_restart_markers = len(models) * len(STATES) * len(BUDGETS) * repeats
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
        "budgets": list(BUDGETS),
        "concurrency": CONCURRENCY,
        "conditions": list(CONDITIONS),
        "victim_configuration": {
            "victim_request_count": VICTIM_REQUEST_COUNT,
            "victim_input_len": VICTIM_INPUT_LEN,
            "victim_output_len": VICTIM_OUTPUT_LEN,
            "victim_temperature": VICTIM_TEMPERATURE,
        },
        "burst_configuration": {
            "burst_parallel": BURST_PARALLEL_REQUESTS,
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
    lines.append("Chunk-Budget-Screen Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"design_version: {DESIGN_VERSION}")
    lines.append(f"schedule_fingerprint: {fingerprint}")
    lines.append(f"seed: {seed}")
    lines.append(f"repeats per cell: {repeats}")
    lines.append(f"models: {', '.join(per_model_episodes.keys())}")
    lines.append(f"total episodes (all models): {len(all_episodes)}")
    lines.append("")

    total_no_burst = sum(1 for ep in all_episodes if ep.condition == "no_burst")
    total_burst = sum(1 for ep in all_episodes if ep.condition == BURST_CONDITION)

    for model, episodes in per_model_episodes.items():
        lines.append(f"--- {model}: PASS ---")
        lines.append(f"total episodes: {len(episodes)}")

        block_ids: list[str] = []
        for ep in episodes:
            if ep.block_id not in block_ids:
                block_ids.append(ep.block_id)
        lines.append(f"blocks: {len(block_ids)}")
        lines.append(f"no_burst episodes: {sum(1 for ep in episodes if ep.condition == 'no_burst')}")
        lines.append(f"prefill_burst episodes: {sum(1 for ep in episodes if ep.condition == BURST_CONDITION)}")

        state_sequence = []
        for block_id in block_ids:
            state_sequence.append(
                next(ep.state_label for ep in episodes if ep.block_id == block_id)
            )
        lines.append(f"state sequence: {', '.join(state_sequence)}")

        cell_counts: dict[tuple[int, int, str], int] = {}
        for ep in episodes:
            key = (ep.offload_gb, ep.max_num_batched_tokens, ep.condition)
            cell_counts[key] = cell_counts.get(key, 0) + 1

        lines.append("cell counts (offload_gb, max_num_batched_tokens, condition):")
        for key in sorted(cell_counts):
            lines.append(f"  {key}: {cell_counts[key]}")

        lines.append(f"episodes per (state, budget): {len(CONDITIONS) * repeats}")
        lines.append(f"blocks per (state, budget): {repeats}")
        lines.append(f"episodes per block: {len(CONDITIONS)}")

        blocks_by_id: dict[str, list[Episode]] = {}
        for ep in episodes:
            blocks_by_id.setdefault(ep.block_id, []).append(ep)

        first_conditions = [
            min(bes, key=lambda e: e.within_block_order).condition
            for bes in blocks_by_id.values()
        ]
        no_burst_first = sum(1 for c in first_conditions if c == "no_burst")
        burst_first = sum(1 for c in first_conditions if c == BURST_CONDITION)
        lines.append(
            f"condition-first balance overall: no_burst={no_burst_first}, "
            f"prefill_burst={burst_first}"
        )

        first_condition_by_cell: dict[tuple[str, int], list[str]] = {}
        for bid, bes in blocks_by_id.items():
            first_ep = min(bes, key=lambda e: e.within_block_order)
            key = (first_ep.state_label, first_ep.max_num_batched_tokens)
            first_condition_by_cell.setdefault(key, []).append(first_ep.condition)

        lines.append("condition-first balance per (state, budget):")
        for key in sorted(first_condition_by_cell):
            conds = first_condition_by_cell[key]
            lines.append(
                f"  state={key[0]}, budget={key[1]}: no_burst="
                f"{conds.count('no_burst')}, prefill_burst="
                f"{conds.count(BURST_CONDITION)}"
            )

        lines.append("budget positions per (state, repeat) (within-repeat order):")
        for repeat_n in range(1, repeats + 1):
            for _offload_gb, state_label in STATES:
                budgets_here = [
                    blocks_by_id[bid][0].max_num_batched_tokens
                    for bid in block_ids
                    if blocks_by_id[bid][0].repeat == repeat_n
                    and blocks_by_id[bid][0].state_label == state_label
                ]
                lines.append(f"  repeat={repeat_n}, state={state_label}: {budgets_here}")

        lines.append(
            "block sequence (each entry implies a server restart before "
            "its first episode):"
        )
        for block_id in block_ids:
            state_label = next(
                ep.state_label for ep in episodes if ep.block_id == block_id
            )
            budget = next(
                ep.max_num_batched_tokens for ep in episodes if ep.block_id == block_id
            )
            lines.append(f"  {block_id} (state={state_label}, budget={budget})")

        lines.append("")

    lines.append("=" * 60)
    lines.append("--- global checks ---")
    lines.append("  episode_id uniqueness across merged models: PASS")
    lines.append("  episode_seed uniqueness across merged models: PASS")
    lines.append("  csv/json consistency: PASS")
    lines.append(f"  total no_burst episodes: {total_no_burst}")
    lines.append(f"  total prefill_burst episodes: {total_burst}")
    lines.append(f"  budgets exactly {{512, 1024, 2048}}: PASS")
    lines.append(f"  offload exactly {{0, 12}}: PASS")
    lines.append(f"  concurrency exclusively 4: PASS")
    lines.append("")

    lines.append(
        "--- stabilization protocol (documented for "
        "run_chunk_budget_screen.py; not executed by this generator) ---"
    )
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
        "with a mix of old and new files. Before use, "
        "run_chunk_budget_screen.py must: (1) recompute the JSON's "
        "schedule_fingerprint from its own canonical payload and confirm "
        "it matches the stored schedule_fingerprint, (2) compare the CSV "
        "episodes against the JSON episodes, and (3) compare this audit "
        "report's schedule_fingerprint against the JSON's "
        "schedule_fingerprint. Any mismatch means the files were not "
        "produced by the same run and must be treated as invalid."
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
    another is not a single transaction. If this process is killed, or
    a later os.replace() call fails after an earlier one already
    succeeded, the output directory can be left with a MIX of old and
    new chunk_budget_schedule.csv / .json / audit.txt files. This
    function does not attempt to roll back an earlier successful
    replace() if a later one fails.

    Consequently, callers (in particular run_chunk_budget_screen.py)
    MUST NOT assume the three published files are mutually consistent
    just because a previous run of this generator exited 0. Before
    using them, run_chunk_budget_screen.py must at minimum:
      1. recompute the JSON's schedule_fingerprint from its own
         canonical payload and confirm it matches the JSON's stored
         schedule_fingerprint,
      2. compare the CSV episodes against the JSON episodes,
      3. compare the audit report's schedule_fingerprint against the
         JSON's schedule_fingerprint.
    Any mismatch means the three files were not produced together by
    one successful run and must be treated as invalid.

    On any failure during this function, already-written temp files
    are removed on a best-effort basis and the exception is re-raised.
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
        description="Generate the frozen, explorative Chunk-Budget-Screen "
        "episode schedule "
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
    The Chunk-Budget-Screen design is frozen: --models/--repeats/--seed
    may be omitted (using the frozen defaults) or passed explicitly, but
    only if they match the official values exactly -- same models, same
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
            "The Chunk-Budget-Screen design is frozen. Allowed invocation: "
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

    out_csv = args.output_dir / "chunk_budget_schedule.csv"
    out_json = args.output_dir / "chunk_budget_schedule.json"
    out_audit = args.output_dir / "chunk_budget_schedule_audit.txt"

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
