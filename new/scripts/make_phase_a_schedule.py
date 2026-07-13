#!/usr/bin/env python3
"""
make_phase_a_schedule.py

Generates a reproducible, block-randomized Phase-A episode schedule.
Does NOT execute any requests -- output is a schedule (CSV + JSON) plus
a human-readable audit report.

Design per model:
  - 2 states (offload0="low", offload12="high")
  - 2 victim concurrencies (4, 8)
  - 2 conditions (no_burst, fixed_burst)
  - N repeats (default 5)
  => N * 2 * 2 * 2 episodes per model (40 at N=5)

Block structure:
  - Each repeat contributes two state-blocks, executed back to back.
  - State order alternates by repeat parity:
      repeat 1 (odd):  low  -> high
      repeat 2 (even): high -> low
      repeat 3 (odd):  low  -> high
      ...
    (avoids "all low episodes, then all high episodes" ordering, since a
    state switch requires a server restart.)
  - Within each repeat, the 4 (concurrency x condition) cells are
    randomized once and that same order is reused for both the
    low-state block and the high-state block of that repeat (using a
    per-model RNG seeded from (global_seed, model) so the order of
    --models on the CLI does not affect reproducibility). This means a
    matched low/high pair of a given (concurrency, condition) always has
    the same order_in_block.
  - Exactly one warmup request is scheduled at the start of each
    state-block (i.e. right after the implied server restart), not
    before every episode. `restart_server_before_block` flags the same
    position explicitly and independently of warmup_requests, since a
    restart is required before *every* block boundary -- including
    consecutive same-state blocks in an ABBA sequence -- even if the
    warmup policy were ever changed to 0.

Seeds (three distinct, deliberately separated):
  - schedule_seed (field: random_seed): the single global seed that
    controls block/cell randomization. Identical for every row; kept as
    a reproducibility record of *how the schedule itself was generated*.
  - episode_seed: a unique, deterministic seed per episode, derived from
    (schedule_seed, episode_id). Intended for anything that should vary
    independently per episode (e.g. per-request sampling noise, if any).
  - victim_workload_seed: deterministic from (schedule_seed, model,
    concurrency, repeat) -- deliberately NOT from offload_gb/state_label
    NOR from condition. This makes the victim workload identical across
    both states AND both conditions of a given (concurrency, repeat)
    cell: the victim sees the same prompt content whether or not a burst
    is present, and whether the state is low or high. This removes
    prompt-content as a confound both in the state x burst
    difference-in-differences analysis and in the burst-vs-no-burst
    comparison itself.
  - burst_workload_seed: same derivation but with an additional "burst"
    tag, so it is a distinct stream from victim_workload_seed while
    still being constant across states and conditions (identical burst
    content in the matched low-state and high-state episode; a
    no_burst episode simply does not use this seed, but it is still
    computed for every episode for schedule symmetry/audit).

Usage:
    python3 make_phase_a_schedule.py \
        --models llama qwen \
        --repeats 5 \
        --seed 20260711 \
        --output-dir /home/rock/projects/DOS/new/runs/phase_a
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Fixed experimental configuration
# ---------------------------------------------------------------------------

STATES: tuple[tuple[int, str], ...] = ((0, "low"), (12, "high"))
CONCURRENCIES: tuple[int, ...] = (4, 8)
CONDITIONS: tuple[str, ...] = ("no_burst", "fixed_burst")

VICTIM_INPUT_LEN = 256
VICTIM_OUTPUT_LEN = 64
VICTIM_TEMPERATURE = 0.0

BURST_PARALLEL_REQUESTS = 4
BURST_INPUT_LEN = 256
BURST_OUTPUT_LEN = 256
BURST_TEMPERATURE = 0.0

WARMUP_REQUESTS_PER_BLOCK = 1

DEFAULT_MODELS = ("llama", "qwen")
DEFAULT_REPEATS = 5
DEFAULT_SEED = 20260711


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
    victim_input_len: int
    victim_output_len: int
    victim_temperature: float
    burst_parallel_requests: int
    burst_input_len: int
    burst_output_len: int
    burst_temperature: float
    warmup_requests: int
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
# Schedule generation
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
                warmup = WARMUP_REQUESTS_PER_BLOCK if is_block_start else 0
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
                        victim_input_len=VICTIM_INPUT_LEN,
                        victim_output_len=VICTIM_OUTPUT_LEN,
                        victim_temperature=VICTIM_TEMPERATURE,
                        burst_parallel_requests=BURST_PARALLEL_REQUESTS,
                        burst_input_len=BURST_INPUT_LEN,
                        burst_output_len=BURST_OUTPUT_LEN,
                        burst_temperature=BURST_TEMPERATURE,
                        warmup_requests=warmup,
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

    design_keys = [
        (ep.offload_gb, ep.concurrency, ep.condition, ep.repeat) for ep in episodes
    ]
    if len(design_keys) != len(set(design_keys)):
        errors.append(f"model={model}: duplicate (state, concurrency, condition, repeat) combination(s) found")

    episode_ids = [ep.episode_id for ep in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append(f"model={model}: duplicate episode_id(s) found")

    for ep in episodes:
        row = asdict(ep)
        missing = [field for field in REQUIRED_FIELDS if row.get(field) is None]
        if missing:
            errors.append(
                f"model={model}: episode {ep.episode_id} missing field(s): "
                f"{', '.join(missing)}"
            )

    # --- Block-level checks ---------------------------------------------
    blocks: dict[str, list[Episode]] = {}
    block_ids_in_order: list[str] = []
    for ep in episodes:
        if ep.block_id not in blocks:
            blocks[ep.block_id] = []
            block_ids_in_order.append(ep.block_id)
        blocks[ep.block_id].append(ep)

    expected_block_size = len(CONCURRENCIES) * len(CONDITIONS)

    for block_id in block_ids_in_order:
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

        warmup_episodes = [ep for ep in block_episodes if ep.warmup_requests > 0]
        if len(warmup_episodes) != 1 or (
            warmup_episodes and warmup_episodes[0].order_in_block != 1
        ):
            errors.append(
                f"model={model}: block {block_id} does not have exactly "
                f"one warmup request at order_in_block=1"
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
        if len(repeats_in_block) != 1:
            errors.append(
                f"model={model}: block {block_id} mixes repeat values: "
                f"{sorted(repeats_in_block)}"
            )

    # --- ABBA state-sequence check ----------------------------------------
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


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(all_episodes: list[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_FIELDS))
        writer.writeheader()
        for ep in all_episodes:
            writer.writerow(asdict(ep))


def write_json(
    all_episodes: list[Episode],
    models: Sequence[str],
    repeats: int,
    seed: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": seed,
        "repeats": repeats,
        "models": list(models),
        "states": [{"offload_gb": o, "state_label": s} for o, s in STATES],
        "concurrencies": list(CONCURRENCIES),
        "conditions": list(CONDITIONS),
        "episode_count": len(all_episodes),
        "episodes": [asdict(ep) for ep in all_episodes],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_audit(
    per_model_episodes: dict[str, list[Episode]],
    per_model_errors: dict[str, list[str]],
    repeats: int,
    seed: int,
    path: Path,
    global_errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("Phase A Schedule Audit")
    lines.append("=" * 60)
    lines.append(f"seed: {seed}")
    lines.append(f"repeats per cell: {repeats}")
    lines.append(f"models: {', '.join(per_model_episodes.keys())}")
    lines.append("")

    overall_pass = not global_errors

    for model, episodes in per_model_episodes.items():
        errors = per_model_errors[model]
        status = "PASS" if not errors else "FAIL"
        overall_pass = overall_pass and not errors

        lines.append(f"--- {model}: {status} ---")
        lines.append(f"total episodes: {len(episodes)}")

        cell_counts: dict[tuple[int, int, str], int] = {}
        for ep in episodes:
            key = (ep.offload_gb, ep.concurrency, ep.condition)
            cell_counts[key] = cell_counts.get(key, 0) + 1

        lines.append("cell counts (offload_gb, concurrency, condition):")
        for key in sorted(cell_counts):
            lines.append(f"  {key}: {cell_counts[key]}")

        lines.append("block sequence (each entry implies a server restart"
                     " before its first episode):")
        seen_blocks: list[str] = []
        for ep in episodes:
            if ep.block_id not in seen_blocks:
                seen_blocks.append(ep.block_id)
        for block_id in seen_blocks:
            state_label = next(
                ep.state_label for ep in episodes if ep.block_id == block_id
            )
            lines.append(f"  {block_id} (state={state_label})")

        if errors:
            lines.append("errors:")
            for err in errors:
                lines.append(f"  - {err}")

        lines.append("")

    lines.append("=" * 60)
    lines.append("--- global checks ---")
    if global_errors:
        for err in global_errors:
            lines.append(f"  - {err}")
    else:
        lines.append("  episode_id uniqueness across merged models: PASS")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible, block-randomized Phase-A "
        "episode schedule (no requests are executed)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Model identifiers (default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Repeats per cell (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Global RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/rock/projects/DOS/new/runs/phase_a"),
        help="Directory for generated CSV/JSON/audit files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if len(args.models) != len(set(args.models)):
        duplicates = sorted(
            {m for m in args.models if args.models.count(m) > 1}
        )
        print(
            f"ERROR: duplicate model(s) in --models: {duplicates}",
            file=sys.stderr,
        )
        return 1

    if args.repeats < 1:
        print("ERROR: --repeats must be >= 1", file=sys.stderr)
        return 1

    per_model_episodes: dict[str, list[Episode]] = {}
    per_model_errors: dict[str, list[str]] = {}
    all_episodes: list[Episode] = []

    for model in args.models:
        episodes = generate_schedule(model, args.repeats, args.seed)
        errors = validate_schedule(episodes, model, args.repeats)

        per_model_episodes[model] = episodes
        per_model_errors[model] = errors
        all_episodes.extend(episodes)

    global_errors: list[str] = []
    all_ids = [ep.episode_id for ep in all_episodes]
    if len(all_ids) != len(set(all_ids)):
        duplicate_ids = sorted({eid for eid in all_ids if all_ids.count(eid) > 1})
        global_errors.append(
            f"duplicate episode_id(s) across merged models: {duplicate_ids}"
        )

    any_errors = (
        any(errors for errors in per_model_errors.values())
        or bool(global_errors)
    )

    out_csv = args.output_dir / "phase_a_schedule.csv"
    out_json = args.output_dir / "phase_a_schedule.json"
    out_audit = args.output_dir / "phase_a_schedule_audit.txt"

    write_csv(all_episodes, out_csv)
    write_json(all_episodes, args.models, args.repeats, args.seed, out_json)
    write_audit(
        per_model_episodes,
        per_model_errors,
        args.repeats,
        args.seed,
        out_audit,
        global_errors,
    )

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

    print()
    print("Generated files:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_audit}")

    if any_errors:
        print()
        print("FAIL: schedule generated but validation failed.")
        return 1

    print()
    print("PASS: schedule generation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
