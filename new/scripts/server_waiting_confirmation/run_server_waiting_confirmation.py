#!/usr/bin/env python3
"""
run_server_waiting_confirmation.py -- server-side WAITING replication of
the audited vLLM Prefill-Confirmation study.

Completely separate from, and never modifies, anything under
new/scripts/prefill_confirmation/. This module imports
run_prefill_confirmation.py (the audited Schema-5 runner) ONLY as a
library, for infrastructure that is genuinely unchanged by this
extension: the HTTP streaming request executor, dependency-injectable
Clock/Sleeper/HTTPTransport/TokenizerAdapter (+ Fakes), server lifecycle
(start/readiness/stop), environment fingerprinting, integrity-manifest
build/verify, atomic JSON writes, and stabilization. See AUDIT.md for
the complete list of what is imported unchanged versus newly written
for this extension, and why.

What is genuinely new here (see
CLAUDE_SERVER_WAITING_IMPLEMENTATION_PROMPT_FINAL.md for the full
design rationale):
  - No client-side concurrency semaphore: all 20 victim requests are
    dispatched immediately.
  - The "active cohort" (which K = server_max_num_seqs requests actually
    ran first) is never assumed from request_index; it is determined
    from observed first-token arrival order (_active_cohort.py).
  - A background vLLM /metrics sampler provides independent aggregate
    corroboration of the streaming-derived cohort (never the primary
    cohort definition).
  - A new result schema (RESULT_SCHEMA_VERSION) with new exposure field
    names: server_exposure_group, was_dispatched_at_trigger,
    had_first_token_at_trigger, decode_tokens_received_at_trigger,
    first_token_perf_ns, dispatch_to_first_token_ms.

Use --self-test, --dry-run, --smoke-test, or --official-run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import csv
import math
import os
import platform
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# ============================================================================
# Import the audited base module (library use only) and the (already
# independently tested) data-driven cohort/trigger module. Both live
# alongside this file.
# ============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent / "prefill_confirmation"
for _p in (str(_THIS_DIR), str(_BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_prefill_confirmation as base  # noqa: E402  (audited Schema-5 runner, library use only)
import _active_cohort as cohort  # noqa: E402


# ============================================================================
# Frozen server-waiting-confirmation contract
# ============================================================================

SCHEMA_VERSION = 1
DESIGN_VERSION = "server-waiting-confirmation-v1"
SEED_NAMESPACE_TAG = "server-waiting-confirmation"

MODEL_KEY = "qwen"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CAMPAIGN_NAME = "qwen-server-waiting-confirmation"
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    MODEL_KEY: {"model_id": MODEL_ID, "campaign_name": CAMPAIGN_NAME},
}

# This value MUST match the fingerprint produced by
# generate_server_waiting_schedule.py for the frozen seed/grid below.
# Recorded once, from the actually-generated bundle, exactly as the
# original OFFICIAL_FINGERPRINTS were -- see AUDIT.md.
OFFICIAL_SEED = 20260720
OFFICIAL_FINGERPRINT = "sha256:7c5a6e411cc35f6c2d12c7d768434d67cbb58862c6435cd87d5db95672089557"

OFFLOAD_VALUES = [0, 12]
SERVER_MAX_NUM_SEQS_VALUES = [4, 8]
TRIGGER_POSITIONS = [16]
CONDITIONS = ["no_burst", base.BURST_CONDITION]
STATE_LABEL_BY_OFFLOAD: dict[int, str] = {0: "low", 12: "high"}

INITIAL_REPEATS = 4
INCLUDED_REPEATS = list(range(1, INITIAL_REPEATS + 1))

VICTIM_CONFIGURATION = {
    "victim_request_count": 20,
    "victim_input_len": 256,
    "victim_output_len": 64,
    "victim_temperature": 0.0,
}
BURST_CONFIGURATION = {
    "burst_parallel_requests": 4,
    "burst_input_len": 2048,
    "burst_output_len": 16,
    "burst_temperature": 0.0,
}
STABILIZATION_CONFIGURATION = {
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
EXTENSION_POLICY = {
    "initial_bundle_repeats": "1-4",
    "extension_policy": "none -- this is a fixed four-repeat robustness study; no "
    "result-dependent extension is defined or permitted",
}

BLOCK_SIZE = len(CONDITIONS)  # 2
OFFICIAL_BLOCK_COUNT = len(OFFLOAD_VALUES) * len(SERVER_MAX_NUM_SEQS_VALUES) * INITIAL_REPEATS  # 16
OFFICIAL_EPISODE_COUNT = OFFICIAL_BLOCK_COUNT * BLOCK_SIZE  # 32

REQUIRED_BUNDLE_FILENAMES = (
    "server_waiting_confirmation_schedule.json",
    "server_waiting_confirmation_schedule.csv",
    "server_waiting_confirmation_schedule_audit.txt",
)

RUN_MODE_MARKER_FILENAME = ".server_waiting_confirmation_run_mode"

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else SCRIPT_PATH.parent
DEFAULT_SCHEDULE_DIR_ROOT = PROJECT_ROOT / "new" / "runs" / "server_waiting_confirmation"


def default_schedule_dir(model_key: str = MODEL_KEY) -> Path:
    return DEFAULT_SCHEDULE_DIR_ROOT / model_key / "schedule"


def default_output_dir(model_key: str, mode: str) -> Path:
    return DEFAULT_SCHEDULE_DIR_ROOT / model_key / "results" / mode


# ============================================================================
# Episode schema (26 fields: identical to the schedule generator's)
# ============================================================================

EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id", "model_key", "model_id", "offload_gb", "state_label",
    "server_max_num_seqs", "trigger_after_decode_tokens", "condition", "repeat",
    "random_seed", "episode_seed", "victim_workload_seed", "burst_workload_seed",
    "victim_request_count", "victim_input_len", "victim_output_len", "victim_temperature",
    "burst_parallel_requests", "burst_input_len", "burst_output_len", "burst_temperature",
    "max_num_batched_tokens", "condition_first_in_block", "restart_server_before_block",
    "block_id", "order_in_block",
)

EPISODE_FIELD_TYPES: dict[str, type] = {
    "episode_id": str, "model_key": str, "model_id": str, "offload_gb": int,
    "state_label": str, "server_max_num_seqs": int, "trigger_after_decode_tokens": int,
    "condition": str, "repeat": int, "random_seed": int, "episode_seed": int,
    "victim_workload_seed": int, "burst_workload_seed": int, "victim_request_count": int,
    "victim_input_len": int, "victim_output_len": int, "victim_temperature": float,
    "burst_parallel_requests": int, "burst_input_len": int, "burst_output_len": int,
    "burst_temperature": float, "max_num_batched_tokens": int, "condition_first_in_block": str,
    "restart_server_before_block": int, "block_id": str, "order_in_block": int,
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


def derive_seed(*parts: str) -> int:
    """Identical recipe to base.derive_seed -- kept local so this
    module has no runtime dependency on the generator, matching the
    existing project convention (see generate_server_waiting_schedule.py)."""
    joined = ":".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


# ============================================================================
# Bundle loading & structural validation (new grid; framework pattern
# mirrors the audited base module's, but bound to THIS module's own
# EPISODE_FIELDS / official contract -- see AUDIT.md for why this could
# not be imported unchanged from base.)
# ============================================================================

class BundleLoadError(Exception):
    pass


def find_bundle_paths(schedule_dir: Path) -> dict[str, Path]:
    if not schedule_dir.is_dir():
        raise BundleLoadError(f"--schedule-dir '{schedule_dir}' does not exist or is not a directory")
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name in REQUIRED_BUNDLE_FILENAMES:
        candidate = schedule_dir / name
        if not candidate.is_file() or not os.access(candidate, os.R_OK):
            missing.append(name)
        else:
            paths[name] = candidate
    if missing:
        raise BundleLoadError(f"--schedule-dir '{schedule_dir}' is missing required, readable file(s): {missing}")
    return paths


def _check_type_strict(value: object, expected_type: type) -> bool:
    return type(value) is expected_type


def check_json_episode_schema(obj: object, index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"episode index {index}: not a JSON object"]
    actual_keys = set(obj.keys())
    expected_keys = set(EPISODE_FIELDS)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        errors.append(f"episode index {index}: missing field(s): {sorted(missing)}")
    if extra:
        errors.append(f"episode index {index}: unexpected extra field(s): {sorted(extra)}")
    if missing or extra:
        return errors
    for f in EPISODE_FIELDS:
        if not _check_type_strict(obj[f], EPISODE_FIELD_TYPES[f]):
            errors.append(
                f"episode index {index} ({f}): wrong type, expected "
                f"{EPISODE_FIELD_TYPES[f].__name__}, got {type(obj[f]).__name__} ({obj[f]!r})"
            )
    return errors


def parse_episode_from_json(obj: dict) -> Episode:
    return Episode(**{f: obj[f] for f in EPISODE_FIELDS})


def check_csv_header(fieldnames: list[str]) -> list[str]:
    if fieldnames != list(EPISODE_FIELDS):
        return [f"csv header {fieldnames} does not match expected field order {list(EPISODE_FIELDS)}"]
    return []


def _normalize_csv_value(raw: str, expected_type: type) -> object:
    if expected_type is int:
        return int(raw)
    if expected_type is float:
        return float(raw)
    return raw


def normalize_csv_row(row: dict[str, str], index: int) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    normalized: dict[str, object] = {}
    for f in EPISODE_FIELDS:
        raw = row.get(f)
        if raw is None:
            errors.append(f"csv row index {index}: missing field {f!r}")
            continue
        try:
            normalized[f] = _normalize_csv_value(raw, EPISODE_FIELD_TYPES[f])
        except (TypeError, ValueError) as exc:
            errors.append(f"csv row index {index} ({f}): could not parse {raw!r}: {exc}")
    return normalized, errors


def check_csv_json_consistency(csv_fieldnames, csv_rows, json_episodes) -> list[str]:
    errors: list[str] = []
    errors.extend(check_csv_header(csv_fieldnames))
    if len(csv_rows) != OFFICIAL_EPISODE_COUNT:
        errors.append(f"csv has {len(csv_rows)} row(s), expected exactly {OFFICIAL_EPISODE_COUNT}")
    if len(json_episodes) != OFFICIAL_EPISODE_COUNT:
        errors.append(f"json has {len(json_episodes)} episode(s), expected exactly {OFFICIAL_EPISODE_COUNT}")
    if len(csv_rows) != len(json_episodes):
        errors.append(f"csv/json episode count mismatch: csv={len(csv_rows)}, json={len(json_episodes)}")
        return errors
    for idx, (csv_row, json_row) in enumerate(zip(csv_rows, json_episodes)):
        normalized_csv, row_errors = normalize_csv_row(csv_row, idx)
        errors.extend(row_errors)
        if not isinstance(json_row, dict):
            errors.append(f"episode index {idx}: json row is not an object")
            continue
        for f in EPISODE_FIELDS:
            if f not in normalized_csv or f not in json_row:
                continue
            if normalized_csv[f] != json_row[f]:
                errors.append(f"episode index {idx} ({f}): csv {normalized_csv[f]!r} != json {json_row[f]!r}")
    return errors


_FP_LINE_RE = re.compile(r"^schedule_fingerprint:\s*(\S+)\s*$", re.MULTILINE)


def check_audit(audit_text: str, expected_fingerprint: object) -> list[str]:
    errors: list[str] = []
    matches = _FP_LINE_RE.findall(audit_text)
    if len(matches) == 0:
        errors.append("audit report has no 'schedule_fingerprint: ...' line")
    elif len(matches) > 1:
        errors.append(f"audit report has {len(matches)} fingerprint lines, expected exactly 1")
    elif matches[0] != expected_fingerprint:
        errors.append("audit report fingerprint does not match JSON")

    required_patterns = (
        (r"^schema_version:\s*1\s*$", "schema_version: 1"),
        (rf"^design_version:\s*{re.escape(DESIGN_VERSION)}\s*$", "design_version"),
        (rf"^campaign_name:\s*{re.escape(CAMPAIGN_NAME)}\s*$", "campaign_name"),
        (rf"^model_key:\s*{re.escape(MODEL_KEY)}\s*$", "model_key"),
        (rf"^block count:\s*{OFFICIAL_BLOCK_COUNT}\s*$", f"block count: {OFFICIAL_BLOCK_COUNT}"),
        (rf"^episode count:\s*{OFFICIAL_EPISODE_COUNT}\s*$", f"episode count: {OFFICIAL_EPISODE_COUNT}"),
        (r"^OVERALL:\s*PASS\s*$", "OVERALL: PASS"),
    )
    for pattern, label in required_patterns:
        if not re.search(pattern, audit_text, re.MULTILINE):
            errors.append(f"audit report does not contain valid {label!r}")
    return errors


def is_valid_fingerprint_format(fp: object) -> bool:
    return base.is_valid_fingerprint_format(fp)


def recompute_fingerprint(json_obj: dict) -> str:
    return base.recompute_fingerprint(json_obj)


def check_official_contract(json_obj: dict, model_key: str) -> list[str]:
    errors: list[str] = []
    if model_key not in MODEL_REGISTRY:
        return [f"unknown model_key {model_key!r}; known: {sorted(MODEL_REGISTRY)}"]
    registry = MODEL_REGISTRY[model_key]

    def check_eq(key: str, expected: object) -> None:
        actual = json_obj.get(key, "<MISSING>")
        if actual != expected:
            errors.append(f"{key} = {actual!r}, expected {expected!r}")

    check_eq("schema_version", SCHEMA_VERSION)
    check_eq("design_version", DESIGN_VERSION)
    check_eq("campaign_name", registry["campaign_name"])
    check_eq("model_key", model_key)
    check_eq("model_id", registry["model_id"])
    check_eq("seed", OFFICIAL_SEED)
    check_eq("offload_values", OFFLOAD_VALUES)
    check_eq("server_max_num_seqs_values", SERVER_MAX_NUM_SEQS_VALUES)
    check_eq("trigger_positions", TRIGGER_POSITIONS)
    check_eq("conditions", CONDITIONS)
    check_eq("initial_repeats", INITIAL_REPEATS)
    check_eq("included_repeats", INCLUDED_REPEATS)
    check_eq("block_count", OFFICIAL_BLOCK_COUNT)
    check_eq("episode_count", OFFICIAL_EPISODE_COUNT)
    check_eq("victim_configuration", VICTIM_CONFIGURATION)
    check_eq("burst_configuration", BURST_CONFIGURATION)
    check_eq("stabilization_configuration", STABILIZATION_CONFIGURATION)
    check_eq("extension_policy", EXTENSION_POLICY)

    fp = json_obj.get("schedule_fingerprint")
    if not is_valid_fingerprint_format(fp):
        errors.append(f"schedule_fingerprint has invalid format: {fp!r}")
    else:
        recomputed = recompute_fingerprint(json_obj)
        if recomputed != fp:
            errors.append(f"recomputed fingerprint {recomputed!r} != stored {fp!r}")
        if fp != OFFICIAL_FINGERPRINT:
            errors.append(
                f"schedule_fingerprint {fp!r} does not match the frozen "
                f"server-waiting-confirmation fingerprint {OFFICIAL_FINGERPRINT!r}"
            )
    return errors


def check_structural_schedule(episodes: list[Episode], seed: object, model_key: str) -> list[str]:
    """Independent-of-fingerprint structural re-derivation check --
    mirrors generate_server_waiting_schedule.validate_schedule's
    forbidden-mutation checks, applied to an already-parsed episode
    list (so a loader-side mutation of e.g. server_max_num_seqs or
    trigger_after_decode_tokens is caught even if the fingerprint were
    somehow bypassed)."""
    errors: list[str] = []
    if not isinstance(seed, int) or isinstance(seed, bool):
        return [f"seed is not an int: {seed!r}"]
    registry = MODEL_REGISTRY.get(model_key)
    if registry is None:
        return [f"unknown model_key {model_key!r}"]
    expected_model_id = registry["model_id"]

    for ep in episodes:
        ctx = f"episode {ep.episode_id}"
        if ep.model_key != model_key:
            errors.append(f"{ctx}: model_key {ep.model_key!r} != {model_key!r}")
        if ep.model_id != expected_model_id:
            errors.append(f"{ctx}: model_id {ep.model_id!r} != expected {expected_model_id!r}")
        if ep.offload_gb not in OFFLOAD_VALUES:
            errors.append(f"{ctx}: forbidden offload_gb {ep.offload_gb!r}")
        if ep.server_max_num_seqs not in SERVER_MAX_NUM_SEQS_VALUES:
            errors.append(f"{ctx}: forbidden server_max_num_seqs {ep.server_max_num_seqs!r}")
        if ep.trigger_after_decode_tokens not in TRIGGER_POSITIONS:
            errors.append(f"{ctx}: forbidden trigger_after_decode_tokens {ep.trigger_after_decode_tokens!r}")
        if ep.condition not in CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.random_seed != seed:
            errors.append(f"{ctx}: random_seed {ep.random_seed!r} != {seed!r}")
        for fname, expected in {**VICTIM_CONFIGURATION, **BURST_CONFIGURATION}.items():
            if getattr(ep, fname) != expected:
                errors.append(f"{ctx}: {fname} != {expected!r}")
        if not (1 <= ep.repeat <= INITIAL_REPEATS):
            errors.append(f"{ctx}: repeat {ep.repeat!r} outside 1..{INITIAL_REPEATS}")

    blocks: dict[str, list[Episode]] = {}
    for ep in episodes:
        blocks.setdefault(ep.block_id, []).append(ep)
    if len(blocks) != OFFICIAL_BLOCK_COUNT:
        errors.append(f"expected {OFFICIAL_BLOCK_COUNT} blocks, found {len(blocks)}")
    for block_id, block_episodes in blocks.items():
        if len(block_episodes) != BLOCK_SIZE:
            errors.append(f"block {block_id!r} has {len(block_episodes)} episode(s), expected {BLOCK_SIZE}")
            continue
        conditions_here = sorted(ep.condition for ep in block_episodes)
        if conditions_here != sorted(CONDITIONS):
            errors.append(f"block {block_id!r} conditions {conditions_here} != {sorted(CONDITIONS)}")
        restarts = {ep.order_in_block: ep.restart_server_before_block for ep in block_episodes}
        if restarts.get(1) != 1 or restarts.get(2) != 0:
            errors.append(f"block {block_id!r} restart sequence {restarts} != {{1: 1, 2: 0}}")

    return errors


@dataclass
class LoadedBundle:
    schedule_dir: Path
    json_obj: dict
    csv_fieldnames: list[str]
    csv_rows: list[dict[str, str]]
    audit_text: str
    episodes: list[Episode]
    fingerprint: str


def load_and_validate_bundle(schedule_dir: Path, model_key: str = MODEL_KEY) -> tuple[LoadedBundle | None, list[str]]:
    if model_key not in MODEL_REGISTRY:
        return None, [f"unknown model_key {model_key!r}; known: {sorted(MODEL_REGISTRY)}"]
    try:
        paths = find_bundle_paths(schedule_dir)
        json_obj = base.load_json_bundle(paths[REQUIRED_BUNDLE_FILENAMES[0]])
        csv_fieldnames, csv_rows = base.load_csv_bundle(paths[REQUIRED_BUNDLE_FILENAMES[1]])
        audit_text = base.load_audit_text(paths[REQUIRED_BUNDLE_FILENAMES[2]])
    except base.BundleLoadError as exc:
        return None, [str(exc)]
    except BundleLoadError as exc:
        return None, [str(exc)]

    errors: list[str] = []
    errors.extend(check_official_contract(json_obj, model_key))

    raw_json_episodes = json_obj.get("episodes")
    if not isinstance(raw_json_episodes, list):
        errors.append("json 'episodes' is missing or not a list")
        return None, errors

    schema_errors: list[str] = []
    for idx, raw_ep in enumerate(raw_json_episodes):
        schema_errors.extend(check_json_episode_schema(raw_ep, idx))
    errors.extend(schema_errors)
    if schema_errors:
        return None, errors

    episodes = [parse_episode_from_json(ep) for ep in raw_json_episodes]
    errors.extend(check_csv_json_consistency(csv_fieldnames, csv_rows, raw_json_episodes))
    fingerprint = json_obj.get("schedule_fingerprint", "")
    errors.extend(check_audit(audit_text, fingerprint))
    errors.extend(check_structural_schedule(episodes, json_obj.get("seed"), model_key))

    if errors:
        return None, errors

    bundle = LoadedBundle(
        schedule_dir=schedule_dir, json_obj=json_obj, csv_fieldnames=csv_fieldnames,
        csv_rows=csv_rows, audit_text=audit_text, episodes=episodes, fingerprint=fingerprint,
    )
    return bundle, []


def _group_into_blocks(episodes: list[Episode]) -> list[dict]:
    blocks: list[dict] = []
    idx = 0
    n = len(episodes)
    while idx < n:
        bid = episodes[idx].block_id
        run_end = idx
        while run_end < n and episodes[run_end].block_id == bid:
            run_end += 1
        run_episodes = episodes[idx:run_end]
        first = run_episodes[0]
        blocks.append({
            "block_id": bid, "model_key": first.model_key, "model_id": first.model_id,
            "offload_gb": first.offload_gb, "state_label": first.state_label,
            "server_max_num_seqs": first.server_max_num_seqs,
            "trigger_after_decode_tokens": first.trigger_after_decode_tokens,
            "repeat": first.repeat, "condition_first_in_block": first.condition_first_in_block,
            "episodes": run_episodes,
        })
        idx = run_end
    return blocks


def build_execution_plan(bundle: LoadedBundle) -> dict:
    episodes = bundle.episodes
    blocks = _group_into_blocks(episodes)
    no_burst_count = sum(1 for ep in episodes if ep.condition == "no_burst")
    burst_condition_count = sum(1 for ep in episodes if ep.condition == base.BURST_CONDITION)
    return {
        "regular_episodes": len(episodes),
        "blocks": blocks,
        "planned_server_starts": len(blocks),
        "planned_stabilization_runs": len(blocks) * STABILIZATION_CONFIGURATION["stabilization_runs_per_block"],
        "no_burst_count": no_burst_count,
        "burst_condition_count": burst_condition_count,
    }


def print_execution_plan(bundle: LoadedBundle, plan: dict) -> None:
    print("Execution plan (dry-run; nothing is executed)")
    print("=" * 60)
    print(f"schedule_fingerprint: {bundle.fingerprint}")
    print(f"regular episodes: {plan['regular_episodes']}")
    print(f"blocks (= planned server restarts): {plan['planned_server_starts']}")
    print(f"planned stabilization runs: {plan['planned_stabilization_runs']}")
    print(f"no_burst episodes: {plan['no_burst_count']}")
    print(f"{base.BURST_CONDITION} episodes: {plan['burst_condition_count']}")
    print()
    for blk in plan["blocks"]:
        print(f"--- {blk['block_id']} ---")
        print(
            f"  model_key: {blk['model_key']}, offload_gb: {blk['offload_gb']}, "
            f"state: {blk['state_label']}, server_max_num_seqs: {blk['server_max_num_seqs']}, "
            f"trigger_after_decode_tokens: {blk['trigger_after_decode_tokens']}, "
            f"repeat: {blk['repeat']}, condition_first: {blk['condition_first_in_block']}"
        )
        for ep in blk["episodes"]:
            print(f"    order {ep.order_in_block}: {ep.episode_id} (condition={ep.condition})")
    print()
    print(
        "DRY RUN: no server was started, no network connection was opened, "
        "no tokenizer was loaded, no result files were written."
    )


# ============================================================================
# Result schema (new, independent of the audited Schema 5 -- see module
# docstring and AUDIT.md for exactly what is and is not comparable to
# the original Prefill-Confirmation result files).
# ============================================================================

RESULT_SCHEMA_VERSION = 6
RUNNER_VERSION = "run_server_waiting_confirmation-v4"
TIMING_INSTRUMENTATION_VERSION = 1
TIMING_INSTRUMENTATION_NAME = "server_max_num_seqs_data_driven_cohort_v1"

RECORD_TYPE_REGULAR_EPISODE = "regular_episode"
RECORD_TYPE_STABILIZATION = "stabilization"

CLASSIFICATION_MISSING = "missing"
CLASSIFICATION_VALID_COMPLETE = "valid_complete"
CLASSIFICATION_PARTIAL = "partial"
CLASSIFICATION_INVALID = "invalid"
CLASSIFICATION_CORRUPTED = "corrupted"

SERVER_EXPOSURE_RUNNING_AT_TRIGGER = "running_at_trigger_observed"
SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT = "dispatched_no_output_at_trigger"
SERVER_EXPOSURE_UNKNOWN = "unknown"


def compute_server_exposure(
    *, request_dispatch_ns, first_token_perf_ns, decode_tokens_received_at_trigger,
    trigger_perf_ns, in_active_cohort: bool,
) -> dict:
    """Requirement (Exposure labels): names reflect what is directly
    observed. `running_at_trigger_observed` is set ONLY for a request
    that is a member of the streaming-derived active cohort (never
    inferred from timestamps alone -- cohort membership is Section
    'Data-driven active cohort's job, not this function's). A
    dispatched request outside the cohort is
    `dispatched_no_output_at_trigger` regardless of its TTFT, since (per
    the frozen design) its TTFT mixes server-side waiting with later
    prefill time and cannot be split without internal scheduler
    timestamps."""
    if type(trigger_perf_ns) is not int:
        return {
            "server_exposure_group": SERVER_EXPOSURE_UNKNOWN,
            "was_dispatched_at_trigger": None,
            "had_first_token_at_trigger": None,
        }
    was_dispatched_at_trigger = type(request_dispatch_ns) is int and request_dispatch_ns <= trigger_perf_ns
    had_first_token_at_trigger = type(first_token_perf_ns) is int and first_token_perf_ns <= trigger_perf_ns
    if in_active_cohort:
        group = SERVER_EXPOSURE_RUNNING_AT_TRIGGER
    elif was_dispatched_at_trigger:
        group = SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT
    else:
        group = SERVER_EXPOSURE_UNKNOWN
    return {
        "server_exposure_group": group,
        "was_dispatched_at_trigger": was_dispatched_at_trigger,
        "had_first_token_at_trigger": had_first_token_at_trigger,
    }


def _enrich_server_exposure_fields(
    victim_results: list[dict], *, trigger_perf_ns, active_indices, k: int,
    progress_by_index: dict[int, "cohort.TokenProgress"],
) -> None:
    trigger_ns = trigger_perf_ns if type(trigger_perf_ns) is int else None
    active_set = set(active_indices or ())
    for record in victim_results:
        if not isinstance(record, dict):
            continue
        idx = record.get("request_index")
        first_token_ns = record.get("first_token_receive_ns")
        dispatch_ns = record.get("request_dispatch_ns")
        decode_tokens_at_trigger = None
        if trigger_ns is not None and idx in progress_by_index:
            decode_tokens_at_trigger = progress_by_index[idx].count_at_or_before(trigger_ns)
        exposure = compute_server_exposure(
            request_dispatch_ns=dispatch_ns, first_token_perf_ns=first_token_ns,
            decode_tokens_received_at_trigger=decode_tokens_at_trigger,
            trigger_perf_ns=trigger_ns, in_active_cohort=idx in active_set,
        )
        record.update(exposure)
        record["decode_tokens_received_at_trigger"] = decode_tokens_at_trigger
        record["first_token_perf_ns"] = first_token_ns
        record["server_max_num_seqs"] = k
        record["dispatch_to_first_token_ms"] = (
            (first_token_ns - dispatch_ns) / 1e6
            if type(first_token_ns) is int and type(dispatch_ns) is int and first_token_ns >= dispatch_ns
            else None
        )


def validate_server_waiting_episode_invariants(
    *, episode: Episode, victim_results: list[dict], burst_results: list[dict],
    trigger_ns, active_indices, k: int,
) -> list[str]:
    """Implements the 'Validation requirements' list from the design
    prompt directly and explicitly (one check per bullet), rather than
    inheriting the audited base module's (differently-shaped, wave/
    semaphore-oriented) validator."""
    errors: list[str] = []
    by_idx = {r.get("request_index"): r for r in victim_results if isinstance(r, dict)}

    if len(victim_results) != episode.victim_request_count:
        errors.append(f"expected {episode.victim_request_count} victim results, found {len(victim_results)}")
    if any(r.get("status") != base.REQUEST_STATUS_COMPLETE for r in victim_results):
        errors.append("not all victim requests completed validly (status == 'complete')")
    if any(r.get("status") != base.REQUEST_STATUS_COMPLETE for r in burst_results):
        errors.append("not all burst requests completed validly (status == 'complete')")

    if trigger_ns is None:
        errors.append("trigger_perf_ns is not known; cannot validate trigger-relative invariants")
        return errors

    undispatched = sorted(
        i for i, r in by_idx.items()
        if not (type(r.get("request_dispatch_ns")) is int and r["request_dispatch_ns"] <= trigger_ns)
    )
    if undispatched:
        errors.append(f"request(s) not dispatched before the trigger: {undispatched}")

    if len(active_indices) != k:
        errors.append(f"frozen active cohort size {len(active_indices)} != server_max_num_seqs {k}")

    for i in sorted(active_indices):
        r = by_idx.get(i)
        if r is None:
            errors.append(f"active-cohort request {i} has no result record")
            continue
        dcount = r.get("decode_tokens_received_at_trigger")
        if not (type(dcount) is int and dcount >= episode.trigger_after_decode_tokens):
            errors.append(
                f"active-cohort request {i}: decode_tokens_received_at_trigger={dcount!r} "
                f"< trigger_after_decode_tokens={episode.trigger_after_decode_tokens}"
            )
        end_ns = r.get("stream_end_ns")
        if type(end_ns) is int and end_ns <= trigger_ns:
            errors.append(
                f"active-cohort request {i} completed (stream_end_ns={end_ns}) at or "
                f"before the trigger ({trigger_ns}) -- output_len must exceed the trigger threshold"
            )

    for i, r in sorted(by_idx.items()):
        if i in active_indices:
            continue
        dcount = r.get("decode_tokens_received_at_trigger")
        if dcount != 0:
            errors.append(
                f"non-cohort request {i}: decode_tokens_received_at_trigger={dcount!r} != exactly 0 "
                f"at the trigger (None is not an acceptable substitute for zero)"
            )

    if episode.condition == base.BURST_CONDITION:
        if len(burst_results) != episode.burst_parallel_requests:
            errors.append(f"expected {episode.burst_parallel_requests} burst requests, found {len(burst_results)}")
        for r in burst_results:
            start_ns = r.get("request_start_ns")
            if type(start_ns) is int and start_ns <= trigger_ns:
                errors.append(
                    f"burst request {r.get('request_index')} started (request_start_ns={start_ns}) "
                    f"at or before the trigger ({trigger_ns})"
                )
    else:
        if burst_results:
            errors.append(f"no_burst episode unexpectedly has {len(burst_results)} burst request(s)")

    return errors


def _build_episode_result(
    *, episode: Episode, schedule_fingerprint: str, server_metadata: dict, stabilization_ref: dict,
    trigger: dict, burst_interval: dict | None, victim_results: list[dict], burst_results: list[dict],
    status: str, validation_errors: list[str], run_mode: str, victim_phase_start_ns: int,
    transport_concurrency_evidence: dict | None = None,
) -> dict:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "record_type": RECORD_TYPE_REGULAR_EPISODE,
        "run_mode": run_mode,
        "schedule_fingerprint": schedule_fingerprint,
        "episode_id": episode.episode_id,
        "schedule_row": asdict(episode),
        "block_id": episode.block_id,
        "server_metadata": server_metadata,
        "stabilization_reference": stabilization_ref,
        "trigger": trigger,
        "burst_interval": burst_interval,
        "victim_requests": victim_results,
        "burst_requests": burst_results,
        "aggregate_metrics": base._aggregate_metrics(victim_results, burst_results, burst_interval),
        "transport_concurrency_evidence": transport_concurrency_evidence or {},
        "status": status,
        "validation_errors": validation_errors,
        "timing_instrumentation_version": TIMING_INSTRUMENTATION_VERSION,
        "timing_instrumentation_name": TIMING_INSTRUMENTATION_NAME,
        "victim_phase_start_ns": victim_phase_start_ns,
        "no_client_admission_semaphore_used": True,
        "server_exposure_scope_note": (
            "For victims classified dispatched_no_output_at_trigger, TTFT / "
            "dispatch_to_first_token_ms contains server-side waiting PLUS later prefill "
            "time and cannot be interpreted as a pure queue duration without internal "
            "vLLM scheduler timestamps. The active cohort (running_at_trigger_observed) "
            "is determined only from observed first-token arrival order, never from "
            "request_index. Client-observed stream_open_or_response_headers_perf_ns and "
            "first_token_perf_ns do NOT directly expose internal vLLM scheduler admission "
            "or prefill-start time -- see transport_concurrency_evidence and AUDIT.md."
        ),
    }


# ============================================================================
# Transport-concurrency evidence and per-request canonical timing aliases
# (2026-07-20 hardening pass, requirements 2, 3, 5). These read whatever
# the transport optionally exposes (get_stream_registry_entry /
# get_diagnostics) without ever inventing a value the real stack cannot
# defensibly provide -- see PersistentCompletionTransport docstring.
# ============================================================================

def _first_crossing_ns(history: list[tuple[int, int]], threshold: int) -> int | None:
    """First timestamp at which cumulative token count reached
    `threshold`, from a TokenProgress.history list. Returns None if the
    threshold was never reached in the recorded history."""
    for ns, count in history:
        if count >= threshold:
            return ns
    return None


def _check_monotonic_sequence(label: str, ordered_fields: list[tuple[str, object]]) -> list[str]:
    """Checks a <= b <= c ... across only the fields that are actually
    present (type is int); a missing (None) field is skipped, never
    treated as a violation by itself."""
    errors: list[str] = []
    present = [(name, ns) for name, ns in ordered_fields if type(ns) is int]
    for (name_a, ns_a), (name_b, ns_b) in zip(present, present[1:]):
        if ns_a > ns_b:
            errors.append(f"{label}: {name_a}={ns_a} > {name_b}={ns_b} (monotonicity violated)")
    return errors


def _enrich_victim_transport_and_timing_fields(
    victim_results: list[dict], *, transport, progress_by_index: dict, trigger_after_decode_tokens: int,
    episode_id: str,
) -> list[str]:
    """Adds, per victim record: stream_open_or_response_headers_perf_ns
    / stream_close_perf_ns (from the transport's stream registry, IF it
    exposes one -- otherwise null, never invented), the canonical alias
    fields task_created_perf_ns / http_dispatch_start_perf_ns /
    stream_end_perf_ns, and token_16_perf_ns (derived from this
    request's own TokenProgress history). Validates
    task_created_perf_ns <= http_dispatch_start_perf_ns <=
    stream_open_or_response_headers_perf_ns <= first_token_perf_ns <=
    stream_end_perf_ns wherever the relevant fields are present, storing
    any violation both on the record (`timestamp_monotonicity_errors`)
    and returning the flattened list for the caller's validation_errors.
    """
    get_entry = getattr(transport, "get_stream_registry_entry", None)
    all_errors: list[str] = []
    for record in victim_results:
        if not isinstance(record, dict):
            continue
        idx = record.get("request_index")
        request_id = f"{episode_id}:victim:{idx}"
        entry = get_entry(request_id) if callable(get_entry) else None
        record["stream_open_or_response_headers_perf_ns"] = (
            entry.get("stream_open_or_response_headers_perf_ns") if entry else None
        )
        record["stream_close_perf_ns"] = entry.get("stream_close_perf_ns") if entry else None
        record["task_created_perf_ns"] = record.get("task_created_ns")
        record["http_dispatch_start_perf_ns"] = record.get("request_dispatch_ns")
        record["stream_end_perf_ns"] = record.get("stream_end_ns")
        if record.get("first_token_perf_ns") is None:
            record["first_token_perf_ns"] = record.get("first_token_receive_ns")
        if progress_by_index is not None and idx in progress_by_index:
            record["token_16_perf_ns"] = _first_crossing_ns(progress_by_index[idx].history, trigger_after_decode_tokens)
        else:
            record["token_16_perf_ns"] = None
        mono_errors = _check_monotonic_sequence(
            f"victim {idx}",
            [
                ("task_created_perf_ns", record["task_created_perf_ns"]),
                ("http_dispatch_start_perf_ns", record["http_dispatch_start_perf_ns"]),
                ("stream_open_or_response_headers_perf_ns", record["stream_open_or_response_headers_perf_ns"]),
                ("first_token_perf_ns", record["first_token_perf_ns"]),
                ("stream_end_perf_ns", record["stream_end_perf_ns"]),
            ],
        )
        record["timestamp_monotonicity_errors"] = mono_errors
        all_errors.extend(mono_errors)
    return all_errors


def _enrich_burst_transport_and_timing_fields(
    burst_results: list[dict], *, transport, episode_id: str,
) -> list[str]:
    """Adds the canonical burst_* timing aliases (requirement 5) and
    validates their monotonicity. For no_burst episodes this is called
    with an empty list and is a no-op."""
    get_entry = getattr(transport, "get_stream_registry_entry", None)
    all_errors: list[str] = []
    for record in burst_results:
        if not isinstance(record, dict):
            continue
        idx = record.get("request_index")
        request_id = f"{episode_id}:burst:{idx}"
        entry = get_entry(request_id) if callable(get_entry) else None
        record["burst_dispatch_start_perf_ns"] = record.get("request_dispatch_ns")
        record["burst_stream_open_or_response_headers_perf_ns"] = (
            entry.get("stream_open_or_response_headers_perf_ns") if entry else None
        )
        record["burst_first_token_perf_ns"] = record.get("first_token_receive_ns")
        record["burst_end_perf_ns"] = record.get("stream_end_ns")
        mono_errors = _check_monotonic_sequence(
            f"burst {idx}",
            [
                ("burst_dispatch_start_perf_ns", record["burst_dispatch_start_perf_ns"]),
                ("burst_stream_open_or_response_headers_perf_ns", record["burst_stream_open_or_response_headers_perf_ns"]),
                ("burst_first_token_perf_ns", record["burst_first_token_perf_ns"]),
                ("burst_end_perf_ns", record["burst_end_perf_ns"]),
            ],
        )
        record["timestamp_monotonicity_errors"] = mono_errors
        all_errors.extend(mono_errors)
    return all_errors


def compute_transport_concurrency_evidence(victim_results: list[dict], transport) -> dict:
    """Requirement 3: the strict criterion is that all 20
    stream_open_or_response_headers_perf_ns are non-null AND every one
    of them is strictly earlier than the earliest victim first-token
    timestamp. Task creation or dispatch-start timestamps alone are
    NEVER treated as transport-level evidence here."""
    open_timestamps = []
    first_token_timestamps = []
    for r in victim_results:
        if not isinstance(r, dict):
            continue
        open_ns = r.get("stream_open_or_response_headers_perf_ns")
        if type(open_ns) is int:
            open_timestamps.append(open_ns)
        ft_ns = r.get("first_token_perf_ns")
        if type(ft_ns) is int:
            first_token_timestamps.append(ft_ns)
    earliest_first_token_ns = min(first_token_timestamps) if first_token_timestamps else None
    victim_stream_open_count = len(open_timestamps)
    all_20_before = bool(
        victim_stream_open_count == len(victim_results) and victim_stream_open_count > 0
        and earliest_first_token_ns is not None
        and all(ns < earliest_first_token_ns for ns in open_timestamps)
    )
    get_diag = getattr(transport, "get_diagnostics", None)
    diagnostics = get_diag() if callable(get_diag) else {}
    return {
        "earliest_victim_first_token_ns": earliest_first_token_ns,
        "victim_stream_open_count": victim_stream_open_count,
        "all_20_streams_open_before_first_token": all_20_before,
        "peak_concurrent_open_completion_streams": diagnostics.get("peak_open_stream_count"),
        "completion_pool_limits": {
            "max_connections": diagnostics.get("max_connections"),
            "max_keepalive_connections": diagnostics.get("max_keepalive_connections"),
        },
    }


# ============================================================================
# vLLM /metrics background sampler (independent aggregate corroboration
# of the streaming-derived cohort -- never the primary cohort
# definition; see module docstring and AUDIT.md).
# ============================================================================

METRICS_ENDPOINT = "/metrics"
RUNNING_METRIC_NAME_CANDIDATES = (
    "vllm:num_requests_running",
    "vllm:num_running_requests",
)
WAITING_METRIC_NAME_CANDIDATES = (
    "vllm:num_requests_waiting",
    "vllm:num_waiting_requests",
)
METRICS_STALENESS_THRESHOLD_MS = 500.0

METRICS_QUALITY_CORROBORATED = "corroborated"
METRICS_QUALITY_UNAVAILABLE = "unavailable"
METRICS_QUALITY_STALE = "stale"
METRICS_QUALITY_CONTRADICTORY = "contradictory"
METRICS_QUALITY_UNPARSABLE = "unparsable"


def _sum_prometheus_metric(text: str, candidate_names: tuple[str, ...]) -> tuple[float | None, str | None]:
    """Tries each candidate metric name in order (oldest/most-common
    first); the first name with at least one matching sample line wins.
    Sums across all matching lines so a multi-worker/multi-label
    exposition (e.g. per engine-index) still yields one aggregate
    number. Returns (None, None) if no candidate name matched at all,
    or (None, matched_name) if the name matched but a value could not
    be parsed as a float (e.g. 'NaN')."""
    for name in candidate_names:
        pattern = re.compile(
            rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)\s*$', re.MULTILINE,
        )
        matches = pattern.findall(text)
        if not matches:
            continue
        total = 0.0
        for m in matches:
            try:
                total += float(m)
            except ValueError:
                return None, name
        return total, name
    return None, None


def parse_vllm_metrics_text(text: str) -> dict:
    running, running_name = _sum_prometheus_metric(text, RUNNING_METRIC_NAME_CANDIDATES)
    waiting, waiting_name = _sum_prometheus_metric(text, WAITING_METRIC_NAME_CANDIDATES)
    return {
        "running": running,
        "waiting": waiting,
        "matched_running_metric_name": running_name,
        "matched_waiting_metric_name": waiting_name,
        "parse_ok": running is not None and waiting is not None,
    }


class MetricsSampler:
    """Lightweight background poller for the local vLLM /metrics
    endpoint. Runs concurrently with the victim phase; NEVER awaited
    synchronously on the trigger/burst-dispatch critical path (the
    caller only ever reads the latest already-collected sample via
    `nearest_sample_before`).

    B2 fix (2026-07-20 hardening pass): a sample's eligibility as
    "pre-trigger" and its staleness are both computed from
    `response_received_perf_ns` -- the instant the HTTP response
    actually arrived -- never from `scrape_start_perf_ns` (when the GET
    was issued). A scrape that started before the trigger but whose
    response arrived after it must NOT be selectable as a pre-trigger
    sample, since its body may describe post-trigger (or even
    post-burst) server state. See test_server_waiting_trigger_timing.py
    for the regression test."""

    def __init__(
        self, *, transport, base_url: str, sleeper, clock,
        poll_interval_s: float = 0.05, request_timeout_s: float = 5.0,
    ) -> None:
        self.transport = transport
        self.base_url = base_url
        self.sleeper = sleeper
        self.clock = clock
        self.poll_interval_s = poll_interval_s
        self.request_timeout_s = request_timeout_s
        self.samples: list[dict] = []
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._stop_event = asyncio.Event()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=max(self.request_timeout_s, 2.0))
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None

    async def _run(self) -> None:
        url = self.base_url.rstrip("/") + METRICS_ENDPOINT
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            scrape_start_ns = self.clock.perf_counter_ns()
            try:
                status, body = await self.transport.get_json(url, {}, self.request_timeout_s)
                response_received_ns = self.clock.perf_counter_ns()
                text = body if isinstance(body, str) else json.dumps(body)
                if status == 200:
                    parsed = parse_vllm_metrics_text(text)
                    self.samples.append({
                        "scrape_start_perf_ns": scrape_start_ns,
                        "response_received_perf_ns": response_received_ns,
                        "http_status": status, "raw_body": text,
                        "parsed_running": parsed["running"], "parsed_waiting": parsed["waiting"],
                        "matched_running_metric_name": parsed["matched_running_metric_name"],
                        "matched_waiting_metric_name": parsed["matched_waiting_metric_name"],
                        "parse_status": "ok" if parsed["parse_ok"] else "unparsable",
                        "error": None,
                    })
                else:
                    self.samples.append({
                        "scrape_start_perf_ns": scrape_start_ns,
                        "response_received_perf_ns": response_received_ns,
                        "http_status": status, "raw_body": text,
                        "parsed_running": None, "parsed_waiting": None,
                        "matched_running_metric_name": None, "matched_waiting_metric_name": None,
                        "parse_status": "http_error", "error": f"http_status={status}",
                    })
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001 -- a sampler failure must never crash the episode
                response_received_ns = self.clock.perf_counter_ns()
                self.samples.append({
                    "scrape_start_perf_ns": scrape_start_ns,
                    "response_received_perf_ns": response_received_ns,
                    "http_status": None, "raw_body": None,
                    "parsed_running": None, "parsed_waiting": None,
                    "matched_running_metric_name": None, "matched_waiting_metric_name": None,
                    "parse_status": "transport_error", "error": type(exc).__name__,
                })
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    def nearest_sample_before(self, ns: int) -> dict | None:
        """A sample is eligible only if its RESPONSE arrived at or
        before `ns` -- a scrape that started before `ns` but whose
        response arrived after it is never selected (B2 fix)."""
        candidates = [s for s in self.samples if s["response_received_perf_ns"] <= ns]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s["response_received_perf_ns"])


def evaluate_metrics_quality(*, nearest_sample: dict | None, trigger_perf_ns, k: int) -> dict:
    if trigger_perf_ns is None or nearest_sample is None:
        return {
            "metrics_quality_status": METRICS_QUALITY_UNAVAILABLE,
            "reason": "no metrics sample was collected before the logical trigger",
            "nearest_pre_trigger_sample": nearest_sample,
            "sample_age_ms": None, "parsed_running": None, "parsed_waiting": None,
        }
    # B2 fix: staleness/age is computed from response_received_perf_ns,
    # never scrape_start_perf_ns.
    response_received_ns = nearest_sample["response_received_perf_ns"]
    age_ms = (trigger_perf_ns - response_received_ns) / 1e6
    if nearest_sample.get("error") is not None or nearest_sample.get("parse_status") != "ok":
        return {
            "metrics_quality_status": METRICS_QUALITY_UNPARSABLE,
            "reason": f"nearest pre-trigger sample could not be parsed (error={nearest_sample.get('error')!r}, "
                      f"parse_status={nearest_sample.get('parse_status')!r})",
            "nearest_pre_trigger_sample": nearest_sample,
            "sample_age_ms": age_ms, "parsed_running": None, "parsed_waiting": None,
        }
    running = nearest_sample.get("parsed_running")
    waiting = nearest_sample.get("parsed_waiting")
    if age_ms > METRICS_STALENESS_THRESHOLD_MS:
        return {
            "metrics_quality_status": METRICS_QUALITY_STALE,
            "reason": f"nearest pre-trigger sample's response is {age_ms:.1f}ms old, exceeds the "
                      f"{METRICS_STALENESS_THRESHOLD_MS}ms staleness threshold",
            "nearest_pre_trigger_sample": nearest_sample,
            "sample_age_ms": age_ms, "parsed_running": running, "parsed_waiting": waiting,
        }
    if running != float(k) or waiting != float(20 - k):
        return {
            "metrics_quality_status": METRICS_QUALITY_CONTRADICTORY,
            "reason": f"parsed running={running!r}/waiting={waiting!r} != expected running={k}/waiting={20 - k}",
            "nearest_pre_trigger_sample": nearest_sample,
            "sample_age_ms": age_ms, "parsed_running": running, "parsed_waiting": waiting,
        }
    return {
        "metrics_quality_status": METRICS_QUALITY_CORROBORATED,
        "reason": "",
        "nearest_pre_trigger_sample": nearest_sample,
        "sample_age_ms": age_ms, "parsed_running": running, "parsed_waiting": waiting,
    }


# ============================================================================
# Persistent HTTP transports (2026-07-20 hardening pass, requirements 1,
# 2, 4). The audited base HttpxTransport creates a NEW httpx.AsyncClient
# per request/get -- unsuitable for (a) exact, auditable connection-pool
# limits across all 20+4 concurrent completion streams, and (b) a
# metrics client that must never perturb, or be perturbed by, the
# completion pool. Both classes below are separate, independent
# httpx.AsyncClient owners with explicit start()/aclose() lifecycles.
# ============================================================================

COMPLETION_POOL_MAX_CONNECTIONS = 32
COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS = 32


class PersistentCompletionTransport:
    """Owns exactly one reusable httpx.AsyncClient for /v1/completions
    streams (and lightweight GETs: /health, /v1/models, /openapi.json),
    with explicit connection-pool limits, for the full duration of a
    diagnostic pair or smoke block (both conditions share one client;
    never recreated per-request or per-episode). Implements the same
    HTTPTransport protocol as base.HttpxTransport/base.FakeTransport, so
    it is a drop-in RunContext.transport.

    Records, per request_id, `stream_open_or_response_headers_perf_ns`
    -- the instant `async with client.stream(...)` has entered and the
    response status/headers are already available, but BEFORE the first
    SSE body line is read. This is the earliest point at which the real
    httpx stack exposes a defensible "the server responded" boundary;
    nothing earlier is invented (requirement 2). Also tracks the peak
    number of simultaneously open completion streams.
    """

    def __init__(
        self, *, clock=None, max_connections: int = COMPLETION_POOL_MAX_CONNECTIONS,
        max_keepalive_connections: int = COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS,
        default_timeout_s: float = 1200.0,
    ) -> None:
        self.clock = clock if clock is not None else base.RealClock()
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.default_timeout_s = default_timeout_s
        self._client = None
        self._registry: dict[str, dict] = {}
        self._registry_lock: "asyncio.Lock | None" = None
        # B3 fix (2026-07-20 second hardening pass): two DISTINCT counters.
        # `_inflight_count`/`peak_inflight_completion_attempts` counts from
        # the moment a request enters stream_completion() (which may still
        # be waiting for a connection-pool slot or for the server to
        # respond) -- this is the OLD, honestly-renamed semantics.
        # `_open_count`/`peak_open_stream_count` counts ONLY from the
        # moment response headers/status are actually available (i.e.
        # `stream_open_or_response_headers_perf_ns` has just been set)
        # until the stream closes -- this is the semantically CORRECT
        # "open completion stream" measurement and is what
        # transport_concurrency_evidence.peak_concurrent_open_completion_streams
        # reports.
        self._inflight_count = 0
        self.peak_inflight_completion_attempts = 0
        self._open_count = 0
        self.peak_open_stream_count = 0

    async def start(self) -> None:
        """Idempotent: a second call while already started is a no-op."""
        if self._client is not None:
            return
        import httpx  # local import: mirrors base.HttpxTransport's lazy-import pattern
        limits = httpx.Limits(
            max_connections=self.max_connections, max_keepalive_connections=self.max_keepalive_connections,
        )
        # Long SSE streams need a generous read timeout; the audited
        # execute_completion_request() already supervises the overall
        # per-request budget via asyncio.wait_for(http_timeout_s), so
        # this client-level timeout is a defensive outer bound, not the
        # primary enforcement mechanism.
        timeout = httpx.Timeout(self.default_timeout_s, connect=30.0)
        self._client = httpx.AsyncClient(limits=limits, timeout=timeout)
        self._registry_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Idempotent; safe to call on success, failure, cancellation,
        or from a SIGINT/SIGTERM-triggered cleanup path."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_started(self) -> bool:
        return self._client is not None

    def get_stream_registry_entry(self, request_id: str) -> dict | None:
        return self._registry.get(request_id)

    def get_diagnostics(self) -> dict:
        return {
            "max_connections": self.max_connections,
            "max_keepalive_connections": self.max_keepalive_connections,
            "peak_open_stream_count": self.peak_open_stream_count,
            "peak_inflight_completion_attempts": self.peak_inflight_completion_attempts,
        }

    async def get_json(self, url: str, headers: dict[str, str], timeout_s: float) -> tuple[int, Any]:
        if self._client is None:
            raise base.ServerLifecycleError("PersistentCompletionTransport.get_json() called before start()")
        resp = await self._client.get(url, headers=headers, timeout=timeout_s)
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body

    async def stream_completion(
        self, url: str, headers: dict[str, str], payload: dict, timeout_s: float, *, request_id: str | None = None,
    ):
        if self._client is None:
            raise base.ServerLifecycleError("PersistentCompletionTransport.stream_completion() called before start()")
        key = request_id or f"<unlabeled-{id(payload)}>"
        entry = {
            "request_id": key, "stream_open_or_response_headers_perf_ns": None,
            "stream_close_perf_ns": None, "http_status": None,
        }
        assert self._registry_lock is not None
        async with self._registry_lock:
            self._registry[key] = entry
            self._inflight_count += 1
            self.peak_inflight_completion_attempts = max(self.peak_inflight_completion_attempts, self._inflight_count)
        opened = False
        try:
            async with self._client.stream("POST", url, headers=headers, json=payload, timeout=timeout_s) as resp:
                # Requirement 2: recorded immediately after the response
                # status/headers are available, before consuming any
                # SSE body line. B3 fix: the TRUE open-stream counter is
                # incremented here too -- only once headers/status exist,
                # never earlier.
                entry["stream_open_or_response_headers_perf_ns"] = self.clock.perf_counter_ns()
                entry["http_status"] = resp.status_code
                async with self._registry_lock:
                    self._open_count += 1
                    self.peak_open_stream_count = max(self.peak_open_stream_count, self._open_count)
                    opened = True
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise base.HTTPStatusError(resp.status_code, body)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    yield line[len("data:"):].strip()
        finally:
            entry["stream_close_perf_ns"] = self.clock.perf_counter_ns()
            async with self._registry_lock:
                self._inflight_count -= 1
                if opened:
                    self._open_count -= 1


class PersistentMetricsTransport:
    """A SEPARATE persistent httpx.AsyncClient dedicated to /metrics
    scraping -- never shared with PersistentCompletionTransport, so
    metrics polling can neither perturb nor be perturbed by the
    completion pool (requirement 4 / N2)."""

    def __init__(self, *, default_timeout_s: float = 10.0) -> None:
        self.default_timeout_s = default_timeout_s
        self._client = None

    async def start(self) -> None:
        if self._client is not None:
            return
        import httpx
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.default_timeout_s))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_started(self) -> bool:
        return self._client is not None

    async def get_json(self, url: str, headers: dict[str, str], timeout_s: float) -> tuple[int, Any]:
        if self._client is None:
            raise base.ServerLifecycleError("PersistentMetricsTransport.get_json() called before start()")
        resp = await self._client.get(url, headers=headers, timeout=timeout_s)
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body


class FakeCompletionTransport:
    """Test-only stand-in for PersistentCompletionTransport: delegates
    scripted responses to an inner base.FakeTransport (so existing
    FakeStreamScript/queue_script/default_script_factory test fixtures
    keep working unchanged) while additionally providing the SAME
    stream-open registry / peak-concurrency-tracking interface the real
    persistent transport provides, so tests can exercise the new
    transport-concurrency-evidence fields without a real network.

    `clock` should be the SAME Clock instance used elsewhere in a test
    (e.g. the FakeClock passed to RunContext) so registry timestamps
    remain comparable to task_created_ns/first_token_perf_ns/etc.
    -- a real persistent transport records genuinely independent
    wall-clock time, but a fake one must share the test's clock to make
    monotonicity assertions meaningful.
    """

    def __init__(
        self, inner: "base.FakeTransport | None" = None, *, clock=None,
        max_connections: int = COMPLETION_POOL_MAX_CONNECTIONS,
        max_keepalive_connections: int = COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS,
    ) -> None:
        self.inner = inner if inner is not None else base.FakeTransport()
        self.clock = clock if clock is not None else base.RealClock()
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._registry: dict[str, dict] = {}
        self._inflight_count = 0
        self.peak_inflight_completion_attempts = 0
        self._open_count = 0
        self.peak_open_stream_count = 0
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True

    def is_started(self) -> bool:
        return self.started and not self.closed

    def get_stream_registry_entry(self, request_id: str) -> dict | None:
        return self._registry.get(request_id)

    def get_diagnostics(self) -> dict:
        return {
            "max_connections": self.max_connections,
            "max_keepalive_connections": self.max_keepalive_connections,
            "peak_open_stream_count": self.peak_open_stream_count,
            "peak_inflight_completion_attempts": self.peak_inflight_completion_attempts,
        }

    async def get_json(self, url, headers, timeout_s):
        return await self.inner.get_json(url, headers, timeout_s)

    async def stream_completion(self, url, headers, payload, timeout_s, *, request_id: str | None = None):
        key = request_id or f"<unlabeled-{id(payload)}>"
        entry = {
            "request_id": key, "stream_open_or_response_headers_perf_ns": None,
            "stream_close_perf_ns": None, "http_status": None,
        }
        self._registry[key] = entry
        self._inflight_count += 1
        self.peak_inflight_completion_attempts = max(self.peak_inflight_completion_attempts, self._inflight_count)
        first_event = True
        opened = False
        try:
            async for raw in self.inner.stream_completion(url, headers, payload, timeout_s, request_id=request_id):
                if first_event:
                    # Earliest defensible equivalent, in the fake world,
                    # of "the response headers/status are available" --
                    # the instant the stream actually starts producing.
                    # B3 fix: the TRUE open-stream counter is incremented
                    # only here, never at dispatch -- a script with
                    # `hang=True` never reaches this point, so it never
                    # inflates peak_open_stream_count (see the dedicated
                    # regression test).
                    entry["stream_open_or_response_headers_perf_ns"] = self.clock.perf_counter_ns()
                    entry["http_status"] = 200
                    first_event = False
                    self._open_count += 1
                    self.peak_open_stream_count = max(self.peak_open_stream_count, self._open_count)
                    opened = True
                yield raw
        finally:
            entry["stream_close_perf_ns"] = self.clock.perf_counter_ns()
            self._inflight_count -= 1
            if opened:
                self._open_count -= 1


# ============================================================================
# Run context + episode runner
# ============================================================================

# RunContext is fully generic (transport/clock/sleeper/base_url/api_key/
# model_full_id/valid_ids/http_timeout_s/trigger_timeout_s) -- reused
# unchanged.
RunContext = base.RunContext


async def run_server_waiting_episode(
    ctx: "base.RunContext",
    episode: Episode,
    *,
    schedule_fingerprint: str,
    server_metadata: dict,
    stabilization_ref: dict,
    run_mode: str,
    metrics_sampler: "MetricsSampler | None" = None,
    cohort_freeze_timeout_s: float | None = None,
    trigger_timeout_s: float | None = None,
) -> dict:
    """The server-waiting replacement for the audited
    run_prefill_confirmation.run_regular_episode(). Reuses
    execute_completion_request (via base._run_victim_request /
    base._run_burst_request) and base._aggregate_metrics unchanged;
    replaces the client-semaphore + static-wave dispatch/trigger with
    the data-driven cohort/trigger barrier in _active_cohort.py, and
    the wave-based exposure fields with server_exposure_group / etc."""
    n = episode.victim_request_count
    k = episode.server_max_num_seqs
    threshold = episode.trigger_after_decode_tokens
    cohort_timeout = cohort_freeze_timeout_s if cohort_freeze_timeout_s is not None else ctx.trigger_timeout_s
    trig_timeout = trigger_timeout_s if trigger_timeout_s is not None else ctx.trigger_timeout_s

    signal = cohort._ProgressSignal()
    progress: dict[int, cohort.TokenProgress] = {
        i: cohort.TokenProgress(request_index=i, signal=signal) for i in range(n)
    }
    first_token_events = {i: asyncio.Event() for i in range(n)}
    victim_tasks: dict[int, asyncio.Task] = {}
    burst_tasks: list[asyncio.Task] = []
    task_metadata: dict[int, dict] = {}

    def request_status_ok(result: dict) -> bool:
        return isinstance(result, dict) and result.get("status") == base.REQUEST_STATUS_COMPLETE

    victim_phase_start_ns = ctx.clock.perf_counter_ns()

    def _make_on_tokens(i: int):
        def _cb(cumulative_count: int, receive_ns: int) -> None:
            progress[i].record(cumulative_count, receive_ns)
            if cumulative_count > 0 and not first_token_events[i].is_set():
                first_token_events[i].set()
        return _cb

    async def _victim(i: int, task_created_ns: int) -> dict:
        result = await base._run_victim_request(
            ctx, episode, i, on_output_tokens=_make_on_tokens(i),
            task_created_ns=task_created_ns,
            victim_phase_start_ns=victim_phase_start_ns,
            semaphore_acquired_ns=None,  # Requirement: no client admission semaphore.
            wave_id=None, wave_position=None,
        )
        end_ns = result.get("stream_end_ns")
        if type(end_ns) is int:
            # Use execute_completion_request's OWN stream_end_ns, not a
            # fresh clock read here -- this is what
            # validate_server_waiting_episode_invariants compares
            # against the logical trigger.
            progress[i].mark_complete(end_ns)
        return result

    active_indices: frozenset[int] = frozenset()
    trigger_ns: int | None = None

    if metrics_sampler is not None:
        metrics_sampler.start()

    try:
        # Requirements 1-3: create all 20 victim tasks, begin all 20 HTTP
        # streaming requests with no client admission semaphore. HTTP
        # connection-pool sizing for >=24 concurrent streams is the
        # transport's responsibility -- see PersistentCompletionTransport
        # (constructed in the CLI section with explicit
        # max_connections=32/max_keepalive_connections=32 limits).
        for i in range(n):
            task_created_ns = ctx.clock.perf_counter_ns()
            task_metadata[i] = {
                "request_index": i, "task_created_ns": task_created_ns,
                "victim_phase_start_ns": victim_phase_start_ns,
            }
            victim_tasks[i] = asyncio.create_task(_victim(i, task_created_ns))
        await asyncio.sleep(0)

        trigger_start_ns = ctx.clock.perf_counter_ns()
        cohort_result = await cohort.watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=threshold,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=cohort_timeout, trigger_timeout_s=trig_timeout,
            signal=signal,
        )
        trigger_observed_ns = ctx.clock.perf_counter_ns()
        trigger_observed_utc = ctx.clock.utcnow_iso()

        trigger_ns = cohort_result.logical_trigger_ns
        active_indices = cohort_result.active_indices or frozenset()

        metrics_pre_trigger_sample = None
        if metrics_sampler is not None:
            metrics_pre_trigger_sample = metrics_sampler.nearest_sample_before(
                trigger_ns if trigger_ns is not None else trigger_observed_ns
            )
        metrics_quality = evaluate_metrics_quality(
            nearest_sample=metrics_pre_trigger_sample, trigger_perf_ns=trigger_ns, k=k,
        )

        trigger = {
            "status": cohort_result.status,
            "detail": cohort_result.detail,
            "trigger_observed_utc": trigger_observed_utc,
            "trigger_perf_ns": trigger_ns,
            "trigger_observed_perf_ns": trigger_observed_ns,
            "trigger_wait_duration_ms": (trigger_observed_ns - trigger_start_ns) / 1e6,
            "trigger_after_decode_tokens": threshold,
            "server_max_num_seqs": k,
            "cohort_freeze_ns": cohort_result.cohort_freeze_ns,
            "active_cohort_request_indices": sorted(active_indices),
            "active_cohort_size": len(active_indices),
            "metrics_quality": metrics_quality,
            "metrics_raw_samples": list(metrics_sampler.samples) if metrics_sampler is not None else [],
        }

        if cohort_result.status != cohort.DYNAMIC_TRIGGER_OK:
            await base.cancel_all(victim_tasks.values())
            victim_results_raw = await asyncio.gather(*[victim_tasks[i] for i in range(n)], return_exceptions=True)
            victim_results = base._enrich_minimal_records(
                base._coerce_task_results(victim_results_raw), episode_id=episode.episode_id, role="victim",
            )
            base._apply_known_task_metadata(victim_results, task_metadata)
            _enrich_server_exposure_fields(
                victim_results, trigger_perf_ns=trigger_ns, active_indices=active_indices, k=k,
                progress_by_index=progress,
            )
            _enrich_victim_transport_and_timing_fields(
                victim_results, transport=ctx.transport, progress_by_index=progress,
                trigger_after_decode_tokens=threshold, episode_id=episode.episode_id,
            )
            transport_concurrency_evidence = compute_transport_concurrency_evidence(victim_results, ctx.transport)
            return _build_episode_result(
                episode=episode, schedule_fingerprint=schedule_fingerprint,
                server_metadata=server_metadata, stabilization_ref=stabilization_ref,
                trigger=trigger, burst_interval=None, victim_results=victim_results,
                burst_results=[], status="failed",
                validation_errors=[f"trigger failed: {cohort_result.status}: {cohort_result.detail}"],
                run_mode=run_mode, victim_phase_start_ns=victim_phase_start_ns,
                transport_concurrency_evidence=transport_concurrency_evidence,
            )

        if episode.condition == base.BURST_CONDITION:
            for j in range(episode.burst_parallel_requests):
                burst_tasks.append(asyncio.create_task(base._run_burst_request(ctx, episode, j)))

        victim_results_raw = await asyncio.gather(*[victim_tasks[i] for i in range(n)], return_exceptions=True)
        burst_results_raw = (
            await asyncio.gather(*burst_tasks, return_exceptions=True) if burst_tasks else []
        )
    except asyncio.CancelledError:
        await base.cancel_all(list(victim_tasks.values()) + burst_tasks)
        raise
    finally:
        if metrics_sampler is not None:
            await metrics_sampler.stop()

    victim_results = base._coerce_task_results(victim_results_raw)
    burst_results = base._coerce_task_results(burst_results_raw)
    _enrich_server_exposure_fields(
        victim_results, trigger_perf_ns=trigger_ns, active_indices=active_indices, k=k,
        progress_by_index=progress,
    )
    monotonicity_errors = _enrich_victim_transport_and_timing_fields(
        victim_results, transport=ctx.transport, progress_by_index=progress,
        trigger_after_decode_tokens=threshold, episode_id=episode.episode_id,
    )
    monotonicity_errors += _enrich_burst_transport_and_timing_fields(
        burst_results, transport=ctx.transport, episode_id=episode.episode_id,
    )
    transport_concurrency_evidence = compute_transport_concurrency_evidence(victim_results, ctx.transport)

    all_complete = all(r.get("status") == base.REQUEST_STATUS_COMPLETE for r in victim_results) and all(
        r.get("status") == base.REQUEST_STATUS_COMPLETE for r in burst_results
    )
    validation_errors: list[str] = []
    if not all_complete:
        validation_errors.append("one or more requests after trigger were not complete")
    validation_errors.extend(
        validate_server_waiting_episode_invariants(
            episode=episode, victim_results=victim_results, burst_results=burst_results,
            trigger_ns=trigger_ns, active_indices=active_indices, k=k,
        )
    )
    # Requirement 2: enforce/validate the canonical timestamp ordering
    # for every victim and burst request.
    validation_errors.extend(monotonicity_errors)

    burst_interval = None
    if burst_results:
        starts = [r["request_start_ns"] for r in burst_results if r.get("request_start_ns") is not None]
        ends = [r["stream_end_ns"] for r in burst_results if r.get("stream_end_ns") is not None]
        if starts and ends:
            burst_interval = {"start_ns": min(starts), "end_ns": max(ends)}

    status = base.REQUEST_STATUS_COMPLETE if not validation_errors else "failed"

    return _build_episode_result(
        episode=episode, schedule_fingerprint=schedule_fingerprint,
        server_metadata=server_metadata, stabilization_ref=stabilization_ref,
        trigger=trigger, burst_interval=burst_interval, victim_results=victim_results,
        burst_results=burst_results, status=status, validation_errors=validation_errors,
        run_mode=run_mode, victim_phase_start_ns=victim_phase_start_ns,
        transport_concurrency_evidence=transport_concurrency_evidence,
    )


async def run_server_waiting_stabilization(
    ctx: "base.RunContext", bundle: LoadedBundle, model_key: str, block_id: str,
    offload_gb: int, state_label: str, *, server_metadata: dict,
) -> dict:
    """Thin wrapper around the audited base.run_stabilization(): the
    REQUEST EXECUTION is reused byte-for-byte unchanged (20 requests,
    internal concurrency-4 semaphore, first/second-half median
    comparison -- stabilization is not part of what this extension
    changes). Only the identity fields that would otherwise falsely
    claim the OLD schema/runner identity are corrected afterward. This
    is a deliberate, minimal, documented deviation -- see AUDIT.md."""
    stab_result = await base.run_stabilization(
        ctx, bundle, model_key, block_id, offload_gb, state_label, server_metadata=server_metadata,
    )
    stab_result["result_schema_version"] = RESULT_SCHEMA_VERSION
    stab_result["runner_version"] = RUNNER_VERSION
    return stab_result


# ============================================================================
# Server lifecycle: new launcher/command, reused start/stop/readiness.
# ============================================================================

import shutil  # noqa: E402


def build_server_command(
    run_server_path: Path, model_key: str, offload_gb: int, server_max_num_seqs: int, host: str, port: int,
) -> list[str]:
    return ["bash", str(run_server_path), model_key, str(offload_gb), str(server_max_num_seqs), host, str(port)]


def resolve_run_server_path(script_path: Path = SCRIPT_PATH) -> Path:
    return script_path.parent / "run_server_waiting_server.sh"


def check_run_server_script(run_server_path: Path) -> None:
    if not run_server_path.is_file() or not os.access(run_server_path, os.R_OK):
        raise base.ServerLifecycleError(
            f"run_server_waiting_server.sh '{run_server_path}' does not exist as a readable regular file"
        )
    if shutil.which("bash") is None:
        raise base.ServerLifecycleError("'bash' is not available in PATH")
    try:
        text = run_server_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise base.ServerLifecycleError(f"could not read run_server_waiting_server.sh: {exc}") from exc
    required_fragments = (
        "--enable-chunked-prefill", "--max-num-batched-tokens", 'MAX_NUM_BATCHED_TOKENS="2048"',
        "--cpu-offload-gb", "--max-num-seqs", "--no-enable-prefix-caching", "--max-model-len", "qwen)",
    )
    missing = [f for f in required_fragments if f not in text]
    if missing:
        raise base.ServerLifecycleError(
            f"run_server_waiting_server.sh is missing required frozen-contract fragment(s): {missing}"
        )
    if "--api-key" in text:
        raise base.ServerLifecycleError(
            "run_server_waiting_server.sh must not place the API key on the process command line"
        )


# is_port_free / stop_server / wait_for_server_ready / check_post_stabilization_health /
# fetch_capability_summary / read_api_key_from_env are fully generic (host/port/handle/
# transport/sleeper-parameterized) -- reused unchanged.
is_port_free = base.is_port_free
stop_server = base.stop_server
wait_for_server_ready = base.wait_for_server_ready
check_post_stabilization_health = base.check_post_stabilization_health
fetch_capability_summary = base.fetch_capability_summary
read_api_key_from_env = base.read_api_key_from_env
compute_valid_token_ids = base.compute_valid_token_ids

# Server/API/capability error classes and dependency-injectable
# Clock/Sleeper/HTTPTransport/TokenizerAdapter (+ Real/Fake
# implementations) are fully generic -- reused unchanged.
ApiKeyError = base.ApiKeyError
ServerLifecycleError = base.ServerLifecycleError
CapabilityError = base.CapabilityError
HTTPStatusError = base.HTTPStatusError
Clock = base.Clock
RealClock = base.RealClock
FakeClock = base.FakeClock
Sleeper = base.Sleeper
RealSleeper = base.RealSleeper
FakeSleeper = base.FakeSleeper
HTTPTransport = base.HTTPTransport
HttpxTransport = base.HttpxTransport
FakeStreamScript = base.FakeStreamScript
FakeTransport = base.FakeTransport
TokenizerAdapter = base.TokenizerAdapter
HFTokenizerAdapter = base.HFTokenizerAdapter
FakeTokenizerAdapter = base.FakeTokenizerAdapter
ServerHandle = base.ServerHandle
ServerProcessAdapter = base.ServerProcessAdapter
RealServerProcessAdapter = base.RealServerProcessAdapter
FakeServerHandle = base.FakeServerHandle
FakeServerProcessAdapter = base.FakeServerProcessAdapter

SERVER_READY_TIMEOUT_S = base.SERVER_READY_TIMEOUT_S
HTTP_REQUEST_TIMEOUT_S = base.HTTP_REQUEST_TIMEOUT_S
# N3 fix (2026-07-20 hardening pass): _active_cohort.watch_dynamic_cohort_and_trigger()
# is called (see run_server_waiting_episode) with cohort_freeze_timeout_s
# and trigger_timeout_s as two INDEPENDENT budgets, each defaulting to
# ctx.trigger_timeout_s (this constant) -- applied separately to Phase 1
# (cohort freeze) and Phase 2 (the token-16 barrier). A previous version
# of this comment incorrectly claimed a single shared budget covering
# both phases; the real worst-case wait for one episode's trigger
# machinery is therefore up to approximately 2 x TRIGGER_TIMEOUT_S, not
# TRIGGER_TIMEOUT_S. Documentation-only fix: _active_cohort.py's actual
# timeout semantics are unchanged by this hardening pass.
TRIGGER_TIMEOUT_S = base.TRIGGER_TIMEOUT_S
SERVER_STOP_TIMEOUT_S = base.SERVER_STOP_TIMEOUT_S
COOLDOWN_S = base.COOLDOWN_S
READINESS_POLL_INTERVAL_S = base.READINESS_POLL_INTERVAL_S
COMPLETIONS_ENDPOINT = base.COMPLETIONS_ENDPOINT
HEALTH_ENDPOINT = base.HEALTH_ENDPOINT
MODELS_ENDPOINT = base.MODELS_ENDPOINT
OPENAPI_ENDPOINT = base.OPENAPI_ENDPOINT
RUN_MODE_SMOKE = base.RUN_MODE_SMOKE
RUN_MODE_OFFICIAL = base.RUN_MODE_OFFICIAL
REQUEST_STATUS_COMPLETE = base.REQUEST_STATUS_COMPLETE
REQUEST_STATUS_INCOMPLETE = base.REQUEST_STATUS_INCOMPLETE
REQUEST_STATUS_FAILED = base.REQUEST_STATUS_FAILED
REQUEST_STATUS_CANCELLED = base.REQUEST_STATUS_CANCELLED
STABILIZATION_CONDITION = base.STABILIZATION_CONDITION
STABILIZATION_CONCURRENCY = base.STABILIZATION_CONCURRENCY
STABILIZATION_REQUEST_COUNT = base.STABILIZATION_REQUEST_COUNT
STABILIZATION_INPUT_LEN = base.STABILIZATION_INPUT_LEN
STABILIZATION_OUTPUT_LEN = base.STABILIZATION_OUTPUT_LEN
STABILIZATION_TEMPERATURE = base.STABILIZATION_TEMPERATURE

write_json_atomic = base.write_json_atomic
BundleLoadErrorBase = base.BundleLoadError
OutputDirConflictError = base.OutputDirConflictError

EPISODES_SUBDIR = base.EPISODES_SUBDIR
STABILIZATION_SUBDIR = base.STABILIZATION_SUBDIR
episode_result_path = base.episode_result_path
stabilization_result_path = base.stabilization_result_path

# N4 fix (2026-07-20 hardening pass): the base module's
# check_output_dir_not_shared/write_run_mode_marker/
# require_output_dir_mode_marker all reference base's OWN
# RUN_MODE_MARKER_FILENAME (".prefill_confirmation_run_mode") internally
# -- aliasing them directly meant this module's own
# RUN_MODE_MARKER_FILENAME constant was defined but never actually used,
# which is misleading even though the two output-dir trees never
# overlap in practice. These local implementations use THIS module's
# own marker filename, so the on-disk artifact identity honestly
# reflects which runner produced it.
def check_output_dir_not_shared(output_dir: Path, mode: str) -> None:
    marker_path = output_dir / RUN_MODE_MARKER_FILENAME
    if not marker_path.exists():
        return
    try:
        existing_mode = marker_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise base.OutputDirConflictError(f"could not read run-mode marker '{marker_path}': {exc}") from exc
    if existing_mode != mode:
        raise base.OutputDirConflictError(
            f"output-dir '{output_dir}' is already marked as belonging to run mode {existing_mode!r}; "
            f"refusing to also use it for run mode {mode!r}."
        )


def write_run_mode_marker(output_dir: Path, mode: str) -> None:
    (output_dir / RUN_MODE_MARKER_FILENAME).write_text(mode, encoding="utf-8")


def require_output_dir_mode_marker(output_dir: Path, mode: str) -> None:
    marker_path = output_dir / RUN_MODE_MARKER_FILENAME
    if not marker_path.is_file():
        raise base.OutputDirConflictError(f"--resume requires an existing run-mode marker at {marker_path}")
    check_output_dir_not_shared(output_dir, mode)


# ============================================================================
# Result-file classification / resume-depth (new schema; framework
# pattern mirrors the audited base module's classify_result_file, bound
# to THIS module's own required-key/type set and to
# validate_server_waiting_episode_invariants instead of the wave/
# semaphore-oriented validator -- see AUDIT.md for exact scope: this
# does NOT re-derive expected prompt token ids from seeds (that needs a
# live tokenizer), it DOES check schema_row/id/fingerprint/counts and
# re-runs the full trigger/cohort/exposure invariant set against the
# stored data.)
# ============================================================================

_RESULT_REQUIRED_KEYS = {
    "result_schema_version", "runner_version", "record_type", "run_mode", "schedule_fingerprint",
    "episode_id", "schedule_row", "block_id", "server_metadata", "stabilization_reference",
    "trigger", "burst_interval", "victim_requests", "burst_requests", "aggregate_metrics",
    "status", "validation_errors", "timing_instrumentation_version", "timing_instrumentation_name",
    "victim_phase_start_ns", "no_client_admission_semaphore_used",
}
_RESULT_FIELD_TYPES: dict[str, type] = {
    "result_schema_version": int, "runner_version": str, "record_type": str, "run_mode": str,
    "schedule_fingerprint": str, "episode_id": str, "schedule_row": dict, "block_id": str,
    "trigger": dict, "victim_requests": list, "burst_requests": list, "status": str,
    "timing_instrumentation_version": int, "timing_instrumentation_name": str, "victim_phase_start_ns": int,
    "validation_errors": list, "no_client_admission_semaphore_used": bool,
}


def classify_result_file(
    path: Path, expected_episode: Episode, expected_fingerprint: str, expected_run_mode: str,
) -> tuple[str, list[str]]:
    """B3 hardening (2026-07-20): in addition to the original checks,
    this now ALSO requires -- as hard, fail-closed conditions, not mere
    advisory notes -- that: `validation_errors == []`,
    `no_client_admission_semaphore_used is True`, `trigger.status ==
    'ok'`, `trigger.active_cohort_size` is internally consistent with
    the actual number of unique `active_cohort_request_indices` and
    equals the episode's own `server_max_num_seqs`, and every victim AND
    burst record individually has `status == 'complete'` (not just the
    right COUNT of records). It also recomputes each victim's
    server_exposure_group / was_dispatched_at_trigger /
    had_first_token_at_trigger / dispatch_to_first_token_ms from that
    record's own stored timestamps and active-cohort membership, and
    rejects the file if the stored value disagrees with the
    recomputation. These four categories are exactly the adversarial
    mutations the independent code audit
    (SERVER_WAITING_CODE_AUDIT_2026-07-20.md, finding B3) demonstrated
    were previously accepted as valid_complete."""
    if not path.exists():
        return CLASSIFICATION_MISSING, []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return CLASSIFICATION_CORRUPTED, [f"could not read file: {exc}"]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return CLASSIFICATION_CORRUPTED, [f"invalid JSON: {exc}"]
    if not isinstance(obj, dict):
        return CLASSIFICATION_CORRUPTED, ["result file is not a JSON object"]

    missing_keys = _RESULT_REQUIRED_KEYS - set(obj.keys())
    if missing_keys:
        return CLASSIFICATION_INVALID, [f"missing required key(s): {sorted(missing_keys)}"]

    type_errors = [
        f"field {f!r} has wrong type: expected {t.__name__}, got {type(obj[f]).__name__}"
        for f, t in _RESULT_FIELD_TYPES.items() if not _check_type_strict(obj[f], t)
    ]
    if type_errors:
        return CLASSIFICATION_INVALID, type_errors

    if obj["status"] != base.REQUEST_STATUS_COMPLETE:
        return CLASSIFICATION_PARTIAL, [f"status={obj['status']!r} != 'complete'"]

    notes: list[str] = []
    if obj["result_schema_version"] != RESULT_SCHEMA_VERSION:
        notes.append(f"result_schema_version {obj['result_schema_version']!r} != {RESULT_SCHEMA_VERSION!r}")
    if obj["runner_version"] != RUNNER_VERSION:
        notes.append(f"runner_version {obj['runner_version']!r} != {RUNNER_VERSION!r}")
    if obj["schedule_fingerprint"] != expected_fingerprint:
        notes.append("schedule_fingerprint mismatch")
    if obj["record_type"] != RECORD_TYPE_REGULAR_EPISODE:
        notes.append(f"record_type != {RECORD_TYPE_REGULAR_EPISODE!r}")
    if obj["run_mode"] != expected_run_mode:
        notes.append("run_mode mismatch")
    if obj["timing_instrumentation_version"] != TIMING_INSTRUMENTATION_VERSION:
        notes.append("timing_instrumentation_version mismatch")
    if obj["timing_instrumentation_name"] != TIMING_INSTRUMENTATION_NAME:
        notes.append("timing_instrumentation_name mismatch")
    if obj["episode_id"] != expected_episode.episode_id:
        notes.append(f"episode_id {obj['episode_id']!r} does not match expected {expected_episode.episode_id!r}")
    expected_row = asdict(expected_episode)
    if obj["schedule_row"] != expected_row:
        notes.append("schedule_row does not exactly match the expected schedule episode")

    # --- B3: a "complete" episode must have declared zero validation
    # errors and an explicit True no-semaphore flag; a file with a
    # non-empty validation_errors or a missing/False semaphore flag can
    # never be valid_complete regardless of anything else. ---------------
    if obj["validation_errors"] != []:
        notes.append(f"validation_errors is not empty: {obj['validation_errors']!r}")
    if obj["no_client_admission_semaphore_used"] is not True:
        notes.append("no_client_admission_semaphore_used is not exactly True")

    victim_requests = obj["victim_requests"]
    if len(victim_requests) != expected_episode.victim_request_count:
        notes.append(f"expected exactly {expected_episode.victim_request_count} victim_requests")

    expected_burst_count = (
        expected_episode.burst_parallel_requests if expected_episode.condition == base.BURST_CONDITION else 0
    )
    burst_requests = obj["burst_requests"]
    if len(burst_requests) != expected_burst_count:
        notes.append(
            f"expected exactly {expected_burst_count} burst_requests for condition "
            f"{expected_episode.condition!r}"
        )

    # --- B3: every individual victim AND burst record must itself be
    # complete -- the original classifier only checked list LENGTH, so
    # e.g. a burst record's status silently changed to "failed" was
    # invisible here. -------------------------------------------------
    if isinstance(victim_requests, list):
        bad_victims = sorted(
            r.get("request_index") for r in victim_requests
            if isinstance(r, dict) and r.get("status") != base.REQUEST_STATUS_COMPLETE
        )
        if bad_victims:
            notes.append(f"victim request(s) with status != 'complete': {bad_victims}")
    if isinstance(burst_requests, list):
        bad_bursts = sorted(
            r.get("request_index") for r in burst_requests
            if isinstance(r, dict) and r.get("status") != base.REQUEST_STATUS_COMPLETE
        )
        if bad_bursts:
            notes.append(f"burst request(s) with status != 'complete': {bad_bursts}")

    if isinstance(victim_requests, list):
        indices = [r.get("request_index") for r in victim_requests if isinstance(r, dict)]
        if len(indices) != len(set(indices)):
            notes.append("duplicate request_index value(s) among victim_requests")
        if sorted(i for i in indices if isinstance(i, int)) != list(range(expected_episode.victim_request_count)):
            notes.append("victim_requests request_index values are not exactly 0..N-1")

    trigger = obj["trigger"]
    k = expected_episode.server_max_num_seqs
    trigger_ns = trigger.get("trigger_perf_ns") if isinstance(trigger, dict) else None
    active_indices_list = trigger.get("active_cohort_request_indices") if isinstance(trigger, dict) else None
    active_indices = frozenset(active_indices_list or []) if isinstance(active_indices_list, list) else frozenset()

    # --- B3: trigger.status must be exactly 'ok', and the reported
    # active_cohort_size must be internally consistent with the actual
    # unique index list and with this episode's own server_max_num_seqs.
    if not isinstance(trigger, dict) or trigger.get("status") != cohort.DYNAMIC_TRIGGER_OK:
        notes.append(
            f"trigger.status != {cohort.DYNAMIC_TRIGGER_OK!r} "
            f"(got {trigger.get('status') if isinstance(trigger, dict) else trigger!r})"
        )
    if not isinstance(active_indices_list, list) or len(set(active_indices_list)) != len(active_indices_list):
        notes.append("trigger.active_cohort_request_indices is missing or contains duplicates")
    elif len(active_indices_list) != k:
        notes.append(f"trigger.active_cohort_request_indices has {len(active_indices_list)} entries, expected {k}")
    if isinstance(trigger, dict) and trigger.get("active_cohort_size") != len(active_indices):
        notes.append(
            f"trigger.active_cohort_size ({trigger.get('active_cohort_size')!r}) != "
            f"len(active_cohort_request_indices) ({len(active_indices)})"
        )
    if isinstance(trigger, dict) and trigger.get("active_cohort_size") != k:
        notes.append(f"trigger.active_cohort_size != server_max_num_seqs ({k})")

    # --- B3: recompute and compare each victim's exposure labels from
    # its own stored timestamps -- a stored server_exposure_group /
    # was_dispatched_at_trigger / had_first_token_at_trigger /
    # dispatch_to_first_token_ms that disagrees with what those same
    # stored timestamps actually imply is rejected. -----------------------
    if isinstance(victim_requests, list) and type(trigger_ns) is int:
        for r in victim_requests:
            if not isinstance(r, dict):
                continue
            idx = r.get("request_index")
            recomputed = compute_server_exposure(
                request_dispatch_ns=r.get("request_dispatch_ns"),
                first_token_perf_ns=r.get("first_token_perf_ns"),
                decode_tokens_received_at_trigger=r.get("decode_tokens_received_at_trigger"),
                trigger_perf_ns=trigger_ns, in_active_cohort=idx in active_indices,
            )
            for key, expected_value in recomputed.items():
                if r.get(key) != expected_value:
                    notes.append(
                        f"victim {idx}: stored {key}={r.get(key)!r} != recomputed {expected_value!r} "
                        f"from its own stored timestamps"
                    )
            dispatch_ns = r.get("request_dispatch_ns")
            first_token_ns = r.get("first_token_perf_ns")
            expected_dtf = (
                (first_token_ns - dispatch_ns) / 1e6
                if type(first_token_ns) is int and type(dispatch_ns) is int and first_token_ns >= dispatch_ns
                else None
            )
            stored_dtf = r.get("dispatch_to_first_token_ms")
            if expected_dtf is None:
                if stored_dtf is not None:
                    notes.append(f"victim {idx}: dispatch_to_first_token_ms should be null, got {stored_dtf!r}")
            elif not isinstance(stored_dtf, (int, float)) or abs(float(stored_dtf) - expected_dtf) > 1e-6:
                notes.append(
                    f"victim {idx}: dispatch_to_first_token_ms={stored_dtf!r} != recomputed {expected_dtf!r}"
                )

    if (
        isinstance(victim_requests, list) and isinstance(burst_requests, list)
        and all(isinstance(r, dict) for r in victim_requests)
    ):
        notes.extend(
            validate_server_waiting_episode_invariants(
                episode=expected_episode, victim_results=victim_requests, burst_results=burst_requests,
                trigger_ns=trigger_ns if type(trigger_ns) is int else None,
                active_indices=active_indices, k=k,
            )
        )

    if notes:
        return CLASSIFICATION_INVALID, notes
    return CLASSIFICATION_VALID_COMPLETE, []


def scan_existing_results(output_dir: Path, bundle: LoadedBundle, run_mode: str) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for ep in bundle.episodes:
        result_path = episode_result_path(output_dir, ep.episode_id)
        cls, _notes = classify_result_file(result_path, ep, bundle.fingerprint, run_mode)
        classifications[ep.episode_id] = cls
    return classifications


# ============================================================================
# Environment fingerprinting: same recipe/machinery as the audited base
# module (compute_environment_fingerprint, git/nvidia-smi/hash helpers
# are fully generic and reused directly), but with THIS module's own
# six frozen file names, since the base module's own
# EXPECTED_ENVIRONMENT_FILE_HASH_NAMES/RealEnvironmentProbe hardcode the
# ORIGINAL prefill_confirmation filenames.
# ============================================================================

EXPECTED_ENVIRONMENT_FILE_HASH_NAMES = frozenset({
    "run_server_waiting_confirmation.py",
    "run_server_waiting_confirmation.sh",
    "run_server_waiting_server.sh",
    "server_waiting_confirmation_schedule.json",
    "server_waiting_confirmation_schedule.csv",
    "server_waiting_confirmation_schedule_audit.txt",
    # B1 fix (2026-07-20 hardening pass): these two executed/imported
    # modules contain the core cohort/trigger algorithm and the audited
    # HTTP/lifecycle machinery this runner imports as a library --
    # editing either one must change environment_fingerprint just as
    # surely as editing the runner itself. See
    # SERVER_WAITING_CODE_AUDIT_2026-07-20.md, finding B1.
    "_active_cohort.py",
    "run_prefill_confirmation.py",
})


class RealEnvironmentProbe:
    def gather(self, schedule_dir: Path) -> dict:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_list = base._query_nvidia_smi_gpus()
        resolved_gpu = base._resolve_visible_gpu(gpu_list, cuda_visible)
        git_commit = base._safe_run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
        git_status = base._safe_run_text(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"]
        )
        tracked_git_dirty = (git_status != "") if git_status is not None else None
        file_hash_targets = {
            "run_server_waiting_confirmation.py": SCRIPT_PATH,
            "run_server_waiting_confirmation.sh": SCRIPT_PATH.parent / "run_server_waiting_confirmation.sh",
            "run_server_waiting_server.sh": SCRIPT_PATH.parent / "run_server_waiting_server.sh",
            "server_waiting_confirmation_schedule.json": schedule_dir / "server_waiting_confirmation_schedule.json",
            "server_waiting_confirmation_schedule.csv": schedule_dir / "server_waiting_confirmation_schedule.csv",
            "server_waiting_confirmation_schedule_audit.txt": schedule_dir / "server_waiting_confirmation_schedule_audit.txt",
            # B1 fix: the data-driven cohort/trigger module and the
            # imported audited base runner, both actually executed by
            # this campaign.
            "_active_cohort.py": SCRIPT_PATH.parent / "_active_cohort.py",
            "run_prefill_confirmation.py": _BASE_DIR / "run_prefill_confirmation.py",
        }
        file_hashes = {name: base._sha256_file(p) for name, p in file_hash_targets.items()}
        return {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "kernel": platform.release(),
            "git_commit": git_commit,
            "git_dirty": tracked_git_dirty,
            "cuda_visible_devices": cuda_visible,
            "vllm_version": base._safe_package_version("vllm"),
            "torch_version": base._safe_package_version("torch"),
            "transformers_version": base._safe_package_version("transformers"),
            "httpx_version": base._safe_package_version("httpx"),
            "gpu_list": gpu_list,
            "resolved_gpu": resolved_gpu,
            "file_hashes": file_hashes,
        }


class FakeEnvironmentProbe:
    def __init__(self, env: dict | None = None) -> None:
        self.env: dict = env if env is not None else {
            "python_executable": "/fake/bin/python3", "python_version": "3.12.0",
            "platform": "FakeLinux-6.0.0-x86_64", "hostname": "fake-host", "kernel": "6.0.0-fake",
            "git_commit": "f" * 40, "git_dirty": False, "cuda_visible_devices": "0",
            "vllm_version": "0.17.1", "torch_version": "2.5.0", "transformers_version": "4.45.0",
            "httpx_version": "0.27.0",
            "gpu_list": [{"index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake-0000-0000",
                          "memory_total": "24576 MiB", "driver_version": "550.00"}],
            "resolved_gpu": {"index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake-0000-0000",
                              "memory_total": "24576 MiB", "driver_version": "550.00"},
            "file_hashes": {
                name: hashlib.sha256(name.encode("utf-8")).hexdigest()
                for name in sorted(EXPECTED_ENVIRONMENT_FILE_HASH_NAMES)
            },
        }

    def gather(self, schedule_dir: Path) -> dict:
        return json.loads(json.dumps(self.env))


compute_environment_fingerprint = base.compute_environment_fingerprint


def _is_hex_of_length(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdefABCDEF" for c in value)


def validate_official_environment(env: dict) -> list[str]:
    errors: list[str] = []
    if env.get("git_dirty") is not False:
        errors.append("git_dirty must be exactly False (tracked working tree must be clean)")
    if not _is_hex_of_length(env.get("git_commit"), 40):
        errors.append("git_commit must be a 40-character hex commit hash")
    resolved_gpu = env.get("resolved_gpu")
    if not isinstance(resolved_gpu, dict):
        errors.append("resolved_gpu could not be resolved to a physical GPU (missing or not a dict)")
    else:
        for key in ("uuid", "name", "memory_total", "driver_version"):
            v = resolved_gpu.get(key)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"resolved_gpu.{key} is missing or empty")
    for key in ("python_version", "vllm_version", "torch_version", "transformers_version", "httpx_version", "kernel"):
        v = env.get(key)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{key} is missing or empty")
    file_hashes = env.get("file_hashes")
    if not isinstance(file_hashes, dict) or set(file_hashes.keys()) != EXPECTED_ENVIRONMENT_FILE_HASH_NAMES:
        errors.append(
            f"file_hashes must contain exactly the {len(EXPECTED_ENVIRONMENT_FILE_HASH_NAMES)} expected files: "
            f"{sorted(EXPECTED_ENVIRONMENT_FILE_HASH_NAMES)}"
        )
    else:
        for name, h in file_hashes.items():
            if not _is_hex_of_length(h, 64):
                errors.append(f"file_hashes[{name!r}] is not a well-formed 64-character SHA-256 hex value")
    return errors


# ============================================================================
# Official run manifest / resume / integrity manifest. build_integrity_manifest
# / verify_integrity_manifest / _iter_output_files_for_integrity /
# validate_resume_manifest / load_json_file_or_none are fully generic
# (parameterized by output_dir/bundle/env/counts, never reference the base
# module's own RESULT_SCHEMA_VERSION/RUNNER_VERSION) -- reused unchanged.
# Only build_official_run_manifest bakes in RUNNER_VERSION/
# RESULT_SCHEMA_VERSION internally, so it is reimplemented here with this
# module's own values.
# ============================================================================

MANIFEST_SCHEMA_VERSION = base.MANIFEST_SCHEMA_VERSION
OFFICIAL_RUN_MANIFEST_FILENAME = base.OFFICIAL_RUN_MANIFEST_FILENAME
OFFICIAL_RUN_SUMMARY_FILENAME = base.OFFICIAL_RUN_SUMMARY_FILENAME
INTEGRITY_MANIFEST_FILENAME = base.INTEGRITY_MANIFEST_FILENAME

load_json_file_or_none = base.load_json_file_or_none
validate_resume_manifest = base.validate_resume_manifest
# NOTE: build_integrity_manifest/verify_integrity_manifest (aliased from
# base above) are used ONLY by the still-deferred, out-of-scope official
# campaign path (run_official_campaign). Blocker A (2026-07-20 fifth
# hardening pass): the base module's own file-iteration helper excludes
# files by BASENAME (`p.name != INTEGRITY_MANIFEST_FILENAME`), which
# would silently ignore ANY same-named file anywhere in the tree, e.g.
# `nested/integrity_manifest.json` -- not just the true root manifest.
# The protected base module is not modified. Instead, run_diagnostic_pair()
# uses the LOCAL, corrected replacements below, which exclude ONLY the
# exact resolved root manifest path.
build_integrity_manifest = base.build_integrity_manifest
verify_integrity_manifest = base.verify_integrity_manifest


def _iter_diagnostic_integrity_files(output_dir: Path) -> list[Path]:
    """Blocker A fix: excludes ONLY the file whose resolved path equals
    the exact resolved root manifest path
    `(output_dir / INTEGRITY_MANIFEST_FILENAME).resolve()` -- never by
    basename alone. A same-named file anywhere else in the tree (e.g.
    `nested/integrity_manifest.json`) is treated as an ordinary file:
    hashed, listed, and flagged as unexpected if the manifest doesn't
    already know about it."""
    root_manifest = (output_dir / INTEGRITY_MANIFEST_FILENAME).resolve()
    files = [
        p for p in output_dir.rglob("*")
        if p.is_file() and p.resolve() != root_manifest
    ]
    return sorted(files, key=lambda p: p.relative_to(output_dir).as_posix())


def build_diagnostic_integrity_manifest(
    output_dir: Path, *, schedule_fingerprint: str, environment_fingerprint: str, clock,
) -> dict:
    """Blocker A fix: identical shape/semantics to
    `base.build_integrity_manifest`, but iterates disk via
    `_iter_diagnostic_integrity_files` (exact-root-path exclusion)."""
    entries: list[dict] = []
    episode_file_count = 0
    stabilization_file_count = 0
    block_summary_count = 0
    # Hash only the frozen pre-manifest whitelist. Any extra file remains
    # unlisted and is therefore rejected by verification instead of being
    # silently blessed into a newly generated manifest.
    for rel in sorted(_expected_diagnostic_relative_files(include_integrity_manifest=False)):
        p = output_dir / rel
        if not p.is_file() or p.is_symlink():
            continue
        entries.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": base._sha256_file(p)})
        if rel.startswith("episodes/"):
            episode_file_count += 1
        elif rel.startswith("stabilization/"):
            stabilization_file_count += 1
        elif rel.startswith("block_summaries/"):
            block_summary_count += 1
    return {
        "file_count": len(entries),
        "episode_file_count": episode_file_count,
        "stabilization_file_count": stabilization_file_count,
        "block_summary_count": block_summary_count,
        "generated_utc": clock.utcnow_iso(),
        "schedule_fingerprint": schedule_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "files": entries,
    }


def verify_diagnostic_integrity_manifest(
    output_dir: Path,
    manifest: object,
    *,
    expected_schedule_fingerprint: str | None = None,
    expected_environment_fingerprint: str | None = None,
    expected_episode_count: int | None = None,
    expected_stabilization_count: int | None = None,
    expected_block_summary_count: int | None = None,
    expected_server_log_count: int | None = None,
) -> tuple[bool, list[str]]:
    """Blocker A fix: identical structural/deep validation contract to
    `base.verify_integrity_manifest`, but compares against disk via
    `_iter_diagnostic_integrity_files` (exact-root-path exclusion)
    instead of the base module's by-basename exclusion. A structurally
    broken manifest is never compared against disk. No filtering of any
    reported error is ever applied by any caller of this function."""
    errors: list[str] = []

    if not isinstance(manifest, dict):
        return False, ["integrity manifest is not a JSON object"]

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("'files' is not a list")
        files = []

    seen_paths: set[str] = set()
    structurally_valid_entries: list[dict] = []
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"files[{i}] is not a JSON object")
            continue
        rel = entry.get("relative_path")
        size = entry.get("size_bytes")
        sha = entry.get("sha256")
        entry_ok = True
        if type(rel) is not str or not rel:
            errors.append(f"files[{i}].relative_path is missing or not a non-empty str")
            entry_ok = False
        if type(size) is not int or isinstance(size, bool) or size < 0:
            errors.append(f"files[{i}].size_bytes is missing or not a non-negative int")
            entry_ok = False
        if not _is_hex_of_length(sha, 64):
            errors.append(f"files[{i}].sha256 is not a well-formed 64-character SHA-256 hex value")
            entry_ok = False
        if entry_ok:
            if rel in seen_paths:
                errors.append(f"duplicate relative_path in integrity manifest: {rel}")
            seen_paths.add(rel)
            structurally_valid_entries.append(entry)

    rel_paths_in_order = [e["relative_path"] for e in files if isinstance(e, dict) and type(e.get("relative_path")) is str]
    if rel_paths_in_order != sorted(rel_paths_in_order):
        errors.append("'files' entries are not in lexicographic relative_path order")

    for key in ("file_count", "episode_file_count", "stabilization_file_count", "block_summary_count"):
        if type(manifest.get(key)) is not int or isinstance(manifest.get(key), bool):
            errors.append(f"'{key}' is missing or not an int")

    if type(manifest.get("file_count")) is int and manifest.get("file_count") != len(files):
        errors.append(f"'file_count' ({manifest.get('file_count')!r}) != len(files) ({len(files)})")

    if not isinstance(manifest.get("schedule_fingerprint"), str) or not manifest.get("schedule_fingerprint"):
        errors.append("'schedule_fingerprint' is missing or not a non-empty str")
    elif expected_schedule_fingerprint is not None and manifest.get("schedule_fingerprint") != expected_schedule_fingerprint:
        errors.append(
            f"schedule_fingerprint mismatch: manifest={manifest.get('schedule_fingerprint')!r} "
            f"expected={expected_schedule_fingerprint!r}"
        )

    if not isinstance(manifest.get("environment_fingerprint"), str) or not manifest.get("environment_fingerprint"):
        errors.append("'environment_fingerprint' is missing or not a non-empty str")
    elif expected_environment_fingerprint is not None and manifest.get("environment_fingerprint") != expected_environment_fingerprint:
        errors.append(
            f"environment_fingerprint mismatch: manifest={manifest.get('environment_fingerprint')!r} "
            f"expected={expected_environment_fingerprint!r}"
        )

    if expected_episode_count is not None and manifest.get("episode_file_count") != expected_episode_count:
        errors.append(f"episode_file_count {manifest.get('episode_file_count')!r} != expected {expected_episode_count!r}")
    if expected_stabilization_count is not None and manifest.get("stabilization_file_count") != expected_stabilization_count:
        errors.append(
            f"stabilization_file_count {manifest.get('stabilization_file_count')!r} != "
            f"expected {expected_stabilization_count!r}"
        )
    if expected_block_summary_count is not None and manifest.get("block_summary_count") != expected_block_summary_count:
        errors.append(
            f"block_summary_count {manifest.get('block_summary_count')!r} != "
            f"expected {expected_block_summary_count!r}"
        )
    if expected_server_log_count is not None:
        actual_server_log_count = sum(1 for rel in rel_paths_in_order if rel.startswith("server_logs/"))
        if actual_server_log_count != expected_server_log_count:
            errors.append(f"server_logs file count {actual_server_log_count} != expected {expected_server_log_count}")

    # A structurally broken manifest is never compared against disk --
    # there is nothing trustworthy left to compare.
    if errors:
        return False, errors

    current_files = {p.relative_to(output_dir).as_posix(): p for p in _iter_diagnostic_integrity_files(output_dir)}
    manifest_files = {e["relative_path"]: e for e in structurally_valid_entries}

    for rel in sorted(set(manifest_files) - set(current_files)):
        errors.append(f"file listed in the integrity manifest is missing on disk: {rel}")
    for rel in sorted(set(current_files) - set(manifest_files)):
        errors.append(f"file present on disk but not listed in the integrity manifest: {rel}")

    for rel, entry in manifest_files.items():
        p = current_files.get(rel)
        if p is None:
            continue
        actual_size = p.stat().st_size
        actual_hash = base._sha256_file(p)
        if actual_size != entry.get("size_bytes"):
            errors.append(f"size mismatch for {rel}: manifest={entry.get('size_bytes')!r} actual={actual_size!r}")
        if actual_hash != entry.get("sha256"):
            errors.append(f"sha256 mismatch for {rel}: manifest={entry.get('sha256')!r} actual={actual_hash!r}")

    return (not errors), errors


def _expected_diagnostic_relative_files(*, include_integrity_manifest: bool) -> set[str]:
    """Return the exact file whitelist for the frozen diagnostic pair."""
    episode_ids = {
        f"{MODEL_KEY}_off{DIAGNOSTIC_OFFLOAD_GB}_k{DIAGNOSTIC_SERVER_MAX_NUM_SEQS}_trigger16_no_burst_rep{DIAGNOSTIC_REPEAT:02d}",
        f"{MODEL_KEY}_off{DIAGNOSTIC_OFFLOAD_GB}_k{DIAGNOSTIC_SERVER_MAX_NUM_SEQS}_trigger16_{base.BURST_CONDITION}_rep{DIAGNOSTIC_REPEAT:02d}",
    }
    expected = {
        RUN_MODE_MARKER_FILENAME,
        DIAGNOSTIC_RUN_MANIFEST_FILENAME,
        "diagnostic_pair_summary.json",
        "diagnostic_pair_summary.txt",
        f"stabilization/{DIAGNOSTIC_BLOCK_ID}.json",
        f"server_logs/{DIAGNOSTIC_BLOCK_ID}.log",
        *(f"episodes/{episode_id}.json" for episode_id in episode_ids),
    }
    if include_integrity_manifest:
        expected.add(INTEGRITY_MANIFEST_FILENAME)
    return expected


def _validate_diagnostic_artifact_whitelist(
    output_dir: Path, *, include_integrity_manifest: bool,
) -> list[str]:
    """Require exactly the frozen diagnostic files and directories.

    This prevents arbitrary pre-existing files from being blessed merely
    because a freshly generated manifest happened to hash them.
    """
    if not output_dir.is_dir():
        return [f"diagnostic output directory does not exist or is not a directory: {output_dir}"]

    errors: list[str] = []
    expected_files = _expected_diagnostic_relative_files(
        include_integrity_manifest=include_integrity_manifest,
    )
    expected_dirs = {"episodes", "stabilization", "server_logs"}
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()

    for path in output_dir.rglob("*"):
        rel = path.relative_to(output_dir).as_posix()
        if path.is_symlink():
            errors.append(f"diagnostic output contains forbidden symlink: {rel}")
        elif path.is_file():
            actual_files.add(rel)
        elif path.is_dir():
            actual_dirs.add(rel)
        else:
            errors.append(f"diagnostic output contains unsupported filesystem entry: {rel}")

    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    missing_dirs = sorted(expected_dirs - actual_dirs)
    extra_dirs = sorted(actual_dirs - expected_dirs)
    if missing_files:
        errors.append(f"diagnostic output is missing expected file(s): {missing_files}")
    if extra_files:
        errors.append(f"diagnostic output contains unexpected file(s): {extra_files}")
    if missing_dirs:
        errors.append(f"diagnostic output is missing expected directories: {missing_dirs}")
    if extra_dirs:
        errors.append(f"diagnostic output contains unexpected directories: {extra_dirs}")
    return errors


def _validate_diagnostic_artifact_counts(output_dir: Path) -> list[str]:
    """Validate the complete final diagnostic tree against its whitelist."""
    return _validate_diagnostic_artifact_whitelist(
        output_dir, include_integrity_manifest=True,
    )


def build_official_run_manifest(
    *, env: dict, bundle: LoadedBundle, run_mode: str, output_dir: Path, host: str, port: int, clock,
) -> dict:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "schedule_fingerprint": bundle.fingerprint,
        "design_version": bundle.json_obj.get("design_version"),
        "schedule_seed": bundle.json_obj.get("seed"),
        "run_mode": run_mode,
        "created_utc": clock.utcnow_iso(),
        "output_dir": str(output_dir),
        "host": host,
        "port": port,
        "python_executable": env.get("python_executable"),
        "python_version": env.get("python_version"),
        "platform": env.get("platform"),
        "hostname": env.get("hostname"),
        "kernel": env.get("kernel"),
        "git_commit": env.get("git_commit"),
        "git_dirty": env.get("git_dirty"),
        "CUDA_VISIBLE_DEVICES": env.get("cuda_visible_devices"),
        "vllm_version": env.get("vllm_version"),
        "torch_version": env.get("torch_version"),
        "transformers_version": env.get("transformers_version"),
        "httpx_version": env.get("httpx_version"),
        "gpu_list": env.get("gpu_list"),
        "resolved_gpu": env.get("resolved_gpu"),
        "file_hashes": env.get("file_hashes"),
        "environment_fingerprint": compute_environment_fingerprint(env),
    }


# ============================================================================
# Block protocol / smoke / official campaign orchestration. Adapted from
# the audited base module's _run_block_protocol / run_smoke_block /
# run_official_campaign pattern (server start -> readiness -> exactly
# one stabilization run -> post-stabilization health gate -> fixed
# cooldown -> episodes in schedule order -> verified server stop), but
# calling run_server_waiting_episode / run_server_waiting_stabilization
# / build_server_command (with server_max_num_seqs) instead of the
# originals. See AUDIT.md for the documented scope of what is
# simplified relative to the original's official-campaign depth.
# ============================================================================

def find_block(bundle: LoadedBundle, block_id: str) -> list[Episode]:
    return [ep for ep in bundle.episodes if ep.block_id == block_id]


def find_and_validate_smoke_block(bundle: LoadedBundle, block_id: str) -> list[Episode]:
    episodes = find_block(bundle, block_id)
    if not episodes:
        raise ValueError(f"--smoke-block {block_id!r} does not exist in the validated schedule bundle")
    if len(episodes) != BLOCK_SIZE:
        raise ValueError(f"--smoke-block {block_id!r} does not form a complete block of size {BLOCK_SIZE}")
    return episodes


def _classify_and_plan_block(
    block_episodes: list[Episode], bundle: LoadedBundle, output_dir: Path, run_mode: str, resume: bool,
) -> tuple[dict[str, str], list[Episode]]:
    block_id = block_episodes[0].block_id
    episode_statuses: dict[str, str] = {}
    if resume:
        for ep in block_episodes:
            result_path = episode_result_path(output_dir, ep.episode_id)
            cls, _notes = classify_result_file(result_path, ep, bundle.fingerprint, run_mode)
            episode_statuses[ep.episode_id] = cls
        bad = {
            eid: cls for eid, cls in episode_statuses.items()
            if cls in (CLASSIFICATION_PARTIAL, CLASSIFICATION_INVALID, CLASSIFICATION_CORRUPTED)
        }
        if bad:
            raise base.ServerLifecycleError(
                f"--resume found non-resumable existing result file(s) for block {block_id!r}: {bad}; "
                f"refusing to silently overwrite. Fix or remove them manually first."
            )
        episodes_to_run = [
            ep for ep in block_episodes if episode_statuses[ep.episode_id] != CLASSIFICATION_VALID_COMPLETE
        ]
    else:
        for ep in block_episodes:
            if episode_result_path(output_dir, ep.episode_id).exists():
                raise base.ServerLifecycleError(
                    f"result file for episode {ep.episode_id!r} already exists and --resume was not given"
                )
        if stabilization_result_path(output_dir, block_id).exists():
            raise base.ServerLifecycleError(
                f"stabilization result for block {block_id!r} already exists and --resume was not given"
            )
        episode_statuses = {ep.episode_id: CLASSIFICATION_MISSING for ep in block_episodes}
        episodes_to_run = list(block_episodes)
    return episode_statuses, episodes_to_run


async def _run_server_waiting_block_protocol(
    *, bundle: LoadedBundle, block_episodes: list[Episode], episodes_to_run: list[Episode], block_id: str,
    output_dir: Path, host: str, port: int, run_mode: str, api_key: str, transport, metrics_transport,
    tokenizer, server_adapter, sleeper, clock, run_server_path: Path, episode_statuses: dict[str, str],
    environment_fingerprint: str | None = None,
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S, stop_kill_confirm_timeout_s: float = 5.0,
    stop_port_poll_timeout_s: float = 30.0, should_abort: Callable[[], bool] | None = None,
    metrics_poll_interval_s: float = 0.05,
) -> dict:
    """`transport` is the (persistent) completion transport used for
    RunContext and server-lifecycle HTTP calls; `metrics_transport` is a
    SEPARATE transport used only by the per-episode MetricsSampler
    (requirement 4 / N2) -- never the same object as `transport`.

    B1 fix (2026-07-20 second hardening pass): `environment_fingerprint`,
    when supplied, is stamped into `server_metadata` and therefore flows
    into every episode result and the stabilization result produced by
    this block -- binding each artifact to the runtime environment that
    produced it, not just to the schedule fingerprint.

    N1 fix (2026-07-20 hardening pass): `should_abort` is now also
    checked immediately before the stabilization run, immediately before
    the cooldown, and before each episode in the loop -- not only before
    the server starts -- so a SIGINT/SIGTERM does not have to wait for
    an entire in-progress block before being honored between its natural
    checkpoints. It still cannot interrupt a single already-dispatched
    episode's own request phase (that would require cancelling the
    episode's task tree mid-flight, which is out of scope for this
    hardening pass and remains a known, documented limitation)."""
    model_key = block_episodes[0].model_key
    offload_gb = block_episodes[0].offload_gb
    state_label = block_episodes[0].state_label
    server_max_num_seqs = block_episodes[0].server_max_num_seqs
    model_full_id = MODEL_REGISTRY[model_key]["model_id"]

    result: dict[str, Any] = {
        "server_start": None, "readiness": None, "stabilization_status": "not_run",
        "post_stabilization_health": None, "cooldown_s": None, "server_stop": None,
        "overall_status": "not_run", "executed_episode_ids": [],
    }
    handle = None
    try:
        if should_abort is not None and should_abort():
            result["overall_status"] = "interrupted"
            return result
        if not is_port_free(host, port):
            raise base.ServerLifecycleError(f"port {port} on host {host!r} is already in use")

        cmd = build_server_command(run_server_path, model_key, offload_gb, server_max_num_seqs, host, port)
        log_path = output_dir / "server_logs" / f"{block_id}.log"

        if should_abort is not None and should_abort():
            result["overall_status"] = "interrupted"
            return result

        handle = server_adapter.start(cmd, log_path)
        result["server_start"] = {
            "cmd": cmd, "pid": handle.pid, "pgid": handle.pgid, "start_utc": handle.start_utc,
        }

        base_url = f"http://{host}:{port}"
        readiness_info = await wait_for_server_ready(transport, handle, base_url, api_key, model_full_id, sleeper)
        readiness_info["capability_summary"] = await fetch_capability_summary(transport, base_url, api_key)
        result["readiness"] = readiness_info

        valid_ids = compute_valid_token_ids(tokenizer)
        ctx = RunContext(
            transport=transport, clock=clock, sleeper=sleeper, base_url=base_url,
            api_key=api_key, model_full_id=model_full_id, valid_ids=valid_ids,
        )
        get_diag = getattr(transport, "get_diagnostics", None)
        completion_pool_diagnostics = get_diag() if callable(get_diag) else {}
        server_metadata = {
            "model_key": model_key, "model_full_id": model_full_id, "offload_gb": offload_gb,
            "server_max_num_seqs": server_max_num_seqs, "host": host, "port": port,
            "server_command": cmd, "pid": handle.pid, "pgid": handle.pgid,
            "server_start_utc": handle.start_utc, "readiness": readiness_info,
            "environment_fingerprint": environment_fingerprint,
            "completion_pool_limits": {
                "max_connections": completion_pool_diagnostics.get("max_connections"),
                "max_keepalive_connections": completion_pool_diagnostics.get("max_keepalive_connections"),
            },
        }

        if should_abort is not None and should_abort():
            result["overall_status"] = "interrupted"
            return result

        stab_result = await run_server_waiting_stabilization(
            ctx, bundle, model_key, block_id, offload_gb, state_label, server_metadata=server_metadata,
        )
        stab_result["run_mode"] = run_mode
        write_json_atomic(stabilization_result_path(output_dir, block_id), stab_result)
        result["stabilization_status"] = stab_result["status"]

        if stab_result["status"] != base.REQUEST_STATUS_COMPLETE:
            result["overall_status"] = "stabilization_failed"
            return result

        post_stabilization_health = await check_post_stabilization_health(transport, base_url)
        readiness_info["post_stabilization_health"] = post_stabilization_health
        result["readiness"] = readiness_info
        result["post_stabilization_health"] = post_stabilization_health
        if not post_stabilization_health["ok"]:
            result["overall_status"] = "post_stabilization_health_failed"
            return result

        if should_abort is not None and should_abort():
            result["overall_status"] = "interrupted"
            return result

        await sleeper.sleep(COOLDOWN_S)
        result["cooldown_s"] = COOLDOWN_S

        stabilization_ref = {
            "block_id": block_id, "path": str(stabilization_result_path(output_dir, block_id)),
            "functional_passed": stab_result["functional_passed"],
        }

        block_aborted = False
        for ep in episodes_to_run:
            if should_abort is not None and should_abort():
                result["overall_status"] = "interrupted"
                return result
            metrics_sampler = MetricsSampler(
                transport=metrics_transport, base_url=base_url, sleeper=sleeper, clock=clock,
                poll_interval_s=metrics_poll_interval_s,
            )
            ep_result = await run_server_waiting_episode(
                ctx, ep, schedule_fingerprint=bundle.fingerprint, server_metadata=server_metadata,
                stabilization_ref=stabilization_ref, run_mode=run_mode, metrics_sampler=metrics_sampler,
            )
            write_json_atomic(episode_result_path(output_dir, ep.episode_id), ep_result)
            result["executed_episode_ids"].append(ep.episode_id)
            ok = ep_result["status"] == base.REQUEST_STATUS_COMPLETE
            episode_statuses[ep.episode_id] = CLASSIFICATION_VALID_COMPLETE if ok else CLASSIFICATION_PARTIAL
            if not ok:
                block_aborted = True
                break

        result["overall_status"] = "block_failed" if block_aborted else "block_complete"
        return result
    except asyncio.CancelledError:
        result["overall_status"] = "interrupted"
        return result
    except BaseException as exc:  # noqa: BLE001 -- must always return, never crash the caller
        result["overall_status"] = "error"
        result["error"] = base._redact_secret(str(exc), api_key)
        return result
    finally:
        if handle is not None:
            stop_result = await stop_server(
                handle, host, port, sleeper, timeout_s=stop_timeout_s,
                kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
                port_poll_timeout_s=stop_port_poll_timeout_s,
            )
            result["server_stop"] = stop_result
            if not stop_result.get("stop_success"):
                result["status_before_server_stop_failure"] = result.get("overall_status")
                result["overall_status"] = "server_stop_failed"


async def run_server_waiting_smoke_block(
    *, bundle: LoadedBundle, block_id: str, output_dir: Path, host: str, port: int, resume: bool,
    api_key: str, transport, metrics_transport, tokenizer, server_adapter, sleeper, clock,
    run_server_path: Path, interrupt_state: "InterruptState | None" = None,
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S,
    stop_kill_confirm_timeout_s: float = 5.0, stop_port_poll_timeout_s: float = 30.0,
    metrics_poll_interval_s: float = 0.05,
) -> dict:
    """B2 fix (2026-07-20 second hardening pass): `interrupt_state`, when
    supplied, is threaded into the block protocol's `should_abort` check
    (the same cooperative mechanism `--official-run` already used) so
    SIGINT/SIGTERM is honored between this block's natural checkpoints
    for `--smoke-test` too, not only for `--official-run`."""
    start_utc = clock.utcnow_iso()
    block_episodes = find_and_validate_smoke_block(bundle, block_id)

    if resume:
        if not output_dir.exists():
            raise base.ServerLifecycleError(
                f"--smoke-test --resume requires an existing output directory; {output_dir} does not exist"
            )
        require_output_dir_mode_marker(output_dir, RUN_MODE_SMOKE)
    else:
        if output_dir.exists():
            existing_entries = sorted(p.name for p in output_dir.iterdir())
            if existing_entries:
                raise base.ServerLifecycleError(
                    f"--smoke-test without --resume requires a new or completely empty output directory; "
                    f"found entries in {output_dir}: {existing_entries}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_mode_marker(output_dir, RUN_MODE_SMOKE)

    episode_statuses, episodes_to_run = _classify_and_plan_block(
        block_episodes, bundle, output_dir, RUN_MODE_SMOKE, resume
    )

    summary: dict[str, Any] = {
        "runner_version": RUNNER_VERSION, "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_mode": RUN_MODE_SMOKE, "schedule_fingerprint": bundle.fingerprint, "smoke_block": block_id,
        "server_start": None, "readiness": None, "stabilization_status": "not_run", "cooldown_s": None,
        "episode_statuses": episode_statuses, "server_stop": None, "overall_status": "not_run",
        "start_utc": start_utc, "end_utc": None,
    }

    if not episodes_to_run:
        summary["stabilization_status"] = "skipped (all episodes already valid_complete)"
        summary["overall_status"] = "already_complete"
        summary["end_utc"] = clock.utcnow_iso()
        return summary

    def _should_abort(state=interrupt_state) -> bool:
        return state is not None and state.event.is_set()

    protocol_result = await _run_server_waiting_block_protocol(
        bundle=bundle, block_episodes=block_episodes, episodes_to_run=episodes_to_run, block_id=block_id,
        output_dir=output_dir, host=host, port=port, run_mode=RUN_MODE_SMOKE, api_key=api_key,
        transport=transport, metrics_transport=metrics_transport, tokenizer=tokenizer,
        server_adapter=server_adapter, sleeper=sleeper, should_abort=_should_abort,
        clock=clock, run_server_path=run_server_path, episode_statuses=episode_statuses,
        stop_timeout_s=stop_timeout_s, stop_kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
        stop_port_poll_timeout_s=stop_port_poll_timeout_s, metrics_poll_interval_s=metrics_poll_interval_s,
    )

    summary["server_start"] = protocol_result["server_start"]
    summary["readiness"] = protocol_result["readiness"]
    summary["stabilization_status"] = protocol_result["stabilization_status"]
    summary["cooldown_s"] = protocol_result["cooldown_s"]
    summary["episode_statuses"] = dict(episode_statuses)
    summary["server_stop"] = protocol_result["server_stop"]
    summary["overall_status"] = protocol_result["overall_status"]
    if "error" in protocol_result:
        summary["error"] = protocol_result["error"]
    summary["end_utc"] = clock.utcnow_iso()
    write_json_atomic(output_dir / "smoke_run_summary.json", summary)
    return summary


def run_dry_run(schedule_dir: Path, model_key: str = MODEL_KEY) -> int:
    """Loads and prints the execution plan only. Starts no process,
    opens no network connection, loads no tokenizer, writes no result
    file -- verified by test_run_server_waiting_confirmation.py."""
    bundle, errors = load_and_validate_bundle(schedule_dir, model_key)
    if bundle is None:
        sys.stderr.write("Schedule bundle FAILED validation:\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        return 1
    plan = build_execution_plan(bundle)
    print_execution_plan(bundle, plan)
    return 0


# ============================================================================
# Official campaign orchestrator. Scoped down relative to the audited
# base module's ~500-line version: full multi-block resume, integrity
# manifest, and environment-fingerprint gating are implemented; the
# elaborate OS-signal edge-case matrix is simplified to a single
# cooperative InterruptState checked between blocks and immediately
# before each server start (see AUDIT.md). NEVER invoked automatically
# by this assistant session -- see main()'s explicit refusal below.
# ============================================================================

@dataclass
class InterruptState:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    signal_name: str | None = None

    def trigger(self, signal_name: str) -> None:
        self.signal_name = signal_name
        self.event.set()


def all_block_ids_in_schedule_order(bundle: LoadedBundle) -> list[str]:
    seen: dict[str, None] = {}
    for ep in bundle.episodes:
        seen.setdefault(ep.block_id, None)
    return list(seen.keys())


async def run_official_campaign(
    *, bundle: LoadedBundle, output_dir: Path, host: str, port: int, resume: bool, api_key: str,
    transport, metrics_transport, tokenizer, server_adapter, sleeper, clock, run_server_path: Path, env: dict,
    interrupt_state: InterruptState | None = None, stop_timeout_s: float = SERVER_STOP_TIMEOUT_S,
    stop_kill_confirm_timeout_s: float = 5.0, stop_port_poll_timeout_s: float = 30.0,
    metrics_poll_interval_s: float = 0.05,
) -> dict:
    start_utc = clock.utcnow_iso()
    run_mode = RUN_MODE_OFFICIAL
    environment_fingerprint = compute_environment_fingerprint(env)

    if resume:
        if not output_dir.exists():
            raise base.ServerLifecycleError(
                f"--official-run --resume requires an existing output directory; {output_dir} does not exist"
            )
        require_output_dir_mode_marker(output_dir, run_mode)
        existing_manifest = load_json_file_or_none(output_dir / OFFICIAL_RUN_MANIFEST_FILENAME)
        if existing_manifest is None:
            raise base.ServerLifecycleError("--resume requires an existing official_run_manifest.json")
        manifest_errors = validate_resume_manifest(existing_manifest, bundle=bundle, run_mode=run_mode, current_env=env)
        if manifest_errors:
            raise base.ServerLifecycleError(f"--resume manifest validation failed: {manifest_errors}")
    else:
        if output_dir.exists():
            existing_entries = sorted(p.name for p in output_dir.iterdir())
            if existing_entries:
                raise base.ServerLifecycleError(
                    f"--official-run without --resume requires a new or completely empty output "
                    f"directory; found entries in {output_dir}: {existing_entries}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_mode_marker(output_dir, run_mode)
        manifest = build_official_run_manifest(
            env=env, bundle=bundle, run_mode=run_mode, output_dir=output_dir, host=host, port=port, clock=clock,
        )
        write_json_atomic(output_dir / OFFICIAL_RUN_MANIFEST_FILENAME, manifest)

    block_ids = all_block_ids_in_schedule_order(bundle)
    block_statuses: dict[str, str] = {}
    completed_blocks = 0
    skipped_blocks = 0
    failed_block: str | None = None
    interrupted_by: str | None = None
    overall_status = "in_progress"

    for block_id in block_ids:
        if interrupt_state is not None and interrupt_state.event.is_set():
            interrupted_by = interrupt_state.signal_name
            overall_status = "interrupted"
            break

        block_episodes = find_block(bundle, block_id)
        episode_statuses, episodes_to_run = _classify_and_plan_block(
            block_episodes, bundle, output_dir, run_mode, resume
        )

        if not episodes_to_run:
            skipped_blocks += 1
            completed_blocks += 1
            block_statuses[block_id] = "already_complete"
            continue

        def _should_abort(state=interrupt_state) -> bool:
            return state is not None and state.event.is_set()

        protocol_result = await _run_server_waiting_block_protocol(
            bundle=bundle, block_episodes=block_episodes, episodes_to_run=episodes_to_run, block_id=block_id,
            output_dir=output_dir, host=host, port=port, run_mode=run_mode, api_key=api_key,
            transport=transport, metrics_transport=metrics_transport, tokenizer=tokenizer,
            server_adapter=server_adapter, sleeper=sleeper,
            clock=clock, run_server_path=run_server_path, episode_statuses=episode_statuses,
            stop_timeout_s=stop_timeout_s, stop_kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
            stop_port_poll_timeout_s=stop_port_poll_timeout_s, should_abort=_should_abort,
            metrics_poll_interval_s=metrics_poll_interval_s,
        )

        block_statuses[block_id] = protocol_result["overall_status"]
        block_summary = {
            "runner_version": RUNNER_VERSION, "result_schema_version": RESULT_SCHEMA_VERSION,
            "run_mode": run_mode, "schedule_fingerprint": bundle.fingerprint, "block_id": block_id,
            **protocol_result, "episode_statuses": dict(episode_statuses),
        }
        (output_dir / "block_summaries").mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "block_summaries" / f"{block_id}.json", block_summary)

        if protocol_result["overall_status"] == "block_complete":
            completed_blocks += 1
        elif protocol_result["overall_status"] == "interrupted":
            interrupted_by = interrupt_state.signal_name if interrupt_state is not None else None
            overall_status = "interrupted"
            break
        else:
            failed_block = block_id
            overall_status = "failed"
            break
    else:
        overall_status = "complete" if completed_blocks == len(block_ids) else "incomplete"

    pending_blocks = len(block_ids) - len(block_statuses)
    valid_complete_episodes = sum(
        1 for ep in bundle.episodes
        if classify_result_file(
            episode_result_path(output_dir, ep.episode_id), ep, bundle.fingerprint, run_mode
        )[0] == CLASSIFICATION_VALID_COMPLETE
    )
    missing_episodes = len(bundle.episodes) - valid_complete_episodes

    integrity_manifest = build_integrity_manifest(
        output_dir, schedule_fingerprint=bundle.fingerprint,
        environment_fingerprint=environment_fingerprint, clock=clock,
    )
    write_json_atomic(output_dir / INTEGRITY_MANIFEST_FILENAME, integrity_manifest)

    summary = {
        "runner_version": RUNNER_VERSION, "result_schema_version": RESULT_SCHEMA_VERSION, "run_mode": run_mode,
        "schedule_fingerprint": bundle.fingerprint, "environment_fingerprint": environment_fingerprint,
        "start_utc": start_utc, "end_utc": clock.utcnow_iso(), "overall_status": overall_status,
        "planned_blocks": len(block_ids), "completed_blocks": completed_blocks, "skipped_blocks": skipped_blocks,
        "pending_blocks": pending_blocks, "planned_episodes": len(bundle.episodes),
        "valid_complete_episodes": valid_complete_episodes, "missing_episodes": missing_episodes,
        "failed_block": failed_block, "interrupted_by": interrupted_by, "block_statuses": block_statuses,
    }
    write_json_atomic(output_dir / OFFICIAL_RUN_SUMMARY_FILENAME, summary)
    return summary


# ============================================================================
# --diagnostic-pair-only mode (2026-07-20 hardening pass, requirements 8,
# 9, 10). Hardens the existing implementation only enough to run ONE
# fresh diagnostic pair -- qwen_off12_k8_rep01 -- fresh-directory-only,
# never resumable, never touching any other block, never calling
# run_official_campaign. The scientific result (A/B/C classification) is
# left genuinely open; nothing here assumes overlap, waiting,
# protection, or a scheduler mechanism.
# ============================================================================

RUN_MODE_DIAGNOSTIC_PAIR = "diagnostic_pair"

DIAGNOSTIC_BLOCK_ID = "qwen_off12_k8_rep01"
DIAGNOSTIC_MODEL_KEY = "qwen"
DIAGNOSTIC_OFFLOAD_GB = 12
DIAGNOSTIC_SERVER_MAX_NUM_SEQS = 8
DIAGNOSTIC_REPEAT = 1

DIAGNOSTIC_CLASSIFICATION_A = "A_OUTPUT_LEVEL_OVERLAP"
DIAGNOSTIC_CLASSIFICATION_B = "B_NO_OUTPUT_LEVEL_OVERLAP_WITH_FIRST_COHORT"
DIAGNOSTIC_CLASSIFICATION_C = "C_BURST_OUTPUT_AFTER_ALL_VICTIMS"
DIAGNOSTIC_CLASSIFICATION_D = "D_AMBIGUOUS_OR_INVALID"

_DIAGNOSTIC_INTERPRETATION_NOTES = (
    "A is sufficient evidence of output-level overlap only.",
    "B does not exclude an earlier internal prefill start.",
    "C is strong configuration-specific evidence that burst output was deferred behind the "
    "pre-existing victim set.",
    "Client timestamps do not directly expose scheduler admission or internal prefill-start time.",
    "An active-victim ITL stall is an effect signal, not by itself admission proof.",
)


def validate_diagnostic_block_selection(bundle: LoadedBundle) -> tuple[list[Episode] | None, list[str]]:
    """Requirement 8: automatically select and REVALIDATE exactly
    qwen_off12_k8_rep01 -- exactly one no_burst and one prefill_burst
    episode, Qwen, offload 12GB, server_max_num_seqs=8, repeat 1.
    Returns (episodes_in_schedule_defined_order, errors); episodes is
    None if selection/validation failed."""
    errors: list[str] = []
    episodes = find_block(bundle, DIAGNOSTIC_BLOCK_ID)
    if len(episodes) != 2:
        errors.append(f"block {DIAGNOSTIC_BLOCK_ID!r} does not have exactly 2 episodes (found {len(episodes)})")
        return None, errors

    conditions = sorted(e.condition for e in episodes)
    if conditions != ["no_burst", base.BURST_CONDITION]:
        errors.append(f"block {DIAGNOSTIC_BLOCK_ID!r} conditions {conditions} != ['no_burst', 'prefill_burst']")
    for ep in episodes:
        if ep.model_key != DIAGNOSTIC_MODEL_KEY:
            errors.append(f"episode {ep.episode_id}: model_key {ep.model_key!r} != {DIAGNOSTIC_MODEL_KEY!r}")
        if ep.offload_gb != DIAGNOSTIC_OFFLOAD_GB:
            errors.append(f"episode {ep.episode_id}: offload_gb {ep.offload_gb!r} != {DIAGNOSTIC_OFFLOAD_GB!r}")
        if ep.server_max_num_seqs != DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
            errors.append(
                f"episode {ep.episode_id}: server_max_num_seqs {ep.server_max_num_seqs!r} != "
                f"{DIAGNOSTIC_SERVER_MAX_NUM_SEQS!r}"
            )
        if ep.repeat != DIAGNOSTIC_REPEAT:
            errors.append(f"episode {ep.episode_id}: repeat {ep.repeat!r} != {DIAGNOSTIC_REPEAT!r}")
    orders = sorted(e.order_in_block for e in episodes)
    if orders != [1, 2]:
        errors.append(f"block {DIAGNOSTIC_BLOCK_ID!r} order_in_block values {orders} != [1, 2]")

    if errors:
        return None, errors

    # Preserve the schedule-defined condition order (order_in_block),
    # never a hardcoded no_burst-first/prefill_burst-first assumption.
    ordered = sorted(episodes, key=lambda e: e.order_in_block)
    return ordered, []


DIAGNOSTIC_RUN_MANIFEST_FILENAME = "diagnostic_run_manifest.json"


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Recursive JSON equality with exact scalar types.

    Python's normal equality accepts ``12.0 == 12`` and ``True == 1``;
    provenance validation must not.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def _validate_diagnostic_run_manifest_artifact(actual: object, expected: dict) -> list[str]:
    if not isinstance(actual, dict):
        return ["diagnostic_run_manifest.json is missing, unreadable, or not a JSON object"]
    if not _strict_json_equal(actual, expected):
        return [
            "diagnostic_run_manifest.json does not exactly match the in-memory manifest "
            "captured before server start"
        ]
    return []


def _validate_diagnostic_server_metadata(
    *, label: str, metadata: object, expected_episode: Episode,
    expected_environment_fingerprint: str, expected_server_command: Sequence[str],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(metadata, dict):
        return [f"{label}: server_metadata is missing or not a dict"]
    expected_values = {
        "environment_fingerprint": expected_environment_fingerprint,
        "model_key": expected_episode.model_key,
        "model_full_id": expected_episode.model_id,
        "offload_gb": expected_episode.offload_gb,
        "server_max_num_seqs": expected_episode.server_max_num_seqs,
    }
    for key, expected in expected_values.items():
        actual = metadata.get(key)
        if type(actual) is not type(expected) or actual != expected:
            reasons.append(f"{label}: server_metadata.{key}={actual!r} != strict expected {expected!r}")
    expected_command = list(expected_server_command)
    command = metadata.get("server_command")
    if not isinstance(command, list) or any(type(x) is not str for x in command) or command != expected_command:
        reasons.append(f"{label}: server_metadata.server_command does not exactly match the expected command")
    if len(expected_command) >= 7:
        expected_host = expected_command[-2]
        try:
            expected_port = int(expected_command[-1])
        except (TypeError, ValueError):
            expected_port = None
        if type(metadata.get("host")) is not str or metadata.get("host") != expected_host:
            reasons.append(f"{label}: server_metadata.host does not match {expected_host!r}")
        if type(metadata.get("port")) is not int or metadata.get("port") != expected_port:
            reasons.append(f"{label}: server_metadata.port does not match strict int {expected_port!r}")
    expected_pool = {
        "max_connections": COMPLETION_POOL_MAX_CONNECTIONS,
        "max_keepalive_connections": COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS,
    }
    pool = metadata.get("completion_pool_limits")
    if not isinstance(pool, dict) or not _strict_json_equal(pool, expected_pool):
        reasons.append(f"{label}: server_metadata.completion_pool_limits != {expected_pool!r}")
    return reasons


def _validate_diagnostic_stabilization_artifact(
    *, obj: object, bundle: LoadedBundle, expected_episode: Episode,
    expected_environment_fingerprint: str, expected_server_command: Sequence[str],
) -> list[str]:
    label = "stabilization artifact"
    if not isinstance(obj, dict):
        return [f"{label} is missing, unreadable, or not a JSON object"]
    reasons: list[str] = []
    expected_top = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "record_type": RECORD_TYPE_STABILIZATION,
        "run_mode": RUN_MODE_DIAGNOSTIC_PAIR,
        "schedule_fingerprint": bundle.fingerprint,
        "block_id": DIAGNOSTIC_BLOCK_ID,
        "model": expected_episode.model_key,
        "offload_gb": expected_episode.offload_gb,
        "state_label": expected_episode.state_label,
        "excluded_from_analysis": True,
        "counted_repeat": False,
        "status": base.REQUEST_STATUS_COMPLETE,
        "functional_passed": True,
        "stabilization_passed": True,
    }
    for key, expected in expected_top.items():
        actual = obj.get(key)
        if type(actual) is not type(expected) or actual != expected:
            reasons.append(f"{label}: {key}={actual!r} != strict expected {expected!r}")

    expected_cfg = {
        "condition": base.STABILIZATION_CONDITION,
        "concurrency": base.STABILIZATION_CONCURRENCY,
        "request_count": base.STABILIZATION_REQUEST_COUNT,
        "input_len": base.STABILIZATION_INPUT_LEN,
        "output_len": base.STABILIZATION_OUTPUT_LEN,
        "temperature": base.STABILIZATION_TEMPERATURE,
    }
    if not _strict_json_equal(obj.get("stabilization_configuration"), expected_cfg):
        reasons.append(f"{label}: stabilization_configuration does not exactly match the frozen configuration")

    reasons.extend(_validate_diagnostic_server_metadata(
        label=label, metadata=obj.get("server_metadata"), expected_episode=expected_episode,
        expected_environment_fingerprint=expected_environment_fingerprint,
        expected_server_command=expected_server_command,
    ))

    request_results = obj.get("request_results")
    if not isinstance(request_results, list) or len(request_results) != base.STABILIZATION_REQUEST_COUNT:
        reasons.append(
            f"{label}: request_results is not a list of exactly {base.STABILIZATION_REQUEST_COUNT} records"
        )
    else:
        bundle_seed = bundle.json_obj.get("seed")
        for index, record in enumerate(request_results):
            for error in base._validate_request_record_fields(
                record,
                episode_id=DIAGNOSTIC_BLOCK_ID,
                role="stabilization",
                request_index=index,
                expected_prompt_seed=base.stabilization_prompt_seed(
                    bundle_seed, expected_episode.model_key, DIAGNOSTIC_BLOCK_ID, index,
                ),
                expected_generation_seed=base.stabilization_generation_seed(
                    bundle_seed, expected_episode.model_key, DIAGNOSTIC_BLOCK_ID, index,
                ),
                expected_prompt_tokens=base.STABILIZATION_INPUT_LEN,
                expected_completion_tokens=base.STABILIZATION_OUTPUT_LEN,
            ):
                reasons.append(f"{label}: {error}")
    return reasons


def _validate_exact_stabilization_references(
    *, results: Sequence[tuple[str, dict | None]], expected_path: Path,
) -> list[str]:
    reasons: list[str] = []
    expected_resolved = expected_path.resolve()
    for label, result in results:
        if not isinstance(result, dict):
            continue
        reference = result.get("stabilization_reference")
        path_value = reference.get("path") if isinstance(reference, dict) else None
        if type(path_value) is not str:
            reasons.append(f"{label}: stabilization_reference.path is missing or not a string")
            continue
        try:
            actual_resolved = Path(path_value).resolve()
        except OSError as exc:
            reasons.append(f"{label}: stabilization_reference.path cannot be resolved: {exc}")
            continue
        if actual_resolved != expected_resolved:
            reasons.append(
                f"{label}: stabilization_reference.path resolves to {actual_resolved}, "
                f"expected {expected_resolved}"
            )
    return reasons


async def run_diagnostic_pair(
    *, bundle: LoadedBundle, output_dir: Path, host: str, port: int, api_key: str, transport, metrics_transport,
    tokenizer, server_adapter, sleeper, clock, run_server_path: Path, env: dict,
    interrupt_state: "InterruptState | None" = None,
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S, stop_kill_confirm_timeout_s: float = 5.0,
    stop_port_poll_timeout_s: float = 30.0, metrics_poll_interval_s: float = 0.05,
) -> dict:
    """Requirement 8. `output_dir` MUST be explicitly supplied by the
    caller (no silent default), MUST be new or completely empty, and
    this function never accepts or implements a --resume path. It
    starts the server exactly once, runs both paired episodes
    (schedule-defined order) on that same server instance, runs exactly
    one stabilization (via the shared block protocol, identical to
    --smoke-test's lifecycle), and shuts the server down cleanly
    afterward -- reusing `_run_server_waiting_block_protocol` unchanged
    rather than re-implementing server lifecycle. Never iterates over or
    starts any other block; never calls run_official_campaign.

    B1 fix (2026-07-20 second hardening pass): `env` (the gathered,
    already-validated runtime environment) is required. Before the
    server starts, a `diagnostic_run_manifest.json` is written containing
    the full environment snapshot (`file_hashes` included) and
    `environment_fingerprint`. `environment_fingerprint` is also threaded
    into `server_metadata` (and therefore into every episode result and
    the stabilization result).

    Blocker 1 fix (2026-07-20 fourth hardening pass): finalization order
    is now: (1) generate every episode/stabilization/provenance
    artifact; (2) write `diagnostic_pair_summary.json`; (3) write
    `diagnostic_pair_summary.txt`; (4) ONLY THEN build the final
    integrity manifest over the *complete* result set (which now
    includes both summary files); (5) immediately verify it against the
    finished output tree; (6) any verification error makes the run
    fail-closed. The on-disk summary therefore never makes a
    self-referential "integrity_verified" claim (that would be circular
    -- the file cannot attest to its own hash before it exists); instead
    it carries non-circular metadata (`integrity_manifest_filename`,
    `integrity_scope`, `integrity_finalization_required`). The actual
    post-hoc verification outcome (`integrity_verified`, `integrity_errors`,
    and the corrected `overall_status`/`diagnostic_valid`) is attached
    only to this function's RETURNED dict (what the CLI prints and uses
    for its exit code) -- the on-disk JSON/TXT files are never rewritten
    after the integrity check, so what the manifest hashed is exactly
    what remains on disk.

    Blocker 2 fix (2026-07-20 fourth hardening pass): `classify_diagnostic_pair()`
    is now called with the two expected `Episode` objects (from this
    exact validated block selection) and the expected schedule/
    environment fingerprints, so it can independently bind each episode
    result to precisely `qwen_off12_k8_rep01` rather than merely
    checking internal self-consistency of whatever was handed to it."""
    start_utc = clock.utcnow_iso()
    environment_fingerprint = compute_environment_fingerprint(env)

    block_episodes, selection_errors = validate_diagnostic_block_selection(bundle)
    if block_episodes is None:
        raise base.ServerLifecycleError(
            f"--diagnostic-pair-only block selection/validation failed: {selection_errors}"
        )
    expected_no_burst_episode = next(e for e in block_episodes if e.condition == "no_burst")
    expected_prefill_burst_episode = next(e for e in block_episodes if e.condition == base.BURST_CONDITION)

    if output_dir.exists():
        existing_entries = sorted(p.name for p in output_dir.iterdir())
        if existing_entries:
            raise base.ServerLifecycleError(
                f"--diagnostic-pair-only requires a new or completely empty output directory; "
                f"found entries in {output_dir}: {existing_entries}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_mode_marker(output_dir, RUN_MODE_DIAGNOSTIC_PAIR)

    for ep in block_episodes:
        if episode_result_path(output_dir, ep.episode_id).exists():
            raise base.ServerLifecycleError(
                f"result file for episode {ep.episode_id!r} already exists in a supposedly empty "
                f"output directory -- refusing to proceed"
            )

    # B1 fix: write the diagnostic run manifest -- environment snapshot,
    # file_hashes, environment_fingerprint -- BEFORE the server starts,
    # so provenance is captured even if the run later fails.
    diagnostic_manifest = build_official_run_manifest(
        env=env, bundle=bundle, run_mode=RUN_MODE_DIAGNOSTIC_PAIR, output_dir=output_dir,
        host=host, port=port, clock=clock,
    )
    diagnostic_manifest["diagnostic_block_id"] = DIAGNOSTIC_BLOCK_ID
    write_json_atomic(output_dir / DIAGNOSTIC_RUN_MANIFEST_FILENAME, diagnostic_manifest)

    episode_statuses = {ep.episode_id: CLASSIFICATION_MISSING for ep in block_episodes}
    episodes_to_run = list(block_episodes)  # always exactly these 2, in schedule order, never resumed

    def _should_abort(state=interrupt_state) -> bool:
        return state is not None and state.event.is_set()

    protocol_result = await _run_server_waiting_block_protocol(
        bundle=bundle, block_episodes=block_episodes, episodes_to_run=episodes_to_run,
        block_id=DIAGNOSTIC_BLOCK_ID, output_dir=output_dir, host=host, port=port,
        run_mode=RUN_MODE_DIAGNOSTIC_PAIR, api_key=api_key, transport=transport,
        metrics_transport=metrics_transport, tokenizer=tokenizer, server_adapter=server_adapter,
        sleeper=sleeper, clock=clock, run_server_path=run_server_path, episode_statuses=episode_statuses,
        environment_fingerprint=environment_fingerprint, should_abort=_should_abort,
        stop_timeout_s=stop_timeout_s, stop_kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
        stop_port_poll_timeout_s=stop_port_poll_timeout_s, metrics_poll_interval_s=metrics_poll_interval_s,
    )

    no_burst_result: dict | None = None
    prefill_burst_result: dict | None = None
    for ep in block_episodes:
        obj = load_json_file_or_none(episode_result_path(output_dir, ep.episode_id))
        if ep.condition == "no_burst":
            no_burst_result = obj
        else:
            prefill_burst_result = obj

    expected_server_command = build_server_command(
        run_server_path, DIAGNOSTIC_MODEL_KEY, DIAGNOSTIC_OFFLOAD_GB,
        DIAGNOSTIC_SERVER_MAX_NUM_SEQS, host, port,
    )

    # Blocker 2 fix: bind classification to the two expected Episode
    # objects and the expected schedule/environment fingerprints -- never
    # merely to whatever happens to be in the result files.
    classification = classify_diagnostic_pair(
        no_burst_result=no_burst_result, prefill_burst_result=prefill_burst_result,
        expected_no_burst_episode=expected_no_burst_episode,
        expected_prefill_burst_episode=expected_prefill_burst_episode,
        expected_schedule_fingerprint=bundle.fingerprint,
        expected_environment_fingerprint=environment_fingerprint,
        expected_server_command=expected_server_command,
    )

    # Integrity hashes are not semantic validation.  Reload and bind the
    # actual provenance/stabilization artifacts before the summary and final
    # manifest can attest to a scientifically valid pair.
    semantic_artifact_errors: list[str] = []
    actual_manifest = load_json_file_or_none(output_dir / DIAGNOSTIC_RUN_MANIFEST_FILENAME)
    semantic_artifact_errors.extend(
        _validate_diagnostic_run_manifest_artifact(actual_manifest, diagnostic_manifest)
    )
    expected_stabilization_path = stabilization_result_path(output_dir, DIAGNOSTIC_BLOCK_ID)
    actual_stabilization = load_json_file_or_none(expected_stabilization_path)
    semantic_artifact_errors.extend(_validate_diagnostic_stabilization_artifact(
        obj=actual_stabilization, bundle=bundle, expected_episode=expected_no_burst_episode,
        expected_environment_fingerprint=environment_fingerprint,
        expected_server_command=expected_server_command,
    ))
    semantic_artifact_errors.extend(_validate_exact_stabilization_references(
        results=(
            ("no_burst episode", no_burst_result),
            ("prefill_burst episode", prefill_burst_result),
        ),
        expected_path=expected_stabilization_path,
    ))
    if semantic_artifact_errors:
        existing_reasons = list(classification.get("reasons") or [])
        classification = {
            "classification": DIAGNOSTIC_CLASSIFICATION_D,
            "reasons": existing_reasons + semantic_artifact_errors,
        }
    if classification.get("classification") == DIAGNOSTIC_CLASSIFICATION_D:
        effect_summary = {
            "available": False,
            "reason": "diagnostic pair classified D_AMBIGUOUS_OR_INVALID",
            "classification_reasons": list(classification.get("reasons") or []),
            "no_ci_note": "No effect estimate is computed from an invalid diagnostic pair.",
        }
    else:
        effect_summary = compute_paired_effect_summary(
            no_burst_result=no_burst_result, prefill_burst_result=prefill_burst_result,
        )

    # Blocker 1 fix: the ON-DISK summary carries the block-lifecycle
    # outcome and non-circular integrity METADATA (never a
    # self-referential pass/fail claim -- that file cannot attest to its
    # own hash before the integrity manifest, built AFTER it, exists).
    disk_summary: dict[str, Any] = {
        "runner_version": RUNNER_VERSION, "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_mode": RUN_MODE_DIAGNOSTIC_PAIR, "schedule_fingerprint": bundle.fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "diagnostic_block_id": DIAGNOSTIC_BLOCK_ID,
        "start_utc": start_utc, "end_utc": clock.utcnow_iso(),
        "server_start": protocol_result["server_start"], "readiness": protocol_result["readiness"],
        "stabilization_status": protocol_result["stabilization_status"], "cooldown_s": protocol_result["cooldown_s"],
        "episode_statuses": dict(episode_statuses), "server_stop": protocol_result["server_stop"],
        "overall_status": protocol_result["overall_status"],
        "classification": classification,
        "paired_effect_summary": effect_summary,
        "integrity_manifest_filename": INTEGRITY_MANIFEST_FILENAME,
        "integrity_scope": (
            "all files under this diagnostic output directory, except the integrity manifest file "
            "itself, as captured by a manifest built and verified AFTER this summary file was written"
        ),
        "integrity_finalization_required": True,
    }
    if "error" in protocol_result:
        disk_summary["error"] = protocol_result["error"]

    write_json_atomic(output_dir / "diagnostic_pair_summary.json", disk_summary)
    (output_dir / "diagnostic_pair_summary.txt").write_text(
        render_diagnostic_text_summary(disk_summary), encoding="utf-8",
    )

    # Exact pre-manifest whitelist gate. Unexpected files must not be
    # accepted merely because a new manifest happens to hash them.
    pre_manifest_artifact_errors = _validate_diagnostic_artifact_whitelist(
        output_dir, include_integrity_manifest=False,
    )

    # Blocker 1 fix: build and verify the final integrity manifest ONLY
    # NOW -- strictly after both summary files exist on disk -- so it
    # actually covers the complete, final result set, not a mid-way
    # snapshot. A verification failure fails the run closed.
    #
    # Blocker A fix (2026-07-20 fifth hardening pass): uses the LOCAL
    # build_diagnostic_integrity_manifest/verify_diagnostic_integrity_manifest
    # (exact-root-path exclusion, not by-basename) and passes the exact
    # expected artifact counts for this frozen diagnostic experiment (2
    # episodes, 1 stabilization file, 0 block summaries, 1 server log).
    integrity_manifest = build_diagnostic_integrity_manifest(
        output_dir, schedule_fingerprint=bundle.fingerprint,
        environment_fingerprint=environment_fingerprint, clock=clock,
    )
    write_json_atomic(output_dir / INTEGRITY_MANIFEST_FILENAME, integrity_manifest)
    integrity_verified, integrity_errors = verify_diagnostic_integrity_manifest(
        output_dir, integrity_manifest, expected_schedule_fingerprint=bundle.fingerprint,
        expected_environment_fingerprint=environment_fingerprint,
        expected_episode_count=2, expected_stabilization_count=1,
        expected_block_summary_count=0, expected_server_log_count=1,
    )
    # Blocker A: additionally enforce the full expected singleton-artifact
    # structure (diagnostic_run_manifest.json, both summary files, the
    # run-mode marker, and the root integrity manifest itself, each
    # exactly once) -- never filtered out of the reported errors.
    artifact_count_errors = _validate_diagnostic_artifact_counts(output_dir)
    combined_artifact_errors = list(pre_manifest_artifact_errors) + list(artifact_count_errors)
    if combined_artifact_errors:
        integrity_verified = False
        integrity_errors = list(integrity_errors) + combined_artifact_errors

    overall_status = protocol_result["overall_status"]
    if overall_status == "block_complete" and not integrity_verified:
        overall_status = "integrity_verification_failed"

    # N4 fix: an explicit, separate scientific-validity flag -- a
    # technically-complete block whose classification is
    # D_AMBIGUOUS_OR_INVALID (or whose integrity failed to verify) must
    # never be indistinguishable from a genuinely valid A/B/C result to
    # any automation reading only `overall_status`.
    diagnostic_valid = (
        overall_status == "block_complete" and integrity_verified
        and classification.get("classification") != DIAGNOSTIC_CLASSIFICATION_D
    )

    # This RETURNED dict (never written back to disk_summary's own file)
    # augments the exact on-disk content with the post-hoc integrity
    # outcome, so the CLI's printed JSON / exit code reflect the true
    # final state without ever rewriting an already-hashed file.
    returned_summary = dict(disk_summary)
    returned_summary["overall_status"] = overall_status
    returned_summary["integrity_verified"] = integrity_verified
    returned_summary["integrity_errors"] = integrity_errors
    returned_summary["diagnostic_valid"] = diagnostic_valid
    return returned_summary


def _is_strict_finite_number(value: object) -> bool:
    """True only for finite JSON-number scalars, excluding bool."""
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_diagnostic_request_functional_fields(
    *, label: str, role: str, request_index: int, record: dict,
    expected_episode: Episode, expected_prompt_tokens: int, expected_completion_tokens: int,
) -> list[str]:
    """Fail-closed functional completeness checks for one stored request.

    This deliberately validates the fields that make ``status='complete'``
    meaningful instead of trusting the status string alone.  It never
    raises on malformed JSON values and returns every independently
    detectable reason.
    """
    prefix = f"{label} episode: {role} {request_index}"
    reasons: list[str] = []

    # Bind the stored request to the exact deterministic request that the
    # frozen schedule and runner were supposed to create.  This is stronger
    # than merely checking output length/usage: the role, request ID, seed
    # derivation and prompt echo/hash must all agree.
    expected_request_id = f"{expected_episode.episode_id}:{role}:{request_index}"
    stored_request_id = record.get("request_id")
    if type(stored_request_id) is not str or stored_request_id != expected_request_id:
        reasons.append(
            f"{prefix} request_id={stored_request_id!r} != expected {expected_request_id!r}"
        )

    stored_role = record.get("role")
    if type(stored_role) is not str or stored_role != role:
        reasons.append(f"{prefix} role={stored_role!r} != expected {role!r}")

    if role == "victim":
        expected_prompt_seed = base.victim_prompt_seed(expected_episode, request_index)
        expected_generation_seed = base.victim_generation_seed(expected_episode, request_index)
    elif role == "burst":
        expected_prompt_seed = base.burst_prompt_seed(expected_episode, request_index)
        expected_generation_seed = base.burst_generation_seed(expected_episode, request_index)
    else:
        expected_prompt_seed = None
        expected_generation_seed = None
        reasons.append(f"{prefix} unknown request role {role!r}")

    stored_prompt_seed = record.get("prompt_seed")
    if type(stored_prompt_seed) is not int or stored_prompt_seed != expected_prompt_seed:
        reasons.append(f"{prefix} prompt_seed does not match the deterministic derivation")
    stored_generation_seed = record.get("generation_seed")
    if type(stored_generation_seed) is not int or stored_generation_seed != expected_generation_seed:
        reasons.append(f"{prefix} generation_seed does not match the deterministic derivation")

    sent_prompt_ids = record.get("prompt_token_ids_sent")
    returned_prompt_ids = record.get("prompt_token_ids_returned")
    sent_valid = (
        isinstance(sent_prompt_ids, list)
        and len(sent_prompt_ids) == expected_prompt_tokens
        and all(type(token_id) is int for token_id in sent_prompt_ids)
    )
    returned_valid = (
        isinstance(returned_prompt_ids, list)
        and len(returned_prompt_ids) == expected_prompt_tokens
        and all(type(token_id) is int for token_id in returned_prompt_ids)
    )
    if not sent_valid:
        reasons.append(
            f"{prefix} prompt_token_ids_sent is not list[int] of length {expected_prompt_tokens}"
        )
    if not returned_valid:
        reasons.append(
            f"{prefix} prompt_token_ids_returned is not list[int] of length {expected_prompt_tokens}"
        )
    if sent_valid and returned_valid and returned_prompt_ids != sent_prompt_ids:
        reasons.append(f"{prefix} prompt_token_ids_returned != prompt_token_ids_sent")

    stored_prompt_sha256 = record.get("prompt_sha256")
    if not sent_valid:
        if type(stored_prompt_sha256) is not str:
            reasons.append(f"{prefix} prompt_sha256 is missing or not a string")
    else:
        expected_prompt_sha256 = base.prompt_sha256(sent_prompt_ids)
        if type(stored_prompt_sha256) is not str or stored_prompt_sha256 != expected_prompt_sha256:
            reasons.append(f"{prefix} prompt_sha256 does not match the canonical sent-token hash")

    validation_errors = record.get("validation_errors", object())
    if validation_errors != []:
        reasons.append(f"{prefix} validation_errors is not exactly []")

    http_status = record.get("http_status")
    if type(http_status) is not int or http_status != 200:
        reasons.append(f"{prefix} http_status={http_status!r} is not the strict int 200")

    for key, expected in (("timed_out", False), ("cancelled", False), ("done_received", True)):
        value = record.get(key)
        if type(value) is not bool or value is not expected:
            reasons.append(f"{prefix} {key}={value!r} is not exactly {expected!r}")

    for key in ("error_type", "error_message"):
        if key not in record or record[key] is not None:
            reasons.append(f"{prefix} {key} is not exactly null")

    if record.get("finish_reason") != "length":
        reasons.append(f"{prefix} finish_reason={record.get('finish_reason')!r} != 'length'")

    stored_expected_prompt = record.get("expected_prompt_tokens")
    if type(stored_expected_prompt) is not int or stored_expected_prompt != expected_prompt_tokens:
        reasons.append(
            f"{prefix} expected_prompt_tokens={stored_expected_prompt!r} != strict int {expected_prompt_tokens}"
        )
    stored_expected_completion = record.get("expected_completion_tokens")
    if type(stored_expected_completion) is not int or stored_expected_completion != expected_completion_tokens:
        reasons.append(
            f"{prefix} expected_completion_tokens={stored_expected_completion!r} != strict int "
            f"{expected_completion_tokens}"
        )

    output_ids = record.get("output_token_ids")
    if not isinstance(output_ids, list) or any(type(token_id) is not int for token_id in output_ids):
        reasons.append(f"{prefix} output_token_ids is not list[int]")
    elif len(output_ids) != expected_completion_tokens:
        reasons.append(
            f"{prefix} output_token_ids has {len(output_ids)} entries, expected {expected_completion_tokens}"
        )

    usage = record.get("usage")
    if not isinstance(usage, dict):
        reasons.append(f"{prefix} usage is missing or not a dict")
    else:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if type(prompt_tokens) is not int or prompt_tokens != expected_prompt_tokens:
            reasons.append(
                f"{prefix} usage.prompt_tokens={prompt_tokens!r} != strict int {expected_prompt_tokens}"
            )
        if type(completion_tokens) is not int or completion_tokens != expected_completion_tokens:
            reasons.append(
                f"{prefix} usage.completion_tokens={completion_tokens!r} != strict int "
                f"{expected_completion_tokens}"
            )

    return reasons


_REQUIRED_VICTIM_FIELDS: tuple[str, ...] = (
    "request_index", "status",
    "task_created_ns", "request_dispatch_ns", "first_token_receive_ns", "stream_end_ns",
    "task_created_perf_ns", "http_dispatch_start_perf_ns", "stream_open_or_response_headers_perf_ns",
    "first_token_perf_ns", "token_16_perf_ns", "stream_end_perf_ns",
    "was_dispatched_at_trigger", "had_first_token_at_trigger", "server_exposure_group",
    "decode_tokens_received_at_trigger", "dispatch_to_first_token_ms", "timestamp_monotonicity_errors",
    "server_max_num_seqs",
)


def _validate_diagnostic_episode_gates(
    label: str, result: dict, expected_episode: Episode, expected_burst_count: int,
) -> list[str]:
    """B2/B4 fix (2026-07-20 second hardening pass), extended by Blocker
    B/C (2026-07-20 fifth hardening pass): independently RECOMPUTES every
    mandatory validity gate from raw per-request fields, rather than
    trusting the episode's own stored aggregate flags
    (`validation_errors`, `all_20_streams_open_before_first_token`,
    `timestamp_monotonicity_errors`, reported `active_cohort_size`,
    `trigger.status`, `server_exposure_group`, `was_dispatched_at_trigger`,
    `had_first_token_at_trigger`, `dispatch_to_first_token_ms`, or victim
    alias fields). The logical trigger itself is independently
    reconstructed as the exact maximum of the eight active-cohort
    members' own `token_16_perf_ns` values (no tolerance, no `>=`) --
    never merely cross-checked against a stored value that happens to
    already agree. Every adversarial mutation demonstrated by both
    independent audits and this fifth hardening pass's own requirements
    is caught here without relying on any corresponding flag having been
    set correctly upstream."""
    reasons: list[str] = []

    if not isinstance(result, dict):
        return [f"{label} episode result is not a dict"]

    if result.get("status") != base.REQUEST_STATUS_COMPLETE:
        reasons.append(f"{label} episode status != 'complete' (got {result.get('status')!r})")
    _missing = object()
    stored_validation_errors = result.get("validation_errors", _missing)
    if stored_validation_errors != []:
        shown = "<MISSING>" if stored_validation_errors is _missing else repr(stored_validation_errors)
        reasons.append(f"{label} episode validation_errors is not exactly [] (got {shown})")
    if result.get("no_client_admission_semaphore_used") is not True:
        reasons.append(f"{label} episode: no_client_admission_semaphore_used is not exactly True")
    if result.get("record_type") != RECORD_TYPE_REGULAR_EPISODE:
        reasons.append(f"{label} episode: record_type {result.get('record_type')!r} != {RECORD_TYPE_REGULAR_EPISODE!r}")

    victims = result.get("victim_requests")
    victims_by_idx: dict[int, dict] = {}
    if not isinstance(victims, list) or len(victims) != 20 or not all(isinstance(r, dict) for r in victims):
        reasons.append(f"{label} episode does not have exactly 20 victim_requests, all dicts")
    else:
        indices = [r.get("request_index") for r in victims]
        if any(type(i) is not int for i in indices):
            reasons.append(f"{label} episode victim_requests request_index values are not all strict integers")
        elif len(set(indices)) != 20 or set(indices) != set(range(20)):
            reasons.append(f"{label} episode victim_requests request_index values are not exactly the unique set 0..19")
        else:
            victims_by_idx = {r["request_index"]: r for r in victims}
        if any(r.get("status") != base.REQUEST_STATUS_COMPLETE for r in victims):
            reasons.append(f"{label} episode has a victim request with status != 'complete'")

    burst_reqs = result.get("burst_requests")
    burst_by_idx: dict[int, dict] = {}
    if not isinstance(burst_reqs, list) or len(burst_reqs) != expected_burst_count or not all(isinstance(r, dict) for r in burst_reqs):
        reasons.append(
            f"{label} episode does not have exactly {expected_burst_count} burst_requests, all dicts "
            f"(found {len(burst_reqs) if isinstance(burst_reqs, list) else burst_reqs!r})"
        )
    else:
        if expected_burst_count > 0:
            b_indices = [r.get("request_index") for r in burst_reqs]
            if any(type(i) is not int for i in b_indices):
                reasons.append(f"{label} episode burst_requests request_index values are not all strict integers")
            elif len(set(b_indices)) != expected_burst_count or set(b_indices) != set(range(expected_burst_count)):
                reasons.append(f"{label} episode burst_requests request_index values are not exactly the unique set 0..{expected_burst_count - 1}")
            else:
                burst_by_idx = {r["request_index"]: r for r in burst_reqs}
        if any(r.get("status") != base.REQUEST_STATUS_COMPLETE for r in burst_reqs):
            reasons.append(f"{label} episode has a burst request with status != 'complete'")

    trigger = result.get("trigger")
    trigger_ns = None
    active_indices: set = set()
    if not isinstance(trigger, dict) or trigger.get("status") != cohort.DYNAMIC_TRIGGER_OK:
        reasons.append(f"{label} episode trigger.status != 'ok'")
    else:
        trigger_ns = trigger.get("trigger_perf_ns")
        if type(trigger_ns) is not int:
            reasons.append(f"{label} episode trigger.trigger_perf_ns is not an int")
            trigger_ns = None
        active_list = trigger.get("active_cohort_request_indices")
        # Independently require: a list of exactly 8 unique ints, each
        # in range 0..19 -- never trust the reported active_cohort_size
        # alone (Audit finding: size==8 with only 7 actual entries).
        if (
            not isinstance(active_list, list) or len(active_list) != DIAGNOSTIC_SERVER_MAX_NUM_SEQS
            or any(type(i) is not int for i in active_list)
            or len(set(active_list)) != DIAGNOSTIC_SERVER_MAX_NUM_SEQS
            or any(i < 0 or i > 19 for i in active_list)
        ):
            reasons.append(
                f"{label} episode trigger.active_cohort_request_indices is not exactly "
                f"{DIAGNOSTIC_SERVER_MAX_NUM_SEQS} unique integers in 0..19 (got {active_list!r})"
            )
        else:
            active_indices = set(active_list)
        stored_active_size = trigger.get("active_cohort_size")
        if type(stored_active_size) is not int or stored_active_size != len(active_indices):
            reasons.append(
                f"{label} episode trigger.active_cohort_size ({stored_active_size!r}) is not the strict int "
                f"equal to the actual number of unique active indices ({len(active_indices)})"
            )

        stored_trigger_k = trigger.get("server_max_num_seqs")
        if type(stored_trigger_k) is not int or stored_trigger_k != expected_episode.server_max_num_seqs:
            reasons.append(
                f"{label} episode trigger.server_max_num_seqs={stored_trigger_k!r} != strict int "
                f"{expected_episode.server_max_num_seqs}"
            )
        stored_threshold = trigger.get("trigger_after_decode_tokens")
        if type(stored_threshold) is not int or stored_threshold != expected_episode.trigger_after_decode_tokens:
            reasons.append(
                f"{label} episode trigger.trigger_after_decode_tokens={stored_threshold!r} != strict int "
                f"{expected_episode.trigger_after_decode_tokens}"
            )
        cohort_freeze_ns = trigger.get("cohort_freeze_ns")
        if type(cohort_freeze_ns) is not int or cohort_freeze_ns < 0:
            reasons.append(
                f"{label} episode trigger.cohort_freeze_ns={cohort_freeze_ns!r} is not a non-negative int"
            )
        elif trigger_ns is not None and cohort_freeze_ns > trigger_ns:
            reasons.append(
                f"{label} episode trigger.cohort_freeze_ns ({cohort_freeze_ns}) is after trigger_perf_ns "
                f"({trigger_ns})"
            )

        metrics_quality = trigger.get("metrics_quality")
        allowed_metrics_statuses = {
            METRICS_QUALITY_CORROBORATED,
            METRICS_QUALITY_UNAVAILABLE,
            METRICS_QUALITY_STALE,
            METRICS_QUALITY_CONTRADICTORY,
            METRICS_QUALITY_UNPARSABLE,
        }
        if not isinstance(metrics_quality, dict):
            reasons.append(f"{label} episode trigger.metrics_quality is missing or not a dict")
        else:
            metrics_status = metrics_quality.get("metrics_quality_status")
            if type(metrics_status) is not str or metrics_status not in allowed_metrics_statuses:
                reasons.append(
                    f"{label} episode trigger.metrics_quality.metrics_quality_status={metrics_status!r} "
                    f"is not one of {sorted(allowed_metrics_statuses)!r}"
                )

    if not victims_by_idx or trigger_ns is None or len(active_indices) != DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
        # Cannot safely recompute anything trigger/index/timestamp-relative
        # without a trustworthy trigger_ns/active_indices/victims_by_idx --
        # stop here rather than risk a false negative on later checks.
        # Section 6 fix (2026-07-20 fifth hardening pass): this check is
        # scoped ONLY to the three prerequisites actually needed for the
        # reconstruction below -- it must NOT bail out merely because some
        # OTHER, unrelated reason (status/validation_errors/no_semaphore/
        # burst-list-shape/record_type) already exists in `reasons`. Every
        # reason collected so far is still returned either way.
        return reasons

    # --- Blocker C (5.1-5.7): full per-victim field presence, alias
    # equality, independently-recomputed monotonicity (including
    # token_16_perf_ns in the ordered chain), and independently
    # recomputed was_dispatched_at_trigger / had_first_token_at_trigger /
    # server_exposure_group / dispatch_to_first_token_ms -- plus Blocker
    # B (4.2-4.4): exact active-cohort token-16/trigger reconstruction
    # and strict non-cohort/active trigger-relative timing. ---------------
    active_token16_values: list[int] = []
    active_first_token_values: list[int] = []
    for idx, r in victims_by_idx.items():
        missing_fields = [f for f in _REQUIRED_VICTIM_FIELDS if f not in r]
        if missing_fields:
            reasons.append(f"{label} episode: victim {idx} is missing required field(s): {missing_fields}")
            continue

        stored_k = r["server_max_num_seqs"]
        if type(stored_k) is not int or stored_k != DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
            reasons.append(
                f"{label} episode: victim {idx} server_max_num_seqs={stored_k!r} != "
                f"the required strict int {DIAGNOSTIC_SERVER_MAX_NUM_SEQS}"
            )

        reasons.extend(_validate_diagnostic_request_functional_fields(
            label=label, role="victim", request_index=idx, record=r,
            expected_episode=expected_episode,
            expected_prompt_tokens=expected_episode.victim_input_len,
            expected_completion_tokens=expected_episode.victim_output_len,
        ))

        # 5.2 Alias equality -- exact, no tolerance.
        if r["task_created_perf_ns"] != r["task_created_ns"]:
            reasons.append(f"{label} episode: victim {idx} task_created_perf_ns != task_created_ns")
        if r["http_dispatch_start_perf_ns"] != r["request_dispatch_ns"]:
            reasons.append(f"{label} episode: victim {idx} http_dispatch_start_perf_ns != request_dispatch_ns")
        if r["first_token_perf_ns"] != r["first_token_receive_ns"]:
            reasons.append(f"{label} episode: victim {idx} first_token_perf_ns != first_token_receive_ns")
        if r["stream_end_perf_ns"] != r["stream_end_ns"]:
            reasons.append(f"{label} episode: victim {idx} stream_end_perf_ns != stream_end_ns")

        raw_timestamps = [
            ("task_created_ns", r["task_created_ns"]),
            ("request_dispatch_ns", r["request_dispatch_ns"]),
            ("stream_open_or_response_headers_perf_ns", r["stream_open_or_response_headers_perf_ns"]),
            ("first_token_receive_ns", r["first_token_receive_ns"]),
            ("token_16_perf_ns", r["token_16_perf_ns"]),
            ("stream_end_ns", r["stream_end_ns"]),
        ]
        canonical = [
            ("task_created_perf_ns", r["task_created_perf_ns"]),
            ("http_dispatch_start_perf_ns", r["http_dispatch_start_perf_ns"]),
            ("stream_open_or_response_headers_perf_ns", r["stream_open_or_response_headers_perf_ns"]),
            ("first_token_perf_ns", r["first_token_perf_ns"]),
            ("token_16_perf_ns", r["token_16_perf_ns"]),
            ("stream_end_perf_ns", r["stream_end_perf_ns"]),
        ]
        bad_raw_types = [name for name, value in raw_timestamps if type(value) is not int]
        bad_canonical_types = [name for name, value in canonical if type(value) is not int]
        if bad_raw_types:
            reasons.append(f"{label} episode: victim {idx} has non-int raw timestamp(s): {bad_raw_types}")
        if bad_canonical_types:
            reasons.append(f"{label} episode: victim {idx} has non-int canonical timestamp(s): {bad_canonical_types}")

        # 5.3 Independently reconstructed monotonicity -- the stored
        # timestamp_monotonicity_errors is ALSO required to be exactly
        # [], but is never trusted on its own.
        if r["timestamp_monotonicity_errors"] != []:
            reasons.append(
                f"{label} episode: victim {idx} stored timestamp_monotonicity_errors is not exactly []: "
                f"{r['timestamp_monotonicity_errors']!r}"
            )
        if bad_raw_types or bad_canonical_types:
            # Unsafe arithmetic is never attempted on damaged JSON. The
            # strict type failures above force D while validation continues
            # with all other records, so the classifier remains fail-closed
            # and exception-free.
            continue

        reasons.extend(_check_monotonic_sequence(f"{label} victim {idx} raw", raw_timestamps))
        reasons.extend(_check_monotonic_sequence(f"{label} victim {idx} canonical", canonical))

        request_dispatch_ns = r["request_dispatch_ns"]
        first_token_ns = r["first_token_perf_ns"]
        token16_ns = r["token_16_perf_ns"]
        end_ns = r["stream_end_perf_ns"]

        # 5.4 Dispatch/first-token booleans, independently reconstructed.
        expected_dispatched = request_dispatch_ns <= trigger_ns
        expected_first_token = first_token_ns <= trigger_ns
        if r["was_dispatched_at_trigger"] is not expected_dispatched:
            reasons.append(
                f"{label} episode: victim {idx} was_dispatched_at_trigger={r['was_dispatched_at_trigger']!r} "
                f"!= reconstructed {expected_dispatched!r}"
            )
        if not expected_dispatched:
            reasons.append(f"{label} episode: victim {idx} was not dispatched before the trigger (all 20 must be)")
        if r["had_first_token_at_trigger"] is not expected_first_token:
            reasons.append(
                f"{label} episode: victim {idx} had_first_token_at_trigger={r['had_first_token_at_trigger']!r} "
                f"!= reconstructed {expected_first_token!r}"
            )

        # 5.6 dispatch_to_first_token_ms, independently recomputed with a
        # tight (representation-only) tolerance -- never a generous one.
        expected_dtf_ms = (first_token_ns - request_dispatch_ns) / 1e6
        stored_dtf = r["dispatch_to_first_token_ms"]
        if not isinstance(stored_dtf, (int, float)) or isinstance(stored_dtf, bool) or not math.isclose(
            float(stored_dtf), expected_dtf_ms, rel_tol=0.0, abs_tol=1e-9,
        ):
            reasons.append(
                f"{label} episode: victim {idx} dispatch_to_first_token_ms={stored_dtf!r} != "
                f"reconstructed {expected_dtf_ms!r}"
            )

        if idx in active_indices:
            # 4.4 Active-cohort state at the trigger.
            dcount = r["decode_tokens_received_at_trigger"]
            if not (
                type(dcount) is int
                and expected_episode.trigger_after_decode_tokens <= dcount <= expected_episode.victim_output_len
            ):
                reasons.append(
                    f"{label} episode: active-cohort victim {idx} decode_tokens_received_at_trigger="
                    f"{dcount!r} is not a strict int in "
                    f"[{expected_episode.trigger_after_decode_tokens}, {expected_episode.victim_output_len}]"
                )
            if not (first_token_ns <= trigger_ns):
                reasons.append(f"{label} episode: active-cohort victim {idx} first_token_perf_ns is not <= the trigger")
            if not (token16_ns <= trigger_ns):
                reasons.append(f"{label} episode: active-cohort victim {idx} token_16_perf_ns is not <= the trigger")
            if r["had_first_token_at_trigger"] is not True:
                reasons.append(f"{label} episode: active-cohort victim {idx} had_first_token_at_trigger is not True")
            if r["server_exposure_group"] != SERVER_EXPOSURE_RUNNING_AT_TRIGGER:
                reasons.append(
                    f"{label} episode: active-cohort victim {idx} server_exposure_group="
                    f"{r['server_exposure_group']!r} != {SERVER_EXPOSURE_RUNNING_AT_TRIGGER!r}"
                )
            if not (end_ns > trigger_ns):
                reasons.append(f"{label} episode: active-cohort victim {idx} completed at/before the trigger")
            active_token16_values.append(token16_ns)
            active_first_token_values.append(first_token_ns)
        else:
            # 4.3 Non-cohort state at the trigger. A first token AT or
            # BEFORE the trigger is D even if decode_tokens_received_at_trigger
            # still (incorrectly) reports 0.
            dcount = r["decode_tokens_received_at_trigger"]
            if type(dcount) is not int or dcount != 0:
                reasons.append(
                    f"{label} episode: non-cohort victim {idx} decode_tokens_received_at_trigger="
                    f"{dcount!r} != the strict int 0"
                )
            if not (first_token_ns > trigger_ns):
                reasons.append(
                    f"{label} episode: non-cohort victim {idx} first_token_perf_ns ({first_token_ns}) is not "
                    f"strictly after the trigger ({trigger_ns})"
                )
            if r["had_first_token_at_trigger"] is not False:
                reasons.append(f"{label} episode: non-cohort victim {idx} had_first_token_at_trigger is not False")
            if r["server_exposure_group"] != SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT:
                reasons.append(
                    f"{label} episode: non-cohort victim {idx} server_exposure_group="
                    f"{r['server_exposure_group']!r} != {SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT!r}"
                )

    # 4.2 Exact trigger reconstruction: trigger_perf_ns must equal the
    # maximum of the eight active-cohort members' own token_16_perf_ns
    # values -- no tolerance, no >=. Only attempted when all eight were
    # actually collected (i.e. no earlier per-victim type/field failure
    # already explains a gap; if one did, that failure is already in
    # `reasons` and this check is skipped rather than risk a misleading
    # comparison against incomplete data).
    if len(active_token16_values) == DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
        expected_trigger_ns = max(active_token16_values)
        if trigger_ns != expected_trigger_ns:
            reasons.append(
                f"{label} episode: trigger_perf_ns ({trigger_ns!r}) != max(active token_16_perf_ns) "
                f"({expected_trigger_ns!r})"
            )
    if len(active_first_token_values) == DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
        expected_freeze_ns = max(active_first_token_values)
        stored_freeze_ns = trigger.get("cohort_freeze_ns")
        if type(stored_freeze_ns) is not int or stored_freeze_ns != expected_freeze_ns:
            reasons.append(
                f"{label} episode: cohort_freeze_ns ({stored_freeze_ns!r}) != "
                f"max(active first_token_perf_ns) ({expected_freeze_ns!r})"
            )

    # --- Recompute all_20_streams_open_before_first_token directly from
    # the 20 raw records; cross-check against (never simply trust) the
    # stored aggregate flag. -----------------------------------------------
    open_timestamps = [r.get("stream_open_or_response_headers_perf_ns") for r in victims_by_idx.values()]
    first_token_timestamps = [r.get("first_token_perf_ns") for r in victims_by_idx.values()]
    if all(type(x) is int for x in open_timestamps) and all(type(x) is int for x in first_token_timestamps):
        earliest_first_token_ns = min(first_token_timestamps)
        recomputed_all_20 = all(x < earliest_first_token_ns for x in open_timestamps)
    else:
        recomputed_all_20 = False
    if recomputed_all_20 is not True:
        reasons.append(f"{label} episode: recomputed all_20_streams_open_before_first_token is not true")
    stored_tce = result.get("transport_concurrency_evidence")
    if not isinstance(stored_tce, dict):
        reasons.append(
            f"{label} episode: transport_concurrency_evidence is not a dict "
            f"(got {type(stored_tce).__name__})"
        )
    else:
        stored_all_20 = stored_tce.get("all_20_streams_open_before_first_token")
        if type(stored_all_20) is not bool or stored_all_20 is not recomputed_all_20:
            reasons.append(
                f"{label} episode: stored all_20_streams_open_before_first_token ({stored_all_20!r}) is not "
                f"the strict bool matching the value recomputed from per-request timestamps ({recomputed_all_20!r})"
            )
        stored_open_count = stored_tce.get("victim_stream_open_count")
        if type(stored_open_count) is not int or stored_open_count != len(open_timestamps):
            reasons.append(
                f"{label} episode: victim_stream_open_count={stored_open_count!r} != strict int "
                f"{len(open_timestamps)} recomputed from victim records"
            )
        stored_earliest = stored_tce.get("earliest_victim_first_token_ns")
        expected_earliest = min(first_token_timestamps) if all(type(x) is int for x in first_token_timestamps) else None
        if type(stored_earliest) is not int or stored_earliest != expected_earliest:
            reasons.append(
                f"{label} episode: earliest_victim_first_token_ns={stored_earliest!r} != "
                f"recomputed {expected_earliest!r}"
            )
        peak_open = stored_tce.get("peak_concurrent_open_completion_streams")
        if expected_burst_count == 0:
            peak_valid = type(peak_open) is int and peak_open == 20
            peak_expectation = "the strict int 20 for no_burst"
        else:
            peak_valid = type(peak_open) is int and 20 <= peak_open <= 24
            peak_expectation = "a strict int in [20, 24] for prefill_burst"
        if not peak_valid:
            reasons.append(
                f"{label} episode: peak_concurrent_open_completion_streams={peak_open!r} is not "
                f"{peak_expectation}"
            )
        expected_pool_limits = {
            "max_connections": COMPLETION_POOL_MAX_CONNECTIONS,
            "max_keepalive_connections": COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS,
        }
        stored_pool_limits = stored_tce.get("completion_pool_limits")
        if not isinstance(stored_pool_limits, dict) or stored_pool_limits != expected_pool_limits or any(
            type(stored_pool_limits.get(key)) is not int for key in expected_pool_limits
        ):
            reasons.append(
                f"{label} episode: transport completion_pool_limits={stored_pool_limits!r} != "
                f"the exact strict-int limits {expected_pool_limits!r}"
            )

    # --- Burst-specific gates: canonical timestamps present, recomputed
    # monotonicity, alias fields equal their canonical source fields
    # exactly, and dispatch strictly AFTER the trigger. ---------------------
    if expected_burst_count > 0 and trigger_ns is not None:
        for idx, r in burst_by_idx.items():
            reasons.extend(_validate_diagnostic_request_functional_fields(
                label=label, role="burst", request_index=idx, record=r,
                expected_episode=expected_episode,
                expected_prompt_tokens=expected_episode.burst_input_len,
                expected_completion_tokens=expected_episode.burst_output_len,
            ))
            if "timestamp_monotonicity_errors" not in r:
                reasons.append(
                    f"{label} episode: burst {idx} is missing required field 'timestamp_monotonicity_errors'"
                )
            elif r.get("timestamp_monotonicity_errors") != []:
                reasons.append(
                    f"{label} episode: burst {idx} stored timestamp_monotonicity_errors is not empty: "
                    f"{r.get('timestamp_monotonicity_errors')!r}"
                )
            canonical = [
                ("burst_dispatch_start_perf_ns", r.get("burst_dispatch_start_perf_ns")),
                ("burst_stream_open_or_response_headers_perf_ns", r.get("burst_stream_open_or_response_headers_perf_ns")),
                ("burst_first_token_perf_ns", r.get("burst_first_token_perf_ns")),
                ("burst_end_perf_ns", r.get("burst_end_perf_ns")),
            ]
            missing = [name for name, value in canonical if type(value) is not int]
            if missing:
                reasons.append(f"{label} episode: burst {idx} is missing required canonical timestamp(s): {missing}")
                continue
            reasons.extend(_check_monotonic_sequence(f"{label} burst {idx}", canonical))

            raw_burst_timestamps = [
                ("request_dispatch_ns", r.get("request_dispatch_ns")),
                ("first_token_receive_ns", r.get("first_token_receive_ns")),
                ("stream_end_ns", r.get("stream_end_ns")),
            ]
            invalid_raw = [name for name, value in raw_burst_timestamps if type(value) is not int]
            if invalid_raw:
                reasons.append(
                    f"{label} episode: burst {idx} has non-int required raw timestamp(s): {invalid_raw}"
                )
            else:
                if r["burst_dispatch_start_perf_ns"] != r["request_dispatch_ns"]:
                    reasons.append(
                        f"{label} episode: burst {idx} burst_dispatch_start_perf_ns != its own request_dispatch_ns"
                    )
                if r["burst_first_token_perf_ns"] != r["first_token_receive_ns"]:
                    reasons.append(
                        f"{label} episode: burst {idx} burst_first_token_perf_ns != its own first_token_receive_ns"
                    )
                if r["burst_end_perf_ns"] != r["stream_end_ns"]:
                    reasons.append(
                        f"{label} episode: burst {idx} burst_end_perf_ns != its own stream_end_ns"
                    )
                reasons.extend(_check_monotonic_sequence(f"{label} burst {idx} raw", raw_burst_timestamps))

            dispatch_ns = r.get("burst_dispatch_start_perf_ns")
            if type(dispatch_ns) is int and dispatch_ns <= trigger_ns:
                reasons.append(f"{label} episode: burst {idx} dispatch ({dispatch_ns}) is not strictly after the trigger ({trigger_ns})")

    return reasons


def _validate_diagnostic_episode_identity(
    label: str, result: dict, expected_episode: Episode, expected_condition: str,
    expected_schedule_fingerprint: str, expected_environment_fingerprint: str,
    expected_server_command: Sequence[str],
) -> list[str]:
    """Blocker 2 fix (2026-07-20 fourth hardening pass): binds a result
    to precisely `qwen_off12_k8_rep01` and to the caller's expected
    fingerprints, rather than only checking the result's own internal
    self-consistency. A syntactically well-formed but semantically wrong
    result (wrong episode_id, mutated schedule_row field, swapped
    condition, wrong run_mode/schedule_fingerprint/environment_fingerprint/
    schema/runner version) must never be classified A, B, or C."""
    reasons: list[str] = []
    if not isinstance(result, dict):
        return [f"{label} episode result is not a dict"]

    # Sanity-check the CALLER's inputs too: the expected episode object
    # itself must actually describe this exact diagnostic cell. This
    # catches a caller passing the wrong expected_episode by mistake,
    # rather than silently validating a result against an already-wrong
    # expectation.
    if expected_episode.block_id != DIAGNOSTIC_BLOCK_ID:
        reasons.append(f"{label}: expected_episode.block_id {expected_episode.block_id!r} != {DIAGNOSTIC_BLOCK_ID!r}")
    if expected_episode.model_key != DIAGNOSTIC_MODEL_KEY:
        reasons.append(f"{label}: expected_episode.model_key {expected_episode.model_key!r} != {DIAGNOSTIC_MODEL_KEY!r}")
    if expected_episode.model_id != MODEL_ID:
        reasons.append(f"{label}: expected_episode.model_id {expected_episode.model_id!r} != {MODEL_ID!r}")
    if expected_episode.offload_gb != DIAGNOSTIC_OFFLOAD_GB:
        reasons.append(f"{label}: expected_episode.offload_gb {expected_episode.offload_gb!r} != {DIAGNOSTIC_OFFLOAD_GB!r}")
    if expected_episode.state_label != STATE_LABEL_BY_OFFLOAD[DIAGNOSTIC_OFFLOAD_GB]:
        reasons.append(
            f"{label}: expected_episode.state_label {expected_episode.state_label!r} != "
            f"{STATE_LABEL_BY_OFFLOAD[DIAGNOSTIC_OFFLOAD_GB]!r}"
        )
    if expected_episode.server_max_num_seqs != DIAGNOSTIC_SERVER_MAX_NUM_SEQS:
        reasons.append(
            f"{label}: expected_episode.server_max_num_seqs {expected_episode.server_max_num_seqs!r} != "
            f"{DIAGNOSTIC_SERVER_MAX_NUM_SEQS!r}"
        )
    if expected_episode.repeat != DIAGNOSTIC_REPEAT:
        reasons.append(f"{label}: expected_episode.repeat {expected_episode.repeat!r} != {DIAGNOSTIC_REPEAT!r}")
    if expected_episode.trigger_after_decode_tokens != 16:
        reasons.append(f"{label}: expected_episode.trigger_after_decode_tokens != 16")
    if expected_episode.condition != expected_condition:
        reasons.append(
            f"{label}: expected_episode.condition {expected_episode.condition!r} != {expected_condition!r} "
            f"(caller passed the wrong expected episode for this slot)"
        )

    # Now bind the ACTUAL result to that (validated) expectation.
    if result.get("episode_id") != expected_episode.episode_id:
        reasons.append(f"{label}: episode_id {result.get('episode_id')!r} != expected {expected_episode.episode_id!r}")
    if result.get("block_id") != DIAGNOSTIC_BLOCK_ID:
        reasons.append(f"{label}: block_id {result.get('block_id')!r} != {DIAGNOSTIC_BLOCK_ID!r}")

    schedule_row = result.get("schedule_row")
    expected_schedule_row = asdict(expected_episode)
    if not isinstance(schedule_row, dict):
        reasons.append(f"{label}: stored schedule_row is missing or not a dict")
    else:
        if set(schedule_row) != set(expected_schedule_row):
            reasons.append(f"{label}: stored schedule_row field set does not exactly match the expected episode object")
        for field_name, expected_value in expected_schedule_row.items():
            if field_name not in schedule_row:
                continue
            actual_value = schedule_row[field_name]
            expected_type = EPISODE_FIELD_TYPES[field_name]
            if type(actual_value) is not expected_type or actual_value != expected_value:
                reasons.append(
                    f"{label}: schedule_row.{field_name}={actual_value!r} is not the expected "
                    f"{expected_type.__name__} value {expected_value!r}"
                )
        if schedule_row.get("condition") != expected_condition:
            reasons.append(
                f"{label}: schedule_row.condition {schedule_row.get('condition')!r} != "
                f"expected {expected_condition!r}"
            )

    if result.get("run_mode") != RUN_MODE_DIAGNOSTIC_PAIR:
        reasons.append(f"{label}: run_mode {result.get('run_mode')!r} != {RUN_MODE_DIAGNOSTIC_PAIR!r}")
    if result.get("schedule_fingerprint") != expected_schedule_fingerprint:
        reasons.append(f"{label}: schedule_fingerprint does not match the expected schedule_fingerprint")
    stored_schema_version = result.get("result_schema_version")
    if type(stored_schema_version) is not int or stored_schema_version != RESULT_SCHEMA_VERSION:
        reasons.append(
            f"{label}: result_schema_version {stored_schema_version!r} is not the strict int "
            f"{RESULT_SCHEMA_VERSION!r}"
        )
    if result.get("runner_version") != RUNNER_VERSION:
        reasons.append(f"{label}: runner_version {result.get('runner_version')!r} != {RUNNER_VERSION!r}")
    stored_instrumentation_name = result.get("timing_instrumentation_name")
    if type(stored_instrumentation_name) is not str or stored_instrumentation_name != TIMING_INSTRUMENTATION_NAME:
        reasons.append(
            f"{label}: timing_instrumentation_name={stored_instrumentation_name!r} != "
            f"{TIMING_INSTRUMENTATION_NAME!r}"
        )
    stored_instrumentation_version = result.get("timing_instrumentation_version")
    if type(stored_instrumentation_version) is not int or stored_instrumentation_version != TIMING_INSTRUMENTATION_VERSION:
        reasons.append(
            f"{label}: timing_instrumentation_version={stored_instrumentation_version!r} != strict int "
            f"{TIMING_INSTRUMENTATION_VERSION}"
        )

    stabilization_reference = result.get("stabilization_reference")
    if not isinstance(stabilization_reference, dict):
        reasons.append(f"{label}: stabilization_reference is missing or not a dict")
    else:
        if stabilization_reference.get("block_id") != expected_episode.block_id:
            reasons.append(
                f"{label}: stabilization_reference.block_id={stabilization_reference.get('block_id')!r} != "
                f"{expected_episode.block_id!r}"
            )
        if stabilization_reference.get("functional_passed") is not True:
            reasons.append(f"{label}: stabilization_reference.functional_passed is not exactly True")
        stabilization_path = stabilization_reference.get("path")
        expected_stabilization_name = f"{expected_episode.block_id}.json"
        if type(stabilization_path) is not str:
            reasons.append(f"{label}: stabilization_reference.path is missing or not a string")
        else:
            stabilization_path_obj = Path(stabilization_path)
            if (
                stabilization_path_obj.name != expected_stabilization_name
                or stabilization_path_obj.parent.name != "stabilization"
            ):
                reasons.append(
                    f"{label}: stabilization_reference.path={stabilization_path!r} does not identify "
                    f"stabilization/{expected_stabilization_name}"
                )

    server_metadata = result.get("server_metadata")
    if not isinstance(server_metadata, dict):
        reasons.append(f"{label}: server_metadata is missing or not a dict")
    else:
        expected_metadata_values = {
            "environment_fingerprint": expected_environment_fingerprint,
            "model_key": expected_episode.model_key,
            "model_full_id": expected_episode.model_id,
            "offload_gb": expected_episode.offload_gb,
            "server_max_num_seqs": expected_episode.server_max_num_seqs,
        }
        for key, expected_value in expected_metadata_values.items():
            actual_value = server_metadata.get(key)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                reasons.append(
                    f"{label}: server_metadata.{key}={actual_value!r} is not the expected strict "
                    f"{type(expected_value).__name__} value {expected_value!r}"
                )

        expected_pool_limits = {
            "max_connections": COMPLETION_POOL_MAX_CONNECTIONS,
            "max_keepalive_connections": COMPLETION_POOL_MAX_KEEPALIVE_CONNECTIONS,
        }
        stored_server_pool_limits = server_metadata.get("completion_pool_limits")
        if not isinstance(stored_server_pool_limits, dict) or stored_server_pool_limits != expected_pool_limits or any(
            type(stored_server_pool_limits.get(key)) is not int for key in expected_pool_limits
        ):
            reasons.append(
                f"{label}: server_metadata.completion_pool_limits={stored_server_pool_limits!r} != "
                f"the exact strict-int limits {expected_pool_limits!r}"
            )

        expected_command = list(expected_server_command)
        stored_command = server_metadata.get("server_command")
        if (
            not isinstance(stored_command, list)
            or any(type(part) is not str for part in stored_command)
            or stored_command != expected_command
        ):
            reasons.append(
                f"{label}: server_metadata.server_command does not exactly match the expected diagnostic command"
            )
        if len(expected_command) >= 7:
            expected_host = expected_command[-2]
            try:
                expected_port = int(expected_command[-1])
            except (TypeError, ValueError):
                expected_port = None
            stored_host = server_metadata.get("host")
            if type(stored_host) is not str or stored_host != expected_host:
                reasons.append(
                    f"{label}: server_metadata.host={stored_host!r} is not the expected strict str "
                    f"{expected_host!r}"
                )
            stored_port = server_metadata.get("port")
            if type(stored_port) is not int or stored_port != expected_port:
                reasons.append(
                    f"{label}: server_metadata.port={stored_port!r} is not the expected strict int "
                    f"{expected_port!r}"
                )


    return reasons


def classify_diagnostic_pair(
    *, no_burst_result: dict | None, prefill_burst_result: dict | None,
    expected_no_burst_episode: Episode, expected_prefill_burst_episode: Episode,
    expected_schedule_fingerprint: str, expected_environment_fingerprint: str,
    expected_server_command: Sequence[str],
) -> dict:
    """Requirement 9. Classifies the prefill_burst episode, using the
    paired no_burst episode only for effect comparison. Every mandatory
    gate is independently RECOMPUTED (see
    `_validate_diagnostic_episode_gates`), and both results are bound to
    the caller's exact expected `Episode` objects and expected schedule/
    environment fingerprints (Blocker 2 fix, see
    `_validate_diagnostic_episode_identity`) before A/B/C is ever
    considered; any failure produces D_AMBIGUOUS_OR_INVALID with the
    complete accumulated reason list (never just the first one found)."""
    reasons: list[str] = []

    if no_burst_result is None:
        reasons.append("no_burst episode result is missing or unreadable")
    if prefill_burst_result is None:
        reasons.append("prefill_burst episode result is missing or unreadable")
    if reasons:
        return {"classification": DIAGNOSTIC_CLASSIFICATION_D, "reasons": reasons}

    reasons.extend(_validate_diagnostic_episode_identity(
        "no_burst", no_burst_result, expected_no_burst_episode, "no_burst",
        expected_schedule_fingerprint, expected_environment_fingerprint, expected_server_command,
    ))
    reasons.extend(_validate_diagnostic_episode_identity(
        "prefill_burst", prefill_burst_result, expected_prefill_burst_episode, base.BURST_CONDITION,
        expected_schedule_fingerprint, expected_environment_fingerprint, expected_server_command,
    ))

    reasons.extend(_validate_diagnostic_episode_gates(
        "no_burst", no_burst_result, expected_no_burst_episode, 0,
    ))
    reasons.extend(_validate_diagnostic_episode_gates(
        "prefill_burst", prefill_burst_result, expected_prefill_burst_episode,
        BURST_CONFIGURATION["burst_parallel_requests"],
    ))

    # The pair's descriptive effect values are part of the scientific
    # output.  They must be internally reconstructible from the raw token
    # timing fields before A/B/C can be considered valid.
    reasons.extend(_validate_condition_effect_input("no_burst", no_burst_result))
    reasons.extend(_validate_condition_effect_input("prefill_burst", prefill_burst_result))

    if reasons:
        return {"classification": DIAGNOSTIC_CLASSIFICATION_D, "reasons": reasons}

    pb_trigger = prefill_burst_result["trigger"]
    active_indices = set(pb_trigger["active_cohort_request_indices"])
    victims_by_idx = {r["request_index"]: r for r in prefill_burst_result["victim_requests"]}
    burst_reqs = prefill_burst_result["burst_requests"]

    first_burst_output_ns = min(r["burst_first_token_perf_ns"] for r in burst_reqs)
    last_active_victim_completion_ns = max(victims_by_idx[i]["stream_end_perf_ns"] for i in active_indices)
    last_all_victim_completion_ns = max(r["stream_end_perf_ns"] for r in victims_by_idx.values())

    if first_burst_output_ns < last_active_victim_completion_ns:
        classification = DIAGNOSTIC_CLASSIFICATION_A
    elif first_burst_output_ns > last_all_victim_completion_ns:
        classification = DIAGNOSTIC_CLASSIFICATION_C
    else:
        classification = DIAGNOSTIC_CLASSIFICATION_B

    return {
        "classification": classification,
        "reasons": [],
        "first_burst_output_ns": first_burst_output_ns,
        "last_active_victim_completion_ns": last_active_victim_completion_ns,
        "last_all_victim_completion_ns": last_all_victim_completion_ns,
        "interpretation_notes": list(_DIAGNOSTIC_INTERPRETATION_NOTES),
    }


def _validate_condition_effect_input(label: str, episode_result: object) -> list[str]:
    """Validate every field consumed by the descriptive effect summary.

    The summary is secondary output and must never turn an already
    fail-closed D classification into an exception.  This validator is
    deliberately independent of the main classifier so direct callers are
    safe as well.
    """
    reasons: list[str] = []
    if not isinstance(episode_result, dict):
        return [f"{label} effect input is not a dict"]

    trigger = episode_result.get("trigger")
    if not isinstance(trigger, dict):
        return [f"{label} effect input trigger is missing or not a dict"]
    active_list = trigger.get("active_cohort_request_indices")
    if (
        not isinstance(active_list, list)
        or len(active_list) != DIAGNOSTIC_SERVER_MAX_NUM_SEQS
        or any(type(index) is not int for index in active_list)
        or len(set(active_list)) != DIAGNOSTIC_SERVER_MAX_NUM_SEQS
    ):
        return [f"{label} effect input active_cohort_request_indices is not eight unique strict integers"]
    active_indices = set(active_list)

    victims = episode_result.get("victim_requests")
    if not isinstance(victims, list) or not all(isinstance(record, dict) for record in victims):
        return [f"{label} effect input victim_requests is missing or not list[dict]"]
    victim_indices = [record.get("request_index") for record in victims]
    if any(type(index) is not int for index in victim_indices):
        return [f"{label} effect input victim request indices are not strict integers"]
    victims_by_idx = {record["request_index"]: record for record in victims}
    missing_active = sorted(active_indices - set(victims_by_idx))
    if missing_active:
        reasons.append(f"{label} effect input is missing active victim(s): {missing_active}")

    for index in sorted(active_indices & set(victims_by_idx)):
        record = victims_by_idx[index]
        expected_completion_tokens = VICTIM_CONFIGURATION["victim_output_len"]

        if record.get("itl_available") is not True:
            reasons.append(f"{label} effect input victim {index} itl_available is not exactly True")

        tpot = record.get("client_observed_tpot_ms")
        if not _is_strict_finite_number(tpot):
            reasons.append(f"{label} effect input victim {index} client_observed_tpot_ms is not finite numeric")

        itl_ms = record.get("itl_ms")
        valid_itls = False
        if not isinstance(itl_ms, list):
            reasons.append(f"{label} effect input victim {index} itl_ms is missing or not a list")
        elif len(itl_ms) != expected_completion_tokens - 1:
            reasons.append(
                f"{label} effect input victim {index} itl_ms has {len(itl_ms)} entries, expected "
                f"{expected_completion_tokens - 1}"
            )
        elif any(not _is_strict_finite_number(value) or float(value) < 0.0 for value in itl_ms):
            reasons.append(
                f"{label} effect input victim {index} itl_ms contains a negative, non-finite, or non-numeric value"
            )
        else:
            valid_itls = True

        first_token_ns = record.get("first_token_receive_ns")
        last_token_ns = record.get("last_token_receive_ns")
        valid_token_interval = (
            type(first_token_ns) is int
            and type(last_token_ns) is int
            and last_token_ns >= first_token_ns
        )
        if not valid_token_interval:
            reasons.append(
                f"{label} effect input victim {index} has invalid first_token_receive_ns/last_token_receive_ns"
            )
        else:
            expected_interval_ms = (last_token_ns - first_token_ns) / 1e6
            if valid_itls:
                stored_itl_sum_ms = math.fsum(float(value) for value in itl_ms)
                if not math.isclose(stored_itl_sum_ms, expected_interval_ms, rel_tol=0.0, abs_tol=1e-9):
                    reasons.append(
                        f"{label} effect input victim {index} sum(itl_ms)={stored_itl_sum_ms!r} "
                        f"!= token interval {expected_interval_ms!r} ms"
                    )
            if _is_strict_finite_number(tpot):
                expected_tpot_ms = expected_interval_ms / (expected_completion_tokens - 1)
                if not math.isclose(float(tpot), expected_tpot_ms, rel_tol=0.0, abs_tol=1e-9):
                    reasons.append(
                        f"{label} effect input victim {index} client_observed_tpot_ms={tpot!r} "
                        f"!= reconstructed {expected_tpot_ms!r}"
                    )

        task_created = record.get("task_created_perf_ns")
        end_ns = record.get("stream_end_perf_ns")
        if type(task_created) is not int or type(end_ns) is not int or end_ns < task_created:
            reasons.append(
                f"{label} effect input victim {index} has invalid task_created_perf_ns/stream_end_perf_ns"
            )

    return reasons


def _compute_condition_active_cohort_stats(episode_result: dict) -> dict:
    trigger = episode_result["trigger"]
    active_indices = set(trigger["active_cohort_request_indices"])
    victims_by_idx = {record["request_index"]: record for record in episode_result["victim_requests"]}
    active = [victims_by_idx[index] for index in sorted(active_indices)]

    tpots = [float(record["client_observed_tpot_ms"]) for record in active]
    max_itls = [max(float(value) for value in record["itl_ms"]) for record in active]
    completions = [
        (record["stream_end_perf_ns"] - record["task_created_perf_ns"]) / 1e6
        for record in active
    ]

    return {
        "active_cohort_median_tpot_ms": base._median(tpots),
        "active_cohort_median_max_itl_ms": base._median(max_itls),
        "active_cohort_median_completion_from_task_creation_ms": base._median(completions),
        "active_cohort_size": len(active),
    }


def _build_condition_diagnostic_detail(result: dict) -> dict:
    stats = _compute_condition_active_cohort_stats(result)
    trigger = result["trigger"]
    tce = result["transport_concurrency_evidence"]
    trigger_ns = trigger["trigger_perf_ns"]
    active_indices = set(trigger["active_cohort_request_indices"])
    victims = result["victim_requests"]

    all_dispatched_before_trigger = all(record["request_dispatch_ns"] <= trigger_ns for record in victims)
    noncohort = [record for record in victims if record["request_index"] not in active_indices]
    noncohort_zero = all(record["decode_tokens_received_at_trigger"] == 0 for record in noncohort)

    active_ends = [
        record["stream_end_perf_ns"] for record in victims if record["request_index"] in active_indices
    ]
    all_ends = [record["stream_end_perf_ns"] for record in victims]

    per_burst = [
        {
            "request_index": record["request_index"],
            "burst_dispatch_start_perf_ns": record["burst_dispatch_start_perf_ns"],
            "burst_stream_open_or_response_headers_perf_ns": record[
                "burst_stream_open_or_response_headers_perf_ns"
            ],
            "burst_first_token_perf_ns": record["burst_first_token_perf_ns"],
            "burst_end_perf_ns": record["burst_end_perf_ns"],
        }
        for record in result["burst_requests"]
    ]
    metrics_quality = trigger.get("metrics_quality")
    if not isinstance(metrics_quality, dict):
        metrics_quality = {}

    return {
        **stats,
        "all_20_dispatched_before_trigger": all_dispatched_before_trigger,
        "all_20_streams_open_before_first_token": tce["all_20_streams_open_before_first_token"],
        "active_cohort_size_reported": trigger["active_cohort_size"],
        "noncohort_zero_tokens_at_trigger": noncohort_zero,
        "logical_trigger_ns": trigger_ns,
        "last_active_victim_completion_ns": max(active_ends),
        "last_all_victim_completion_ns": max(all_ends),
        "per_burst_timestamps": per_burst,
        "metrics_quality_status": metrics_quality.get("metrics_quality_status"),
        "nearest_eligible_pre_trigger_metrics_sample": metrics_quality.get("nearest_pre_trigger_sample"),
    }


def compute_paired_effect_summary(*, no_burst_result: dict | None, prefill_burst_result: dict | None) -> dict:
    """Return a descriptive pair summary or a fail-closed unavailable result.

    No inferential CI is attached to a single pair.  Malformed records never
    raise from this secondary path; they produce ``available: false`` with all
    safely detectable reasons.
    """
    if no_burst_result is None or prefill_burst_result is None:
        return {"available": False, "reason": "one or both episode results are missing or unreadable"}

    reasons = _validate_condition_effect_input("no_burst", no_burst_result)
    reasons.extend(_validate_condition_effect_input("prefill_burst", prefill_burst_result))
    if reasons:
        return {
            "available": False,
            "reason": "effect summary input is invalid",
            "reasons": reasons,
            "no_ci_note": "No effect estimate is computed from an invalid diagnostic pair.",
        }

    no_burst_detail = _build_condition_diagnostic_detail(no_burst_result)
    prefill_burst_detail = _build_condition_diagnostic_detail(prefill_burst_result)

    def _ratio(a: float, b: float) -> float | None:
        return (a / b) if b != 0 else None

    def _diff(a: float, b: float) -> float:
        return a - b

    return {
        "available": True,
        "no_burst": no_burst_detail,
        "prefill_burst": prefill_burst_detail,
        "burst_over_no_burst_active_median_tpot_ratio": _ratio(
            prefill_burst_detail["active_cohort_median_tpot_ms"],
            no_burst_detail["active_cohort_median_tpot_ms"],
        ),
        "burst_over_no_burst_active_median_max_itl_ratio": _ratio(
            prefill_burst_detail["active_cohort_median_max_itl_ms"],
            no_burst_detail["active_cohort_median_max_itl_ms"],
        ),
        "absolute_active_completion_delay_difference_ms": _diff(
            prefill_burst_detail["active_cohort_median_completion_from_task_creation_ms"],
            no_burst_detail["active_cohort_median_completion_from_task_creation_ms"],
        ),
        "no_ci_note": (
            "Descriptive ratios/differences for a single diagnostic pair only -- "
            "no inferential confidence interval is attached (requirement 10)."
        ),
    }


def render_diagnostic_text_summary(summary: dict) -> str:
    lines: list[str] = []
    lines.append("Server-Waiting-Confirmation Diagnostic Pair Summary")
    lines.append("=" * 60)
    lines.append(f"block_id: {summary.get('diagnostic_block_id')}")
    lines.append(f"schedule_fingerprint: {summary.get('schedule_fingerprint')}")
    lines.append(f"environment_fingerprint: {summary.get('environment_fingerprint')}")
    lines.append(f"overall_status: {summary.get('overall_status')}")
    lines.append(f"start_utc: {summary.get('start_utc')}")
    lines.append(f"end_utc: {summary.get('end_utc')}")
    lines.append("")
    lines.append(f"integrity_manifest_filename: {summary.get('integrity_manifest_filename')}")
    lines.append(f"integrity_scope: {summary.get('integrity_scope')}")
    lines.append(f"integrity_finalization_required: {summary.get('integrity_finalization_required')}")
    # These two are only present on the RETURNED (post-integrity-check)
    # summary, never on the on-disk diagnostic_pair_summary.json/.txt
    # themselves (Blocker 1 fix -- see run_diagnostic_pair docstring for
    # why that would otherwise be circular).
    if "integrity_verified" in summary:
        lines.append(f"integrity_verified (post-hoc, not part of the on-disk file): {summary.get('integrity_verified')}")
    if "diagnostic_valid" in summary:
        lines.append(f"diagnostic_valid (post-hoc, not part of the on-disk file): {summary.get('diagnostic_valid')}")
    lines.append("")

    classification = summary.get("classification") or {}
    lines.append(f"Classification: {classification.get('classification')}")
    if classification.get("reasons"):
        lines.append("Reasons:")
        for r in classification["reasons"]:
            lines.append(f"  - {r}")
    if classification.get("classification") not in (None, DIAGNOSTIC_CLASSIFICATION_D):
        lines.append(f"first_burst_output_ns: {classification.get('first_burst_output_ns')}")
        lines.append(f"last_active_victim_completion_ns: {classification.get('last_active_victim_completion_ns')}")
        lines.append(f"last_all_victim_completion_ns: {classification.get('last_all_victim_completion_ns')}")
        lines.append("")
        lines.append("Interpretation notes:")
        for note in classification.get("interpretation_notes", []):
            lines.append(f"  - {note}")
    lines.append("")

    effect = summary.get("paired_effect_summary") or {}
    if effect.get("available"):
        lines.append("Paired descriptive effect summary (n=1 pair; no inferential CI):")
        lines.append(f"  burst/no_burst active median TPOT ratio: {effect.get('burst_over_no_burst_active_median_tpot_ratio')}")
        lines.append(f"  burst/no_burst active median max-ITL ratio: {effect.get('burst_over_no_burst_active_median_max_itl_ratio')}")
        lines.append(
            f"  absolute active completion-delay difference (ms): "
            f"{effect.get('absolute_active_completion_delay_difference_ms')}"
        )
        for label in ("no_burst", "prefill_burst"):
            detail = effect.get(label) or {}
            lines.append(f"  [{label}]")
            lines.append(f"    active_cohort_median_tpot_ms: {detail.get('active_cohort_median_tpot_ms')}")
            lines.append(f"    active_cohort_median_max_itl_ms: {detail.get('active_cohort_median_max_itl_ms')}")
            lines.append(
                f"    active_cohort_median_completion_from_task_creation_ms: "
                f"{detail.get('active_cohort_median_completion_from_task_creation_ms')}"
            )
            lines.append(f"    all_20_dispatched_before_trigger: {detail.get('all_20_dispatched_before_trigger')}")
            lines.append(
                f"    all_20_streams_open_before_first_token: {detail.get('all_20_streams_open_before_first_token')}"
            )
            lines.append(f"    active_cohort_size: {detail.get('active_cohort_size')}")
            lines.append(f"    noncohort_zero_tokens_at_trigger: {detail.get('noncohort_zero_tokens_at_trigger')}")
            lines.append(f"    logical_trigger_ns: {detail.get('logical_trigger_ns')}")
            lines.append(f"    last_active_victim_completion_ns: {detail.get('last_active_victim_completion_ns')}")
            lines.append(f"    last_all_victim_completion_ns: {detail.get('last_all_victim_completion_ns')}")
            lines.append(f"    metrics_quality_status: {detail.get('metrics_quality_status')}")
    else:
        lines.append(f"Paired effect summary unavailable: {effect.get('reason')}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Self-test / CLI
# ============================================================================

def run_self_test() -> int:
    print("run_server_waiting_confirmation.py self-test")
    print("=" * 78)
    cohort_rc = cohort.run_self_test()
    print("-" * 78)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))
        print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    print("run_server_waiting_confirmation.py structural self-checks")
    check("EPISODE_FIELDS has 26 fields", len(EPISODE_FIELDS) == 26)
    check("OFFICIAL_BLOCK_COUNT == 16", OFFICIAL_BLOCK_COUNT == 16)
    check("OFFICIAL_EPISODE_COUNT == 32", OFFICIAL_EPISODE_COUNT == 32)
    check("OFFLOAD_VALUES == [0, 12]", OFFLOAD_VALUES == [0, 12])
    check("SERVER_MAX_NUM_SEQS_VALUES == [4, 8]", SERVER_MAX_NUM_SEQS_VALUES == [4, 8])
    check("MODEL_REGISTRY is qwen-only", set(MODEL_REGISTRY) == {"qwen"})
    check(
        "seed/design_version differ from the original prefill-confirmation study",
        OFFICIAL_SEED != base.OFFICIAL_SEED_BY_MODEL.get("qwen") and DESIGN_VERSION != base.DESIGN_VERSION,
    )

    # RESULT_SCHEMA_VERSION/RUNNER_VERSION independence from base module.
    check(
        "result schema/runner identity is independent of the base module's",
        RESULT_SCHEMA_VERSION != base.RESULT_SCHEMA_VERSION or RUNNER_VERSION != base.RUNNER_VERSION,
    )

    # compute_server_exposure sanity.
    exp = compute_server_exposure(
        request_dispatch_ns=100, first_token_perf_ns=150, decode_tokens_received_at_trigger=16,
        trigger_perf_ns=200, in_active_cohort=True,
    )
    check("compute_server_exposure: cohort member -> running_at_trigger_observed",
          exp["server_exposure_group"] == SERVER_EXPOSURE_RUNNING_AT_TRIGGER)
    exp2 = compute_server_exposure(
        request_dispatch_ns=100, first_token_perf_ns=None, decode_tokens_received_at_trigger=0,
        trigger_perf_ns=200, in_active_cohort=False,
    )
    check("compute_server_exposure: dispatched non-member -> dispatched_no_output_at_trigger",
          exp2["server_exposure_group"] == SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT)

    # Prometheus metrics parser sanity.
    sample_text = (
        "# HELP vllm:num_requests_running foo\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="x"} 8.0\n'
        'vllm:num_requests_waiting{model_name="x"} 12.0\n'
    )
    parsed = parse_vllm_metrics_text(sample_text)
    check("parse_vllm_metrics_text: running==8.0", parsed["running"] == 8.0, str(parsed))
    check("parse_vllm_metrics_text: waiting==12.0", parsed["waiting"] == 12.0, str(parsed))
    check("parse_vllm_metrics_text: parse_ok", parsed["parse_ok"] is True)

    unparsable = parse_vllm_metrics_text("# no matching metric lines here\n")
    check("parse_vllm_metrics_text: missing metric -> parse_ok False", unparsable["parse_ok"] is False)

    fake_sample = {
        "scrape_start_perf_ns": 0, "response_received_perf_ns": 0, "http_status": 200, "error": None,
        "parse_status": "ok", "parsed_running": parsed["running"], "parsed_waiting": parsed["waiting"],
        "raw_body": sample_text,
    }
    quality = evaluate_metrics_quality(nearest_sample=fake_sample, trigger_perf_ns=int(50e6), k=8)
    check("evaluate_metrics_quality: corroborated for matching running/waiting",
          quality["metrics_quality_status"] == METRICS_QUALITY_CORROBORATED, str(quality))

    stale_quality = evaluate_metrics_quality(nearest_sample=fake_sample, trigger_perf_ns=int(5000e6), k=8)
    check("evaluate_metrics_quality: stale for an old sample",
          stale_quality["metrics_quality_status"] == METRICS_QUALITY_STALE, str(stale_quality))

    contradictory_quality = evaluate_metrics_quality(nearest_sample=fake_sample, trigger_perf_ns=int(50e6), k=4)
    check("evaluate_metrics_quality: contradictory when running/waiting don't match K",
          contradictory_quality["metrics_quality_status"] == METRICS_QUALITY_CONTRADICTORY, str(contradictory_quality))

    unavailable_quality = evaluate_metrics_quality(nearest_sample=None, trigger_perf_ns=int(50e6), k=4)
    check("evaluate_metrics_quality: unavailable with no sample",
          unavailable_quality["metrics_quality_status"] == METRICS_QUALITY_UNAVAILABLE)

    # B2 regression: a scrape that STARTS before the trigger but whose
    # RESPONSE arrives after it must never be selected as a pre-trigger
    # sample.
    sampler_probe = MetricsSampler(transport=None, base_url="http://x", sleeper=None, clock=base.RealClock())
    sampler_probe.samples = [
        {"scrape_start_perf_ns": 0, "response_received_perf_ns": int(100e6),
         "http_status": 200, "raw_body": sample_text, "parsed_running": 8.0, "parsed_waiting": 12.0,
         "matched_running_metric_name": "vllm:num_requests_running",
         "matched_waiting_metric_name": "vllm:num_requests_waiting", "parse_status": "ok", "error": None},
    ]
    late_response_sample = sampler_probe.nearest_sample_before(int(50e6))
    check(
        "MetricsSampler.nearest_sample_before: excludes a sample whose scrape started pre-trigger "
        "but whose response arrived post-trigger",
        late_response_sample is None, str(late_response_sample),
    )
    check(
        "MetricsSampler.nearest_sample_before: the same sample IS eligible once the trigger is after its response",
        sampler_probe.nearest_sample_before(int(150e6)) is not None,
    )

    print("-" * 78)
    passed = sum(1 for _, c, _ in checks if c)
    print(f"{passed}/{len(checks)} structural checks passed")
    return 0 if (cohort_rc == 0 and passed == len(checks)) else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Run the offline self-test suite")
    mode.add_argument("--dry-run", action="store_true", help="Load/validate the bundle and print the execution plan; no I/O")
    mode.add_argument("--smoke-test", action="store_true", help="Run one real paired block (requires --smoke-block, vLLM, VLLM_API_KEY)")
    mode.add_argument(
        "--diagnostic-pair-only", action="store_true",
        help="Run exactly the qwen_off12_k8_rep01 diagnostic pair (requires --output-dir, vLLM, VLLM_API_KEY; "
             "fresh directory only, never resumable)",
    )
    mode.add_argument("--official-run", action="store_true", help="Run the full campaign (requires vLLM, VLLM_API_KEY)")
    p.add_argument("--model-key", default=MODEL_KEY, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--schedule-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--smoke-block", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--host", default=base.DEFAULT_SMOKE_HOST)
    p.add_argument("--port", type=int, default=base.DEFAULT_SMOKE_PORT)
    p.add_argument("--metrics-poll-interval-s", type=float, default=0.05)
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


async def run_with_transactional_transports(transport, metrics_transport, coro_factory: Callable[[], Any]) -> Any:
    """N1 fix (2026-07-20 second/third hardening pass): transactional
    transport startup. Previously, `_start_transports()` was called
    OUTSIDE the try/finally in `main()`, so if `transport.start()`
    succeeded but `metrics_transport.start()` then raised, the
    already-started completion transport was never closed (the whole
    `_start_transports()` call raised before the try block was even
    entered). Here, each transport's own started-flag gates whether its
    `aclose()` is attempted.

    Blocker 3 fix (2026-07-20 fourth hardening pass): the two `aclose()`
    calls were previously coupled serially (`await metrics_transport.aclose()`
    directly followed by `await transport.aclose()` in the same
    unguarded `finally` block) -- if the FIRST call raised, the SECOND
    was never reached, violating the required best-effort-cleanup
    guarantee. Each close is now wrapped in its own try/except so a
    failure in either can NEVER suppress the other's close attempt.
    Both errors (if any) are collected, written to stderr for
    visibility, and -- if the underlying call actually produced a
    result (i.e. `coro_factory()` itself did not raise) and that result
    is a dict -- attached to it as `transport_close_errors`, so CLI
    output surfaces them too. If `coro_factory()` itself raised, that
    original exception still propagates unchanged (close errors are
    logged to stderr only in that path, since there is no successful
    result to attach them to)."""
    started_completion = False
    started_metrics = False
    close_errors: list[dict[str, str]] = []
    _no_result = object()
    result: Any = _no_result
    try:
        await transport.start()
        started_completion = True
        await metrics_transport.start()
        started_metrics = True
        result = await coro_factory()
        return result
    finally:
        if started_metrics:
            try:
                await metrics_transport.aclose()
            except BaseException as exc:  # noqa: BLE001 -- must never suppress the completion-transport close below
                close_errors.append({
                    "transport": "metrics_transport", "error_type": type(exc).__name__, "error_message": str(exc),
                })
        if started_completion:
            try:
                await transport.aclose()
            except BaseException as exc:  # noqa: BLE001 -- must never suppress reporting the metrics-transport error above
                close_errors.append({
                    "transport": "completion_transport", "error_type": type(exc).__name__, "error_message": str(exc),
                })
        for entry in close_errors:
            sys.stderr.write(
                f"Warning: {entry['transport']} failed to close cleanly: "
                f"{entry['error_type']}: {entry['error_message']}\n"
            )
        if close_errors and isinstance(result, dict):
            result.setdefault("transport_close_errors", []).extend(close_errors)



def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    schedule_dir = args.schedule_dir if args.schedule_dir is not None else default_schedule_dir(args.model_key)

    if args.dry_run:
        return run_dry_run(schedule_dir, args.model_key)

    if args.diagnostic_pair_only and args.resume:
        sys.stderr.write("Fehler: --resume wird von --diagnostic-pair-only nicht unterstützt (fresh-directory-only).\n")
        return 1
    if args.diagnostic_pair_only and args.output_dir is None:
        sys.stderr.write("Fehler: --diagnostic-pair-only erfordert ein explizit angegebenes --output-dir.\n")
        return 1

    # --smoke-test, --diagnostic-pair-only, and --official-run all need a
    # real vLLM server, a real tokenizer, and VLLM_API_KEY; the assistant
    # session never exercises these paths (--self-test/--dry-run/the
    # offline test suite's fake-server integration tests cover this code
    # instead).
    bundle, errors = load_and_validate_bundle(schedule_dir, args.model_key)
    if bundle is None:
        sys.stderr.write("Schedule bundle FAILED validation:\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        return 1

    api_key = read_api_key_from_env()
    env = RealEnvironmentProbe().gather(schedule_dir)
    env_errors = validate_official_environment(env)
    if env_errors:
        sys.stderr.write("Environment FAILED the official pre-flight gate:\n")
        for e in env_errors:
            sys.stderr.write(f"  - {e}\n")
        return 1

    run_server_path = resolve_run_server_path()
    check_run_server_script(run_server_path)
    tokenizer = HFTokenizerAdapter(MODEL_REGISTRY[args.model_key]["model_id"])

    # Requirements 1/4: one persistent completion-pool transport (32/32
    # connection limits) and one SEPARATE persistent metrics transport,
    # both real for every non-offline mode below.
    transport = PersistentCompletionTransport()
    metrics_transport = PersistentMetricsTransport()

    # Audit2 B2 fix / Audit1 N1 fix (2026-07-20 second hardening pass):
    # ALL real modes (smoke-test, diagnostic-pair-only, official-run) now
    # share ONE signal-handling + transactional-transport-startup wrapper
    # -- previously only --official-run installed SIGINT/SIGTERM handlers
    # and cleanup was not transactional (a completion-transport start
    # succeeding while metrics-transport start then failed would leak the
    # already-started completion client, because _start_transports() was
    # called OUTSIDE the try/finally).
    interrupt_state = InterruptState()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handler(signum, _frame):
        interrupt_state.trigger(signal.Signals(signum).name)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig, None)
        except (NotImplementedError, RuntimeError):
            pass  # platform without loop signal handlers; Ctrl-C still raises KeyboardInterrupt

    async def _run_with_transports(coro_factory: Callable[[], Any]) -> dict:
        return await run_with_transactional_transports(transport, metrics_transport, coro_factory)

    try:
        if args.smoke_test:
            if not args.smoke_block:
                sys.stderr.write("Fehler: --smoke-test erfordert --smoke-block <block_id>\n")
                return 1
            output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.model_key, RUN_MODE_SMOKE)

            async def _factory() -> dict:
                return await run_server_waiting_smoke_block(
                    bundle=bundle, block_id=args.smoke_block, output_dir=output_dir, host=args.host, port=args.port,
                    resume=args.resume, api_key=api_key, transport=transport, metrics_transport=metrics_transport,
                    tokenizer=tokenizer, server_adapter=RealServerProcessAdapter(), sleeper=RealSleeper(),
                    clock=RealClock(), run_server_path=run_server_path, interrupt_state=interrupt_state,
                    metrics_poll_interval_s=args.metrics_poll_interval_s,
                )

            summary = loop.run_until_complete(_run_with_transports(_factory))
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))
            return 0 if summary.get("overall_status") in ("block_complete", "already_complete") else 1

        if args.diagnostic_pair_only:
            output_dir = args.output_dir

            async def _factory() -> dict:
                return await run_diagnostic_pair(
                    bundle=bundle, output_dir=output_dir, host=args.host, port=args.port, api_key=api_key,
                    transport=transport, metrics_transport=metrics_transport, tokenizer=tokenizer,
                    server_adapter=RealServerProcessAdapter(), sleeper=RealSleeper(), clock=RealClock(),
                    run_server_path=run_server_path, env=env, interrupt_state=interrupt_state,
                    metrics_poll_interval_s=args.metrics_poll_interval_s,
                )

            summary = loop.run_until_complete(_run_with_transports(_factory))
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))
            # N4 fix: a technically-complete block whose scientific
            # classification is D_AMBIGUOUS_OR_INVALID, or whose final
            # integrity manifest failed to verify, must never exit 0 --
            # automation must not be able to mistake it for a genuine
            # A/B/C pass. Exit codes: 0 = ran AND scientifically valid;
            # 2 = the block ran to completion but is not
            # `diagnostic_valid` (D-classified or integrity failure);
            # 1 = the block itself did not complete (server/stabilization/
            # request failure, interruption, etc.).
            if summary.get("overall_status") not in ("block_complete", "integrity_verification_failed"):
                return 1
            return 0 if summary.get("diagnostic_valid") else 2

        if args.official_run:
            output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.model_key, RUN_MODE_OFFICIAL)

            async def _factory() -> dict:
                return await run_official_campaign(
                    bundle=bundle, output_dir=output_dir, host=args.host, port=args.port, resume=args.resume,
                    api_key=api_key, transport=transport, metrics_transport=metrics_transport, tokenizer=tokenizer,
                    server_adapter=RealServerProcessAdapter(), sleeper=RealSleeper(), clock=RealClock(),
                    run_server_path=run_server_path, env=env, interrupt_state=interrupt_state,
                    metrics_poll_interval_s=args.metrics_poll_interval_s,
                )

            summary = loop.run_until_complete(_run_with_transports(_factory))
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))
            return 0 if summary.get("overall_status") == "complete" else 1

        return 1
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
