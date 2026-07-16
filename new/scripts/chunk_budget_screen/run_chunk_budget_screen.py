#!/usr/bin/env python3
"""
run_chunk_budget_screen.py -- frozen exploratory Chunk-Budget-Screen runner.

This runner executes a fully separate 36-episode screening campaign:
  - model: Llama-3.1-8B-Instruct
  - states: cpu_offload_gb 0 (low) and 12 (high)
  - chunked-prefill budgets (max_num_batched_tokens): 512, 1024, 2048
  - victim concurrency: 4 (fixed)
  - conditions: no_burst and prefill_burst
  - repeats: 3
  - prefill burst: 4 parallel requests, 2048 input tokens, 16 output tokens

Each of the eighteen (state, budget) blocks follows the same hardened
protocol used by the frozen Phase-A/Prefill-Screen infrastructure:
server start (with this block's max_num_batched_tokens), readiness
validation, a server-log check confirming chunked prefill is actually
enabled with the expected budget, one excluded stabilization run,
health gate, fixed cooldown, two regular episodes in schedule order,
atomic outputs, and verified process-group shutdown with port-release
polling.

The generator is not imported at runtime. The operative bundle consists
of exactly:
  chunk_budget_schedule.json
  chunk_budget_schedule.csv
  chunk_budget_schedule_audit.txt

CLI modes:
  --self-test     Run focused no-GPU/no-network contract and lifecycle tests.
  --dry-run       Validate the complete bundle and print the execution plan.
  --smoke-test    Execute exactly one selected two-episode block.
  --official-run Execute all eighteen blocks / 36 regular episodes.

Examples:
  python3 run_chunk_budget_screen.py --self-test
  python3 run_chunk_budget_screen.py --dry-run \
      --schedule-dir /path/to/new/runs/chunk_budget_screen
  VLLM_API_KEY=... GPU_DEVICE=0 python3 run_chunk_budget_screen.py --official-run \
      --schedule-dir /path/to/new/runs/chunk_budget_screen \
      --output-dir /path/to/new/runs/chunk_budget_screen/results/official
  VLLM_API_KEY=... GPU_DEVICE=0 python3 run_chunk_budget_screen.py --smoke-test \
      --smoke-block llama_block01_low_budget512 \
      --schedule-dir /path/to/new/runs/chunk_budget_screen \
      --output-dir /path/to/new/runs/chunk_budget_screen/results/smoke

The API key is read only from the environment on real execution paths.
It is never accepted as a CLI argument, written to result artifacts, or
included in fingerprints.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Protocol, Sequence, runtime_checkable


# ============================================================================
# Deterministic seed derivation (local, self-contained -- the generator is
# not imported at runtime; see module docstring)
# ============================================================================

def derive_seed(*parts: str) -> int:
    joined = ":".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


# ============================================================================
# Frozen Schema v2 episode schema (25 fields, exact set in both directions)
# ============================================================================

EPISODE_FIELDS: tuple[str, ...] = (
    "design_version",
    "schedule_seed",
    "block_id",
    "block_index",
    "episode_id",
    "state_label",
    "offload_gb",
    "max_num_batched_tokens",
    "concurrency",
    "condition",
    "repeat",
    "within_block_order",
    "model",
    "victim_input_len",
    "victim_output_len",
    "victim_request_count",
    "victim_temperature",
    "burst_parallel",
    "burst_input_len",
    "burst_output_len",
    "burst_temperature",
    "episode_seed",
    "victim_workload_seed",
    "burst_workload_seed",
    "restart_server_before_block",
)

EPISODE_FIELD_TYPES: dict[str, type] = {
    "design_version": str,
    "schedule_seed": int,
    "block_id": str,
    "block_index": int,
    "episode_id": str,
    "state_label": str,
    "offload_gb": int,
    "max_num_batched_tokens": int,
    "concurrency": int,
    "condition": str,
    "repeat": int,
    "within_block_order": int,
    "model": str,
    "victim_input_len": int,
    "victim_output_len": int,
    "victim_request_count": int,
    "victim_temperature": float,
    "burst_parallel": int,
    "burst_input_len": int,
    "burst_output_len": int,
    "burst_temperature": float,
    "episode_seed": int,
    "victim_workload_seed": int,
    "burst_workload_seed": int,
    "restart_server_before_block": int,
}


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


# ============================================================================
# Frozen official Chunk-Budget-Screen contract
#
# This is a fully separate, explorative screening design -- mechanically
# adapted from the frozen Prefill-Screen contract, but with its own
# block dimension (chunked-prefill budget), fixed concurrency, and
# fingerprint. It does not read, write, or share any file with
# new/runs/prefill_screen/ or new/runs/phase_a/.
# ============================================================================

SCHEMA_VERSION = 2
DESIGN_VERSION = "chunk-budget-screen-v1"
OFFICIAL_FINGERPRINT = (
    "sha256:3c35a72ee08e289258c254be6be9566d299b95ba8a2b38817be36bd4c79459cd"
)
OFFICIAL_SEED = 20260716
OFFICIAL_EPISODE_COUNT = 36
OFFICIAL_MODELS = ["llama"]
OFFICIAL_REPEATS = 3
OFFICIAL_BUDGETS = [512, 1024, 2048]
OFFICIAL_CONCURRENCY = 4
# Prefill-heavy, bounded burst condition -- unchanged from Prefill-Screen.
# Kept as a single named constant so every burst-shape/-count/-interval
# check below derives from one place.
BURST_CONDITION = "prefill_burst"
OFFICIAL_CONDITIONS = ["no_burst", BURST_CONDITION]
OFFICIAL_STATES = [
    {"offload_gb": 0, "state_label": "low"},
    {"offload_gb": 12, "state_label": "high"},
]
STATE_LABEL_BY_OFFLOAD: dict[int, str] = {0: "low", 12: "high"}

OFFICIAL_VICTIM_CONFIGURATION = {
    "victim_request_count": 20,
    "victim_input_len": 256,
    "victim_output_len": 64,
    "victim_temperature": 0.0,
}
# Prefill-heavy, bounded burst: large input, small output -- unchanged
# from Prefill-Screen; only the server's chunked-prefill budget varies
# in this screen.
OFFICIAL_BURST_CONFIGURATION = {
    "burst_parallel": 4,
    "burst_input_len": 2048,
    "burst_output_len": 16,
    "burst_temperature": 0.0,
}
OFFICIAL_STABILIZATION_CONFIGURATION = {
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
    "chunked_prefill_log_verification_required": True,
}

# 18 blocks total, 6 per repeat: low/high blocks interleaved
# budget-by-budget within each repeat (never 6, or even 3, consecutive
# blocks of the same state), and budgets rotated per repeat via a
# Latin square so each budget occupies each within-repeat position
# (1st/2nd/3rd) exactly once across the 3 repeats, independently per
# state. See make_chunk_budget_schedule.py's module docstring for the
# full rationale -- this function independently re-derives the exact
# same rotation from the same formula, not by importing the generator.


def budget_order_for_repeat(repeat: int) -> tuple[int, ...]:
    n = len(OFFICIAL_BUDGETS)
    shift = (repeat - 1) % n
    return tuple(OFFICIAL_BUDGETS[(i + shift) % n] for i in range(n))


BLOCK_SIZE = len(OFFICIAL_CONDITIONS)  # 2
BLOCKS_PER_MODEL = len(OFFICIAL_STATES) * len(OFFICIAL_BUDGETS) * OFFICIAL_REPEATS  # 18
EPISODES_PER_MODEL = BLOCK_SIZE * BLOCKS_PER_MODEL  # 36
TOTAL_RESTART_MARKERS = BLOCKS_PER_MODEL * len(OFFICIAL_MODELS)  # 18

REQUIRED_BUNDLE_FILENAMES = (
    "chunk_budget_schedule.json",
    "chunk_budget_schedule.csv",
    "chunk_budget_schedule_audit.txt",
)

RUN_MODE_MARKER_FILENAME = ".chunk_budget_screen_run_mode"


# ============================================================================
# Portable paths (derived from this file's own location, not hardcoded)
# Expected location: <PROJECT_ROOT>/new/scripts/chunk_budget_screen/run_chunk_budget_screen.py
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SCHEDULE_DIR = PROJECT_ROOT / "new" / "runs" / "chunk_budget_screen"


def default_output_dir(mode: str) -> Path:
    return PROJECT_ROOT / "new" / "runs" / "chunk_budget_screen" / "results" / mode


# ============================================================================
# Exceptions
# ============================================================================

class BundleLoadError(Exception):
    """Fatal, unrecoverable bundle loading failure (missing/unreadable
    file, malformed JSON/CSV). Further structural validation would be
    meaningless once this fires."""


class OutputDirConflictError(Exception):
    """The requested output directory is already owned by a different
    run mode (official vs smoke)."""


# ============================================================================
# Bundle file discovery & raw loading
# ============================================================================

def find_bundle_paths(schedule_dir: Path) -> dict[str, Path]:
    if not schedule_dir.is_dir():
        raise BundleLoadError(
            f"--schedule-dir '{schedule_dir}' does not exist or is not a "
            f"directory"
        )
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name in REQUIRED_BUNDLE_FILENAMES:
        candidate = schedule_dir / name
        if not candidate.is_file() or not os.access(candidate, os.R_OK):
            missing.append(name)
        else:
            paths[name] = candidate
    if missing:
        raise BundleLoadError(
            f"--schedule-dir '{schedule_dir}' is missing required, "
            f"readable, regular file(s): {missing}"
        )
    return paths


def load_json_bundle(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleLoadError(f"could not read '{path}': {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleLoadError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BundleLoadError(
            f"'{path}' does not contain a JSON object at the top level"
        )
    return obj


def load_csv_bundle(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleLoadError(f"could not read '{path}': {exc}") from exc
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    except csv.Error as exc:
        raise BundleLoadError(f"'{path}' is not valid CSV: {exc}") from exc
    return fieldnames, rows


def load_audit_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleLoadError(f"could not read '{path}': {exc}") from exc


# ============================================================================
# Canonical fingerprint (same documented derivation as the generator, but
# re-implemented locally -- no generator import)
# ============================================================================

_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_valid_fingerprint_format(fp: object) -> bool:
    return isinstance(fp, str) and bool(_FINGERPRINT_RE.match(fp))


def recompute_fingerprint(json_obj: dict) -> str:
    canonical = {k: v for k, v in json_obj.items() if k != "schedule_fingerprint"}
    serialized = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ============================================================================
# Strict episode schema checks (exact field set in both directions)
# ============================================================================

def _check_type_strict(value: object, expected_type: type) -> bool:
    # `type(x) is T` (not isinstance) so that bool is never silently
    # accepted where int is expected (bool is a subclass of int in
    # Python), and so that int is never silently accepted where float
    # is expected.
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
        errors.append(
            f"episode index {index}: unexpected extra field(s): {sorted(extra)}"
        )
    if missing or extra:
        return errors

    for field in EPISODE_FIELDS:
        if not _check_type_strict(obj[field], EPISODE_FIELD_TYPES[field]):
            errors.append(
                f"episode index {index} ({field}): wrong type, expected "
                f"{EPISODE_FIELD_TYPES[field].__name__}, got "
                f"{type(obj[field]).__name__} ({obj[field]!r})"
            )
    return errors


def parse_episode_from_json(obj: dict) -> Episode:
    return Episode(**{field: obj[field] for field in EPISODE_FIELDS})


def check_csv_header(fieldnames: list[str]) -> list[str]:
    if fieldnames != list(EPISODE_FIELDS):
        return [
            f"csv header {fieldnames} does not match the expected frozen "
            f"field order {list(EPISODE_FIELDS)}"
        ]
    return []


def _normalize_csv_value(raw: str, expected_type: type) -> object:
    if expected_type is int:
        return int(raw)
    if expected_type is float:
        return float(raw)
    return raw


def normalize_csv_row(
    row: dict[str, str], index: int
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    normalized: dict[str, object] = {}
    for field in EPISODE_FIELDS:
        raw = row.get(field)
        if raw is None:
            errors.append(f"csv row index {index}: missing field {field!r}")
            continue
        try:
            normalized[field] = _normalize_csv_value(raw, EPISODE_FIELD_TYPES[field])
        except (TypeError, ValueError) as exc:
            errors.append(
                f"csv row index {index} ({field}): could not parse "
                f"{raw!r} as {EPISODE_FIELD_TYPES[field].__name__}: {exc}"
            )
    return normalized, errors


def check_csv_json_consistency(
    csv_fieldnames: list[str],
    csv_rows: list[dict[str, str]],
    json_episodes: list[dict],
) -> list[str]:
    """
    Re-parses the actual CSV rows (via csv.DictReader, already done by the
    caller) and compares them field-by-field, in list order, against the
    actual JSON episode objects -- header, row count, field names, and
    values, all independently.
    """
    errors: list[str] = []
    errors.extend(check_csv_header(csv_fieldnames))

    if len(csv_rows) != OFFICIAL_EPISODE_COUNT:
        errors.append(
            f"csv has {len(csv_rows)} row(s), expected exactly "
            f"{OFFICIAL_EPISODE_COUNT}"
        )
    if len(json_episodes) != OFFICIAL_EPISODE_COUNT:
        errors.append(
            f"json has {len(json_episodes)} episode(s), expected exactly "
            f"{OFFICIAL_EPISODE_COUNT}"
        )
    if len(csv_rows) != len(json_episodes):
        errors.append(
            f"csv/json episode count mismatch: csv={len(csv_rows)}, "
            f"json={len(json_episodes)}"
        )
        return errors

    for idx, (csv_row, json_row) in enumerate(zip(csv_rows, json_episodes)):
        normalized_csv, row_errors = normalize_csv_row(csv_row, idx)
        errors.extend(row_errors)
        if not isinstance(json_row, dict):
            errors.append(f"episode index {idx}: json row is not an object")
            continue
        for field in EPISODE_FIELDS:
            if field not in normalized_csv or field not in json_row:
                continue
            if normalized_csv[field] != json_row[field]:
                errors.append(
                    f"episode index {idx} ({field}): csv value "
                    f"{normalized_csv[field]!r} != json value "
                    f"{json_row[field]!r}"
                )
    return errors


# ============================================================================
# Audit report checks
# ============================================================================

_FP_LINE_RE = re.compile(r"^schedule_fingerprint:\s*(\S+)\s*$", re.MULTILINE)


def check_audit(audit_text: str, expected_fingerprint: object) -> list[str]:
    errors: list[str] = []

    matches = _FP_LINE_RE.findall(audit_text)
    if len(matches) == 0:
        errors.append("audit report has no 'schedule_fingerprint: ...' line")
    elif len(matches) > 1:
        errors.append(
            f"audit report has {len(matches)} 'schedule_fingerprint: ...' "
            f"lines, expected exactly 1"
        )
    elif matches[0] != expected_fingerprint:
        errors.append(
            f"audit report schedule_fingerprint {matches[0]!r} does not "
            f"match the JSON's schedule_fingerprint {expected_fingerprint!r}"
        )

    if not re.search(r"^schema_version:\s*2\s*$", audit_text, re.MULTILINE):
        errors.append("audit report does not contain 'schema_version: 2'")
    if not re.search(
        r"^design_version:\s*chunk-budget-screen-v1\s*$", audit_text, re.MULTILINE
    ):
        errors.append(
            "audit report does not contain 'design_version: chunk-budget-screen-v1'"
        )
    if not re.search(
        r"^total episodes \(all models\):\s*36\s*$", audit_text, re.MULTILINE
    ):
        errors.append(
            "audit report does not contain 'total episodes (all models): 36'"
        )
    if not re.search(r"^OVERALL:\s*PASS\s*$", audit_text, re.MULTILINE):
        errors.append("audit report does not contain 'OVERALL: PASS'")

    return errors


# ============================================================================
# Official top-level contract check
# ============================================================================

def check_official_contract(json_obj: dict) -> list[str]:
    errors: list[str] = []

    def check_eq(key: str, expected: object) -> None:
        actual = json_obj.get(key, "<MISSING>")
        if actual != expected:
            errors.append(f"{key} = {actual!r}, expected {expected!r}")

    check_eq("schema_version", SCHEMA_VERSION)
    check_eq("design_version", DESIGN_VERSION)
    check_eq("seed", OFFICIAL_SEED)
    check_eq("repeats", OFFICIAL_REPEATS)
    check_eq("models", OFFICIAL_MODELS)
    check_eq("budgets", OFFICIAL_BUDGETS)
    check_eq("concurrency", OFFICIAL_CONCURRENCY)
    check_eq("conditions", OFFICIAL_CONDITIONS)
    check_eq("states", OFFICIAL_STATES)
    check_eq("victim_configuration", OFFICIAL_VICTIM_CONFIGURATION)
    check_eq("burst_configuration", OFFICIAL_BURST_CONFIGURATION)
    check_eq("stabilization_configuration", OFFICIAL_STABILIZATION_CONFIGURATION)
    check_eq("episode_count", OFFICIAL_EPISODE_COUNT)

    fp = json_obj.get("schedule_fingerprint")
    if not is_valid_fingerprint_format(fp):
        errors.append(f"schedule_fingerprint has invalid format: {fp!r}")
    else:
        recomputed = recompute_fingerprint(json_obj)
        if recomputed != fp:
            errors.append(
                f"recomputed schedule_fingerprint {recomputed!r} does not "
                f"match the stored schedule_fingerprint {fp!r}"
            )
        if fp != OFFICIAL_FINGERPRINT:
            errors.append(
                f"schedule_fingerprint {fp!r} does not match the frozen "
                f"Chunk-Budget-Screen fingerprint {OFFICIAL_FINGERPRINT!r}"
            )

    return errors


# ============================================================================
# Structural schedule validation (independent of the fingerprint)
# ============================================================================

def _check_model_structure(
    model: str,
    episodes: list[Episode],
    schedule_seed: object,
    *,
    repeats: int,
    episodes_per_model: int,
    blocks_per_model: int,
) -> list[str]:
    errors: list[str] = []

    if len(episodes) != episodes_per_model:
        errors.append(
            f"model={model}: expected {episodes_per_model} episodes, "
            f"found {len(episodes)}"
        )

    # --- per-episode exact value + deterministic re-derivation checks -----
    for ep in episodes:
        ctx = f"model={model}, episode={ep.episode_id}"

        if ep.model != model:
            errors.append(f"{ctx}: episode.model {ep.model!r} != {model!r}")
        if ep.design_version != DESIGN_VERSION:
            errors.append(
                f"{ctx}: design_version {ep.design_version!r} != "
                f"{DESIGN_VERSION!r}"
            )

        expected_state_label = STATE_LABEL_BY_OFFLOAD.get(ep.offload_gb)
        if expected_state_label is None:
            errors.append(f"{ctx}: invalid offload_gb {ep.offload_gb!r}")
        elif ep.state_label != expected_state_label:
            errors.append(
                f"{ctx}: state_label {ep.state_label!r} != expected "
                f"{expected_state_label!r} for offload_gb {ep.offload_gb}"
            )

        if ep.max_num_batched_tokens not in OFFICIAL_BUDGETS:
            errors.append(
                f"{ctx}: invalid max_num_batched_tokens "
                f"{ep.max_num_batched_tokens!r}"
            )
        if ep.concurrency != OFFICIAL_CONCURRENCY:
            errors.append(
                f"{ctx}: concurrency {ep.concurrency!r} != frozen value "
                f"{OFFICIAL_CONCURRENCY!r}"
            )
        if ep.condition not in OFFICIAL_CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.schedule_seed != schedule_seed:
            errors.append(
                f"{ctx}: schedule_seed {ep.schedule_seed!r} != schedule "
                f"seed {schedule_seed!r}"
            )

        if ep.victim_request_count != OFFICIAL_VICTIM_CONFIGURATION["victim_request_count"]:
            errors.append(f"{ctx}: victim_request_count != 20")
        if ep.victim_input_len != OFFICIAL_VICTIM_CONFIGURATION["victim_input_len"]:
            errors.append(f"{ctx}: victim_input_len != 256")
        if ep.victim_output_len != OFFICIAL_VICTIM_CONFIGURATION["victim_output_len"]:
            errors.append(f"{ctx}: victim_output_len != 64")
        if ep.victim_temperature != OFFICIAL_VICTIM_CONFIGURATION["victim_temperature"]:
            errors.append(f"{ctx}: victim_temperature != 0.0")

        if ep.burst_parallel != OFFICIAL_BURST_CONFIGURATION["burst_parallel"]:
            errors.append(f"{ctx}: burst_parallel != 4")
        if ep.burst_input_len != OFFICIAL_BURST_CONFIGURATION["burst_input_len"]:
            errors.append(f"{ctx}: burst_input_len != {OFFICIAL_BURST_CONFIGURATION['burst_input_len']}")
        if ep.burst_output_len != OFFICIAL_BURST_CONFIGURATION["burst_output_len"]:
            errors.append(f"{ctx}: burst_output_len != {OFFICIAL_BURST_CONFIGURATION['burst_output_len']}")
        if ep.burst_temperature != OFFICIAL_BURST_CONFIGURATION["burst_temperature"]:
            errors.append(f"{ctx}: burst_temperature != 0.0")

        if ep.within_block_order == 1:
            if ep.restart_server_before_block != 1:
                errors.append(
                    f"{ctx}: within_block_order=1 requires "
                    f"restart_server_before_block==1"
                )
        else:
            if ep.restart_server_before_block != 0:
                errors.append(
                    f"{ctx}: within_block_order={ep.within_block_order} "
                    f"requires restart_server_before_block==0"
                )

        expected_episode_id = (
            f"{model}_off{ep.offload_gb}_budget{ep.max_num_batched_tokens}_"
            f"conc{ep.concurrency}_{ep.condition}_rep{ep.repeat}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(
                f"{ctx}: episode_id does not match the expected "
                f"derivation {expected_episode_id!r}"
            )

        expected_episode_seed = derive_seed(str(schedule_seed), ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(
                f"{ctx}: episode_seed does not match "
                f"derive_seed(seed, episode_id)"
            )

        expected_victim_seed = derive_seed(str(schedule_seed), model, str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(
                f"{ctx}: victim_workload_seed does not match the "
                f"expected derivation"
            )

        expected_burst_seed = derive_seed(str(schedule_seed), model, str(ep.repeat), "burst")
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(
                f"{ctx}: burst_workload_seed does not match the "
                f"expected derivation"
            )

    # --- exact repeat set per design cell -----------------------------------
    repeats_by_cell: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.max_num_batched_tokens, ep.condition)
        repeats_by_cell.setdefault(key, set()).add(ep.repeat)

    expected_repeat_set = set(range(1, repeats + 1))
    expected_cells = {
        (offload_gb, budget, condition)
        for offload_gb in STATE_LABEL_BY_OFFLOAD
        for budget in OFFICIAL_BUDGETS
        for condition in OFFICIAL_CONDITIONS
    }
    for key in sorted(expected_cells):
        actual = repeats_by_cell.get(key, set())
        if actual != expected_repeat_set:
            errors.append(
                f"model={model}: cell {key} has repeat value(s) "
                f"{sorted(actual)}, expected exactly "
                f"{sorted(expected_repeat_set)}"
            )

    # --- contiguous block / execution-order checks --------------------------
    # The order of `episodes` IS the intended execution order: a block must
    # be exactly BLOCK_SIZE (2) immediately consecutive episodes, a
    # block_id must never reappear once left, and within_block_order /
    # restart_server_before_block must match 1,2 / 1,0 in that exact list
    # order.
    seen_block_ids: set[str] = set()
    idx = 0
    n = len(episodes)
    block_position = 0
    blocks_per_repeat = len(STATE_LABEL_BY_OFFLOAD) * len(OFFICIAL_BUDGETS)

    # Independently re-derive the exact block sequence the generator
    # produces: Latin-square budget rotation per repeat, low/high blocks
    # interleaved budget-by-budget within each repeat (see
    # make_chunk_budget_schedule.py's module docstring / budget_order_for_repeat
    # above -- same formula, not imported).
    expected_state_budget_sequence: list[tuple[str, int]] = []
    for r in range(1, repeats + 1):
        order = OFFICIAL_STATES if r % 2 == 1 else list(reversed(OFFICIAL_STATES))
        for budget in budget_order_for_repeat(r):
            for state_entry in order:
                expected_state_budget_sequence.append((state_entry["state_label"], budget))

    while idx < n:
        bid = episodes[idx].block_id
        if bid in seen_block_ids:
            errors.append(
                f"model={model}: block_id {bid!r} reappears at a later, "
                f"non-contiguous position in the episode list"
            )
        seen_block_ids.add(bid)

        run_end = idx
        while run_end < n and episodes[run_end].block_id == bid:
            run_end += 1
        run_episodes = episodes[idx:run_end]
        block_position += 1

        if len(run_episodes) != BLOCK_SIZE:
            errors.append(
                f"model={model}: contiguous block {bid!r} has "
                f"{len(run_episodes)} immediately consecutive episode(s), "
                f"expected exactly {BLOCK_SIZE}"
            )
        else:
            order_sequence = [ep.within_block_order for ep in run_episodes]
            expected_order_sequence = list(range(1, BLOCK_SIZE + 1))
            if order_sequence != expected_order_sequence:
                errors.append(
                    f"model={model}: block {bid!r} within_block_order "
                    f"sequence {order_sequence} in list order != "
                    f"{expected_order_sequence}"
                )

            restart_sequence = [
                ep.restart_server_before_block for ep in run_episodes
            ]
            expected_restart_sequence = [1] + [0] * (BLOCK_SIZE - 1)
            if restart_sequence != expected_restart_sequence:
                errors.append(
                    f"model={model}: block {bid!r} "
                    f"restart_server_before_block sequence "
                    f"{restart_sequence} in list order != "
                    f"{expected_restart_sequence}"
                )

            state_labels = {ep.state_label for ep in run_episodes}
            offloads = {ep.offload_gb for ep in run_episodes}
            budgets_in_block = {ep.max_num_batched_tokens for ep in run_episodes}
            repeats_in_block = {ep.repeat for ep in run_episodes}
            block_indices = {ep.block_index for ep in run_episodes}

            if len(state_labels) != 1 or len(offloads) != 1 or len(budgets_in_block) != 1:
                errors.append(
                    f"model={model}: block {bid!r} mixes state/offload/"
                    f"budget values: states={sorted(state_labels)}, "
                    f"offloads={sorted(offloads)}, "
                    f"budgets={sorted(budgets_in_block)}"
                )
            if len(block_indices) != 1 or next(iter(block_indices)) != block_position:
                errors.append(
                    f"model={model}: block {bid!r} has block_index "
                    f"{sorted(block_indices)}, expected exactly "
                    f"{{{block_position}}}"
                )

            expected_repeat_for_block = ((block_position - 1) // blocks_per_repeat) + 1
            if repeats_in_block != {expected_repeat_for_block}:
                errors.append(
                    f"model={model}: block {bid!r} (position "
                    f"{block_position}) has repeat value(s) "
                    f"{sorted(repeats_in_block)}, expected exactly "
                    f"{{{expected_repeat_for_block}}}"
                )

            if block_position <= len(expected_state_budget_sequence) and len(
                state_labels
            ) == 1:
                expected_state, expected_budget = expected_state_budget_sequence[block_position - 1]
                actual_state = next(iter(state_labels))
                if actual_state != expected_state:
                    errors.append(
                        f"model={model}: block at position "
                        f"{block_position} has state {actual_state!r}, "
                        f"expected {expected_state!r}"
                    )
                actual_budget = next(iter(budgets_in_block)) if budgets_in_block else None
                if actual_budget != expected_budget:
                    errors.append(
                        f"model={model}: block at position "
                        f"{block_position} has budget {actual_budget!r}, "
                        f"expected {expected_budget!r} (Latin-square rotation)"
                    )
                expected_block_id = (
                    f"{model}_block{block_position:02d}_{expected_state}_"
                    f"budget{expected_budget}"
                )
                if bid != expected_block_id:
                    errors.append(
                        f"model={model}: block at position "
                        f"{block_position} has block_id {bid!r}, "
                        f"expected {expected_block_id!r}"
                    )


        idx = run_end

    if block_position != blocks_per_model:
        errors.append(
            f"model={model}: expected {blocks_per_model} blocks, found "
            f"{block_position}"
        )

    # --- budget-order-within-(repeat,state) check ----------------------------
    # Within each repeat, filtering to just one state's (non-contiguous,
    # interleaved) blocks, the budgets encountered in schedule order must
    # equal that repeat's Latin-square rotation.
    blocks_seen: list[tuple[str, int, str, int]] = []  # (block_id, repeat, state, budget)
    idx = 0
    n2 = len(episodes)
    while idx < n2:
        bid = episodes[idx].block_id
        run_end = idx
        while run_end < n2 and episodes[run_end].block_id == bid:
            run_end += 1
        ep0 = episodes[idx]
        blocks_seen.append((bid, ep0.repeat, ep0.state_label, ep0.max_num_batched_tokens))
        idx = run_end

    for r in range(1, repeats + 1):
        expected_budget_order = list(budget_order_for_repeat(r))
        for state_entry in OFFICIAL_STATES:
            state_label = state_entry["state_label"]
            budgets_here = [
                entry[3] for entry in blocks_seen if entry[1] == r and entry[2] == state_label
            ]
            if budgets_here != expected_budget_order:
                errors.append(
                    f"model={model}: repeat={r} state={state_label!r} has "
                    f"budget order {budgets_here}, expected the "
                    f"Latin-square rotation {expected_budget_order}"
                )

    # --- budget-position balance check ---------------------------------------
    # Each budget must occupy each of the 3 within-repeat positions exactly
    # once across the repeats, independently per state.
    budget_position_by_state: dict[str, dict[int, list[int]]] = {
        state_entry["state_label"]: {budget: [] for budget in OFFICIAL_BUDGETS}
        for state_entry in OFFICIAL_STATES
    }
    for r in range(1, repeats + 1):
        for state_entry in OFFICIAL_STATES:
            state_label = state_entry["state_label"]
            budgets_here = [
                entry[3] for entry in blocks_seen if entry[1] == r and entry[2] == state_label
            ]
            for position, budget in enumerate(budgets_here, start=1):
                budget_position_by_state[state_label][budget].append(position)

    for state_label, per_budget in budget_position_by_state.items():
        for budget, positions in per_budget.items():
            if sorted(positions) != list(range(1, len(OFFICIAL_BUDGETS) + 1)):
                errors.append(
                    f"model={model}: state={state_label!r} budget={budget} "
                    f"occupies within-repeat position(s) {positions}, "
                    f"expected each of {list(range(1, len(OFFICIAL_BUDGETS) + 1))} "
                    f"exactly once across the {repeats} repeats"
                )

    # --- condition-first balance checks --------------------------------------
    blocks_by_id: dict[str, list[Episode]] = {}
    for ep in episodes:
        blocks_by_id.setdefault(ep.block_id, []).append(ep)

    first_conditions = [
        min(bes, key=lambda e: e.within_block_order).condition
        for bes in blocks_by_id.values()
    ]
    no_burst_first_count = sum(1 for c in first_conditions if c == "no_burst")
    burst_first_count = sum(1 for c in first_conditions if c == BURST_CONDITION)
    expected_half = (len(OFFICIAL_STATES) * len(OFFICIAL_BUDGETS) * repeats) // 2
    if no_burst_first_count != expected_half or burst_first_count != expected_half:
        errors.append(
            f"model={model}: condition-first balance is "
            f"no_burst={no_burst_first_count}/prefill_burst={burst_first_count}, "
            f"expected exactly {expected_half}/{expected_half}"
        )

    first_condition_by_cell: dict[tuple[str, int], list[str]] = {}
    for bid, bes in blocks_by_id.items():
        first_ep = min(bes, key=lambda e: e.within_block_order)
        key = (first_ep.state_label, first_ep.max_num_batched_tokens)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)

    for key, conditions_seen in sorted(first_condition_by_cell.items()):
        if len(conditions_seen) == repeats and len(set(conditions_seen)) < 2:
            errors.append(
                f"model={model}: cell state={key[0]!r} budget={key[1]} has "
                f"condition {conditions_seen[0]!r} first in all {repeats} "
                f"repeats; every cell must have a mixed split, never all "
                f"the same"
            )

    # --- workload seed constancy (per repeat, across state/budget/condition)
    # + victim/burst independence --------------------------------------------
    victim_seeds_by_repeat: dict[int, set[int]] = {}
    burst_seeds_by_repeat: dict[int, set[int]] = {}
    for ep in episodes:
        victim_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.burst_workload_seed)

    for repeat_key, seeds in victim_seeds_by_repeat.items():
        if len(seeds) != 1:
            errors.append(
                f"model={model}: victim_workload_seed not constant across "
                f"states/budgets/conditions for repeat={repeat_key}: "
                f"{sorted(seeds)}"
            )
    for repeat_key, seeds in burst_seeds_by_repeat.items():
        if len(seeds) != 1:
            errors.append(
                f"model={model}: burst_workload_seed not constant across "
                f"states/budgets/conditions for repeat={repeat_key}: "
                f"{sorted(seeds)}"
            )
    for ep in episodes:
        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(
                f"model={model}: episode {ep.episode_id} has identical "
                f"victim_workload_seed and burst_workload_seed"
            )

    return errors


def check_structural_schedule(episodes: list[Episode], schedule_seed: object) -> list[str]:
    errors: list[str] = []

    if len(episodes) != OFFICIAL_EPISODE_COUNT:
        errors.append(
            f"expected {OFFICIAL_EPISODE_COUNT} episodes total, found "
            f"{len(episodes)}"
        )

    by_model: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_model.setdefault(ep.model, []).append(ep)

    if sorted(by_model.keys()) != sorted(OFFICIAL_MODELS):
        errors.append(
            f"models present in schedule {sorted(by_model.keys())} != "
            f"expected {sorted(OFFICIAL_MODELS)}"
        )

    total_restart_markers = sum(
        1 for ep in episodes if ep.restart_server_before_block == 1
    )
    if total_restart_markers != TOTAL_RESTART_MARKERS:
        errors.append(
            f"expected {TOTAL_RESTART_MARKERS} restart_server_before_block "
            f"markers total, found {total_restart_markers}"
        )

    all_episode_ids = [ep.episode_id for ep in episodes]
    if len(all_episode_ids) != len(set(all_episode_ids)):
        dup = sorted({e for e in all_episode_ids if all_episode_ids.count(e) > 1})
        errors.append(f"duplicate episode_id(s) across the schedule: {dup}")

    all_episode_seeds = [ep.episode_seed for ep in episodes]
    if len(all_episode_seeds) != len(set(all_episode_seeds)):
        errors.append("duplicate episode_seed(s) found across the schedule")

    for model in OFFICIAL_MODELS:
        model_episodes = by_model.get(model, [])
        errors.extend(
            _check_model_structure(
                model,
                model_episodes,
                schedule_seed,
                repeats=OFFICIAL_REPEATS,
                episodes_per_model=EPISODES_PER_MODEL,
                blocks_per_model=BLOCKS_PER_MODEL,
            )
        )

    return errors


# ============================================================================
# Full bundle loading + validation orchestration
# ============================================================================

@dataclass
class LoadedBundle:
    schedule_dir: Path
    json_obj: dict
    csv_fieldnames: list[str]
    csv_rows: list[dict[str, str]]
    audit_text: str
    episodes: list[Episode]
    fingerprint: str


def load_and_validate_bundle(schedule_dir: Path) -> tuple[LoadedBundle | None, list[str]]:
    try:
        paths = find_bundle_paths(schedule_dir)
        json_obj = load_json_bundle(paths["chunk_budget_schedule.json"])
        csv_fieldnames, csv_rows = load_csv_bundle(paths["chunk_budget_schedule.csv"])
        audit_text = load_audit_text(paths["chunk_budget_schedule_audit.txt"])
    except BundleLoadError as exc:
        return None, [str(exc)]

    errors: list[str] = []
    errors.extend(check_official_contract(json_obj))

    raw_json_episodes = json_obj.get("episodes")
    if not isinstance(raw_json_episodes, list):
        errors.append("json 'episodes' is missing or not a list")
        return None, errors

    schema_errors: list[str] = []
    for idx, raw_ep in enumerate(raw_json_episodes):
        schema_errors.extend(check_json_episode_schema(raw_ep, idx))
    errors.extend(schema_errors)

    # A broken per-episode schema (missing/extra field, wrong type) makes
    # constructing Episode objects and every check below meaningless -- and
    # parse_episode_from_json() would KeyError on a genuinely missing
    # field, so bail out here. Note this is guarded by `schema_errors`
    # specifically, NOT by the cumulative `errors` list -- an official-
    # contract error (e.g. a fingerprint mismatch) must not prevent the
    # csv/json/audit/structural checks below from also running and being
    # reported.
    if schema_errors:
        return None, errors

    episodes = [parse_episode_from_json(ep) for ep in raw_json_episodes]

    errors.extend(
        check_csv_json_consistency(csv_fieldnames, csv_rows, raw_json_episodes)
    )

    fingerprint = json_obj.get("schedule_fingerprint", "")
    errors.extend(check_audit(audit_text, fingerprint))

    errors.extend(check_structural_schedule(episodes, json_obj.get("seed")))

    if errors:
        return None, errors

    bundle = LoadedBundle(
        schedule_dir=schedule_dir,
        json_obj=json_obj,
        csv_fieldnames=csv_fieldnames,
        csv_rows=csv_rows,
        audit_text=audit_text,
        episodes=episodes,
        fingerprint=fingerprint,
    )
    return bundle, []


# ============================================================================
# Execution plan (dry-run)
# ============================================================================

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
        blocks.append(
            {
                "block_id": bid,
                "model": first.model,
                "offload_gb": first.offload_gb,
                "state_label": first.state_label,
                "max_num_batched_tokens": first.max_num_batched_tokens,
                "repeat": first.repeat,
                "episodes": run_episodes,
            }
        )
        idx = run_end
    return blocks


def build_execution_plan(bundle: LoadedBundle) -> dict:
    episodes = bundle.episodes
    blocks = _group_into_blocks(episodes)

    no_burst_count = sum(1 for ep in episodes if ep.condition == "no_burst")
    burst_condition_count = sum(1 for ep in episodes if ep.condition == BURST_CONDITION)
    stabilization_runs_per_block = OFFICIAL_STABILIZATION_CONFIGURATION[
        "stabilization_runs_per_block"
    ]

    return {
        "regular_episodes": len(episodes),
        "blocks": blocks,
        "planned_server_starts": len(blocks),
        "planned_stabilization_runs": len(blocks) * stabilization_runs_per_block,
        "planned_warmups": 0,
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
    print(f"planned warmups: {plan['planned_warmups']}")
    print(f"no_burst episodes: {plan['no_burst_count']}")
    print(f"{BURST_CONDITION} episodes: {plan['burst_condition_count']}")
    print()
    for block in plan["blocks"]:
        print(f"--- {block['block_id']} ---")
        print(
            f"  model: {block['model']}, offload_gb: {block['offload_gb']}, "
            f"state: {block['state_label']}, "
            f"max_num_batched_tokens: {block['max_num_batched_tokens']}, "
            f"repeat: {block['repeat']}"
        )
        for ep in block["episodes"]:
            print(
                f"    order {ep.within_block_order}: {ep.episode_id} "
                f"(concurrency={ep.concurrency}, condition={ep.condition})"
            )
        print(
            f"  stabilization: "
            f"{OFFICIAL_STABILIZATION_CONFIGURATION['stabilization_runs_per_block']} "
            f"run(s), condition="
            f"{OFFICIAL_STABILIZATION_CONFIGURATION['stabilization_condition']!r}, "
            f"concurrency="
            f"{OFFICIAL_STABILIZATION_CONFIGURATION['stabilization_concurrency']}"
        )
    print()
    print(
        "DRY RUN: no server was started, no network connection was opened, "
        "no tokenizer was loaded, no result files were written."
    )


# ============================================================================
# Result-file schema + resume-depth classification (Section 11)
# ============================================================================

RESULT_SCHEMA_VERSION = 2
RUNNER_VERSION = "run_chunk_budget_screen-v1"
RECORD_TYPE_REGULAR_EPISODE = "regular_episode"
RECORD_TYPE_STABILIZATION = "stabilization"

CLASSIFICATION_MISSING = "missing"
CLASSIFICATION_VALID_COMPLETE = "valid_complete"
CLASSIFICATION_PARTIAL = "partial"
CLASSIFICATION_INVALID = "invalid"
CLASSIFICATION_CORRUPTED = "corrupted"

_RESULT_REQUIRED_KEYS = {
    "result_schema_version",
    "runner_version",
    "run_mode",
    "schedule_fingerprint",
    "episode_id",
    "schedule_row",
    "record_type",
    "status",
    "victim_requests",
    "burst_requests",
}


_RESULT_FIELD_TYPES: dict[str, type] = {
    "result_schema_version": int,
    "runner_version": str,
    "run_mode": str,
    "schedule_fingerprint": str,
    "episode_id": str,
    "schedule_row": dict,
    "record_type": str,
    "status": str,
    "victim_requests": list,
    "burst_requests": list,
}


def _validate_request_record_fields(
    record: object,
    *,
    episode_id: str,
    role: str,
    request_index: int,
    expected_prompt_seed: int,
    expected_generation_seed: int,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> list[str]:
    """Section 4: full structural + semantic depth validation of a single
    victim/burst request record, as stored in an episode result file.
    Never raises -- always returns a (possibly empty) list of textual
    error descriptions. A record is only depth-valid if this returns []."""
    if not isinstance(record, dict):
        return [f"{role} request record at index {request_index} is not a dict"]

    errors: list[str] = []

    def _get(key: str, expected_type: type):
        if key not in record:
            errors.append(f"{role}[{request_index}]: missing key {key!r}")
            return None
        v = record[key]
        if not _check_type_strict(v, expected_type):
            errors.append(
                f"{role}[{request_index}]: {key!r} has wrong type: expected "
                f"{expected_type.__name__}, got {type(v).__name__}"
            )
            return None
        return v

    expected_request_id = f"{episode_id}:{role}:{request_index}"
    rid = _get("request_id", str)
    if rid is not None and rid != expected_request_id:
        errors.append(f"{role}[{request_index}]: request_id {rid!r} != expected {expected_request_id!r}")

    r = _get("role", str)
    if r is not None and r != role:
        errors.append(f"{role}[{request_index}]: role {r!r} != expected {role!r}")

    ridx = _get("request_index", int)
    if ridx is not None and ridx != request_index:
        errors.append(f"{role}[{request_index}]: request_index {ridx!r} != expected {request_index!r}")

    pseed = _get("prompt_seed", int)
    if pseed is not None and pseed != expected_prompt_seed:
        errors.append(f"{role}[{request_index}]: prompt_seed does not match the deterministic derivation")

    gseed = _get("generation_seed", int)
    if gseed is not None and gseed != expected_generation_seed:
        errors.append(f"{role}[{request_index}]: generation_seed does not match the deterministic derivation")

    status = _get("status", str)
    if status is not None and status != REQUEST_STATUS_COMPLETE:
        errors.append(f"{role}[{request_index}]: status {status!r} != 'complete'")

    for bool_key, expected_bool in (
        ("timed_out", False), ("cancelled", False), ("done_received", True),
    ):
        v = record.get(bool_key, "<MISSING>")
        if type(v) is not bool:
            errors.append(f"{role}[{request_index}]: {bool_key!r} is missing or not a bool")
        elif v != expected_bool:
            errors.append(f"{role}[{request_index}]: {bool_key!r} == {v!r}, expected {expected_bool!r}")

    if "error_type" not in record or record["error_type"] is not None:
        errors.append(f"{role}[{request_index}]: error_type is not null")
    if "error_message" not in record or record["error_message"] is not None:
        errors.append(f"{role}[{request_index}]: error_message is not null")

    verrs = record.get("validation_errors", "<MISSING>")
    if not isinstance(verrs, list) or verrs:
        errors.append(f"{role}[{request_index}]: validation_errors is not an empty list")

    fr = record.get("finish_reason")
    if fr != "length":
        errors.append(f"{role}[{request_index}]: finish_reason {fr!r} != 'length'")

    sent = _get("prompt_token_ids_sent", list)
    if sent is not None and not all(type(x) is int for x in sent):
        errors.append(f"{role}[{request_index}]: prompt_token_ids_sent is not list[int]")
        sent = None
    returned = _get("prompt_token_ids_returned", list)
    if returned is not None and not all(type(x) is int for x in returned):
        errors.append(f"{role}[{request_index}]: prompt_token_ids_returned is not list[int]")
        returned = None
    if sent is not None and len(sent) != expected_prompt_tokens:
        errors.append(
            f"{role}[{request_index}]: prompt_token_ids_sent has {len(sent)} "
            f"entries, expected {expected_prompt_tokens}"
        )
    if returned is not None and len(returned) != expected_prompt_tokens:
        errors.append(
            f"{role}[{request_index}]: prompt_token_ids_returned has "
            f"{len(returned)} entries, expected {expected_prompt_tokens}"
        )
    if sent is not None and returned is not None and sent != returned:
        errors.append(f"{role}[{request_index}]: prompt_token_ids_sent != prompt_token_ids_returned")

    psha = record.get("prompt_sha256")
    if sent is not None:
        if psha != prompt_sha256(sent):
            errors.append(f"{role}[{request_index}]: prompt_sha256 does not match the canonical recomputation")
    elif not isinstance(psha, str):
        errors.append(f"{role}[{request_index}]: prompt_sha256 is missing or not a string")

    ept = record.get("expected_prompt_tokens")
    if ept != expected_prompt_tokens:
        errors.append(f"{role}[{request_index}]: expected_prompt_tokens {ept!r} != {expected_prompt_tokens!r}")

    output_ids = _get("output_token_ids", list)
    if output_ids is not None and not all(type(x) is int for x in output_ids):
        errors.append(f"{role}[{request_index}]: output_token_ids is not list[int]")
        output_ids = None
    if output_ids is not None and len(output_ids) != expected_completion_tokens:
        errors.append(
            f"{role}[{request_index}]: output_token_ids has {len(output_ids)} "
            f"entries, expected {expected_completion_tokens}"
        )

    ect = record.get("expected_completion_tokens")
    if ect != expected_completion_tokens:
        errors.append(
            f"{role}[{request_index}]: expected_completion_tokens {ect!r} != "
            f"{expected_completion_tokens!r}"
        )

    usage = _get("usage", dict)
    if usage is not None:
        pt = usage.get("prompt_tokens", "<MISSING>")
        if type(pt) is not int or pt != expected_prompt_tokens:
            errors.append(f"{role}[{request_index}]: usage.prompt_tokens is missing, not an int, or wrong")
        ct = usage.get("completion_tokens", "<MISSING>")
        if type(ct) is not int or ct != expected_completion_tokens:
            errors.append(f"{role}[{request_index}]: usage.completion_tokens is missing, not an int, or wrong")

    ns_fields = ("request_start_ns", "first_token_receive_ns", "last_token_receive_ns", "stream_end_ns")
    ns_values: dict[str, int] = {}
    for f in ns_fields:
        v = record.get(f, "<MISSING>")
        if type(v) is not int:
            errors.append(f"{role}[{request_index}]: {f!r} is missing or not an int")
        else:
            ns_values[f] = v
    if len(ns_values) == len(ns_fields):
        ordered = [ns_values[f] for f in ns_fields]
        if ordered != sorted(ordered):
            errors.append(
                f"{role}[{request_index}]: timestamps are not in non-decreasing "
                f"order (request_start_ns <= first_token_receive_ns <= "
                f"last_token_receive_ns <= stream_end_ns)"
            )

    return errors


def validate_complete_request_record(
    record: object, *, episode: Episode, role: str, request_index: int
) -> list[str]:
    """Public entry point for Section 4: validates a single victim/burst
    request record against the deterministic derivation and functional
    completeness rules for `episode`/`role`/`request_index`."""
    if role == "victim":
        expected_prompt_seed = victim_prompt_seed(episode, request_index)
        expected_generation_seed = victim_generation_seed(episode, request_index)
        expected_prompt_tokens = episode.victim_input_len
        expected_completion_tokens = episode.victim_output_len
    elif role == "burst":
        expected_prompt_seed = burst_prompt_seed(episode, request_index)
        expected_generation_seed = burst_generation_seed(episode, request_index)
        expected_prompt_tokens = episode.burst_input_len
        expected_completion_tokens = episode.burst_output_len
    else:
        return [f"unknown role {role!r}"]

    return _validate_request_record_fields(
        record,
        episode_id=episode.episode_id,
        role=role,
        request_index=request_index,
        expected_prompt_seed=expected_prompt_seed,
        expected_generation_seed=expected_generation_seed,
        expected_prompt_tokens=expected_prompt_tokens,
        expected_completion_tokens=expected_completion_tokens,
    )


def classify_result_file(
    path: Path,
    expected_episode: Episode,
    expected_fingerprint: str,
    expected_run_mode: str,
) -> tuple[str, list[str]]:
    """
    Classifies a single episode result file as one of: missing,
    valid_complete, partial, invalid, corrupted.

    `expected_episode` is the specific schedule episode this file is
    supposed to be the result of; it is always passed in explicitly by
    the caller (one exact episode per expected result filename). This
    function never picks an episode out of the schedule based on the
    file's own content -- a file that is an otherwise fully valid,
    complete result for a *different* episode is still classified
    `invalid` here, because its episode_id/schedule_row will not match
    `expected_episode`.

    Every required field is type-checked strictly (`type(x) is T`, so
    e.g. a JSON `true` is never silently accepted where an int is
    expected, and a list is never accepted where a str/dict is
    expected) BEFORE any dictionary lookup or value comparison is
    attempted below. This guarantees a malformed field can never raise
    an exception here -- it is always classified as `invalid`, never a
    crash.

    Full per-request depth validation (Section 4 of the Stage 2 patch)
    is applied to every victim_requests/burst_requests entry: deterministic
    seed/prompt/hash/usage/timestamp/status checks via
    validate_complete_request_record(). A request record is never
    accepted on list-length alone.
    """
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
        return CLASSIFICATION_INVALID, [
            f"missing required key(s): {sorted(missing_keys)}"
        ]

    # Strict type checks for every required field, before any lookup or
    # comparison below relies on that field's shape (e.g. len(), dict
    # equality, or a str comparison). This is what prevents e.g.
    # episode_id=[] or result_schema_version=true from ever reaching a
    # comparison that could behave unexpectedly or crash.
    type_errors = [
        f"field {field!r} has wrong type: expected {expected_type.__name__}, "
        f"got {type(obj[field]).__name__} ({obj[field]!r})"
        for field, expected_type in _RESULT_FIELD_TYPES.items()
        if not _check_type_strict(obj[field], expected_type)
    ]
    if type_errors:
        return CLASSIFICATION_INVALID, type_errors

    if obj["status"] != "complete":
        return CLASSIFICATION_PARTIAL, [f"status={obj['status']!r} != 'complete'"]

    notes: list[str] = []

    if obj["result_schema_version"] != RESULT_SCHEMA_VERSION:
        notes.append(
            f"result_schema_version {obj['result_schema_version']!r} != "
            f"{RESULT_SCHEMA_VERSION!r}"
        )
    if obj["runner_version"] != RUNNER_VERSION:
        notes.append(
            f"runner_version {obj['runner_version']!r} != {RUNNER_VERSION!r}"
        )
    if obj["schedule_fingerprint"] != expected_fingerprint:
        notes.append("schedule_fingerprint mismatch")
    if obj["record_type"] != RECORD_TYPE_REGULAR_EPISODE:
        notes.append(f"record_type != {RECORD_TYPE_REGULAR_EPISODE!r}")
    if obj["run_mode"] != expected_run_mode:
        notes.append("run_mode mismatch")

    if obj["episode_id"] != expected_episode.episode_id:
        notes.append(
            f"episode_id {obj['episode_id']!r} does not match the expected "
            f"episode {expected_episode.episode_id!r} for this result file"
        )

    expected_row = asdict(expected_episode)
    if obj["schedule_row"] != expected_row:
        notes.append(
            "schedule_row does not exactly match the expected schedule "
            "episode"
        )

    victim_requests = obj["victim_requests"]
    if len(victim_requests) != expected_episode.victim_request_count:
        notes.append(
            f"expected exactly {expected_episode.victim_request_count} "
            f"victim_requests"
        )
    else:
        for i, record in enumerate(victim_requests):
            notes.extend(
                validate_complete_request_record(record, episode=expected_episode, role="victim", request_index=i)
            )

    expected_burst_count = (
        expected_episode.burst_parallel
        if expected_episode.condition == BURST_CONDITION
        else 0
    )
    burst_requests = obj["burst_requests"]
    if len(burst_requests) != expected_burst_count:
        notes.append(
            f"expected exactly {expected_burst_count} burst_requests "
            f"for condition {expected_episode.condition!r}"
        )
    else:
        for j, record in enumerate(burst_requests):
            notes.extend(
                validate_complete_request_record(record, episode=expected_episode, role="burst", request_index=j)
            )

    # request_index must additionally be pairwise unique within each role
    # (positional validation above already implies this for well-formed
    # files, but a corrupted file could still smuggle duplicates past a
    # naive check -- verify explicitly).
    if isinstance(victim_requests, list):
        victim_indices = [
            r.get("request_index") for r in victim_requests if isinstance(r, dict)
        ]
        if len(victim_indices) != len(set(victim_indices)):
            notes.append("duplicate request_index value(s) among victim_requests")
    if isinstance(burst_requests, list):
        burst_indices = [
            r.get("request_index") for r in burst_requests if isinstance(r, dict)
        ]
        if len(burst_indices) != len(set(burst_indices)):
            notes.append("duplicate request_index value(s) among burst_requests")

    trigger = obj.get("trigger")
    if not isinstance(trigger, dict) or trigger.get("status") != "ok":
        notes.append("trigger.status != 'ok'")

    burst_interval = obj.get("burst_interval")
    if expected_episode.condition == "no_burst":
        if burst_interval is not None:
            notes.append("burst_interval must be null for condition 'no_burst'")
    else:
        if not (
            isinstance(burst_interval, dict)
            and type(burst_interval.get("start_ns")) is int
            and type(burst_interval.get("end_ns")) is int
        ):
            notes.append(
                f"burst_interval must be a valid {{start_ns, end_ns}} dict "
                f"for condition {BURST_CONDITION!r}"
            )

    if notes:
        return CLASSIFICATION_INVALID, notes

    return CLASSIFICATION_VALID_COMPLETE, []


EPISODES_SUBDIR = "episodes"
STABILIZATION_SUBDIR = "stabilization"


def episode_result_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / EPISODES_SUBDIR / f"{episode_id}.json"


def stabilization_result_path(output_dir: Path, block_id: str) -> Path:
    return output_dir / STABILIZATION_SUBDIR / f"{block_id}.json"


def scan_existing_results(
    output_dir: Path, bundle: LoadedBundle, run_mode: str
) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for ep in bundle.episodes:
        result_path = episode_result_path(output_dir, ep.episode_id)
        classification, _notes = classify_result_file(
            result_path, ep, bundle.fingerprint, run_mode
        )
        classifications[ep.episode_id] = classification
    return classifications


# ============================================================================
# Output-dir mode marker (official/smoke must never share a directory)
# ============================================================================

def check_output_dir_not_shared(output_dir: Path, mode: str) -> None:
    marker_path = output_dir / RUN_MODE_MARKER_FILENAME
    if not marker_path.exists():
        return
    try:
        existing_mode = marker_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OutputDirConflictError(
            f"could not read run-mode marker '{marker_path}': {exc}"
        ) from exc
    if existing_mode != mode:
        raise OutputDirConflictError(
            f"output-dir '{output_dir}' is already marked as belonging to "
            f"run mode {existing_mode!r}; refusing to also use it for run "
            f"mode {mode!r}. official and smoke runs must never share an "
            f"output directory."
        )


def write_run_mode_marker(output_dir: Path, mode: str) -> None:
    (output_dir / RUN_MODE_MARKER_FILENAME).write_text(mode, encoding="utf-8")


# ============================================================================
# Stage 2 constants (spec sections 4, 8, 15)
# ============================================================================

SERVER_READY_TIMEOUT_S = 900.0
HTTP_REQUEST_TIMEOUT_S = 1200.0
TRIGGER_TIMEOUT_S = 180.0
SERVER_STOP_TIMEOUT_S = 30.0
COOLDOWN_S = 5.0
READINESS_POLL_INTERVAL_S = 2.0

DEFAULT_SMOKE_HOST = "127.0.0.1"
DEFAULT_SMOKE_PORT = 8000

MODEL_FULL_ID: dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

COMPLETIONS_ENDPOINT = "/v1/completions"
HEALTH_ENDPOINT = "/health"
MODELS_ENDPOINT = "/v1/models"
OPENAPI_ENDPOINT = "/openapi.json"

RUN_MODE_SMOKE = "smoke"
RUN_MODE_OFFICIAL = "official"

REQUEST_STATUS_COMPLETE = "complete"
REQUEST_STATUS_INCOMPLETE = "incomplete"
REQUEST_STATUS_FAILED = "failed"
REQUEST_STATUS_CANCELLED = "cancelled"

STABILIZATION_CONDITION = "no_burst"
STABILIZATION_CONCURRENCY = 4
STABILIZATION_REQUEST_COUNT = 20
STABILIZATION_INPUT_LEN = 256
STABILIZATION_OUTPUT_LEN = 64
STABILIZATION_TEMPERATURE = 0.0


class ApiKeyError(Exception):
    """VLLM_API_KEY missing/empty. Never includes the key's value."""


class ServerLifecycleError(Exception):
    """Server start/readiness/stop/resume-precondition failure."""


class CapabilityError(Exception):
    """Local server/tooling does not support a required request feature
    (token-id prompts / return_token_ids / tokenizer availability /
    etc). Never triggers a silent fallback to chat-completions or text
    prompts -- callers must abort the smoke test instead."""


class HTTPStatusError(Exception):
    def __init__(self, status_code: int, body: object) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


# ============================================================================
# Atomic JSON writer (Section 21)
# ============================================================================

def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


# ============================================================================
# Dependency-injectable components (Section 25): HTTP transport, clock,
# sleeper, server-process adapter, tokenizer adapter. Real implementations
# lazily import optional third-party packages (httpx, transformers) so
# that --self-test and the fake-block integration test never need them,
# a GPU, or a real vLLM server.
# ============================================================================

@runtime_checkable
class Clock(Protocol):
    def perf_counter_ns(self) -> int: ...
    def utcnow_iso(self) -> str: ...


class RealClock:
    def perf_counter_ns(self) -> int:
        return time.perf_counter_ns()

    def utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class FakeClock:
    """Deterministic clock for tests: perf_counter_ns() advances by a
    fixed step on every call (so ordering/deltas stay meaningful)."""

    def __init__(self, step_ns: int = 1_000_000) -> None:
        self._t = 0
        self._step = step_ns
        self._u = 0

    def perf_counter_ns(self) -> int:
        self._t += self._step
        return self._t

    def utcnow_iso(self) -> str:
        self._u += 1
        return f"1970-01-01T00:{self._u // 60:02d}:{self._u % 60:02d}Z"


@runtime_checkable
class Sleeper(Protocol):
    async def sleep(self, seconds: float) -> None: ...


class RealSleeper:
    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeSleeper:
    """Never actually sleeps for a meaningful duration -- records
    requested durations so tests can assert on them without slowing
    down the self-test suite (spec section 26: 'darf nicht schlafen')."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        await asyncio.sleep(0)


@runtime_checkable
class HTTPTransport(Protocol):
    async def get_json(
        self, url: str, headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any]: ...

    def stream_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        timeout_s: float,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]: ...


class HttpxTransport:
    """Real HTTP transport (Section 12: httpx preferred). `request_id`
    is accepted for interface parity with FakeTransport and ignored."""

    def __init__(self) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for real smoke-test execution but is "
                "not installed in this environment"
            ) from exc
        self._httpx = httpx

    async def get_json(
        self, url: str, headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any]:
        async with self._httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, headers=headers)
            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text
            return resp.status_code, body

    async def stream_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        timeout_s: float,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        async with self._httpx.AsyncClient(timeout=timeout_s) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPStatusError(resp.status_code, body)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    yield line[len("data:") :].strip()


@dataclass
class FakeStreamScript:
    """Canned SSE event stream for exactly one fake request."""

    prompt_token_ids_echo: list[int] | None
    token_events: list[list[int]]
    finish_reason: str | None = "length"
    usage: dict | None = None
    include_done: bool = True
    hang: bool = False
    http_status: int = 200
    raise_after_n_events: int | None = None
    extra_keepalives: int = 0
    extra_raw_events_before_finish: list[str] = field(default_factory=list)


class FakeTransport:
    """In-memory transport for --self-test and the fake-block
    integration test. Scripts can be queued per `request_id` (exact
    match); otherwise `default_script_factory(payload)` is used to
    auto-generate a functionally-complete response."""

    def __init__(self) -> None:
        self.scripts_by_request_id: dict[str, list[FakeStreamScript]] = {}
        self._counts: dict[str, int] = {}
        self.get_responses: dict[str, tuple[int, Any]] = {}
        self.get_json_error_queue: dict[str, list[BaseException]] = {}
        self.get_json_call_log: list[str] = []
        self.active_stream_count = 0
        self.max_active_stream_count = 0
        self.default_script_factory: Callable[[dict], FakeStreamScript] | None = None
        self.seen_headers: list[dict[str, str]] = []
        self.seen_payloads: list[dict] = []
        self.call_order: list[str] = []

    def queue_script(self, request_id: str, script: FakeStreamScript) -> None:
        self.scripts_by_request_id.setdefault(request_id, []).append(script)

    def set_get_response(self, url_suffix: str, status: int, body: Any) -> None:
        self.get_responses[url_suffix] = (status, body)

    def queue_get_error(self, url_suffix: str, exc: BaseException) -> None:
        """Makes the next matching get_json() call for this suffix raise
        `exc` instead of returning a response. Queued items are
        consumed in FIFO order per suffix; once exhausted, get_json()
        falls back to the normal set_get_response()/404 behavior."""
        self.get_json_error_queue.setdefault(url_suffix, []).append(exc)

    def queue_get_status(self, url_suffix: str, status: int, body: Any = None) -> None:
        """Queues a one-shot (status, body) response for the next
        matching get_json() call, e.g. to simulate '503 then 200'
        sequences. Shares the same FIFO queue as queue_get_error()."""
        self.get_json_error_queue.setdefault(url_suffix, []).append((status, body if body is not None else {}))

    async def get_json(self, url, headers, timeout_s):
        for suffix in self.get_json_error_queue:
            if url.endswith(suffix) and self.get_json_error_queue[suffix]:
                self.get_json_call_log.append(url)
                item = self.get_json_error_queue[suffix].pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
        self.get_json_call_log.append(url)
        for suffix, resp in self.get_responses.items():
            if url.endswith(suffix):
                return resp
        return 404, {"error": "not found (fake transport)"}

    async def stream_completion(
        self, url, headers, payload, timeout_s, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        self.seen_headers.append(dict(headers))
        self.seen_payloads.append(payload)
        key = request_id or "<default>"
        self.call_order.append(key)
        idx = self._counts.get(key, 0)
        self._counts[key] = idx + 1
        queued = self.scripts_by_request_id.get(key)
        if queued and idx < len(queued):
            script = queued[idx]
        elif self.default_script_factory is not None:
            script = self.default_script_factory(payload)
        else:
            script = FakeStreamScript(
                prompt_token_ids_echo=list(payload["prompt"]),
                token_events=[[1]] * payload["max_tokens"],
                usage={
                    "prompt_tokens": len(payload["prompt"]),
                    "completion_tokens": payload["max_tokens"],
                },
            )
        self.active_stream_count += 1
        self.max_active_stream_count = max(self.max_active_stream_count, self.active_stream_count)
        try:
            if script.http_status != 200:
                raise HTTPStatusError(script.http_status, "fake error")
            if script.hang:
                await asyncio.Event().wait()
                return
            event_i = 0
            if script.prompt_token_ids_echo is not None:
                yield json.dumps(
                    {
                        "choices": [
                            {"index": 0, "text": "", "token_ids": [], "finish_reason": None}
                        ],
                        "prompt_token_ids": script.prompt_token_ids_echo,
                        "usage": None,
                    }
                )
                event_i += 1
                await asyncio.sleep(0)
            for raw_event in script.extra_raw_events_before_finish:
                yield raw_event
                event_i += 1
                await asyncio.sleep(0)
            for te in script.token_events:
                if (
                    script.raise_after_n_events is not None
                    and event_i >= script.raise_after_n_events
                ):
                    raise ConnectionError("fake mid-stream failure")
                yield json.dumps(
                    {
                        "choices": [
                            {"index": 0, "text": "", "token_ids": te, "finish_reason": None}
                        ],
                        "usage": None,
                    }
                )
                event_i += 1
                await asyncio.sleep(0)
            for _ in range(script.extra_keepalives):
                yield ""
                await asyncio.sleep(0)
            yield json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "token_ids": [],
                            "finish_reason": script.finish_reason,
                        }
                    ],
                    "usage": script.usage,
                }
            )
            if script.include_done:
                yield "[DONE]"
        finally:
            self.active_stream_count -= 1


@runtime_checkable
class TokenizerAdapter(Protocol):
    def vocab_size(self) -> int: ...
    def special_token_ids(self) -> set[int]: ...


class HFTokenizerAdapter:
    """Real adapter: loads the Hugging Face tokenizer for the given full
    model id lazily. Only ever constructed on the real smoke-test path."""

    def __init__(self, model_full_id: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise CapabilityError(
                "the 'transformers' package is required to load the "
                f"tokenizer for '{model_full_id}' but is not installed"
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(model_full_id)
        self._vocab_size = len(self._tok)
        specials = set(self._tok.all_special_ids or [])
        added = getattr(self._tok, "added_tokens_encoder", {}) or {}
        specials.update(added.values())
        self._special_ids = specials

    def vocab_size(self) -> int:
        return self._vocab_size

    def special_token_ids(self) -> set[int]:
        return self._special_ids


class FakeTokenizerAdapter:
    """Deterministic, tiny synthetic vocabulary for tests -- no
    network, no GPU, no real tokenizer files."""

    def __init__(
        self, vocab_size: int = 1000, special_token_ids: Iterable[int] = (0, 1, 2)
    ) -> None:
        self._vocab_size = vocab_size
        self._special_ids = set(special_token_ids)

    def vocab_size(self) -> int:
        return self._vocab_size

    def special_token_ids(self) -> set[int]:
        return self._special_ids


def compute_valid_token_ids(tokenizer: TokenizerAdapter) -> list[int]:
    specials = tokenizer.special_token_ids()
    return [i for i in range(tokenizer.vocab_size()) if i not in specials]


def generate_token_id_prompt(seed: int, valid_ids: Sequence[int], length: int) -> list[int]:
    if not valid_ids:
        raise CapabilityError("tokenizer has no non-special token ids available")
    rng = random.Random(seed)
    n = len(valid_ids)
    return [valid_ids[rng.randrange(n)] for _ in range(length)]


# ============================================================================
# Deterministic request seed derivation (Section 10). derive_seed() itself
# is Stage 1's, unchanged.
# ============================================================================

def victim_prompt_seed(episode: Episode, i: int) -> int:
    return derive_seed(str(episode.victim_workload_seed), "victim-prompt", str(i))


def victim_generation_seed(episode: Episode, i: int) -> int:
    return derive_seed(str(episode.victim_workload_seed), "victim-generation", str(i))


def burst_prompt_seed(episode: Episode, i: int) -> int:
    return derive_seed(str(episode.burst_workload_seed), "burst-prompt", str(i))


def burst_generation_seed(episode: Episode, i: int) -> int:
    return derive_seed(str(episode.burst_workload_seed), "burst-generation", str(i))


def stabilization_prompt_seed(bundle_seed: object, model_key: str, block_id: str, i: int) -> int:
    return derive_seed(str(bundle_seed), model_key, block_id, "stabilization-prompt", str(i))


def stabilization_generation_seed(bundle_seed: object, model_key: str, block_id: str, i: int) -> int:
    return derive_seed(str(bundle_seed), model_key, block_id, "stabilization-generation", str(i))


def prompt_sha256(prompt_token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ============================================================================
# Percentiles
# ============================================================================

def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _median(values: list[float]) -> float | None:
    return _percentile(values, 0.5)


# ============================================================================
# Single streaming /v1/completions request (Sections 8-14)
# ============================================================================

def _completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + COMPLETIONS_ENDPOINT


def extract_prompt_token_ids(
    obj: dict, choice0: dict, event_index: int
) -> tuple[list[int] | None, list[str]]:
    """Real-vLLM patch: prompt_token_ids can appear at the SSE event's
    top level (`obj["prompt_token_ids"]`) or inside the first choice
    (`choice0["prompt_token_ids"]`) -- observed with real vLLM 0.17.1,
    which reports it only in `choices[0]`. Neither position is silently
    preferred over the other:
      - only one position has a valid list[int]  -> use it
      - both have valid, IDENTICAL list[int]     -> use it
      - both have valid but DIFFERENT list[int]   -> protocol error
      - a present value that is neither null nor list[int] -> protocol error
      - null (or the key missing) at a position carries no information
        and is never an error by itself
    Cross-event contradiction (a later event's list differing from an
    earlier one) is intentionally NOT handled here -- the caller already
    tracks `server_prompt_token_ids` across the whole stream and applies
    that check uniformly regardless of which position each event used.
    Never raises on malformed input; always returns (list_or_None, errors).
    """
    errors: list[str] = []

    def _valid_list_or_none(value: object, where: str) -> tuple[list[int] | None, bool]:
        if value is None:
            return None, False
        if isinstance(value, list) and all(type(x) is int for x in value):
            return value, False
        errors.append(f"event {event_index}: '{where}.prompt_token_ids' is present but not list[int]")
        return None, True

    top_raw = obj.get("prompt_token_ids") if isinstance(obj, dict) else None
    choice_raw = choice0.get("prompt_token_ids") if isinstance(choice0, dict) else None

    top_list, top_invalid = _valid_list_or_none(top_raw, "top-level")
    choice_list, choice_invalid = _valid_list_or_none(choice_raw, "choices[0]")

    if top_invalid or choice_invalid:
        return None, errors

    if top_list is not None and choice_list is not None:
        if top_list != choice_list:
            errors.append(
                f"event {event_index}: top-level prompt_token_ids and "
                f"choices[0].prompt_token_ids are both present but differ"
            )
            return None, errors
        return top_list, errors

    if top_list is not None:
        return top_list, errors
    if choice_list is not None:
        return choice_list, errors

    return None, errors


async def execute_completion_request(
    *,
    transport: HTTPTransport,
    clock: Clock,
    url: str,
    api_key: str,
    model_full_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    min_tokens: int,
    temperature: float,
    request_seed: int,
    request_id: str,
    role: str,
    request_index: int,
    prompt_seed: int,
    generation_seed: int,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
    http_timeout_s: float,
    on_first_token: Callable[[], None] | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "model": model_full_id,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "ignore_eos": True,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "return_token_ids": True,
        "seed": request_seed,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_start_utc = clock.utcnow_iso()
    request_start_ns = clock.perf_counter_ns()

    raw_sse_events: list[dict] = []
    server_prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] = []
    output_text_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict | None = None
    done_received = False
    first_token_receive_ns: int | None = None
    last_token_receive_ns: int | None = None
    token_event_receipts: list[tuple[int, int]] = []  # (receive_ns, batch_size)
    http_status: int | None = None
    timed_out = False
    cancelled = False
    error_type: str | None = None
    error_message: str | None = None
    event_index = 0
    protocol_errors: list[str] = []

    async def _consume() -> None:
        nonlocal server_prompt_token_ids, finish_reason, usage, done_received
        nonlocal first_token_receive_ns, last_token_receive_ns, http_status, event_index
        http_status = 200
        stream_gen = transport.stream_completion(
            url, headers, payload, http_timeout_s, request_id=request_id
        )
        try:
            async for raw in stream_gen:
                now_ns = clock.perf_counter_ns()
                elapsed_ms = (now_ns - request_start_ns) / 1e6

                if raw == "":
                    raw_sse_events.append(
                        {
                            "event_index": event_index,
                            "receive_perf_counter_ns": now_ns,
                            "elapsed_since_request_start_ms": elapsed_ms,
                            "raw_data": raw,
                            "parse_status": "keepalive",
                            "token_ids": [],
                            "prompt_token_ids": None,
                            "text_delta": None,
                            "finish_reason": None,
                            "usage": None,
                        }
                    )
                    event_index += 1
                    continue

                if raw == "[DONE]":
                    done_received = True
                    raw_sse_events.append(
                        {
                            "event_index": event_index,
                            "receive_perf_counter_ns": now_ns,
                            "elapsed_since_request_start_ms": elapsed_ms,
                            "raw_data": raw,
                            "parse_status": "done",
                            "token_ids": [],
                            "prompt_token_ids": None,
                            "text_delta": None,
                            "finish_reason": None,
                            "usage": None,
                        }
                    )
                    event_index += 1
                    break

                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    protocol_errors.append(f"event {event_index}: JSON parse error: {exc}")
                    raw_sse_events.append(
                        {
                            "event_index": event_index,
                            "receive_perf_counter_ns": now_ns,
                            "elapsed_since_request_start_ms": elapsed_ms,
                            "raw_data": raw,
                            "parse_status": "parse_error",
                            "token_ids": [],
                            "prompt_token_ids": None,
                            "text_delta": None,
                            "finish_reason": None,
                            "usage": None,
                        }
                    )
                    event_index += 1
                    continue

                if not isinstance(obj, dict):
                    protocol_errors.append(f"event {event_index}: SSE payload is not a JSON object")
                    raw_sse_events.append(
                        {
                            "event_index": event_index,
                            "receive_perf_counter_ns": now_ns,
                            "elapsed_since_request_start_ms": elapsed_ms,
                            "raw_data": raw,
                            "parse_status": "protocol_error",
                            "token_ids": [],
                            "prompt_token_ids": None,
                            "text_delta": None,
                            "finish_reason": None,
                            "usage": None,
                        }
                    )
                    event_index += 1
                    continue

                # `choices`: must be a list of dicts if present at all.
                choices_raw = obj.get("choices")
                event_is_malformed = False
                if choices_raw is not None and not isinstance(choices_raw, list):
                    protocol_errors.append(f"event {event_index}: 'choices' has the wrong type (expected a list)")
                    event_is_malformed = True
                    choices_raw = []
                choices = choices_raw or []

                choice0: dict = {}
                if choices:
                    c0 = choices[0]
                    if not isinstance(c0, dict):
                        protocol_errors.append(f"event {event_index}: choices[0] is not a JSON object")
                        event_is_malformed = True
                    else:
                        choice0 = c0

                # `token_ids`: must be list[int] if present at all.
                token_ids_raw = choice0.get("token_ids")
                if token_ids_raw is None:
                    token_ids_here: list[int] = []
                elif isinstance(token_ids_raw, list) and all(type(x) is int for x in token_ids_raw):
                    token_ids_here = list(token_ids_raw)
                else:
                    protocol_errors.append(f"event {event_index}: 'token_ids' is not list[int]")
                    event_is_malformed = True
                    token_ids_here = []

                text_delta = choice0.get("text")
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = fr

                # `prompt_token_ids`: vLLM may report this at the SSE
                # event's top level OR inside choices[0] (observed with
                # real vLLM 0.17.1); both positions are accepted, must
                # agree with each other within the same event, and must
                # never contradict a value already seen earlier in the
                # stream (regardless of which position supplied it).
                prompt_ids_here, extract_errors = extract_prompt_token_ids(obj, choice0, event_index)
                if extract_errors:
                    protocol_errors.extend(extract_errors)
                    event_is_malformed = True
                if prompt_ids_here is not None:
                    if server_prompt_token_ids is None:
                        server_prompt_token_ids = prompt_ids_here
                    elif prompt_ids_here != server_prompt_token_ids:
                        protocol_errors.append(
                            f"event {event_index}: contradicting prompt_token_ids across multiple events"
                        )
                        event_is_malformed = True

                # `usage`: must be a dict if present, with real-int counters.
                # Only ever assigned to the outer `usage` (used later for
                # completeness checks) when it is genuinely a well-formed dict,
                # so a malformed usage payload can never crash a later
                # `usage.get(...)` call.
                usage_raw = obj.get("usage")
                usage_here: dict | None = None
                if usage_raw is not None:
                    if not isinstance(usage_raw, dict):
                        protocol_errors.append(f"event {event_index}: 'usage' is not a dict")
                        event_is_malformed = True
                    else:
                        bad_counter = False
                        for counter_key in ("prompt_tokens", "completion_tokens"):
                            if counter_key in usage_raw and type(usage_raw[counter_key]) is not int:
                                bad_counter = True
                        if bad_counter:
                            protocol_errors.append(f"event {event_index}: usage counter(s) are not real ints")
                            event_is_malformed = True
                        usage_here = usage_raw
                        usage = usage_here

                raw_sse_events.append(
                    {
                        "event_index": event_index,
                        "receive_perf_counter_ns": now_ns,
                        "elapsed_since_request_start_ms": elapsed_ms,
                        "raw_data": raw,
                        "parse_status": "protocol_error" if event_is_malformed else "ok",
                        "token_ids": token_ids_here,
                        "prompt_token_ids": prompt_ids_here,
                        "text_delta": text_delta,
                        "finish_reason": fr,
                        "usage": usage_here,
                    }
                )
                event_index += 1

                if token_ids_here:
                    output_token_ids.extend(token_ids_here)
                    if text_delta:
                        output_text_parts.append(text_delta)
                    if first_token_receive_ns is None:
                        first_token_receive_ns = now_ns
                        if on_first_token is not None:
                            on_first_token()
                    last_token_receive_ns = now_ns
                    token_event_receipts.append((now_ns, len(token_ids_here)))
        finally:
            # Always close the underlying stream -- both for FakeTransport's
            # bookkeeping (active_stream_count) and, on the real httpx
            # transport, to actually release the HTTP connection/response.
            # `async for` does NOT do this automatically on an early
            # `break` (e.g. right after [DONE]).
            await stream_gen.aclose()

    try:
        await asyncio.wait_for(_consume(), timeout=http_timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        error_type = "timeout"
        error_message = "request exceeded http_timeout_s"
    except asyncio.CancelledError:
        cancelled = True
    except HTTPStatusError as exc:
        http_status = exc.status_code
        error_type = "http_status"
        error_message = f"HTTP {exc.status_code}"
    except Exception as exc:  # noqa: BLE001 -- must never crash the caller/block
        error_type = type(exc).__name__
        error_message = str(exc)

    stream_end_ns = clock.perf_counter_ns()
    request_end_utc = clock.utcnow_iso()

    ttft_ms = (
        (first_token_receive_ns - request_start_ns) / 1e6
        if first_token_receive_ns is not None
        else None
    )
    e2el_ms = (stream_end_ns - request_start_ns) / 1e6

    completion_token_count = len(output_token_ids)
    client_observed_tpot_ms = None
    if (
        first_token_receive_ns is not None
        and last_token_receive_ns is not None
        and completion_token_count > 1
    ):
        client_observed_tpot_ms = (
            (last_token_receive_ns - first_token_receive_ns) / (completion_token_count - 1)
        ) / 1e6

    itl_available = False
    itl_ms: list[float] | None = None
    token_batch_sizes: list[int] | None = None
    token_batch_interarrival_ms: list[float] | None = None
    chunk_interarrival_ms: list[float] | None = None
    if token_event_receipts:
        sizes = [sz for _, sz in token_event_receipts]
        itl_available = all(sz == 1 for sz in sizes)
        deltas_ms = [
            (token_event_receipts[i][0] - token_event_receipts[i - 1][0]) / 1e6
            for i in range(1, len(token_event_receipts))
        ]
        if itl_available:
            itl_ms = deltas_ms
        else:
            token_batch_sizes = sizes
            token_batch_interarrival_ms = deltas_ms
            chunk_interarrival_ms = deltas_ms

    validation_errors: list[str] = list(protocol_errors)
    if len(prompt_token_ids) != expected_prompt_tokens:
        validation_errors.append(
            f"sent prompt length {len(prompt_token_ids)} != expected {expected_prompt_tokens}"
        )
    if server_prompt_token_ids is None:
        validation_errors.append("no server-side prompt_token_ids observed")
    elif server_prompt_token_ids != prompt_token_ids:
        validation_errors.append("server-side prompt_token_ids != sent prompt_token_ids")
    if usage is None or usage.get("prompt_tokens") != expected_prompt_tokens:
        validation_errors.append("usage.prompt_tokens != expected_prompt_tokens")
    if completion_token_count != expected_completion_tokens:
        validation_errors.append(
            f"collected output token_ids count {completion_token_count} != "
            f"expected {expected_completion_tokens}"
        )
    if usage is None or usage.get("completion_tokens") != expected_completion_tokens:
        validation_errors.append("usage.completion_tokens != expected_completion_tokens")
    if finish_reason != "length":
        validation_errors.append(f"finish_reason {finish_reason!r} != 'length'")
    if not done_received:
        validation_errors.append("[DONE] was not received")

    if cancelled:
        status = REQUEST_STATUS_CANCELLED
    elif timed_out or error_type == "http_status" or error_type is not None:
        status = REQUEST_STATUS_FAILED
        if error_message:
            validation_errors.append(f"stream error ({error_type}): {error_message}")
    elif validation_errors:
        status = REQUEST_STATUS_INCOMPLETE
    else:
        status = REQUEST_STATUS_COMPLETE

    return {
        "request_id": request_id,
        "role": role,
        "request_index": request_index,
        "prompt_seed": prompt_seed,
        "generation_seed": generation_seed,
        "prompt_token_ids_sent": prompt_token_ids,
        "prompt_token_ids_returned": server_prompt_token_ids,
        "prompt_sha256": prompt_sha256(prompt_token_ids),
        "expected_prompt_tokens": expected_prompt_tokens,
        "expected_completion_tokens": expected_completion_tokens,
        "usage": usage,
        "output_token_ids": output_token_ids,
        "output_text": "".join(output_text_parts),
        "finish_reason": finish_reason,
        "raw_sse_events": raw_sse_events,
        "done_received": done_received,
        "request_start_utc": request_start_utc,
        "request_end_utc": request_end_utc,
        "request_start_ns": request_start_ns,
        "first_token_receive_ns": first_token_receive_ns,
        "last_token_receive_ns": last_token_receive_ns,
        "stream_end_ns": stream_end_ns,
        "ttft_ms": ttft_ms,
        "client_observed_tpot_ms": client_observed_tpot_ms,
        "e2el_ms": e2el_ms,
        "itl_available": itl_available,
        "itl_ms": itl_ms,
        "token_batch_sizes": token_batch_sizes,
        "token_batch_interarrival_ms": token_batch_interarrival_ms,
        "chunk_interarrival_ms": chunk_interarrival_ms,
        "http_status": http_status,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "error_type": error_type,
        "error_message": error_message,
        "validation_errors": validation_errors,
        "status": status,
    }


# ============================================================================
# Centralized cancellation (Section 17)
# ============================================================================

async def cancel_all(tasks: Iterable[asyncio.Task]) -> None:
    task_list = [t for t in tasks if t is not None]
    for t in task_list:
        if not t.done():
            t.cancel()
    if task_list:
        await asyncio.gather(*task_list, return_exceptions=True)


def _coerce_task_results(raw_results: list) -> list[dict]:
    """Turns a list of asyncio.gather(..., return_exceptions=True)
    results into plain result dicts. A raw asyncio.CancelledError (a
    task that was cancelled before execute_completion_request() ever
    produced its own dict, e.g. while still waiting on the concurrency
    semaphore) becomes status='cancelled'; any other raised exception
    becomes status='failed'. Neither ever crashes the caller."""
    out: list[dict] = []
    for r in raw_results:
        if isinstance(r, asyncio.CancelledError):
            out.append(
                {
                    "status": REQUEST_STATUS_CANCELLED,
                    "cancelled": True,
                    "timed_out": False,
                    "error_type": None,
                    "error_message": None,
                    "validation_errors": ["request task was cancelled before it could start"],
                }
            )
        elif isinstance(r, BaseException):
            out.append(
                {
                    "status": REQUEST_STATUS_FAILED,
                    "error_type": type(r).__name__,
                    "error_message": str(r),
                    "validation_errors": [f"unhandled exception: {r!r}"],
                }
            )
        else:
            out.append(r)
    return out


def _enrich_minimal_records(records: list[dict], *, episode_id: str, role: str) -> list[dict]:
    """Fills in identity fields (request_id/role/request_index) and safe
    defaults on synthetic minimal records produced by _coerce_task_results()
    for tasks that never got far enough to build their own full record,
    keyed by their position in the (already index-ordered) list. Never
    overwrites fields a real record already set."""
    enriched: list[dict] = []
    for i, r in enumerate(records):
        if "request_id" not in r:
            r = dict(r)
            r.setdefault("request_id", f"{episode_id}:{role}:{i}")
            r.setdefault("role", role)
            r.setdefault("request_index", i)
            r.setdefault("prompt_seed", None)
            r.setdefault("generation_seed", None)
            r.setdefault("prompt_token_ids_sent", None)
            r.setdefault("prompt_token_ids_returned", None)
            r.setdefault("prompt_sha256", None)
            r.setdefault("expected_prompt_tokens", None)
            r.setdefault("expected_completion_tokens", None)
            r.setdefault("usage", None)
            r.setdefault("output_token_ids", [])
            r.setdefault("output_text", "")
            r.setdefault("finish_reason", None)
            r.setdefault("raw_sse_events", [])
            r.setdefault("done_received", False)
            r.setdefault("request_start_utc", None)
            r.setdefault("request_end_utc", None)
            r.setdefault("request_start_ns", None)
            r.setdefault("first_token_receive_ns", None)
            r.setdefault("last_token_receive_ns", None)
            r.setdefault("stream_end_ns", None)
            r.setdefault("ttft_ms", None)
            r.setdefault("client_observed_tpot_ms", None)
            r.setdefault("e2el_ms", None)
            r.setdefault("itl_available", False)
            r.setdefault("itl_ms", None)
            r.setdefault("token_batch_sizes", None)
            r.setdefault("token_batch_interarrival_ms", None)
            r.setdefault("chunk_interarrival_ms", None)
            r.setdefault("http_status", None)
            r.setdefault("timed_out", False)
            r.setdefault("cancelled", r.get("status") == REQUEST_STATUS_CANCELLED)
            r.setdefault("error_type", None)
            r.setdefault("error_message", None)
            r.setdefault("validation_errors", [])
        enriched.append(r)
    return enriched


# ============================================================================
# Run context + per-role request builders
# ============================================================================

@dataclass
class RunContext:
    transport: HTTPTransport
    clock: Clock
    sleeper: Sleeper
    base_url: str
    api_key: str
    model_full_id: str
    valid_ids: list[int]
    http_timeout_s: float = HTTP_REQUEST_TIMEOUT_S
    trigger_timeout_s: float = TRIGGER_TIMEOUT_S


async def _run_victim_request(
    ctx: RunContext, episode: Episode, i: int, on_first_token: Callable[[], None] | None = None
) -> dict:
    p_seed = victim_prompt_seed(episode, i)
    g_seed = victim_generation_seed(episode, i)
    prompt_ids = generate_token_id_prompt(p_seed, ctx.valid_ids, episode.victim_input_len)
    return await execute_completion_request(
        transport=ctx.transport,
        clock=ctx.clock,
        url=_completions_url(ctx.base_url),
        api_key=ctx.api_key,
        model_full_id=ctx.model_full_id,
        prompt_token_ids=prompt_ids,
        max_tokens=episode.victim_output_len,
        min_tokens=episode.victim_output_len,
        temperature=episode.victim_temperature,
        request_seed=g_seed,
        request_id=f"{episode.episode_id}:victim:{i}",
        role="victim",
        request_index=i,
        prompt_seed=p_seed,
        generation_seed=g_seed,
        expected_prompt_tokens=episode.victim_input_len,
        expected_completion_tokens=episode.victim_output_len,
        http_timeout_s=ctx.http_timeout_s,
        on_first_token=on_first_token,
    )


async def _run_burst_request(ctx: RunContext, episode: Episode, j: int) -> dict:
    p_seed = burst_prompt_seed(episode, j)
    g_seed = burst_generation_seed(episode, j)
    prompt_ids = generate_token_id_prompt(p_seed, ctx.valid_ids, episode.burst_input_len)
    return await execute_completion_request(
        transport=ctx.transport,
        clock=ctx.clock,
        url=_completions_url(ctx.base_url),
        api_key=ctx.api_key,
        model_full_id=ctx.model_full_id,
        prompt_token_ids=prompt_ids,
        max_tokens=episode.burst_output_len,
        min_tokens=episode.burst_output_len,
        temperature=episode.burst_temperature,
        request_seed=g_seed,
        request_id=f"{episode.episode_id}:burst:{j}",
        role="burst",
        request_index=j,
        prompt_seed=p_seed,
        generation_seed=g_seed,
        expected_prompt_tokens=episode.burst_input_len,
        expected_completion_tokens=episode.burst_output_len,
        http_timeout_s=ctx.http_timeout_s,
    )


async def _run_stabilization_request(
    ctx: RunContext, bundle_seed: object, model_key: str, block_id: str, i: int
) -> dict:
    p_seed = stabilization_prompt_seed(bundle_seed, model_key, block_id, i)
    g_seed = stabilization_generation_seed(bundle_seed, model_key, block_id, i)
    prompt_ids = generate_token_id_prompt(p_seed, ctx.valid_ids, STABILIZATION_INPUT_LEN)
    return await execute_completion_request(
        transport=ctx.transport,
        clock=ctx.clock,
        url=_completions_url(ctx.base_url),
        api_key=ctx.api_key,
        model_full_id=ctx.model_full_id,
        prompt_token_ids=prompt_ids,
        max_tokens=STABILIZATION_OUTPUT_LEN,
        min_tokens=STABILIZATION_OUTPUT_LEN,
        temperature=STABILIZATION_TEMPERATURE,
        request_seed=g_seed,
        request_id=f"{block_id}:stabilization:{i}",
        role="stabilization",
        request_index=i,
        prompt_seed=p_seed,
        generation_seed=g_seed,
        expected_prompt_tokens=STABILIZATION_INPUT_LEN,
        expected_completion_tokens=STABILIZATION_OUTPUT_LEN,
        http_timeout_s=ctx.http_timeout_s,
    )


# ============================================================================
# Trigger watcher (Sections 16-18)
# ============================================================================

async def _watch_trigger(
    first_wave_indices: Iterable[int],
    first_token_events: dict[int, asyncio.Event],
    victim_tasks: dict[int, asyncio.Task],
    timeout_s: float,
) -> str:
    """Returns 'ok' once every first-wave request has received its first
    output token; 'timeout' if timeout_s elapses first; or
    'pretrigger_failure' if a first-wave request finishes without ever
    setting its first-token event, OR finishes with a non-'complete'
    status even after having set it (Section 17/18: a request that
    delivers its first token and then fails/becomes incomplete before
    the overall trigger is reached must also abort the trigger)."""
    start = time.monotonic()
    all_events_task = asyncio.ensure_future(
        asyncio.gather(*(first_token_events[i].wait() for i in first_wave_indices))
    )
    pending = {victim_tasks[i]: i for i in first_wave_indices}
    try:
        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                return "timeout"
            waitables = {all_events_task, *pending.keys()}
            done, _pending = await asyncio.wait(
                waitables, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if all_events_task in done:
                return "ok"
            for t in [t for t in list(pending) if t in done]:
                idx = pending.pop(t)
                if not first_token_events[idx].is_set():
                    return "pretrigger_failure"
                try:
                    result = t.result()
                except BaseException:
                    return "pretrigger_failure"
                if not isinstance(result, dict) or result.get("status") != REQUEST_STATUS_COMPLETE:
                    return "pretrigger_failure"
            # Otherwise asyncio.wait's own timeout elapsed with nothing
            # newly finished -- loop back and let the remaining-time
            # check above return 'timeout'.
    finally:
        if not all_events_task.done():
            all_events_task.cancel()
            try:
                await all_events_task
            except (asyncio.CancelledError, Exception):
                pass


# ============================================================================
# Burst overlap + aggregate metrics (Sections 19-20)
# ============================================================================

def _victim_burst_overlap(victim: dict, burst_interval: dict | None) -> dict:
    if burst_interval is None:
        return {"overlap_ms": 0.0, "overlaps_burst": False}
    v_start = victim.get("request_start_ns")
    v_end = victim.get("stream_end_ns")
    if v_start is None or v_end is None:
        return {"overlap_ms": 0.0, "overlaps_burst": False}
    overlap_start = max(v_start, burst_interval["start_ns"])
    overlap_end = min(v_end, burst_interval["end_ns"])
    overlap_ns = max(0, overlap_end - overlap_start)
    overlap_ms = overlap_ns / 1e6
    return {"overlap_ms": overlap_ms, "overlaps_burst": overlap_ms > 0}


def _aggregate_metrics(
    victim_results: list[dict], burst_results: list[dict], burst_interval: dict | None
) -> dict:
    complete_victims = [r for r in victim_results if r.get("status") == REQUEST_STATUS_COMPLETE]
    complete_bursts = [r for r in burst_results if r.get("status") == REQUEST_STATUS_COMPLETE]

    ttfts = [r["ttft_ms"] for r in complete_victims if r.get("ttft_ms") is not None]
    tpots = [
        r["client_observed_tpot_ms"]
        for r in complete_victims
        if r.get("client_observed_tpot_ms") is not None
    ]
    e2els = [r["e2el_ms"] for r in complete_victims if r.get("e2el_ms") is not None]

    itl_all: list[float] = []
    for r in complete_victims:
        if r.get("itl_available") and r.get("itl_ms"):
            itl_all.extend(r["itl_ms"])

    batch_interarrival_all: list[float] = []
    for r in complete_victims:
        if r.get("token_batch_interarrival_ms"):
            batch_interarrival_all.extend(r["token_batch_interarrival_ms"])

    complete_output_tokens = sum(len(r.get("output_token_ids") or []) for r in complete_victims)
    if complete_victims:
        span_s = (
            max(r["stream_end_ns"] for r in complete_victims)
            - min(r["request_start_ns"] for r in complete_victims)
        ) / 1e9
    else:
        span_s = None
    throughput = (complete_output_tokens / span_s) if span_s and span_s > 0 else None

    per_victim_overlap = [_victim_burst_overlap(r, burst_interval) for r in victim_results]

    burst_output_tokens = sum(len(r.get("output_token_ids") or []) for r in complete_bursts)
    burst_duration_ms = None
    if burst_interval is not None:
        burst_duration_ms = (burst_interval["end_ns"] - burst_interval["start_ns"]) / 1e6

    # --- Chunk-Budget-Screen additional burst aggregates --------------------
    # Computed purely from the burst requests' own timestamps/records (not
    # burst_interval, which reflects the victim-observed injection window).
    burst_e2els = [r["e2el_ms"] for r in complete_bursts if r.get("e2el_ms") is not None]
    burst_ttfts = [r["ttft_ms"] for r in complete_bursts if r.get("ttft_ms") is not None]
    burst_tpots = [
        r["client_observed_tpot_ms"]
        for r in complete_bursts
        if r.get("client_observed_tpot_ms") is not None
    ]
    if complete_bursts:
        burst_makespan_ms = (
            max(r["stream_end_ns"] for r in complete_bursts)
            - min(r["request_start_ns"] for r in complete_bursts)
        ) / 1e6
    else:
        burst_makespan_ms = None
    burst_input_tokens_total = sum(
        r.get("expected_prompt_tokens") or 0 for r in burst_results
    )
    burst_output_tokens_total = sum(len(r.get("output_token_ids") or []) for r in burst_results)

    return {
        "victim_ttft_ms": {
            "median": _median(ttfts), "p95": _percentile(ttfts, 0.95), "p99": _percentile(ttfts, 0.99),
        },
        "victim_client_observed_tpot_ms": {
            "median": _median(tpots), "p95": _percentile(tpots, 0.95), "p99": _percentile(tpots, 0.99),
        },
        "victim_e2el_ms": {
            "median": _median(e2els), "p95": _percentile(e2els, 0.95), "p99": _percentile(e2els, 0.99),
        },
        "victim_itl_ms": {
            "median": _median(itl_all), "p95": _percentile(itl_all, 0.95),
            "p99": _percentile(itl_all, 0.99), "n": len(itl_all),
        },
        "victim_token_batch_interarrival_ms": {
            "median": _median(batch_interarrival_all), "n": len(batch_interarrival_all),
        },
        "victim_complete_output_tokens": complete_output_tokens,
        "victim_throughput_tokens_per_s": throughput,
        "victim_complete_request_count": len(complete_victims),
        "victim_incomplete_request_count": len(victim_results) - len(complete_victims),
        "victim_overlap": per_victim_overlap,
        "burst_output_tokens": burst_output_tokens,
        "burst_duration_ms": burst_duration_ms,
        "burst_complete_request_count": len(complete_bursts),
        "burst_incomplete_request_count": len(burst_results) - len(complete_bursts),
        "burst_cost_requests": len(burst_results),
        "burst_cost_tokens": sum(len(r.get("output_token_ids") or []) for r in burst_results),
        "burst_makespan_ms": burst_makespan_ms,
        "burst_e2el_median_ms": _median(burst_e2els),
        "burst_e2el_p95_ms": _percentile(burst_e2els, 0.95),
        "burst_ttft_median_ms": _median(burst_ttfts),
        "burst_tpot_median_ms": _median(burst_tpots),
        "burst_input_tokens_total": burst_input_tokens_total,
        "burst_output_tokens_total": burst_output_tokens_total,
    }


def _build_episode_result(
    *,
    episode: Episode,
    schedule_fingerprint: str,
    server_metadata: dict,
    stabilization_ref: dict,
    trigger: dict,
    burst_interval: dict | None,
    victim_results: list[dict],
    burst_results: list[dict],
    status: str,
    validation_errors: list[str],
) -> dict:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "record_type": RECORD_TYPE_REGULAR_EPISODE,
        "run_mode": RUN_MODE_SMOKE,
        "schedule_fingerprint": schedule_fingerprint,
        "episode_id": episode.episode_id,
        "schedule_row": asdict(episode),
        "block_id": episode.block_id,
        "server_metadata": server_metadata,
        "stabilization_reference": stabilization_ref,
        "timestamps": {"trigger_utc": trigger.get("trigger_utc")},
        "trigger": trigger,
        "burst_interval": burst_interval,
        "victim_requests": victim_results,
        "burst_requests": burst_results,
        "aggregate_metrics": _aggregate_metrics(victim_results, burst_results, burst_interval),
        "status": status,
        "validation_errors": validation_errors,
    }


# ============================================================================
# Regular episode runner (Sections 16-20)
# ============================================================================

async def run_regular_episode(
    ctx: RunContext,
    episode: Episode,
    *,
    schedule_fingerprint: str,
    server_metadata: dict,
    stabilization_ref: dict,
) -> dict:
    concurrency = episode.concurrency
    n = episode.victim_request_count
    first_wave_size = min(concurrency, n)
    first_wave = set(range(first_wave_size))
    first_token_events = {i: asyncio.Event() for i in first_wave}
    sem = asyncio.Semaphore(concurrency)
    victim_tasks: dict[int, asyncio.Task] = {}
    burst_tasks: list[asyncio.Task] = []

    def _cb_factory(i: int) -> Callable[[], None]:
        def _cb() -> None:
            ev = first_token_events.get(i)
            if ev is not None and not ev.is_set():
                ev.set()

        return _cb

    async def _victim(i: int) -> dict:
        async with sem:
            return await _run_victim_request(ctx, episode, i, on_first_token=_cb_factory(i))

    try:
        # Section 7: explicit first wave. Do not rely solely on implicit
        # semaphore-acquisition fairness -- start exactly victim indices
        # 0..concurrency-1 first, yield control so they actually get a
        # chance to acquire their semaphore slots, and only then create/
        # activate the remaining deterministic queue (indices
        # concurrency..n-1), which will block on the same semaphore until a
        # first-wave slot frees up. This does not change the scientific
        # schedule order or any seed derivation -- only the startup
        # sequencing guarantee.
        for i in range(first_wave_size):
            victim_tasks[i] = asyncio.create_task(_victim(i))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        for i in range(first_wave_size, n):
            victim_tasks[i] = asyncio.create_task(_victim(i))
        await asyncio.sleep(0)

        trigger_start_ns = ctx.clock.perf_counter_ns()
        trigger_status = await _watch_trigger(
            first_wave, first_token_events, victim_tasks, ctx.trigger_timeout_s
        )
        trigger_ns = ctx.clock.perf_counter_ns()
        trigger_utc = ctx.clock.utcnow_iso()
        trigger = {
            "status": trigger_status,
            "trigger_utc": trigger_utc,
            "trigger_perf_ns": trigger_ns,
            "waited_ms": (trigger_ns - trigger_start_ns) / 1e6,
        }

        if trigger_status != "ok":
            # Section 6: cancel everything, wait for cancellation to fully
            # settle, then collect *all* 20 victim task outcomes (not just
            # the first wave) -- tasks that never got far enough to build
            # their own record become minimal, clearly-identified 'cancelled'
            # records, and no raw data from tasks that did start is discarded.
            await cancel_all(victim_tasks.values())
            victim_results_raw = await asyncio.gather(
                *[victim_tasks[i] for i in range(n)], return_exceptions=True
            )
            victim_results = _enrich_minimal_records(
                _coerce_task_results(victim_results_raw), episode_id=episode.episode_id, role="victim"
            )
            return _build_episode_result(
                episode=episode,
                schedule_fingerprint=schedule_fingerprint,
                server_metadata=server_metadata,
                stabilization_ref=stabilization_ref,
                trigger=trigger,
                burst_interval=None,
                victim_results=victim_results,
                burst_results=[],
                status="failed",
                validation_errors=[f"trigger failed: {trigger_status}"],
            )

        if episode.condition == BURST_CONDITION:
            for j in range(episode.burst_parallel):
                burst_tasks.append(asyncio.create_task(_run_burst_request(ctx, episode, j)))

        victim_results_raw = await asyncio.gather(
            *[victim_tasks[i] for i in range(n)], return_exceptions=True
        )
        burst_results_raw = (
            await asyncio.gather(*burst_tasks, return_exceptions=True) if burst_tasks else []
        )
    except asyncio.CancelledError:
        # Section 9: a cancellation can arrive at ANY point above -- while
        # tasks are still being created, while waiting for the trigger
        # (asyncio.wait(), used inside _watch_trigger, does NOT auto-cancel
        # the tasks it was waiting on the way gather() does), or while
        # awaiting either results gather. Regardless of phase, make sure
        # every victim/burst task actually created so far is cancelled and
        # its cleanup (including closing its SSE generator) is awaited
        # before the cancellation propagates to the caller.
        await cancel_all(list(victim_tasks.values()) + burst_tasks)
        raise

    victim_results = _coerce_task_results(victim_results_raw)
    burst_results = _coerce_task_results(burst_results_raw)

    all_complete = all(r.get("status") == REQUEST_STATUS_COMPLETE for r in victim_results) and all(
        r.get("status") == REQUEST_STATUS_COMPLETE for r in burst_results
    )

    validation_errors: list[str] = []
    if not all_complete:
        validation_errors.append("one or more requests after trigger were not complete")

    burst_interval = None
    if burst_results:
        starts = [r["request_start_ns"] for r in burst_results if r.get("request_start_ns") is not None]
        ends = [r["stream_end_ns"] for r in burst_results if r.get("stream_end_ns") is not None]
        if starts and ends:
            burst_interval = {"start_ns": min(starts), "end_ns": max(ends)}

    status = REQUEST_STATUS_COMPLETE if all_complete else "failed"

    return _build_episode_result(
        episode=episode,
        schedule_fingerprint=schedule_fingerprint,
        server_metadata=server_metadata,
        stabilization_ref=stabilization_ref,
        trigger=trigger,
        burst_interval=burst_interval,
        victim_results=victim_results,
        burst_results=burst_results,
        status=status,
        validation_errors=validation_errors,
    )


# ============================================================================
# Stabilization runner (Section 15)
# ============================================================================

async def run_stabilization(
    ctx: RunContext,
    bundle: LoadedBundle,
    model_key: str,
    block_id: str,
    offload_gb: int,
    state_label: str,
    *,
    server_metadata: dict,
) -> dict:
    bundle_seed = bundle.json_obj["seed"]
    sem = asyncio.Semaphore(STABILIZATION_CONCURRENCY)

    async def _one(i: int) -> dict:
        async with sem:
            return await _run_stabilization_request(ctx, bundle_seed, model_key, block_id, i)

    tasks = [asyncio.create_task(_one(i)) for i in range(STABILIZATION_REQUEST_COUNT)]
    try:
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        # Section 9: same protection principle as run_regular_episode --
        # explicitly cancel and await cleanup (including SSE generator
        # closing) before letting the cancellation propagate.
        await cancel_all(tasks)
        raise
    results = _coerce_task_results(raw_results)

    functional_passed = all(r.get("status") == REQUEST_STATUS_COMPLETE for r in results)

    first_half = list(range(0, 10))
    second_half = list(range(10, 20))

    def _half_medians(indices: list[int]) -> dict:
        subset = [results[i] for i in indices]
        ttfts = [r["ttft_ms"] for r in subset if r.get("ttft_ms") is not None]
        tpots = [
            r["client_observed_tpot_ms"] for r in subset if r.get("client_observed_tpot_ms") is not None
        ]
        e2els = [r["e2el_ms"] for r in subset if r.get("e2el_ms") is not None]
        return {
            "median_ttft_ms": _median(ttfts),
            "median_tpot_ms": _median(tpots),
            "median_e2el_ms": _median(e2els),
        }

    first_half_medians = _half_medians(first_half)
    second_half_medians = _half_medians(second_half)

    def _rel_change(key: str) -> float | None:
        a = first_half_medians[key]
        b = second_half_medians[key]
        if a is None or b is None or a == 0:
            return None
        return (b - a) / a

    relative_change = {
        k: _rel_change(k) for k in ("median_ttft_ms", "median_tpot_ms", "median_e2el_ms")
    }

    status = REQUEST_STATUS_COMPLETE if functional_passed else "failed"

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "record_type": RECORD_TYPE_STABILIZATION,
        "run_mode": RUN_MODE_SMOKE,
        "schedule_fingerprint": bundle.fingerprint,
        "block_id": block_id,
        "model": model_key,
        "offload_gb": offload_gb,
        "state_label": state_label,
        "excluded_from_analysis": True,
        "counted_repeat": False,
        "stabilization_configuration": {
            "condition": STABILIZATION_CONDITION,
            "concurrency": STABILIZATION_CONCURRENCY,
            "request_count": STABILIZATION_REQUEST_COUNT,
            "input_len": STABILIZATION_INPUT_LEN,
            "output_len": STABILIZATION_OUTPUT_LEN,
            "temperature": STABILIZATION_TEMPERATURE,
        },
        "server_metadata": server_metadata,
        "request_results": results,
        "functional_passed": functional_passed,
        "stabilization_passed": functional_passed,
        "first_half": first_half,
        "second_half": second_half,
        "first_half_medians": first_half_medians,
        "second_half_medians": second_half_medians,
        "relative_change": relative_change,
        "status": status,
    }


# ============================================================================
# Server lifecycle (Section 6-7)
# ============================================================================

class ServerHandle:
    def __init__(
        self, process: subprocess.Popen, pid: int, pgid: int, log_fh, cmd: list[str], start_utc: str
    ) -> None:
        self.process = process
        self.pid = pid
        self.pgid = pgid
        self.log_fh = log_fh
        self.cmd = cmd
        self.start_utc = start_utc

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def terminate_group(self) -> None:
        os.killpg(self.pgid, signal.SIGTERM)

    def kill_group(self) -> None:
        os.killpg(self.pgid, signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def close_log(self) -> None:
        try:
            self.log_fh.close()
        except OSError:
            pass


@runtime_checkable
class ServerProcessAdapter(Protocol):
    def start(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> Any: ...


class RealServerProcessAdapter:
    def start(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> ServerHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab", buffering=0)
        process = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True,
            env=env,
        )
        pgid = os.getpgid(process.pid)
        return ServerHandle(
            process, process.pid, pgid, log_fh, cmd, datetime.now(timezone.utc).isoformat()
        )


class FakeServerHandle:
    """In-process stand-in for ServerHandle: no real subprocess, no GPU.
    `alive` starts True and flips False once terminate_group()/
    kill_group() is called, simulating a cooperative process. Test-only
    knobs: `raise_on_terminate`/`raise_on_kill` simulate a signal
    hitting an already-gone process; `dies_on_kill=False` simulates a
    process that never actually exits even after SIGKILL."""

    def __init__(
        self,
        cmd: list[str],
        *,
        raise_on_terminate: BaseException | None = None,
        raise_on_kill: BaseException | None = None,
        dies_on_terminate: bool = True,
        dies_on_kill: bool = True,
        already_dead: bool = False,
    ) -> None:
        self.cmd = cmd
        self.pid = 424242
        self.pgid = 424242
        self.alive = not already_dead
        self.terminated = False
        self.killed = False
        self.start_utc = "1970-01-01T00:00:00Z"
        self.log_closed = False
        self._raise_on_terminate = raise_on_terminate
        self._raise_on_kill = raise_on_kill
        self._dies_on_terminate = dies_on_terminate
        self._dies_on_kill = dies_on_kill

    def is_alive(self) -> bool:
        return self.alive

    def terminate_group(self) -> None:
        self.terminated = True
        if self._raise_on_terminate is not None:
            self.alive = False  # a ProcessLookupError means the process is confirmed gone
            raise self._raise_on_terminate
        if self._dies_on_terminate:
            self.alive = False

    def kill_group(self) -> None:
        self.killed = True
        if self._raise_on_kill is not None:
            self.alive = False
            raise self._raise_on_kill
        if self._dies_on_kill:
            self.alive = False

    def wait(self, timeout: float | None = None) -> int | None:
        return 0

    def close_log(self) -> None:
        self.log_closed = True


class FakeServerProcessAdapter:
    def __init__(self) -> None:
        self.started: list[FakeServerHandle] = []
        self.started_envs: list[dict[str, str] | None] = []

    def start(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> FakeServerHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        budget = (env or {}).get("MAX_NUM_BATCHED_TOKENS")
        fake_log_text = (
            f"Chunked prefill is enabled with max_num_batched_tokens={budget}.\n"
            if budget is not None
            else ""
        )
        log_path.write_text(fake_log_text, encoding="utf-8")
        handle = FakeServerHandle(cmd)
        self.started.append(handle)
        self.started_envs.append(env)
        return handle


def build_server_command(
    run_server_path: Path, model_key: str, offload_gb: int, host: str, port: int
) -> list[str]:
    return ["bash", str(run_server_path), model_key, str(offload_gb), host, str(port)]


CHUNKED_PREFILL_LOG_PATTERN = re.compile(
    r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)\."
)
ENABLE_CHUNKED_PREFILL_PATTERN = re.compile(r"enable_chunked_prefill=(True|False)")


def check_chunked_prefill_log(log_path: Path, expected_budget: int) -> list[str]:
    """Design requirement (SERVERWRAPPER section): before stabilization,
    the server log must confirm chunked prefill is enabled with exactly
    the block's max_num_batched_tokens budget. A missing or mismatched
    confirmation is a functional failure and must abort the whole
    block -- never silently proceed on an unconfirmed or wrong budget.
    Also cross-checks enable_chunked_prefill=True wherever that flag
    appears in the log (optional -- only enforced if present at all)."""
    errors: list[str] = []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [
            f"could not read server log {log_path} to verify the "
            f"chunked-prefill budget confirmation: {exc}"
        ]

    matches = CHUNKED_PREFILL_LOG_PATTERN.findall(text)
    if not matches:
        errors.append(
            "server log does not contain a 'Chunked prefill is enabled "
            f"with max_num_batched_tokens=...' confirmation line; "
            f"expected budget {expected_budget}"
        )
    else:
        found_budgets = {int(m) for m in matches}
        if found_budgets != {expected_budget}:
            errors.append(
                f"server log confirms chunked-prefill budget(s) "
                f"{sorted(found_budgets)}, expected exactly "
                f"{{{expected_budget}}}"
            )

    enable_matches = ENABLE_CHUNKED_PREFILL_PATTERN.findall(text)
    if enable_matches and any(m != "True" for m in enable_matches):
        errors.append(
            "server log contains enable_chunked_prefill=False somewhere "
            f"(found: {sorted(set(enable_matches))}), expected True"
        )

    return errors


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            return False
        except OSError:
            return True


async def stop_server(
    handle,
    host: str,
    port: int,
    sleeper: Sleeper,
    timeout_s: float = SERVER_STOP_TIMEOUT_S,
    port_free_check: Callable[[str, int], bool] = is_port_free,
    kill_confirm_timeout_s: float = 5.0,
    port_poll_timeout_s: float = 30.0,
) -> dict:
    """Section 3: targeted, verified server-process-group stop. Signals
    only the server's own PGID, never a global kill. ProcessLookupError
    on either signal is treated as 'process already gone', not an
    error. Declares success only once the process is confirmed dead
    AND the port is confirmed free (polled for a short bounded time)."""
    stop_start_utc = datetime.now(timezone.utc).isoformat()
    term_sent = False
    forced_kill = False
    stop_error: str | None = None

    try:
        if handle.is_alive():
            try:
                handle.terminate_group()
                term_sent = True
            except ProcessLookupError:
                term_sent = False
            except Exception as exc:  # noqa: BLE001 -- the stop path must never crash
                stop_error = f"SIGTERM error: {type(exc).__name__}"

        stop_deadline = time.monotonic() + timeout_s
        while handle.is_alive() and time.monotonic() < stop_deadline:
            await sleeper.sleep(0.2)

        if handle.is_alive():
            try:
                handle.kill_group()
                forced_kill = True
            except ProcessLookupError:
                forced_kill = False
            except Exception as exc:  # noqa: BLE001
                stop_error = (
                    (stop_error + "; ") if stop_error else ""
                ) + f"SIGKILL error: {type(exc).__name__}"

            kill_deadline = time.monotonic() + kill_confirm_timeout_s
            while handle.is_alive() and time.monotonic() < kill_deadline:
                await sleeper.sleep(0.2)

        try:
            handle.wait(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            stop_error = (
                (stop_error + "; ") if stop_error else ""
            ) + f"wait() error: {type(exc).__name__}"

        alive_after_stop = handle.is_alive()

        port_free_after_stop = port_free_check(host, port)
        if not port_free_after_stop:
            port_poll_deadline = time.monotonic() + port_poll_timeout_s
            while not port_free_after_stop and time.monotonic() < port_poll_deadline:
                await sleeper.sleep(0.2)
                port_free_after_stop = port_free_check(host, port)

        stop_success = (not alive_after_stop) and port_free_after_stop

        return {
            "stop_start_utc": stop_start_utc,
            "stop_end_utc": datetime.now(timezone.utc).isoformat(),
            "pid": handle.pid,
            "pgid": handle.pgid,
            "term_sent": term_sent,
            "forced_kill": forced_kill,
            "alive_after_stop": alive_after_stop,
            "port_free_after_stop": port_free_after_stop,
            "stop_success": stop_success,
            "stop_error": stop_error,
        }
    finally:
        handle.close_log()


async def wait_for_server_ready(
    transport: HTTPTransport,
    handle,
    base_url: str,
    api_key: str,
    model_full_id: str,
    sleeper: Sleeper,
    *,
    timeout_s: float = SERVER_READY_TIMEOUT_S,
    poll_interval_s: float = READINESS_POLL_INTERVAL_S,
) -> dict:
    """Polls /health then /v1/models until the expected model is up, or
    raises ServerLifecycleError. Transient connection-level errors
    (connection refused, OS-level timeouts, transport errors) while the
    server process is still alive and the overall timeout has not
    elapsed are expected during startup and are logged + polled through,
    never raised. asyncio.CancelledError/KeyboardInterrupt/SystemExit are
    always re-raised immediately. No error message here ever includes
    the API key -- only exception *type names* are recorded, never
    str(exc), since transport-level exceptions can echo request details."""
    readiness_start_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    poll_count = 0
    last_health_status: object = None
    last_models_status: object = None
    last_transient_error_type: str | None = None

    while True:
        if not handle.is_alive():
            raise ServerLifecycleError(
                "server process exited before becoming ready "
                f"(poll_count={poll_count})"
            )
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            raise ServerLifecycleError(
                f"server did not become ready within {timeout_s}s "
                f"(poll_count={poll_count}, last_health_status={last_health_status!r}, "
                f"last_models_status={last_models_status!r}, "
                f"last_transient_error_type={last_transient_error_type!r})"
            )

        poll_count += 1

        try:
            health_status, _ = await transport.get_json(
                base_url.rstrip("/") + HEALTH_ENDPOINT, {}, 5.0
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 -- transient startup errors are expected and polled through
            last_transient_error_type = type(exc).__name__
            last_health_status = f"error:{type(exc).__name__}"
            await sleeper.sleep(poll_interval_s)
            continue

        last_health_status = health_status
        if health_status == 200:
            try:
                models_status, models_body = await transport.get_json(
                    base_url.rstrip("/") + MODELS_ENDPOINT, auth_headers, 5.0
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001
                last_transient_error_type = type(exc).__name__
                last_models_status = f"error:{type(exc).__name__}"
                await sleeper.sleep(poll_interval_s)
                continue

            last_models_status = models_status
            if models_status == 200:
                model_ids: set[str] = set()
                if isinstance(models_body, dict):
                    for m in models_body.get("data", []) or []:
                        if isinstance(m, dict) and "id" in m:
                            model_ids.add(m["id"])
                if model_full_id in model_ids:
                    return {
                        "server_start_utc": handle.start_utc,
                        "readiness_start_utc": readiness_start_utc,
                        "ready_utc": datetime.now(timezone.utc).isoformat(),
                        "readiness_duration_s": elapsed,
                        "health_status": health_status,
                        "models_response": {"data": sorted(model_ids)},
                        "detected_model": model_full_id,
                        "poll_count": poll_count,
                        "last_health_status": last_health_status,
                        "last_models_status": last_models_status,
                        "last_transient_error_type": last_transient_error_type,
                    }
        await sleeper.sleep(poll_interval_s)


async def check_post_stabilization_health(transport: HTTPTransport, base_url: str) -> dict:
    """Section 2: exactly one /health check between a fully-written
    stabilization result and the cooldown. Only HTTP 200 releases the
    block. Connection-level exceptions are caught and reported as a
    failed gate, never propagated as a raw crash."""
    checked_utc = datetime.now(timezone.utc).isoformat()
    try:
        status, _ = await transport.get_json(base_url.rstrip("/") + HEALTH_ENDPOINT, {}, 10.0)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 -- must gate cleanly, not crash the block
        return {
            "checked_utc": checked_utc,
            "http_status": None,
            "ok": False,
            "error_type": type(exc).__name__,
        }
    return {
        "checked_utc": checked_utc,
        "http_status": status,
        "ok": status == 200,
        "error_type": None,
    }


async def fetch_capability_summary(transport: HTTPTransport, base_url: str, api_key: str) -> dict:
    try:
        status, body = await transport.get_json(
            base_url.rstrip("/") + OPENAPI_ENDPOINT, {"Authorization": f"Bearer {api_key}"}, 10.0
        )
    except Exception as exc:  # noqa: BLE001 -- informational only, never fatal
        return {"available": False, "error": str(exc)}
    if status != 200 or not isinstance(body, dict):
        return {"available": False, "http_status": status}
    paths = body.get("paths", {})
    return {
        "available": True,
        "http_status": status,
        "has_completions_path": isinstance(paths, dict) and COMPLETIONS_ENDPOINT in paths,
        "path_count": len(paths) if isinstance(paths, dict) else None,
        "note": "informational only -- the definitive capability check is the stabilization run",
    }


def resolve_run_server_path(script_path: Path = SCRIPT_PATH) -> Path:
    return script_path.parent / "run_server.sh"


def check_run_server_script(run_server_path: Path) -> None:
    if not run_server_path.is_file() or not os.access(run_server_path, os.R_OK):
        raise ServerLifecycleError(
            f"run_server.sh '{run_server_path}' does not exist as a readable regular file"
        )
    if shutil.which("bash") is None:
        raise ServerLifecycleError("'bash' is not available in PATH")


def read_api_key_from_env(env: dict[str, str] | None = None) -> str:
    e = env if env is not None else os.environ
    key = e.get("VLLM_API_KEY", "")
    if not key:
        raise ApiKeyError(
            "VLLM_API_KEY is missing or empty in the environment; refusing to "
            "start a server without an API key"
        )
    return key


# ============================================================================
# Environment fingerprinting (Stage 3, Section 7 + precision addendum)
#
# The environment_fingerprint exists so a --resume can never silently
# continue on a different physical GPU or a different relevant code/
# software state. It is computed from: the resolved *physical* GPU
# (UUID/model/memory/driver -- never the raw CUDA_VISIBLE_DEVICES
# string, which is just an index/alias), interpreter/library versions,
# kernel, git commit, and explicit SHA-256 hashes of the seven frozen
# runner/schedule files. Timestamps, the output path, and the tracked
# git-dirty boolean are deliberately excluded from the hash (dirty is
# still recorded, informatively, in the manifest). The API key is never
# part of this module at all.
# ============================================================================

def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _safe_run_text(cmd: list[str], timeout_s: float = 10.0, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(cwd) if cwd else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _safe_package_version(name: str) -> str | None:
    try:
        import importlib.metadata as _im
        return _im.version(name)
    except Exception:  # noqa: BLE001 -- an uninstalled/unintrospectable package is not fatal
        return None


def _query_nvidia_smi_gpus() -> list[dict]:
    out = _safe_run_text(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"]
    )
    gpus: list[dict] = []
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, uuid, memory_total, driver_version = parts
        gpus.append(
            {
                "index": index, "name": name, "uuid": uuid,
                "memory_total": memory_total, "driver_version": driver_version,
            }
        )
    return gpus


def _resolve_visible_gpu(gpu_list: list[dict], cuda_visible_devices: str | None) -> dict | None:
    """Resolves CUDA_VISIBLE_DEVICES (an index, a UUID, or unset) against
    the full nvidia-smi GPU list to find the one *physical* GPU that is
    actually visible/measuring. This is what goes into the fingerprint --
    never the raw env var string, which is meaningless across machines."""
    if not gpu_list:
        return None
    if not cuda_visible_devices:
        for g in gpu_list:
            if g["index"] == "0":
                return g
        return gpu_list[0]
    first = cuda_visible_devices.split(",")[0].strip()
    for g in gpu_list:
        if g["index"] == first:
            return g
    for g in gpu_list:
        if g["uuid"] == first or g["uuid"] == first.replace("GPU-", "") or first.endswith(g["uuid"]):
            return g
    return None


@runtime_checkable
class EnvironmentProbe(Protocol):
    def gather(self, schedule_dir: Path) -> dict: ...


class RealEnvironmentProbe:
    """Real environment probe: shells out to nvidia-smi/git, reads
    interpreter/library versions, and hashes the seven frozen files. Never
    raises -- every field independently degrades to None on failure."""

    def gather(self, schedule_dir: Path) -> dict:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_list = _query_nvidia_smi_gpus()
        resolved_gpu = _resolve_visible_gpu(gpu_list, cuda_visible)

        git_commit = _safe_run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
        git_status = _safe_run_text(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"]
        )
        tracked_git_dirty = (git_status != "") if git_status is not None else None

        file_hash_targets = {
            "make_chunk_budget_schedule.py": SCRIPT_PATH.parent / "make_chunk_budget_schedule.py",
            "run_chunk_budget_screen.py": SCRIPT_PATH,
            "run_chunk_budget_screen.sh": SCRIPT_PATH.parent / "run_chunk_budget_screen.sh",
            "run_server.sh": SCRIPT_PATH.parent / "run_server.sh",
            "chunk_budget_schedule.json": schedule_dir / "chunk_budget_schedule.json",
            "chunk_budget_schedule.csv": schedule_dir / "chunk_budget_schedule.csv",
            "chunk_budget_schedule_audit.txt": schedule_dir / "chunk_budget_schedule_audit.txt",
        }
        file_hashes = {name: _sha256_file(p) for name, p in file_hash_targets.items()}

        return {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "kernel": platform.release(),
            "git_commit": git_commit,
            "git_dirty": tracked_git_dirty,
            "cuda_visible_devices": cuda_visible,
            "vllm_version": _safe_package_version("vllm"),
            "torch_version": _safe_package_version("torch"),
            "transformers_version": _safe_package_version("transformers"),
            "httpx_version": _safe_package_version("httpx"),
            "gpu_list": gpu_list,
            "resolved_gpu": resolved_gpu,
            "file_hashes": file_hashes,
        }


class FakeEnvironmentProbe:
    """Deterministic environment probe for tests -- no nvidia-smi, no
    git, no real file hashing, no network."""

    def __init__(self, env: dict | None = None) -> None:
        self.env: dict = env if env is not None else {
            "python_executable": "/fake/bin/python3",
            "python_version": "3.12.0",
            "platform": "FakeLinux-6.0.0-x86_64",
            "hostname": "fake-host",
            "kernel": "6.0.0-fake",
            "git_commit": "f" * 40,
            "git_dirty": False,
            "cuda_visible_devices": "0",
            "vllm_version": "0.17.1",
            "torch_version": "2.5.0",
            "transformers_version": "4.45.0",
            "httpx_version": "0.27.0",
            "gpu_list": [
                {
                    "index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake-0000-0000",
                    "memory_total": "24576 MiB", "driver_version": "550.00",
                }
            ],
            "resolved_gpu": {
                "index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake-0000-0000",
                "memory_total": "24576 MiB", "driver_version": "550.00",
            },
            "file_hashes": {
                "make_chunk_budget_schedule.py": "9" * 64,
                "run_chunk_budget_screen.py": "a" * 64,
                "run_chunk_budget_screen.sh": "b" * 64,
                "run_server.sh": "c" * 64,
                "chunk_budget_schedule.json": "d" * 64,
                "chunk_budget_schedule.csv": "e" * 64,
                "chunk_budget_schedule_audit.txt": "0" * 64,
            },
        }

    def gather(self, schedule_dir: Path) -> dict:
        return json.loads(json.dumps(self.env))  # deep copy, no shared mutable state across calls


def compute_environment_fingerprint(env: dict) -> str:
    resolved = env.get("resolved_gpu") or {}
    payload = {
        "gpu_uuid": resolved.get("uuid"),
        "gpu_model": resolved.get("name"),
        "gpu_memory_total": resolved.get("memory_total"),
        "gpu_driver_version": resolved.get("driver_version"),
        "python_version": env.get("python_version"),
        "vllm_version": env.get("vllm_version"),
        "torch_version": env.get("torch_version"),
        "transformers_version": env.get("transformers_version"),
        "httpx_version": env.get("httpx_version"),
        "kernel": env.get("kernel"),
        "git_commit": env.get("git_commit"),
        "file_hashes": dict(sorted((env.get("file_hashes") or {}).items())),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


EXPECTED_ENVIRONMENT_FILE_HASH_NAMES = frozenset(
    {
        "make_chunk_budget_schedule.py",
        "run_chunk_budget_screen.py",
        "run_chunk_budget_screen.sh",
        "run_server.sh",
        "chunk_budget_schedule.json",
        "chunk_budget_schedule.csv",
        "chunk_budget_schedule_audit.txt",
    }
)


def _is_hex_of_length(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdefABCDEF" for c in value)
    )


def validate_official_environment(env: dict) -> list[str]:
    """Section 1 (Stage-3 patch): a strict, isolated pre-flight gate. Both
    a fresh run and a resume must abort -- before any output-directory
    change and before any server start -- unless every one of these holds.
    None, empty strings, and an unresolved GPU identity are hard errors,
    never silently tolerated or defaulted."""
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

    for key in (
        "python_version", "vllm_version", "torch_version", "transformers_version",
        "httpx_version", "kernel",
    ):
        v = env.get(key)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{key} is missing or empty")

    file_hashes = env.get("file_hashes")
    if not isinstance(file_hashes, dict) or set(file_hashes.keys()) != EXPECTED_ENVIRONMENT_FILE_HASH_NAMES:
        errors.append(
            f"file_hashes must contain exactly the seven expected files: "
            f"{sorted(EXPECTED_ENVIRONMENT_FILE_HASH_NAMES)} (got {sorted(file_hashes.keys()) if isinstance(file_hashes, dict) else file_hashes!r})"
        )
    else:
        for name, h in file_hashes.items():
            if not _is_hex_of_length(h, 64):
                errors.append(f"file_hashes[{name!r}] is not a well-formed 64-character SHA-256 hex value")

    return errors


# ============================================================================
# Official run manifest (Section 7)
# ============================================================================

MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_RUN_MANIFEST_FILENAME = "official_run_manifest.json"
OFFICIAL_RUN_SUMMARY_FILENAME = "official_run_summary.json"
INTEGRITY_MANIFEST_FILENAME = "integrity_manifest.json"


def build_official_run_manifest(
    *, env: dict, bundle: LoadedBundle, run_mode: str, output_dir: Path, host: str, port: int, clock: Clock,
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


def load_json_file_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def validate_resume_manifest(
    existing: dict, *, bundle: LoadedBundle, run_mode: str, current_env: dict,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(existing, dict):
        return ["existing official_run_manifest.json is not a JSON object"]
    if existing.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest_schema_version mismatch")
    if existing.get("schedule_fingerprint") != bundle.fingerprint:
        errors.append("schedule_fingerprint mismatch")
    if existing.get("run_mode") != run_mode:
        errors.append("run_mode mismatch")
    current_fingerprint = compute_environment_fingerprint(current_env)
    if existing.get("environment_fingerprint") != current_fingerprint:
        errors.append(
            f"environment_fingerprint mismatch (manifest={existing.get('environment_fingerprint')!r}, "
            f"current={current_fingerprint!r}) -- refusing to resume on what may be "
            f"different code, GPU, or software"
        )
    return errors


# ============================================================================
# Integrity manifest (Section 11)
# ============================================================================

def _iter_output_files_for_integrity(output_dir: Path) -> list[Path]:
    files = [
        p for p in output_dir.rglob("*")
        if p.is_file() and p.name != INTEGRITY_MANIFEST_FILENAME
    ]
    return sorted(files, key=lambda p: p.relative_to(output_dir).as_posix())


def build_integrity_manifest(
    output_dir: Path, *, schedule_fingerprint: str, environment_fingerprint: str, clock: Clock,
) -> dict:
    entries: list[dict] = []
    episode_file_count = 0
    stabilization_file_count = 0
    block_summary_count = 0
    for p in _iter_output_files_for_integrity(output_dir):
        rel = p.relative_to(output_dir).as_posix()
        entries.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": _sha256_file(p)})
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


def verify_integrity_manifest(
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
    """Section 5: deep structural validation of the integrity manifest
    itself (not just a hash re-check against disk). A manifest that
    fails ANY of these structural/metadata checks is never treated as
    valid, regardless of whether the underlying files happen to still
    hash-match -- callers must never reseal such a manifest automatically."""
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
        actual_server_log_count = sum(
            1 for rel in rel_paths_in_order if rel.startswith("server_logs/")
        )
        if actual_server_log_count != expected_server_log_count:
            errors.append(
                f"server_logs file count {actual_server_log_count} != expected {expected_server_log_count}"
            )

    # A structurally broken manifest is never compared against disk --
    # there is nothing trustworthy left to compare.
    if errors:
        return False, errors

    current_files = {p.relative_to(output_dir).as_posix(): p for p in _iter_output_files_for_integrity(output_dir)}
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
        actual_hash = _sha256_file(p)
        if actual_size != entry.get("size_bytes"):
            errors.append(f"size mismatch for {rel}: manifest={entry.get('size_bytes')!r} actual={actual_size!r}")
        if actual_hash != entry.get("sha256"):
            errors.append(f"sha256 mismatch for {rel}: manifest={entry.get('sha256')!r} actual={actual_hash!r}")

    return (not errors), errors


# ============================================================================
# Deep validators for stabilization files and block summaries
# (Stage-3 patch, Sections 6/7). A bare status=="complete"/
# overall_status=="block_complete" is never sufficient on its own --
# these are used wherever a block is treated as already finished:
# skipping a complete block, no-op resume, reconstructing a missing
# integrity manifest, and the final campaign-wide complete decision.
# ============================================================================

def validate_complete_stabilization_file(
    obj: object,
    *,
    bundle: LoadedBundle,
    block_id: str,
    model_key: str,
    offload_gb: int,
    state_label: str,
    run_mode: str,
) -> list[str]:
    if not isinstance(obj, dict):
        return ["stabilization result is not a JSON object"]
    errors: list[str] = []

    def _eq(key: str, expected: object) -> None:
        if obj.get(key) != expected:
            errors.append(f"{key} {obj.get(key)!r} != expected {expected!r}")

    _eq("result_schema_version", RESULT_SCHEMA_VERSION)
    _eq("runner_version", RUNNER_VERSION)
    _eq("record_type", RECORD_TYPE_STABILIZATION)
    _eq("run_mode", run_mode)
    _eq("schedule_fingerprint", bundle.fingerprint)
    _eq("block_id", block_id)
    _eq("model", model_key)
    _eq("offload_gb", offload_gb)
    _eq("state_label", state_label)

    if obj.get("excluded_from_analysis") is not True:
        errors.append("excluded_from_analysis must be exactly True")
    if obj.get("counted_repeat") is not False:
        errors.append("counted_repeat must be exactly False")
    if obj.get("status") != REQUEST_STATUS_COMPLETE:
        errors.append(f"status {obj.get('status')!r} != 'complete'")
    if obj.get("functional_passed") is not True:
        errors.append("functional_passed must be exactly True")
    if obj.get("stabilization_passed") is not True:
        errors.append("stabilization_passed must be exactly True")

    expected_cfg = {
        "condition": STABILIZATION_CONDITION,
        "concurrency": STABILIZATION_CONCURRENCY,
        "request_count": STABILIZATION_REQUEST_COUNT,
        "input_len": STABILIZATION_INPUT_LEN,
        "output_len": STABILIZATION_OUTPUT_LEN,
        "temperature": STABILIZATION_TEMPERATURE,
    }
    if obj.get("stabilization_configuration") != expected_cfg:
        errors.append(
            f"stabilization_configuration {obj.get('stabilization_configuration')!r} != "
            f"expected {expected_cfg!r}"
        )

    results = obj.get("request_results")
    if not isinstance(results, list) or len(results) != STABILIZATION_REQUEST_COUNT:
        errors.append(f"request_results must be a list of exactly {STABILIZATION_REQUEST_COUNT} entries")
    else:
        bundle_seed = bundle.json_obj.get("seed")
        for i, record in enumerate(results):
            errors.extend(
                _validate_request_record_fields(
                    record,
                    episode_id=block_id,
                    role="stabilization",
                    request_index=i,
                    expected_prompt_seed=stabilization_prompt_seed(bundle_seed, model_key, block_id, i),
                    expected_generation_seed=stabilization_generation_seed(bundle_seed, model_key, block_id, i),
                    expected_prompt_tokens=STABILIZATION_INPUT_LEN,
                    expected_completion_tokens=STABILIZATION_OUTPUT_LEN,
                )
            )

    return errors


def validate_complete_block_summary(
    obj: object,
    *,
    bundle: LoadedBundle,
    block_id: str,
    model_key: str,
    offload_gb: int,
    state_label: str,
    repeat: int,
    run_mode: str,
) -> list[str]:
    if not isinstance(obj, dict):
        return ["block summary is not a JSON object"]
    errors: list[str] = []

    def _eq(key: str, expected: object) -> None:
        if obj.get(key) != expected:
            errors.append(f"{key} {obj.get(key)!r} != expected {expected!r}")

    _eq("block_id", block_id)
    _eq("model", model_key)
    _eq("offload_gb", offload_gb)
    _eq("state_label", state_label)
    _eq("repeat", repeat)
    _eq("run_mode", run_mode)
    _eq("schedule_fingerprint", bundle.fingerprint)

    expected_planned = [ep.episode_id for ep in find_block(bundle, block_id)]
    if obj.get("planned_episode_ids") != expected_planned:
        errors.append(
            f"planned_episode_ids {obj.get('planned_episode_ids')!r} != "
            f"expected schedule order {expected_planned!r}"
        )

    episode_statuses = obj.get("episode_statuses")
    if not isinstance(episode_statuses, dict) or any(
        episode_statuses.get(eid) != CLASSIFICATION_VALID_COMPLETE for eid in expected_planned
    ):
        errors.append("not every planned episode has episode_statuses == 'valid_complete'")

    if obj.get("stabilization_status") != REQUEST_STATUS_COMPLETE:
        errors.append(f"stabilization_status {obj.get('stabilization_status')!r} != 'complete'")

    psh = obj.get("post_stabilization_health")
    if not (isinstance(psh, dict) and psh.get("ok") is True):
        errors.append("post_stabilization_health.ok must be exactly True")

    if obj.get("cooldown_s") != COOLDOWN_S:
        errors.append(f"cooldown_s {obj.get('cooldown_s')!r} != expected {COOLDOWN_S!r}")

    if obj.get("overall_status") != "block_complete":
        errors.append(f"overall_status {obj.get('overall_status')!r} != 'block_complete'")

    server_stop = obj.get("server_stop")
    if not isinstance(server_stop, dict):
        errors.append("server_stop is missing or not a dict")
    else:
        if server_stop.get("stop_success") is not True:
            errors.append("server_stop.stop_success must be exactly True")
        if server_stop.get("alive_after_stop") is not False:
            errors.append("server_stop.alive_after_stop must be exactly False")
        if server_stop.get("port_free_after_stop") is not True:
            errors.append("server_stop.port_free_after_stop must be exactly True")

    return errors


def _validate_block_finalization(
    block_id: str, output_dir: Path, bundle: LoadedBundle, run_mode: str,
) -> tuple[list[str], list[str]]:
    """Returns (stabilization_errors, block_summary_errors) -- the deep
    validation results for a single block's two finalization artifacts."""
    block_episodes = find_block(bundle, block_id)
    model_key = block_episodes[0].model
    offload_gb = block_episodes[0].offload_gb
    state_label = block_episodes[0].state_label
    repeat = block_episodes[0].repeat

    stab_obj = load_json_file_or_none(stabilization_result_path(output_dir, block_id))
    if stab_obj is None:
        stab_errors = ["stabilization file missing or unreadable"]
    else:
        stab_errors = validate_complete_stabilization_file(
            stab_obj, bundle=bundle, block_id=block_id, model_key=model_key,
            offload_gb=offload_gb, state_label=state_label, run_mode=run_mode,
        )

    summary_obj = load_json_file_or_none(output_dir / "block_summaries" / f"{block_id}.json")
    if summary_obj is None:
        summary_errors = ["block_summary file missing or unreadable"]
    else:
        summary_errors = validate_complete_block_summary(
            summary_obj, bundle=bundle, block_id=block_id, model_key=model_key,
            offload_gb=offload_gb, state_label=state_label, repeat=repeat, run_mode=run_mode,
        )

    return stab_errors, summary_errors


# ============================================================================
# Smoke-block orchestration (Sections 3, 15-23)
# ============================================================================

def find_block(bundle: LoadedBundle, block_id: str) -> list[Episode]:
    return [ep for ep in bundle.episodes if ep.block_id == block_id]


def find_and_validate_smoke_block(bundle: LoadedBundle, block_id: str) -> list[Episode]:
    episodes = find_block(bundle, block_id)
    if not episodes:
        raise ValueError(
            f"--smoke-block {block_id!r} does not exist in the validated schedule bundle"
        )
    if len(episodes) != BLOCK_SIZE:
        raise ValueError(
            f"--smoke-block {block_id!r} does not form a complete block of size {BLOCK_SIZE}"
        )
    return episodes


def _classify_and_plan_block(
    block_episodes: list[Episode],
    bundle: LoadedBundle,
    output_dir: Path,
    run_mode: str,
    resume: bool,
) -> tuple[dict[str, str], list[Episode]]:
    """Section 2/5/6: shared resume-classification and fresh-run
    precondition logic for exactly one block's 4 episodes. Identical for
    smoke and official (only `run_mode` differs). Raises
    ServerLifecycleError -- before any server is started -- on:
      - fresh run: any existing episode/stabilization file for this block
      - resume: any episode classified partial/invalid/corrupted
    Returns (episode_statuses, episodes_to_run) in schedule order.
    """
    block_id = block_episodes[0].block_id
    episode_statuses: dict[str, str] = {}
    if resume:
        for ep in block_episodes:
            result_path = episode_result_path(output_dir, ep.episode_id)
            cls, _notes = classify_result_file(result_path, ep, bundle.fingerprint, run_mode)
            episode_statuses[ep.episode_id] = cls
        bad = {
            eid: cls
            for eid, cls in episode_statuses.items()
            if cls in (CLASSIFICATION_PARTIAL, CLASSIFICATION_INVALID, CLASSIFICATION_CORRUPTED)
        }
        if bad:
            raise ServerLifecycleError(
                f"--resume found non-resumable existing result file(s) for block "
                f"{block_id!r}: {bad}; refusing to silently overwrite. Fix or "
                f"remove them manually first."
            )
        episodes_to_run = [
            ep for ep in block_episodes if episode_statuses[ep.episode_id] != CLASSIFICATION_VALID_COMPLETE
        ]
    else:
        for ep in block_episodes:
            if episode_result_path(output_dir, ep.episode_id).exists():
                raise ServerLifecycleError(
                    f"result file for episode {ep.episode_id!r} already exists and "
                    f"--resume was not given; refusing to silently overwrite"
                )
        if stabilization_result_path(output_dir, block_id).exists():
            raise ServerLifecycleError(
                f"stabilization result for block {block_id!r} already exists and "
                f"--resume was not given"
            )
        episode_statuses = {ep.episode_id: CLASSIFICATION_MISSING for ep in block_episodes}
        episodes_to_run = list(block_episodes)
    return episode_statuses, episodes_to_run


async def _run_block_protocol(
    *,
    bundle: LoadedBundle,
    block_episodes: list[Episode],
    episodes_to_run: list[Episode],
    block_id: str,
    output_dir: Path,
    host: str,
    port: int,
    run_mode: str,
    api_key: str,
    transport: HTTPTransport,
    tokenizer: TokenizerAdapter,
    server_adapter: ServerProcessAdapter,
    sleeper: Sleeper,
    clock: Clock,
    run_server_path: Path,
    episode_statuses: dict[str, str],
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S,
    stop_kill_confirm_timeout_s: float = 5.0,
    stop_port_poll_timeout_s: float = 30.0,
    should_abort: Callable[[], bool] | None = None,
) -> dict:
    """Section 2: the reusable internal block executor. Identical
    scientific execution for --smoke-test and --official-run: server
    start -> readiness -> exactly one stabilization run (atomically
    written) -> post-stabilization health gate -> fixed cooldown ->
    `episodes_to_run` in schedule order (atomically written, aborting
    the block at the first non-complete episode) -> a verified server
    stop. Only `run_mode` (threaded into every written record) differs
    between callers.

    `episode_statuses` is the SAME dict the caller already built via
    `_classify_and_plan_block()`; it is mutated in place as episodes
    complete, so the caller always observes progress even if this
    function is cancelled.

    `should_abort`, if given, is checked immediately before
    `server_adapter.start()` (Section 8): if it returns True at that
    point, no server is ever started and this returns
    overall_status='interrupted' straight away.

    Never raises for execution-phase failures (readiness timeout,
    stabilization failure, episode failure, stop failure, or any other
    unexpected exception) -- those are reported via the returned dict's
    `overall_status`/`error`. The one exception is asyncio.CancelledError,
    which is caught, recorded as overall_status='interrupted', and then
    deliberately swallowed (not re-raised) so a cooperative caller (the
    official campaign orchestrator, cancelling this coroutine's own task
    to implement signal handling) always gets a complete, well-formed
    result dict back -- including a verified server stop -- instead of a
    propagating exception.
    """
    model_key = block_episodes[0].model
    offload_gb = block_episodes[0].offload_gb
    state_label = block_episodes[0].state_label
    budget = block_episodes[0].max_num_batched_tokens
    model_full_id = MODEL_FULL_ID[model_key]

    result: dict[str, Any] = {
        "server_start": None,
        "readiness": None,
        "chunked_prefill_log_check": None,
        "stabilization_status": "not_run",
        "post_stabilization_health": None,
        "cooldown_s": None,
        "server_stop": None,
        "overall_status": "not_run",
        "executed_episode_ids": [],
    }

    handle = None
    try:
        if should_abort is not None and should_abort():
            result["overall_status"] = "interrupted"
            return result

        if not is_port_free(host, port):
            raise ServerLifecycleError(f"port {port} on host {host!r} is already in use")

        cmd = build_server_command(run_server_path, model_key, offload_gb, host, port)
        log_path = output_dir / "server_logs" / f"{block_id}.log"
        # MAX_NUM_BATCHED_TOKENS is REQUIRED by run_server.sh and is
        # passed as an environment variable (not a CLI arg), matching
        # the design's SERVERWRAPPER contract.
        server_env = {**os.environ, "MAX_NUM_BATCHED_TOKENS": str(budget)}

        if should_abort is not None and should_abort():
            # Checked again immediately before the actual server start --
            # a signal may have arrived while resolving the port/command.
            result["overall_status"] = "interrupted"
            return result

        handle = server_adapter.start(cmd, log_path, env=server_env)
        result["server_start"] = {
            "cmd": cmd, "pid": handle.pid, "pgid": handle.pgid, "start_utc": handle.start_utc,
            "max_num_batched_tokens": budget,
        }

        base_url = f"http://{host}:{port}"
        readiness_info = await wait_for_server_ready(
            transport, handle, base_url, api_key, model_full_id, sleeper
        )
        readiness_info["capability_summary"] = await fetch_capability_summary(
            transport, base_url, api_key
        )
        result["readiness"] = readiness_info

        # Design requirement (SERVERWRAPPER): verify, before stabilization,
        # that the server log confirms chunked prefill is actually enabled
        # with THIS block's budget. A missing/wrong confirmation aborts
        # the whole block -- it must never silently proceed with an
        # unconfirmed or wrong chunked-prefill configuration.
        log_check_errors = check_chunked_prefill_log(log_path, budget)
        result["chunked_prefill_log_check"] = {
            "expected_max_num_batched_tokens": budget,
            "passed": not log_check_errors,
            "errors": log_check_errors,
        }
        if log_check_errors:
            raise ServerLifecycleError(
                f"chunked-prefill log verification failed for block "
                f"{block_id!r} (expected max_num_batched_tokens={budget}): "
                f"{log_check_errors}"
            )

        valid_ids = compute_valid_token_ids(tokenizer)
        ctx = RunContext(
            transport=transport, clock=clock, sleeper=sleeper, base_url=base_url,
            api_key=api_key, model_full_id=model_full_id, valid_ids=valid_ids,
        )
        server_metadata = {
            "model_key": model_key,
            "model_full_id": model_full_id,
            "offload_gb": offload_gb,
            "host": host,
            "port": port,
            "server_command": cmd,
            "pid": handle.pid,
            "pgid": handle.pgid,
            "server_start_utc": handle.start_utc,
            "readiness": readiness_info,
        }

        stab_result = await run_stabilization(
            ctx, bundle, model_key, block_id, offload_gb, state_label, server_metadata=server_metadata
        )
        stab_result["run_mode"] = run_mode
        write_json_atomic(stabilization_result_path(output_dir, block_id), stab_result)
        result["stabilization_status"] = stab_result["status"]

        if stab_result["status"] != REQUEST_STATUS_COMPLETE:
            result["overall_status"] = "stabilization_failed"
            return result

        post_stabilization_health = await check_post_stabilization_health(transport, base_url)
        readiness_info["post_stabilization_health"] = post_stabilization_health
        result["readiness"] = readiness_info
        result["post_stabilization_health"] = post_stabilization_health

        if not post_stabilization_health["ok"]:
            result["overall_status"] = "post_stabilization_health_failed"
            return result

        await sleeper.sleep(COOLDOWN_S)
        result["cooldown_s"] = COOLDOWN_S

        stabilization_ref = {
            "block_id": block_id,
            "path": str(stabilization_result_path(output_dir, block_id)),
            "functional_passed": stab_result["functional_passed"],
        }

        block_aborted = False
        for ep in episodes_to_run:
            ep_result = await run_regular_episode(
                ctx, ep, schedule_fingerprint=bundle.fingerprint,
                server_metadata=server_metadata, stabilization_ref=stabilization_ref,
            )
            ep_result["run_mode"] = run_mode
            write_json_atomic(episode_result_path(output_dir, ep.episode_id), ep_result)
            result["executed_episode_ids"].append(ep.episode_id)
            ok = ep_result["status"] == REQUEST_STATUS_COMPLETE
            episode_statuses[ep.episode_id] = (
                CLASSIFICATION_VALID_COMPLETE if ok else CLASSIFICATION_PARTIAL
            )
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
        result["error"] = str(exc)
        return result
    finally:
        if handle is not None:
            stop_result = await stop_server(
                handle, host, port, sleeper,
                timeout_s=stop_timeout_s,
                kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
                port_poll_timeout_s=stop_port_poll_timeout_s,
            )
            result["server_stop"] = stop_result
            if not stop_result.get("stop_success") and result.get("overall_status") == "block_complete":
                # A successful block must never be reported as complete if
                # the server itself was not verifiably stopped afterward.
                result["overall_status"] = "server_stop_failed"


async def run_smoke_block(
    *,
    bundle: LoadedBundle,
    block_id: str,
    output_dir: Path,
    host: str,
    port: int,
    resume: bool,
    api_key: str,
    transport: HTTPTransport,
    tokenizer: TokenizerAdapter,
    server_adapter: ServerProcessAdapter,
    sleeper: Sleeper,
    clock: Clock,
    run_server_path: Path,
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S,
    stop_kill_confirm_timeout_s: float = 5.0,
    stop_port_poll_timeout_s: float = 30.0,
) -> dict:
    """Thin, --smoke-test-specific wrapper around the shared block
    executor (Section 2). Output shape (smoke_run_summary.json) is
    unchanged from Stage 2."""
    start_utc = clock.utcnow_iso()
    block_episodes = find_and_validate_smoke_block(bundle, block_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    check_output_dir_not_shared(output_dir, RUN_MODE_SMOKE)
    write_run_mode_marker(output_dir, RUN_MODE_SMOKE)

    episode_statuses, episodes_to_run = _classify_and_plan_block(
        block_episodes, bundle, output_dir, RUN_MODE_SMOKE, resume
    )

    summary: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_mode": RUN_MODE_SMOKE,
        "schedule_fingerprint": bundle.fingerprint,
        "smoke_block": block_id,
        "max_num_batched_tokens": block_episodes[0].max_num_batched_tokens,
        "server_start": None,
        "readiness": None,
        "chunked_prefill_log_check": None,
        "stabilization_status": "not_run",
        "cooldown_s": None,
        "episode_statuses": episode_statuses,
        "server_stop": None,
        "overall_status": "not_run",
        "start_utc": start_utc,
        "end_utc": None,
    }

    if not episodes_to_run:
        summary["stabilization_status"] = "skipped (all episodes already valid_complete)"
        summary["overall_status"] = "already_complete"
        summary["end_utc"] = clock.utcnow_iso()
        write_json_atomic(output_dir / "smoke_run_summary.json", summary)
        return summary

    protocol_result = await _run_block_protocol(
        bundle=bundle,
        block_episodes=block_episodes,
        episodes_to_run=episodes_to_run,
        block_id=block_id,
        output_dir=output_dir,
        host=host,
        port=port,
        run_mode=RUN_MODE_SMOKE,
        api_key=api_key,
        transport=transport,
        tokenizer=tokenizer,
        server_adapter=server_adapter,
        sleeper=sleeper,
        clock=clock,
        run_server_path=run_server_path,
        episode_statuses=episode_statuses,
        stop_timeout_s=stop_timeout_s,
        stop_kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
        stop_port_poll_timeout_s=stop_port_poll_timeout_s,
    )

    summary["server_start"] = protocol_result["server_start"]
    summary["readiness"] = protocol_result["readiness"]
    summary["chunked_prefill_log_check"] = protocol_result.get("chunked_prefill_log_check")
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



# ============================================================================
# Official campaign orchestrator (Stage 3, Sections 3-11)
# ============================================================================

@dataclass
class InterruptState:
    """Cooperative signal-handling state shared between the CLI's real
    signal handlers (or a test's simulated ones) and run_official_campaign().
    `event` is awaited/polled at safe points; `signal_name` records which
    signal (if any) triggered it, for the persisted summary."""

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


def _check_all_blocks_finalized(
    block_ids: list[str], output_dir: Path, bundle: LoadedBundle, run_mode: str,
) -> tuple[bool, bool]:
    """Returns (stabilization_ok, block_summaries_ok) using full deep
    validation (Sections 6/7 of the Stage-3 contract-blocker patch) --
    a bare status=='complete' or overall_status=='block_complete' is
    never sufficient on its own."""
    stab_ok = True
    block_summary_ok = True
    for bid in block_ids:
        stab_errors, summary_errors = _validate_block_finalization(bid, output_dir, bundle, run_mode)
        if stab_errors:
            stab_ok = False
        if summary_errors:
            block_summary_ok = False

    return stab_ok, block_summary_ok


def _build_official_summary(
    *, runner_version: str, run_mode: str, schedule_fingerprint: str, environment_fingerprint: str,
    start_utc: str, end_utc: str | None, overall_status: str, planned_blocks: int, completed_blocks: int,
    skipped_blocks: int, pending_blocks: int, planned_episodes: int, valid_complete_episodes: int,
    missing_episodes: int, failed_block: str | None, interrupted_by: str | None, block_statuses: dict[str, str],
) -> dict:
    return {
        "runner_version": runner_version,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_mode": run_mode,
        "schedule_fingerprint": schedule_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "overall_status": overall_status,
        "planned_blocks": planned_blocks,
        "completed_blocks": completed_blocks,
        "skipped_blocks": skipped_blocks,
        "pending_blocks": pending_blocks,
        "planned_episodes": planned_episodes,
        "valid_complete_episodes": valid_complete_episodes,
        "missing_episodes": missing_episodes,
        "failed_block": failed_block,
        "interrupted_by": interrupted_by,
        "block_statuses": dict(block_statuses),
    }


async def run_official_campaign(
    *,
    bundle: LoadedBundle,
    output_dir: Path,
    host: str,
    port: int,
    resume: bool,
    api_key: str,
    transport: HTTPTransport,
    tokenizer_factory: Callable[[str], TokenizerAdapter],
    server_adapter: ServerProcessAdapter,
    sleeper: Sleeper,
    clock: Clock,
    run_server_path: Path,
    environment_probe: EnvironmentProbe,
    interrupt_state: InterruptState | None = None,
    stop_timeout_s: float = SERVER_STOP_TIMEOUT_S,
    stop_kill_confirm_timeout_s: float = 5.0,
    stop_port_poll_timeout_s: float = 30.0,
) -> dict:
    """Section 3: runs every block of the validated schedule bundle, in
    exact schedule order, via the shared block executor
    (_run_block_protocol, run_mode='official'). Tokenizers are loaded at
    most once per model (cached locally). Never starts more than one
    server at a time; a block is only ever begun once the previous one's
    server has been verifiably stopped."""
    if interrupt_state is None:
        interrupt_state = InterruptState()

    start_utc = clock.utcnow_iso()
    run_mode = RUN_MODE_OFFICIAL

    block_ids = all_block_ids_in_schedule_order(bundle)
    episodes_by_block: dict[str, list[Episode]] = {bid: find_block(bundle, bid) for bid in block_ids}
    for bid, eps in episodes_by_block.items():
        if len(eps) != BLOCK_SIZE:
            raise ValueError(f"block {bid!r} does not form a complete block of size {BLOCK_SIZE}")

    manifest_path = output_dir / OFFICIAL_RUN_MANIFEST_FILENAME
    summary_path = output_dir / OFFICIAL_RUN_SUMMARY_FILENAME
    integrity_path = output_dir / INTEGRITY_MANIFEST_FILENAME

    # --- Section 1: strict environment gate ---------------------------------
    # Gathered and validated before ANY output-directory change of any
    # kind (not even mkdir/marker/manifest) and before any server start,
    # for both a fresh run and a resume.
    env = environment_probe.gather(bundle.schedule_dir)
    env_errors = validate_official_environment(env)
    if env_errors:
        raise ServerLifecycleError(
            f"official-run environment gate failed; refusing to start or resume: {env_errors}"
        )
    env_fingerprint = compute_environment_fingerprint(env)

    if resume:
        if not output_dir.exists():
            raise ServerLifecycleError(
                f"--official-run --resume requires an existing output directory; "
                f"{output_dir} does not exist"
            )
        check_output_dir_not_shared(output_dir, run_mode)
        existing_manifest = load_json_file_or_none(manifest_path)
        if existing_manifest is None:
            raise ServerLifecycleError(
                "--official-run --resume requires an existing official_run_manifest.json "
                "in the output directory; none found or unreadable"
            )
        manifest_errors = validate_resume_manifest(
            existing_manifest, bundle=bundle, run_mode=run_mode, current_env=env
        )
        if manifest_errors:
            raise ServerLifecycleError(
                f"--resume manifest validation failed; refusing to continue on "
                f"possibly different code, GPU, or software: {manifest_errors}"
            )
        # Resume never writes or overwrites the marker or the manifest.
    else:
        # --- Section 2: real fresh-run contract ---
        # output_dir must not exist, OR exist and contain *no entries at
        # all*. No exceptions for "known" artifacts (server_logs/,
        # integrity_manifest.json, .chunk_budget_screen_run_mode) or unknown ones --
        # every single entry aborts the run before any change is made.
        if output_dir.exists():
            existing_entries = sorted(p.name for p in output_dir.iterdir())
            if existing_entries:
                raise ServerLifecycleError(
                    f"--official-run without --resume requires a new or "
                    f"completely empty output directory; found existing "
                    f"entr{'y' if len(existing_entries) == 1 else 'ies'} in "
                    f"{output_dir}: {existing_entries}; refusing to silently overwrite"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_mode_marker(output_dir, run_mode)
        manifest = build_official_run_manifest(
            env=env, bundle=bundle, run_mode=run_mode, output_dir=output_dir, host=host, port=port, clock=clock,
        )
        write_json_atomic(manifest_path, manifest)

    # Campaign-wide classification scan: any partial/invalid/corrupted
    # episode file anywhere aborts the whole resume before any server is
    # started.
    episode_statuses_by_block: dict[str, dict[str, str]] = {}
    all_bad: dict[str, str] = {}
    for bid in block_ids:
        statuses: dict[str, str] = {}
        for ep in episodes_by_block[bid]:
            cls, _notes = classify_result_file(
                episode_result_path(output_dir, ep.episode_id), ep, bundle.fingerprint, run_mode
            )
            statuses[ep.episode_id] = cls
            if cls in (CLASSIFICATION_PARTIAL, CLASSIFICATION_INVALID, CLASSIFICATION_CORRUPTED):
                all_bad[ep.episode_id] = cls
        episode_statuses_by_block[bid] = statuses

    if all_bad:
        raise ServerLifecycleError(
            f"resume found non-resumable existing result file(s) across the "
            f"campaign: {all_bad}; refusing to silently overwrite. Fix or remove "
            f"them manually first."
        )

    planned_episodes = sum(len(v) for v in episode_statuses_by_block.values())
    valid_complete_count = sum(
        1 for statuses in episode_statuses_by_block.values() for c in statuses.values()
        if c == CLASSIFICATION_VALID_COMPLETE
    )
    block_fully_done = {
        bid: all(c == CLASSIFICATION_VALID_COMPLETE for c in episode_statuses_by_block[bid].values())
        for bid in block_ids
    }

    integrity_expected_kwargs = dict(
        expected_schedule_fingerprint=bundle.fingerprint,
        expected_environment_fingerprint=env_fingerprint,
        expected_episode_count=planned_episodes,
        expected_stabilization_count=len(block_ids),
        expected_block_summary_count=len(block_ids),
        expected_server_log_count=len(block_ids),
    )

    # --- Idempotency short-circuits (Sections 3/4): every scientific
    # episode output already present -> never start a server again. ---
    if valid_complete_count == planned_episodes and all(block_fully_done.values()):
        stab_ok, block_summary_ok = _check_all_blocks_finalized(block_ids, output_dir, bundle, run_mode)

        if stab_ok and block_summary_ok:
            if integrity_path.exists():
                # Section 4: an *existing* integrity manifest that fails
                # deep verification is NEVER silently replaced -- that
                # would paper over a genuine integrity problem. Hard
                # abort instead, touching nothing.
                existing_integrity_raw = load_json_file_or_none(integrity_path)
                if existing_integrity_raw is None:
                    raise ServerLifecycleError(
                        f"existing {INTEGRITY_MANIFEST_FILENAME} is unreadable or not "
                        f"valid JSON; refusing to silently reseal it"
                    )
                verified, verify_errors = verify_integrity_manifest(
                    output_dir, existing_integrity_raw, **integrity_expected_kwargs
                )
                if not verified:
                    raise ServerLifecycleError(
                        f"existing {INTEGRITY_MANIFEST_FILENAME} failed deep verification; "
                        f"refusing to silently reseal it: {verify_errors}"
                    )
                # Section 3: a genuine no-op. Return an in-memory-only
                # summary -- touch nothing on disk, not even a timestamp.
                return _build_official_summary(
                    runner_version=RUNNER_VERSION, run_mode=run_mode,
                    schedule_fingerprint=bundle.fingerprint, environment_fingerprint=env_fingerprint,
                    start_utc=start_utc, end_utc=clock.utcnow_iso(), overall_status="already_complete",
                    planned_blocks=len(block_ids), completed_blocks=0, skipped_blocks=len(block_ids),
                    pending_blocks=0, planned_episodes=planned_episodes,
                    valid_complete_episodes=valid_complete_count, missing_episodes=0,
                    failed_block=None, interrupted_by=None,
                    block_statuses={bid: "already_complete" for bid in block_ids},
                )

            # Section 4: the manifest is genuinely MISSING (never
            # present-but-invalid -- that already hard-aborted above).
            # Only now, after every scientific artifact has independently
            # passed deep validation, may a fresh integrity manifest be built.
            summary_out = _build_official_summary(
                runner_version=RUNNER_VERSION, run_mode=run_mode,
                schedule_fingerprint=bundle.fingerprint, environment_fingerprint=env_fingerprint,
                start_utc=start_utc, end_utc=clock.utcnow_iso(), overall_status="complete",
                planned_blocks=len(block_ids), completed_blocks=len(block_ids), skipped_blocks=0,
                pending_blocks=0, planned_episodes=planned_episodes,
                valid_complete_episodes=valid_complete_count, missing_episodes=0,
                failed_block=None, interrupted_by=None,
                block_statuses={bid: "block_complete" for bid in block_ids},
            )
            write_json_atomic(summary_path, summary_out)
            integrity = build_integrity_manifest(
                output_dir, schedule_fingerprint=bundle.fingerprint,
                environment_fingerprint=env_fingerprint, clock=clock,
            )
            write_json_atomic(integrity_path, integrity)
            verified, verify_errors = verify_integrity_manifest(output_dir, integrity, **integrity_expected_kwargs)
            if not verified:
                summary_out["overall_status"] = "error"
                summary_out["error"] = f"integrity manifest self-verification failed: {verify_errors}"
                write_json_atomic(summary_path, summary_out)
            return summary_out
        # else: fall through to the normal loop -- some blocks are
        # episode-complete but not yet (validly) finalized, so re-enter
        # that block's protocol below.

    # --- main execution loop ------------------------------------------------
    block_statuses: dict[str, str] = {}
    completed_blocks = 0
    skipped_blocks = 0
    failed_block: str | None = None
    interrupted_by: str | None = None
    tokenizer_cache: dict[str, TokenizerAdapter] = {}

    def _should_abort() -> bool:
        return interrupt_state.event.is_set()

    summary_out = _build_official_summary(
        runner_version=RUNNER_VERSION, run_mode=run_mode, schedule_fingerprint=bundle.fingerprint,
        environment_fingerprint=env_fingerprint, start_utc=start_utc, end_utc=None, overall_status="running",
        planned_blocks=len(block_ids), completed_blocks=0, skipped_blocks=0, pending_blocks=len(block_ids),
        planned_episodes=planned_episodes, valid_complete_episodes=valid_complete_count,
        missing_episodes=planned_episodes - valid_complete_count, failed_block=None, interrupted_by=None,
        block_statuses=block_statuses,
    )
    write_json_atomic(summary_path, summary_out)

    for bid in block_ids:
        if _should_abort():
            interrupted_by = interrupt_state.signal_name
            break

        block_episodes = episodes_by_block[bid]

        if block_fully_done[bid]:
            # Section 7: deep-validate, never accept a bare status flag.
            stab_errors, summary_errors = _validate_block_finalization(bid, output_dir, bundle, run_mode)
            already_finalized = not stab_errors and not summary_errors
            if already_finalized:
                block_statuses[bid] = "already_complete"
                skipped_blocks += 1
                summary_out["skipped_blocks"] = skipped_blocks
                summary_out["pending_blocks"] = len(block_ids) - len(block_statuses)
                summary_out["block_statuses"] = dict(block_statuses)
                summary_out["end_utc"] = clock.utcnow_iso()
                write_json_atomic(summary_path, summary_out)
                continue
            # All 4 episodes are valid_complete but this block's
            # stabilization/summary is missing or not deep-valid (e.g.
            # interrupted right after its last episode write, or a
            # server_stop that never actually verified) -- fall through
            # and let the protocol run again. Section 6: stabilization is
            # mandatory again whenever a block is re-entered;
            # _classify_and_plan_block() below correctly produces an
            # empty episodes_to_run (nothing to re-run, existing episode
            # files stay byte-untouched) while the protocol still
            # performs one fresh stabilization + health + cooldown pass
            # and (re)writes a corrected block_summary.

        if _should_abort():  # Section 8: after resume planning is about to begin
            interrupted_by = interrupt_state.signal_name
            break

        model_key = block_episodes[0].model
        if model_key not in tokenizer_cache:
            tokenizer_cache[model_key] = tokenizer_factory(model_key)
        tokenizer = tokenizer_cache[model_key]

        if _should_abort():  # Section 8: immediately after tokenizer loading
            interrupted_by = interrupt_state.signal_name
            break

        # The campaign-wide scan above already proved there is nothing
        # partial/invalid/corrupted anywhere, so resume=True here is
        # always safe and simply (re)classifies this block's 4 files.
        episode_statuses, episodes_to_run = _classify_and_plan_block(
            block_episodes, bundle, output_dir, run_mode, resume=True
        )

        if _should_abort():  # Section 8: after resume planning, before any task exists
            interrupted_by = interrupt_state.signal_name
            break

        block_start_utc = clock.utcnow_iso()

        if _should_abort():  # Section 8: immediately before block-task creation/start
            interrupted_by = interrupt_state.signal_name
            break

        block_task: asyncio.Task = asyncio.ensure_future(
            _run_block_protocol(
                bundle=bundle, block_episodes=block_episodes, episodes_to_run=episodes_to_run,
                block_id=bid, output_dir=output_dir, host=host, port=port, run_mode=run_mode,
                api_key=api_key, transport=transport, tokenizer=tokenizer, server_adapter=server_adapter,
                sleeper=sleeper, clock=clock, run_server_path=run_server_path, episode_statuses=episode_statuses,
                stop_timeout_s=stop_timeout_s, stop_kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
                stop_port_poll_timeout_s=stop_port_poll_timeout_s, should_abort=_should_abort,
            )
        )
        interrupt_wait_task = asyncio.ensure_future(interrupt_state.event.wait())
        done, _pending = await asyncio.wait({block_task, interrupt_wait_task}, return_when=asyncio.FIRST_COMPLETED)
        if block_task not in done:
            # Signal arrived mid-block: cancel the block's task in a
            # controlled way (this cascades into cancelling every active
            # request task) and wait for its own cleanup (verified server
            # stop) to finish before proceeding.
            block_task.cancel()
        protocol_result = await block_task
        if not interrupt_wait_task.done():
            interrupt_wait_task.cancel()
            try:
                await interrupt_wait_task
            except asyncio.CancelledError:
                pass

        block_overall_status = protocol_result.get("overall_status")

        if block_overall_status == "interrupted" and protocol_result.get("server_start") is None:
            # Aborted before any server was ever started (should_abort()
            # fired inside _run_block_protocol itself) -- no block_summary
            # is written for a block that was never actually attempted.
            interrupted_by = interrupt_state.signal_name
            break

        block_end_utc = clock.utcnow_iso()
        executed_ids = set(protocol_result.get("executed_episode_ids") or [])
        block_summary = {
            "block_id": bid,
            "model": model_key,
            "offload_gb": block_episodes[0].offload_gb,
            "state_label": block_episodes[0].state_label,
            "max_num_batched_tokens": block_episodes[0].max_num_batched_tokens,
            "repeat": block_episodes[0].repeat,
            "run_mode": run_mode,
            "schedule_fingerprint": bundle.fingerprint,
            "start_utc": block_start_utc,
            "end_utc": block_end_utc,
            "planned_episode_ids": [ep.episode_id for ep in block_episodes],
            "executed_episode_ids": protocol_result.get("executed_episode_ids") or [],
            "skipped_episode_ids": [ep.episode_id for ep in block_episodes if ep.episode_id not in executed_ids],
            "episode_statuses": dict(episode_statuses),
            "stabilization_status": protocol_result.get("stabilization_status"),
            "chunked_prefill_log_check": protocol_result.get("chunked_prefill_log_check"),
            "post_stabilization_health": protocol_result.get("post_stabilization_health"),
            "cooldown_s": protocol_result.get("cooldown_s"),
            "server_start": protocol_result.get("server_start"),
            "server_stop": protocol_result.get("server_stop"),
            "overall_status": block_overall_status,
        }
        write_json_atomic(output_dir / "block_summaries" / f"{bid}.json", block_summary)

        block_statuses[bid] = block_overall_status

        valid_complete_count += sum(
            1 for eid in [ep.episode_id for ep in episodes_to_run]
            if episode_statuses.get(eid) == CLASSIFICATION_VALID_COMPLETE
        )

        summary_out["valid_complete_episodes"] = valid_complete_count
        summary_out["missing_episodes"] = planned_episodes - valid_complete_count
        summary_out["block_statuses"] = dict(block_statuses)
        summary_out["end_utc"] = clock.utcnow_iso()

        if block_overall_status == "interrupted":
            interrupted_by = interrupt_state.signal_name
            summary_out["pending_blocks"] = len(block_ids) - len(block_statuses)
            summary_out["interrupted_by"] = interrupted_by
            summary_out["overall_status"] = "interrupted"
            write_json_atomic(summary_path, summary_out)
            return summary_out

        if block_overall_status != "block_complete":
            failed_block = bid
            summary_out["failed_block"] = failed_block
            summary_out["pending_blocks"] = len(block_ids) - len(block_statuses)
            summary_out["overall_status"] = (
                block_overall_status if block_overall_status in ("server_stop_failed", "error") else "block_failed"
            )
            write_json_atomic(summary_path, summary_out)
            return summary_out

        completed_blocks += 1
        summary_out["completed_blocks"] = completed_blocks
        summary_out["pending_blocks"] = len(block_ids) - len(block_statuses)
        write_json_atomic(summary_path, summary_out)

    if interrupted_by is not None:
        summary_out["overall_status"] = "interrupted"
        summary_out["interrupted_by"] = interrupted_by
        summary_out["pending_blocks"] = len(block_ids) - len(block_statuses)
        summary_out["end_utc"] = clock.utcnow_iso()
        write_json_atomic(summary_path, summary_out)
        return summary_out

    # --- final integrity proof (Section 11 base + Section 5 of this patch) --
    stab_ok, block_summary_ok = _check_all_blocks_finalized(block_ids, output_dir, bundle, run_mode)
    all_valid = valid_complete_count == planned_episodes
    all_verified_stopped = all(block_statuses.get(bid) in ("block_complete", "already_complete") for bid in block_ids)

    if all_valid and stab_ok and block_summary_ok and all_verified_stopped:
        summary_out["overall_status"] = "complete"
        summary_out["end_utc"] = clock.utcnow_iso()
        write_json_atomic(summary_path, summary_out)

        integrity = build_integrity_manifest(
            output_dir, schedule_fingerprint=bundle.fingerprint, environment_fingerprint=env_fingerprint, clock=clock,
        )
        write_json_atomic(integrity_path, integrity)
        verified, verify_errors = verify_integrity_manifest(output_dir, integrity, **integrity_expected_kwargs)
        if not verified:
            summary_out["overall_status"] = "error"
            summary_out["error"] = f"integrity manifest self-verification failed: {verify_errors}"
            write_json_atomic(summary_path, summary_out)
    else:
        summary_out["overall_status"] = "error"
        summary_out["error"] = "campaign loop finished without every block verifiably complete"
        summary_out["end_utc"] = clock.utcnow_iso()
        write_json_atomic(summary_path, summary_out)

    return summary_out




# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_chunk_budget_screen.py",
        description=(
            "Chunk-Budget-Screen runner (frozen exploratory campaign: schedule-bundle validation, "
            "resume-depth checks, self-test, dry-run, real execution of a "
            "single selectable smoke-test block, and the complete "
            "official 18-block/36-episode screening campaign). --official-run is "
            "enabled."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--self-test", action="store_true",
        help="Run this runner's own internal validation self-tests.",
    )
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Load and fully validate --schedule-dir, print the execution "
        "plan, and exit. No side effects.",
    )
    mode_group.add_argument(
        "--official-run", action="store_true",
        help="Really run the complete frozen screening campaign (18 server blocks, "
        "36 regular episodes) in exact schedule order: strict environment "
        "gate, per-block server start/readiness/chunked-prefill-log-check/"
        "stabilization/health/cooldown/episodes/verified stop, atomic "
        "manifest/summary/block-summary output, and a final self-verifying "
        "integrity manifest on genuine completion.",
    )
    mode_group.add_argument(
        "--smoke-test", action="store_true",
        help="Really execute exactly one --smoke-block: server start, "
        "readiness, chunked-prefill-log check, one stabilization run, "
        "cooldown, its two regular episodes, and a targeted server stop.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Only valid with --official-run or --smoke-test: resume from "
        "existing result files after a strict environment-fingerprint and "
        "resume-depth validation; never silently overwrites anything.",
    )
    parser.add_argument(
        "--schedule-dir", type=Path, default=DEFAULT_SCHEDULE_DIR,
        help=f"Bundle directory containing {list(REQUIRED_BUNDLE_FILENAMES)} "
        f"(default: {DEFAULT_SCHEDULE_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Result output directory for --official-run/--smoke-test "
        "(default: results/official or results/smoke under the schedule "
        "runs directory). Without --resume this directory must not exist "
        "or must be completely empty.",
    )
    parser.add_argument(
        "--smoke-block", type=str, default=None,
        help="Required with --smoke-test (and disallowed otherwise): the "
        "exact block_id to execute, e.g. 'llama_block01_low_budget512'.",
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_SMOKE_HOST,
        help=f"Server host for --smoke-test and --official-run "
        f"(default: {DEFAULT_SMOKE_HOST}).",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_SMOKE_PORT,
        help=f"Server port for --smoke-test and --official-run, 1-65535 "
        f"(default: {DEFAULT_SMOKE_PORT}).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.resume and not (args.official_run or args.smoke_test):
        parser.error(
            "--resume is only valid together with --official-run or "
            "--smoke-test"
        )
    if args.smoke_block is not None and not args.smoke_test:
        parser.error(
            "--smoke-block is only valid together with --smoke-test (not "
            "with --self-test, --dry-run, or --official-run)"
        )
    if args.smoke_test and args.smoke_block is None:
        parser.error("--smoke-test requires --smoke-block")
    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be an integer in 1-65535, got {args.port}")
    return args




# ============================================================================
# Self-test
# ============================================================================
#
# Focused no-GPU/no-network checks are intended to live in a sibling
# chunk_budget_screen_tests/ package, mirroring the phase_a_tests/ and
# prefill_screen_tests/ pattern used by the other two runners in this
# project. That package has NOT been built yet in this round (explicitly
# out of scope -- see the round's task description: "keine große neue
# Testinfrastruktur"). --self-test therefore fails cleanly with an
# honest message instead of crashing on an ImportError against a
# nonexistent package; see main() below.



# ============================================================================
# main
# ============================================================================

def official_run_exit_code(summary: dict) -> int:
    """Pure mapping from a campaign summary to the CLI exit code
    (Section 10 precision addendum): 130 for SIGINT, 143 for SIGTERM,
    0 for complete/already_complete, else 1. Never calls sys.exit
    itself -- kept as a pure function so it is trivially unit-testable."""
    if summary.get("overall_status") == "interrupted":
        signal_name = summary.get("interrupted_by")
        if signal_name == "SIGINT":
            return 130
        if signal_name == "SIGTERM":
            return 143
        return 1
    return 0 if summary.get("overall_status") in ("complete", "already_complete") else 1


async def _run_official_cli(
    *,
    bundle: LoadedBundle,
    output_dir: Path,
    host: str,
    port: int,
    resume: bool,
    api_key: str,
    transport: HTTPTransport,
    tokenizer_factory: Callable[[str], TokenizerAdapter],
    run_server_path: Path,
) -> dict:
    """CLI-only glue: installs real SIGINT/SIGTERM handlers for the
    duration of the campaign (Section 10), then runs it. Signal handling
    itself lives in run_official_campaign()/InterruptState -- this
    function only wires real OS signals to that mechanism and guarantees
    the handlers are removed afterward either way."""
    interrupt_state = InterruptState()
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for sig, name in ((signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")):
        loop.add_signal_handler(sig, lambda name=name: interrupt_state.trigger(name))
        installed.append(sig)
    try:
        return await run_official_campaign(
            bundle=bundle,
            output_dir=output_dir,
            host=host,
            port=port,
            resume=resume,
            api_key=api_key,
            transport=transport,
            tokenizer_factory=tokenizer_factory,
            server_adapter=RealServerProcessAdapter(),
            sleeper=RealSleeper(),
            clock=RealClock(),
            run_server_path=run_server_path,
            environment_probe=RealEnvironmentProbe(),
            interrupt_state=interrupt_state,
        )
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        # See the "Self-test" section marker above: the sibling test
        # package (chunk_budget_screen_tests/) has not been built yet in
        # this round. Register this module under its own stable name
        # first regardless, so that whenever that package IS added, its
        # `import run_chunk_budget_screen` reuses this exact module
        # object instead of re-executing this file under a second,
        # distinct module identity.
        sys.modules.setdefault("run_chunk_budget_screen", sys.modules[__name__])
        try:
            from chunk_budget_screen_tests.selftest_runner import run_self_test as _run_self_test
        except ModuleNotFoundError:
            print(
                "ERROR: --self-test is not yet available for "
                "chunk-budget-screen-v1: the chunk_budget_screen_tests/ "
                "package has not been built in this round (explicitly "
                "out of scope). Use --dry-run to validate the schedule "
                "bundle and print the execution plan instead.",
                file=sys.stderr,
            )
            return 1

        return _run_self_test()

    bundle, errors = load_and_validate_bundle(args.schedule_dir)
    if errors:
        print("Schedule bundle validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    assert bundle is not None

    if args.dry_run:
        plan = build_execution_plan(bundle)
        print_execution_plan(bundle, plan)
        print()
        print("PASS: schedule bundle is valid.")
        return 0

    if args.official_run:
        mode = RUN_MODE_OFFICIAL
        output_dir = (
            args.output_dir if args.output_dir is not None else default_output_dir(mode)
        )

        try:
            api_key = read_api_key_from_env()
        except ApiKeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        run_server_path = resolve_run_server_path(SCRIPT_PATH)
        try:
            check_run_server_script(run_server_path)
        except ServerLifecycleError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        try:
            transport: HTTPTransport = HttpxTransport()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        def tokenizer_factory(model_key: str) -> TokenizerAdapter:
            return HFTokenizerAdapter(MODEL_FULL_ID[model_key])

        print(f"schedule_fingerprint: {bundle.fingerprint}")
        print(f"run_mode: {mode}")
        print(f"output_dir: {output_dir}")
        print(f"host:port: {args.host}:{args.port}")
        print()

        try:
            summary = asyncio.run(
                _run_official_cli(
                    bundle=bundle, output_dir=output_dir, host=args.host, port=args.port,
                    resume=args.resume, api_key=api_key, transport=transport,
                    tokenizer_factory=tokenizer_factory, run_server_path=run_server_path,
                )
            )
        except (ApiKeyError, ServerLifecycleError, CapabilityError, ValueError) as exc:
            print(f"ERROR: official run aborted: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 -- always report, never crash silently
            print(f"ERROR: official run failed with an unexpected error: {exc}", file=sys.stderr)
            return 1

        print(json.dumps(summary, indent=2, sort_keys=True, default=str))

        exit_code = official_run_exit_code(summary)
        if summary.get("overall_status") == "interrupted":
            # Section 10 / precision addendum: exit explicitly with the
            # conventional 128+signum code; never re-raise or re-deliver
            # the signal itself.
            sys.exit(exit_code)
        return exit_code

    # --- --smoke-test: real execution of exactly one block (Stage 2) -------
    assert args.smoke_test
    mode = "smoke"
    output_dir = (
        args.output_dir if args.output_dir is not None else default_output_dir(mode)
    )

    try:
        find_and_validate_smoke_block(bundle, args.smoke_block)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        api_key = read_api_key_from_env()
    except ApiKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_server_path = resolve_run_server_path(SCRIPT_PATH)
    try:
        check_run_server_script(run_server_path)
    except ServerLifecycleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    block_model = find_block(bundle, args.smoke_block)[0].model
    try:
        transport: HTTPTransport = HttpxTransport()
        tokenizer: TokenizerAdapter = HFTokenizerAdapter(MODEL_FULL_ID[block_model])
    except (RuntimeError, CapabilityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"schedule_fingerprint: {bundle.fingerprint}")
    print(f"run_mode: {mode}")
    print(f"output_dir: {output_dir}")
    print(f"smoke_block: {args.smoke_block}")
    print(f"host:port: {args.host}:{args.port}")
    print()

    try:
        summary = asyncio.run(
            run_smoke_block(
                bundle=bundle,
                block_id=args.smoke_block,
                output_dir=output_dir,
                host=args.host,
                port=args.port,
                resume=args.resume,
                api_key=api_key,
                transport=transport,
                tokenizer=tokenizer,
                server_adapter=RealServerProcessAdapter(),
                sleeper=RealSleeper(),
                clock=RealClock(),
                run_server_path=run_server_path,
            )
        )
    except (ApiKeyError, ServerLifecycleError, CapabilityError, ValueError) as exc:
        print(f"ERROR: smoke test aborted: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- always report, never crash silently
        print(f"ERROR: smoke test failed with an unexpected error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary.get("overall_status") in ("block_complete", "already_complete") else 1


if __name__ == "__main__":
    sys.exit(main())
