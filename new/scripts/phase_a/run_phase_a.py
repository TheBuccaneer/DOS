#!/usr/bin/env python3
"""
run_phase_a.py -- Phase A runner, Stage 2 of 3.

Stage 1 scope (retained, unchanged): schedule-bundle validation
(structural + deterministic + canonical fingerprint), CLI, resume-depth
validation of already-existing episode result files.

Stage 2 scope (added on top of Stage 1): the full real execution path
for exactly one selectable smoke-test block:
    start server (bash run_server.sh ...) -> API readiness -> request
    capability check -> one full stabilization run -> atomic
    stabilization output -> health check + fixed cooldown -> four
    regular episodes in schedule order -> atomic episode outputs ->
    targeted server-process-group stop with verified shutdown.

--official-run remains explicitly blocked in Stage 2:
    "Official execution is disabled until Stage 3."
No server is started and no result file is written for --official-run.

Stage 2 does not implement a full 80-episode run.

Realpath-hardening patch (applied on top of the initial Stage 2 cut):
    - Readiness polls through transient connection errors instead of
      raising on the first ConnectionRefusedError/timeout/OSError.
    - An additional /health gate runs after stabilization is written and
      before the cooldown; only HTTP 200 releases the block.
    - stop_server() is host/port-aware, treats ProcessLookupError as
      "already gone", and only declares success once the process is
      confirmed dead AND the port is confirmed free.
    - classify_result_file() now depth-validates every victim/burst
      request record (seeds, prompt hash, usage, token counts, status,
      timestamp ordering) instead of accepting a file on list length alone.
    - execute_completion_request() devalues a request on any malformed
      SSE/JSON structure (bad JSON, wrong types, contradicting
      prompt_token_ids), never just on the token-count-based checks.
    - A failed trigger now preserves full raw data for all N victim
      requests (including minimal, clearly-identified records for tasks
      that were cancelled before they could start).
    - The first wave of victim requests is started explicitly, rather
      than relying solely on implicit semaphore-acquisition fairness.

Architecture note -- no runtime dependency on the generator:
    This module does NOT import make_phase_a_schedule.py. The generator
    is frozen and produces the operative artifacts on disk:
        phase_a_schedule.json   <- operative source of truth
        phase_a_schedule.csv    <- independent consistency artifact
        phase_a_schedule_audit.txt <- independent consistency artifact
    This runner treats that on-disk bundle as ground truth: it loads,
    independently re-derives (via a local, self-contained copy of the
    same documented seed-derivation formula), and re-validates it from
    scratch every time. It never (re)generates a schedule itself.

Bundle interface:
    --schedule-dir <dir>
        Directory that must contain exactly the three files above. This
        is the ONLY schedule-input option; there is no --schedule,
        --schedule-json, or --schedule-csv flag.

CLI modes (mutually exclusive, exactly one required):
    --self-test     Exercises this runner's own validation logic (Stage 1)
                     plus its request/trigger/stabilization/smoke-block
                     logic against fakes (Stage 2). Does not require
                     --schedule-dir, a GPU, or a real server.
    --dry-run       Loads and fully validates the given --schedule-dir
                     bundle, prints the resulting execution plan, and
                     exits. Opens no network connection, starts no
                     server, loads no tokenizer, writes no result files.
    --official-run  Same full bundle validation as --dry-run, plus
                     output-dir preparation and (if --resume) a resume
                     scan of already-existing result files, then aborts
                     with "Official execution is disabled until Stage 3."
                     No server is started; no result file is written.
    --smoke-test    Really executes exactly one --smoke-block: starts
                     the server via run_server.sh, waits for readiness,
                     runs one full stabilization pass, cools down, runs
                     that block's four regular episodes in schedule
                     order, and stops the server. Requires --smoke-block.

--resume is only valid together with --official-run or --smoke-test.
--smoke-block/--host/--port are only valid together with --smoke-test.

The API key (VLLM_API_KEY) is read from the environment only inside
read_api_key_from_env(), which is only ever called on the real
--smoke-test path -- never for --self-test, --dry-run, or
--official-run. It is never accepted as a CLI argument, never stored in
any server command, manifest, or JSON result, and never logged.

Usage:
    python3 run_phase_a.py --self-test
    python3 run_phase_a.py --dry-run --schedule-dir /path/to/runs/phase_a
    python3 run_phase_a.py --official-run --schedule-dir /path/to/runs/phase_a \\
        --output-dir /path/to/runs/phase_a/results/official
    VLLM_API_KEY=... GPU_DEVICE=0 python3 run_phase_a.py --smoke-test \\
        --smoke-block llama_block01_low \\
        --schedule-dir /path/to/runs/phase_a \\
        --output-dir /path/to/runs/phase_a/results/smoke
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
# Frozen Schema v2 episode schema (22 fields, exact set in both directions)
# ============================================================================

EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "model",
    "offload_gb",
    "state_label",
    "concurrency",
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
    "restart_server_before_block",
    "block_id",
    "order_in_block",
)

EPISODE_FIELD_TYPES: dict[str, type] = {
    "episode_id": str,
    "model": str,
    "offload_gb": int,
    "state_label": str,
    "concurrency": int,
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
    "restart_server_before_block": int,
    "block_id": str,
    "order_in_block": int,
}


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


# ============================================================================
# Frozen official Phase A contract
# ============================================================================

SCHEMA_VERSION = 2
DESIGN_VERSION = "phase-a-v1"
OFFICIAL_FINGERPRINT = (
    "sha256:1c0e58879c64bc399c772f5b5931f10d940fb2204129e869223622e3a3cf4fc7"
)
OFFICIAL_SEED = 20260711
OFFICIAL_EPISODE_COUNT = 80
OFFICIAL_MODELS = ["llama", "qwen"]
OFFICIAL_REPEATS = 5
OFFICIAL_CONCURRENCIES = [4, 8]
OFFICIAL_CONDITIONS = ["no_burst", "fixed_burst"]
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
OFFICIAL_BURST_CONFIGURATION = {
    "burst_parallel_requests": 4,
    "burst_input_len": 256,
    "burst_output_len": 256,
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
}

EXPECTED_STATE_SEQUENCE: list[str] = [
    "low", "high", "high", "low", "low", "high", "high", "low", "low", "high",
]
BLOCK_SIZE = len(OFFICIAL_CONCURRENCIES) * len(OFFICIAL_CONDITIONS)  # 4
BLOCKS_PER_MODEL = 2 * OFFICIAL_REPEATS  # 10
EPISODES_PER_MODEL = BLOCK_SIZE * BLOCKS_PER_MODEL  # 40
TOTAL_RESTART_MARKERS = BLOCKS_PER_MODEL * len(OFFICIAL_MODELS)  # 20

REQUIRED_BUNDLE_FILENAMES = (
    "phase_a_schedule.json",
    "phase_a_schedule.csv",
    "phase_a_schedule_audit.txt",
)

RUN_MODE_MARKER_FILENAME = ".phase_a_run_mode"


# ============================================================================
# Portable paths (derived from this file's own location, not hardcoded)
# Expected location: <PROJECT_ROOT>/new/scripts/phase_a/run_phase_a.py
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SCHEDULE_DIR = PROJECT_ROOT / "new" / "runs" / "phase_a"


def default_output_dir(mode: str) -> Path:
    return PROJECT_ROOT / "new" / "runs" / "phase_a" / "results" / mode


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
        r"^design_version:\s*phase-a-v1\s*$", audit_text, re.MULTILINE
    ):
        errors.append(
            "audit report does not contain 'design_version: phase-a-v1'"
        )
    if not re.search(
        r"^total episodes \(all models\):\s*80\s*$", audit_text, re.MULTILINE
    ):
        errors.append(
            "audit report does not contain 'total episodes (all models): 80'"
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
    check_eq("concurrencies", OFFICIAL_CONCURRENCIES)
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
                f"official pilot fingerprint {OFFICIAL_FINGERPRINT!r}"
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
    expected_state_sequence: list[str],
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

        expected_state_label = STATE_LABEL_BY_OFFLOAD.get(ep.offload_gb)
        if expected_state_label is None:
            errors.append(f"{ctx}: invalid offload_gb {ep.offload_gb!r}")
        elif ep.state_label != expected_state_label:
            errors.append(
                f"{ctx}: state_label {ep.state_label!r} != expected "
                f"{expected_state_label!r} for offload_gb {ep.offload_gb}"
            )

        if ep.concurrency not in OFFICIAL_CONCURRENCIES:
            errors.append(f"{ctx}: invalid concurrency {ep.concurrency!r}")
        if ep.condition not in OFFICIAL_CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.random_seed != schedule_seed:
            errors.append(
                f"{ctx}: random_seed {ep.random_seed!r} != schedule seed "
                f"{schedule_seed!r}"
            )

        if ep.victim_request_count != OFFICIAL_VICTIM_CONFIGURATION["victim_request_count"]:
            errors.append(f"{ctx}: victim_request_count != 20")
        if ep.victim_input_len != OFFICIAL_VICTIM_CONFIGURATION["victim_input_len"]:
            errors.append(f"{ctx}: victim_input_len != 256")
        if ep.victim_output_len != OFFICIAL_VICTIM_CONFIGURATION["victim_output_len"]:
            errors.append(f"{ctx}: victim_output_len != 64")
        if ep.victim_temperature != OFFICIAL_VICTIM_CONFIGURATION["victim_temperature"]:
            errors.append(f"{ctx}: victim_temperature != 0.0")

        if ep.burst_parallel_requests != OFFICIAL_BURST_CONFIGURATION["burst_parallel_requests"]:
            errors.append(f"{ctx}: burst_parallel_requests != 4")
        if ep.burst_input_len != OFFICIAL_BURST_CONFIGURATION["burst_input_len"]:
            errors.append(f"{ctx}: burst_input_len != 256")
        if ep.burst_output_len != OFFICIAL_BURST_CONFIGURATION["burst_output_len"]:
            errors.append(f"{ctx}: burst_output_len != 256")
        if ep.burst_temperature != OFFICIAL_BURST_CONFIGURATION["burst_temperature"]:
            errors.append(f"{ctx}: burst_temperature != 0.0")

        if ep.order_in_block == 1:
            if ep.restart_server_before_block != 1:
                errors.append(
                    f"{ctx}: order_in_block=1 requires "
                    f"restart_server_before_block==1"
                )
        else:
            if ep.restart_server_before_block != 0:
                errors.append(
                    f"{ctx}: order_in_block={ep.order_in_block} requires "
                    f"restart_server_before_block==0"
                )

        expected_episode_id = (
            f"{model}_off{ep.offload_gb}_conc{ep.concurrency}_"
            f"{ep.condition}_rep{ep.repeat}"
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

        expected_victim_seed = derive_seed(
            str(schedule_seed), model, str(ep.concurrency), str(ep.repeat)
        )
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(
                f"{ctx}: victim_workload_seed does not match the "
                f"expected derivation"
            )

        expected_burst_seed = derive_seed(
            str(schedule_seed), model, str(ep.concurrency), str(ep.repeat), "burst"
        )
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(
                f"{ctx}: burst_workload_seed does not match the "
                f"expected derivation"
            )

    # --- exact repeat set per design cell -----------------------------------
    repeats_by_cell: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.concurrency, ep.condition)
        repeats_by_cell.setdefault(key, set()).add(ep.repeat)

    expected_repeat_set = set(range(1, repeats + 1))
    expected_cells = {
        (offload_gb, concurrency, condition)
        for offload_gb in STATE_LABEL_BY_OFFLOAD
        for concurrency in OFFICIAL_CONCURRENCIES
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
    # be exactly BLOCK_SIZE immediately consecutive episodes, a block_id
    # must never reappear once left, and order_in_block /
    # restart_server_before_block must match 1,2,3,4 / 1,0,0,0 in that
    # exact list order.
    seen_block_ids: set[str] = set()
    idx = 0
    n = len(episodes)
    block_position = 0
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
            order_sequence = [ep.order_in_block for ep in run_episodes]
            expected_order_sequence = list(range(1, BLOCK_SIZE + 1))
            if order_sequence != expected_order_sequence:
                errors.append(
                    f"model={model}: block {bid!r} order_in_block "
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
            repeats_in_block = {ep.repeat for ep in run_episodes}

            if len(state_labels) != 1 or len(offloads) != 1:
                errors.append(
                    f"model={model}: block {bid!r} mixes state/offload "
                    f"values: states={sorted(state_labels)}, "
                    f"offloads={sorted(offloads)}"
                )

            expected_repeat_for_block = ((block_position - 1) // 2) + 1
            if repeats_in_block != {expected_repeat_for_block}:
                errors.append(
                    f"model={model}: block {bid!r} (position "
                    f"{block_position}) has repeat value(s) "
                    f"{sorted(repeats_in_block)}, expected exactly "
                    f"{{{expected_repeat_for_block}}}"
                )

            if block_position <= len(expected_state_sequence) and len(
                state_labels
            ) == 1:
                expected_state = expected_state_sequence[block_position - 1]
                actual_state = next(iter(state_labels))
                if actual_state != expected_state:
                    errors.append(
                        f"model={model}: block at position "
                        f"{block_position} has state {actual_state!r}, "
                        f"expected {expected_state!r}"
                    )
                expected_block_id = (
                    f"{model}_block{block_position:02d}_{expected_state}"
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

    # --- low/high order_in_block matching -----------------------------------
    order_by_match_key: dict[tuple[int, str, int], int] = {}
    for ep in episodes:
        key = (ep.concurrency, ep.condition, ep.repeat)
        if key not in order_by_match_key:
            order_by_match_key[key] = ep.order_in_block
        elif order_by_match_key[key] != ep.order_in_block:
            errors.append(
                f"model={model}: order_in_block mismatch between matched "
                f"low/high episodes for concurrency={ep.concurrency}, "
                f"condition={ep.condition!r}, repeat={ep.repeat}"
            )

    # --- workload seed cell constancy + victim/burst independence ---------
    victim_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
    burst_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
    for ep in episodes:
        key = (ep.concurrency, ep.repeat)
        victim_seeds_by_cell.setdefault(key, set()).add(ep.victim_workload_seed)
        burst_seeds_by_cell.setdefault(key, set()).add(ep.burst_workload_seed)

    for key, seeds in victim_seeds_by_cell.items():
        if len(seeds) != 1:
            errors.append(
                f"model={model}: victim_workload_seed not constant for "
                f"concurrency/repeat {key}: {sorted(seeds)}"
            )
    for key, seeds in burst_seeds_by_cell.items():
        if len(seeds) != 1:
            errors.append(
                f"model={model}: burst_workload_seed not constant for "
                f"concurrency/repeat {key}: {sorted(seeds)}"
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
                expected_state_sequence=EXPECTED_STATE_SEQUENCE,
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
        json_obj = load_json_bundle(paths["phase_a_schedule.json"])
        csv_fieldnames, csv_rows = load_csv_bundle(paths["phase_a_schedule.csv"])
        audit_text = load_audit_text(paths["phase_a_schedule_audit.txt"])
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
    fixed_burst_count = sum(1 for ep in episodes if ep.condition == "fixed_burst")
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
        "fixed_burst_count": fixed_burst_count,
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
    print(f"fixed_burst episodes: {plan['fixed_burst_count']}")
    print()
    for block in plan["blocks"]:
        print(f"--- {block['block_id']} ---")
        print(
            f"  model: {block['model']}, offload_gb: {block['offload_gb']}, "
            f"state: {block['state_label']}, repeat: {block['repeat']}"
        )
        for ep in block["episodes"]:
            print(
                f"    order {ep.order_in_block}: {ep.episode_id} "
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
RUNNER_VERSION = "run_phase_a-stage2"
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
        expected_episode.burst_parallel_requests
        if expected_episode.condition == "fixed_burst"
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
            notes.append("burst_interval must be a valid {start_ns, end_ns} dict for condition 'fixed_burst'")

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
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

COMPLETIONS_ENDPOINT = "/v1/completions"
HEALTH_ENDPOINT = "/health"
MODELS_ENDPOINT = "/v1/models"
OPENAPI_ENDPOINT = "/openapi.json"

RUN_MODE_SMOKE = "smoke"

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
    goodput = (complete_output_tokens / span_s) if span_s and span_s > 0 else None

    per_victim_overlap = [_victim_burst_overlap(r, burst_interval) for r in victim_results]

    burst_output_tokens = sum(len(r.get("output_token_ids") or []) for r in complete_bursts)
    burst_duration_ms = None
    if burst_interval is not None:
        burst_duration_ms = (burst_interval["end_ns"] - burst_interval["start_ns"]) / 1e6

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
        "victim_goodput_tokens_per_s": goodput,
        "victim_complete_request_count": len(complete_victims),
        "victim_incomplete_request_count": len(victim_results) - len(complete_victims),
        "victim_overlap": per_victim_overlap,
        "burst_output_tokens": burst_output_tokens,
        "burst_duration_ms": burst_duration_ms,
        "burst_complete_request_count": len(complete_bursts),
        "burst_incomplete_request_count": len(burst_results) - len(complete_bursts),
        "burst_cost_requests": len(burst_results),
        "burst_cost_tokens": sum(len(r.get("output_token_ids") or []) for r in burst_results),
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

    def _cb_factory(i: int) -> Callable[[], None]:
        def _cb() -> None:
            ev = first_token_events.get(i)
            if ev is not None and not ev.is_set():
                ev.set()

        return _cb

    async def _victim(i: int) -> dict:
        async with sem:
            return await _run_victim_request(ctx, episode, i, on_first_token=_cb_factory(i))

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

    burst_tasks: list[asyncio.Task] = []
    if episode.condition == "fixed_burst":
        for j in range(episode.burst_parallel_requests):
            burst_tasks.append(asyncio.create_task(_run_burst_request(ctx, episode, j)))

    victim_results_raw = await asyncio.gather(
        *[victim_tasks[i] for i in range(n)], return_exceptions=True
    )
    burst_results_raw = (
        await asyncio.gather(*burst_tasks, return_exceptions=True) if burst_tasks else []
    )

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
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
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
    def start(self, cmd: list[str], log_path: Path) -> Any: ...


class RealServerProcessAdapter:
    def start(self, cmd: list[str], log_path: Path) -> ServerHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab", buffering=0)
        process = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True
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

    def start(self, cmd: list[str], log_path: Path) -> FakeServerHandle:
        handle = FakeServerHandle(cmd)
        self.started.append(handle)
        return handle


def build_server_command(
    run_server_path: Path, model_key: str, offload_gb: int, host: str, port: int
) -> list[str]:
    return ["bash", str(run_server_path), model_key, str(offload_gb), host, str(port)]


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
    port_poll_timeout_s: float = 5.0,
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
    stop_port_poll_timeout_s: float = 5.0,
) -> dict:
    start_utc = clock.utcnow_iso()
    block_episodes = find_and_validate_smoke_block(bundle, block_id)

    model_key = block_episodes[0].model
    offload_gb = block_episodes[0].offload_gb
    state_label = block_episodes[0].state_label
    model_full_id = MODEL_FULL_ID[model_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    check_output_dir_not_shared(output_dir, RUN_MODE_SMOKE)
    write_run_mode_marker(output_dir, RUN_MODE_SMOKE)

    episode_statuses: dict[str, str] = {}
    if resume:
        for ep in block_episodes:
            result_path = episode_result_path(output_dir, ep.episode_id)
            cls, _notes = classify_result_file(result_path, ep, bundle.fingerprint, RUN_MODE_SMOKE)
            episode_statuses[ep.episode_id] = cls
        bad = {
            eid: cls
            for eid, cls in episode_statuses.items()
            if cls in (CLASSIFICATION_PARTIAL, CLASSIFICATION_INVALID, CLASSIFICATION_CORRUPTED)
        }
        if bad:
            raise ServerLifecycleError(
                f"--resume found non-resumable existing result file(s): {bad}; "
                f"refusing to silently overwrite. Fix or remove them manually first."
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

    summary: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_mode": RUN_MODE_SMOKE,
        "schedule_fingerprint": bundle.fingerprint,
        "smoke_block": block_id,
        "server_start": None,
        "readiness": None,
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

    handle = None
    try:
        if not is_port_free(host, port):
            raise ServerLifecycleError(f"port {port} on host {host!r} is already in use")

        cmd = build_server_command(run_server_path, model_key, offload_gb, host, port)
        log_path = output_dir / "server_logs" / f"{block_id}.log"
        handle = server_adapter.start(cmd, log_path)
        summary["server_start"] = {
            "cmd": cmd, "pid": handle.pid, "pgid": handle.pgid, "start_utc": handle.start_utc,
        }

        base_url = f"http://{host}:{port}"
        readiness_info = await wait_for_server_ready(
            transport, handle, base_url, api_key, model_full_id, sleeper
        )
        readiness_info["capability_summary"] = await fetch_capability_summary(
            transport, base_url, api_key
        )
        summary["readiness"] = readiness_info

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
        write_json_atomic(stabilization_result_path(output_dir, block_id), stab_result)
        summary["stabilization_status"] = stab_result["status"]

        if stab_result["status"] != REQUEST_STATUS_COMPLETE:
            summary["overall_status"] = "stabilization_failed"
            return summary

        post_stabilization_health = await check_post_stabilization_health(transport, base_url)
        readiness_info["post_stabilization_health"] = post_stabilization_health
        summary["readiness"] = readiness_info

        if not post_stabilization_health["ok"]:
            summary["overall_status"] = "post_stabilization_health_failed"
            return summary

        await sleeper.sleep(COOLDOWN_S)
        summary["cooldown_s"] = COOLDOWN_S

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
            write_json_atomic(episode_result_path(output_dir, ep.episode_id), ep_result)
            ok = ep_result["status"] == REQUEST_STATUS_COMPLETE
            episode_statuses[ep.episode_id] = (
                CLASSIFICATION_VALID_COMPLETE if ok else CLASSIFICATION_PARTIAL
            )
            summary["episode_statuses"] = dict(episode_statuses)
            if not ok:
                block_aborted = True
                break

        summary["overall_status"] = "block_failed" if block_aborted else "block_complete"
        return summary
    except BaseException as exc:
        summary["overall_status"] = "error"
        summary["error"] = str(exc)
        raise
    finally:
        if handle is not None:
            stop_result = await stop_server(
                handle, host, port, sleeper,
                timeout_s=stop_timeout_s,
                kill_confirm_timeout_s=stop_kill_confirm_timeout_s,
                port_poll_timeout_s=stop_port_poll_timeout_s,
            )
            summary["server_stop"] = stop_result
            if not stop_result.get("stop_success") and summary.get("overall_status") == "block_complete":
                # A successful block must never be reported as complete if
                # the server itself was not verifiably stopped afterward.
                summary["overall_status"] = "server_stop_failed"
        summary["end_utc"] = clock.utcnow_iso()
        write_json_atomic(output_dir / "smoke_run_summary.json", summary)



# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase_a.py",
        description=(
            "Phase A runner (stage 2: schedule-bundle validation, "
            "resume-depth checks, self-test, dry-run, and real execution "
            "of a single selectable smoke-test block). --official-run "
            "remains disabled until stage 3."
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
        help="Validate everything for a real official run, then abort: "
        "official execution is disabled until stage 3.",
    )
    mode_group.add_argument(
        "--smoke-test", action="store_true",
        help="Really execute exactly one --smoke-block: server start, "
        "readiness, one stabilization run, cooldown, its four regular "
        "episodes, and a targeted server stop.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Only valid with --official-run or --smoke-test: scan the "
        "output directory and classify existing result files.",
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
        "runs directory).",
    )
    parser.add_argument(
        "--smoke-block", type=str, default=None,
        help="Required with --smoke-test (and disallowed otherwise): the "
        "exact block_id to execute, e.g. 'llama_block01_low'.",
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_SMOKE_HOST,
        help=f"Server host for --smoke-test (default: {DEFAULT_SMOKE_HOST}).",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_SMOKE_PORT,
        help=f"Server port for --smoke-test, 1-65535 (default: {DEFAULT_SMOKE_PORT}).",
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
# Self-test (Stage 1: exercises this runner's own validation logic against
# small synthetic fixtures it builds itself; does not touch --schedule-dir)
# ============================================================================

def _build_fixture_episodes(model: str, seed: int) -> list[Episode]:
    """
    Self-test-only fixture: builds a small, internally-consistent
    1-repeat (2-block, 8-episode) synthetic schedule for a single model,
    used purely to exercise `_check_model_structure()`'s logic in
    isolation. This is NOT the frozen generator, is NOT used by any real
    validation path, and does not claim to match the official 80-episode
    contract (which needs 5 repeats x 2 models).
    """
    episodes: list[Episode] = []
    states = [(0, "low"), (12, "high")]
    cells = [
        (concurrency, condition)
        for concurrency in OFFICIAL_CONCURRENCIES
        for condition in OFFICIAL_CONDITIONS
    ]
    block_number = 0
    for offload_gb, state_label in states:
        block_number += 1
        block_id = f"{model}_block{block_number:02d}_{state_label}"
        for order_in_block, (concurrency, condition) in enumerate(cells, start=1):
            repeat = 1
            episode_id = (
                f"{model}_off{offload_gb}_conc{concurrency}_{condition}_"
                f"rep{repeat}"
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
                    episode_seed=derive_seed(str(seed), episode_id),
                    victim_workload_seed=derive_seed(
                        str(seed), model, str(concurrency), str(repeat)
                    ),
                    burst_workload_seed=derive_seed(
                        str(seed), model, str(concurrency), str(repeat), "burst"
                    ),
                    victim_request_count=20,
                    victim_input_len=256,
                    victim_output_len=64,
                    victim_temperature=0.0,
                    burst_parallel_requests=4,
                    burst_input_len=256,
                    burst_output_len=256,
                    burst_temperature=0.0,
                    restart_server_before_block=1 if order_in_block == 1 else 0,
                    block_id=block_id,
                    order_in_block=order_in_block,
                )
            )
    return episodes


def _make_fixture_block_bundle(model: str, seed: int) -> tuple["LoadedBundle", str]:
    episodes = _build_fixture_episodes(model, seed)
    block_id = episodes[0].block_id
    bundle = LoadedBundle(
        schedule_dir=Path("/nonexistent-fixture-only"),
        json_obj={"seed": seed},
        csv_fieldnames=[],
        csv_rows=[],
        audit_text="",
        episodes=episodes,
        fingerprint="sha256:" + "a" * 64,
    )
    return bundle, block_id


def _success_script_factory(payload: dict) -> "FakeStreamScript":
    return FakeStreamScript(
        prompt_token_ids_echo=list(payload["prompt"]),
        token_events=[[9000 + k] for k in range(payload["max_tokens"])],
        usage={"prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"]},
    )


def _make_success_transport() -> "FakeTransport":
    t = FakeTransport()
    t.default_script_factory = _success_script_factory
    t.set_get_response(HEALTH_ENDPOINT, 200, {})
    t.set_get_response(
        MODELS_ENDPOINT, 200,
        {"data": [{"id": MODEL_FULL_ID["llama"]}, {"id": MODEL_FULL_ID["qwen"]}]},
    )
    t.set_get_response(OPENAPI_ENDPOINT, 200, {"paths": {COMPLETIONS_ENDPOINT: {}}})
    return t


async def _stage2_async_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    fake_clock = RealClock()

    # --- 1. Token-ID prompt has exactly 256 ids -----------------------------
    tok = FakeTokenizerAdapter(vocab_size=300, special_token_ids={0, 1, 2})
    valid_ids = compute_valid_token_ids(tok)
    p256 = generate_token_id_prompt(seed=1, valid_ids=valid_ids, length=256)
    check("(1) token-id prompt has exactly 256 ids", len(p256) == 256)

    # --- 2. Special ids are never chosen -------------------------------------
    tok2 = FakeTokenizerAdapter(vocab_size=50, special_token_ids={0, 1, 2, 3, 4})
    valid_ids2 = compute_valid_token_ids(tok2)
    p_many = generate_token_id_prompt(seed=7, valid_ids=valid_ids2, length=2000)
    check(
        "(2) special token ids are never chosen",
        all(x not in {0, 1, 2, 3, 4} for x in p_many),
    )

    # --- 3. Same seed -> identical prompt-id lists ---------------------------
    a1 = generate_token_id_prompt(seed=42, valid_ids=valid_ids, length=256)
    a2 = generate_token_id_prompt(seed=42, valid_ids=valid_ids, length=256)
    a3 = generate_token_id_prompt(seed=43, valid_ids=valid_ids, length=256)
    check("(3) identical seeds produce identical prompt-id lists", a1 == a2 and a1 != a3)

    # --- 4/5. Matched low/high episodes share victim/burst sequences --------
    fixture_seed = 555001
    fixture_eps = _build_fixture_episodes("matchtest", fixture_seed)
    by_key: dict[tuple[int, str, int], list[Episode]] = {}
    for ep in fixture_eps:
        by_key.setdefault((ep.concurrency, ep.condition, ep.repeat), []).append(ep)
    matched_pairs = [v for v in by_key.values() if len(v) == 2]
    check("(4/5 setup) fixture has matched low/high pairs", len(matched_pairs) > 0)
    victim_match_ok = True
    burst_match_ok = True
    for pair in matched_pairs:
        low_ep, high_ep = pair[0], pair[1]
        for i in range(3):
            if victim_prompt_seed(low_ep, i) != victim_prompt_seed(high_ep, i):
                victim_match_ok = False
            if victim_generation_seed(low_ep, i) != victim_generation_seed(high_ep, i):
                victim_match_ok = False
            if burst_prompt_seed(low_ep, i) != burst_prompt_seed(high_ep, i):
                burst_match_ok = False
            if burst_generation_seed(low_ep, i) != burst_generation_seed(high_ep, i):
                burst_match_ok = False
        low_victims = [
            generate_token_id_prompt(victim_prompt_seed(low_ep, i), valid_ids, 8) for i in range(3)
        ]
        high_victims = [
            generate_token_id_prompt(victim_prompt_seed(high_ep, i), valid_ids, 8) for i in range(3)
        ]
        if low_victims != high_victims:
            victim_match_ok = False
        low_bursts = [
            generate_token_id_prompt(burst_prompt_seed(low_ep, j), valid_ids, 8) for j in range(2)
        ]
        high_bursts = [
            generate_token_id_prompt(burst_prompt_seed(high_ep, j), valid_ids, 8) for j in range(2)
        ]
        if low_bursts != high_bursts:
            burst_match_ok = False
    check("(4) matched low/high episodes produce identical victim sequences", victim_match_ok)
    check("(5) matched low/high episodes produce identical burst sequences", burst_match_ok)

    # --- 6. Stabilization uses its own seed domain ---------------------------
    stab_p_seed = stabilization_prompt_seed(fixture_seed, "matchtest", "matchtest_block01_low", 0)
    victim_p_seed_same_i = victim_prompt_seed(fixture_eps[0], 0)
    expected_stab_seed = derive_seed(
        str(fixture_seed), "matchtest", "matchtest_block01_low", "stabilization-prompt", "0"
    )
    check(
        "(6) stabilization_prompt_seed matches its own documented derivation "
        "and differs from the victim-prompt seed domain",
        stab_p_seed == expected_stab_seed and stab_p_seed != victim_p_seed_same_i,
    )

    # --- 7-13. Server-side completeness validation ---------------------------
    async def _exec(script: FakeStreamScript, expected_prompt: int = 10, expected_completion: int = 4) -> dict:
        t = FakeTransport()
        t.queue_script("t713", script)
        return await execute_completion_request(
            transport=t, clock=fake_clock, url="http://x/v1/completions",
            api_key="k", model_full_id="m", prompt_token_ids=list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t713", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    r7 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(7) matching server-side prompt ids -> complete",
        r7["status"] == REQUEST_STATUS_COMPLETE, str(r7["validation_errors"]),
    )

    r8 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=[99] * 10, token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check("(8) mismatched server-side prompt ids -> incomplete", r8["status"] == REQUEST_STATUS_INCOMPLETE)

    r9 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 999, "completion_tokens": 4},
    ))
    check("(9) wrong usage.prompt_tokens -> incomplete", r9["status"] == REQUEST_STATUS_INCOMPLETE)

    r10 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 999},
    ))
    check("(10) wrong usage.completion_tokens -> incomplete", r10["status"] == REQUEST_STATUS_INCOMPLETE)

    r11 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3]],
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    ))
    check("(11) too-short output-id list -> incomplete", r11["status"] == REQUEST_STATUS_INCOMPLETE)

    r12 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4}, include_done=False,
    ))
    check("(12) missing [DONE] -> incomplete", r12["status"] == REQUEST_STATUS_INCOMPLETE)

    r13 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4}, finish_reason="stop",
    ))
    check("(13) wrong finish_reason -> incomplete", r13["status"] == REQUEST_STATUS_INCOMPLETE)

    # --- 14/15. ITL availability rule -----------------------------------------
    r14 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(14) one token per event -> itl_available=true",
        r14["itl_available"] is True and r14["itl_ms"] is not None,
    )

    r15 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1, 2], [3, 4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(15) multiple tokens in one event -> itl_available=false",
        r15["itl_available"] is False and r15["itl_ms"] is None and r15["token_batch_sizes"] == [2, 2],
    )

    # --- 16/17. TPOT ends at last token event (not [DONE]); E2EL ends at stream end
    step_clock = FakeClock(step_ns=1_000_000)  # 1ms advance per clock call
    t1617 = FakeTransport()
    t1617.queue_script("t1617", FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2]],
        usage={"prompt_tokens": 10, "completion_tokens": 2}, extra_keepalives=3,
    ))
    r1617 = await execute_completion_request(
        transport=t1617, clock=step_clock, url="http://x/v1/completions",
        api_key="k", model_full_id="m", prompt_token_ids=list(range(10)),
        max_tokens=2, min_tokens=2, temperature=0.0, request_seed=1,
        request_id="t1617", role="victim", request_index=0, prompt_seed=1,
        generation_seed=1, expected_prompt_tokens=10, expected_completion_tokens=2,
        http_timeout_s=5.0,
    )
    manual_tpot_ms = (r1617["last_token_receive_ns"] - r1617["first_token_receive_ns"]) / 1e6
    check(
        "(16) client-observed TPOT is derived from first/last token "
        "timestamps, not from [DONE]",
        r1617["client_observed_tpot_ms"] == manual_tpot_ms
        and r1617["last_token_receive_ns"] < r1617["stream_end_ns"],
    )
    check(
        "(17) E2EL is derived from stream end, which is later than the last token event",
        r1617["e2el_ms"] == (r1617["stream_end_ns"] - r1617["request_start_ns"]) / 1e6
        and r1617["e2el_ms"] > manual_tpot_ms,
    )

    # --- 18. Trigger fires only after every first-wave request's first token
    ev0, ev1 = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def _fast_set() -> str:
        ev0.set()
        order.append("fast")
        await asyncio.Event().wait()  # hang forever (real task keeps running)

    async def _slow_set() -> str:
        await asyncio.sleep(0.05)
        ev1.set()
        order.append("slow")
        await asyncio.Event().wait()

    t_fast = asyncio.create_task(_fast_set())
    t_slow = asyncio.create_task(_slow_set())
    events18 = {0: ev0, 1: ev1}
    tasks18 = {0: t_fast, 1: t_slow}
    start18 = time.monotonic()
    status18 = await _watch_trigger({0, 1}, events18, tasks18, timeout_s=2.0)
    elapsed18 = time.monotonic() - start18
    await cancel_all([t_fast, t_slow])
    check(
        "(18) trigger only fires once every first-wave request's first "
        "token has arrived (waits for the slow one)",
        status18 == "ok" and elapsed18 >= 0.04 and order == ["fast", "slow"],
    )

    # --- 19. Trigger timeout cancels all tasks --------------------------------
    async def _hangs_forever() -> None:
        await asyncio.Event().wait()

    ev_a, ev_b = asyncio.Event(), asyncio.Event()
    t_a = asyncio.create_task(_hangs_forever())
    t_b = asyncio.create_task(_hangs_forever())
    status19 = await _watch_trigger(
        {0, 1}, {0: ev_a, 1: ev_b}, {0: t_a, 1: t_b}, timeout_s=0.05
    )
    await cancel_all([t_a, t_b])
    check(
        "(19) trigger timeout is reported and both first-wave tasks can "
        "then be cancelled",
        status19 == "timeout" and t_a.done() and t_b.done(),
    )

    # --- 20. No request task keeps running after cancellation ---------------
    t20 = FakeTransport()
    t20.default_script_factory = lambda payload: FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])

    async def _hanging_request(i: int) -> dict:
        return await execute_completion_request(
            transport=t20, clock=fake_clock, url="http://x", api_key="k", model_full_id="m",
            prompt_token_ids=[1, 2, 3], max_tokens=4, min_tokens=4, temperature=0.0,
            request_seed=1, request_id=f"hang{i}", role="victim", request_index=i,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=3,
            expected_completion_tokens=4, http_timeout_s=30.0,
        )

    hang_tasks = [asyncio.create_task(_hanging_request(i)) for i in range(4)]
    await asyncio.sleep(0.02)
    active_before = t20.active_stream_count
    await cancel_all(hang_tasks)
    check(
        "(20) after cancel_all(), no fake stream is still active and all "
        "tasks are done",
        active_before == 4 and t20.active_stream_count == 0 and all(t.done() for t in hang_tasks),
    )

    # --- 21/22. Burst starts only after trigger; no_burst issues no bursts ---
    tok21 = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
    valid21 = compute_valid_token_ids(tok21)
    transport21 = _make_success_transport()
    ctx21 = RunContext(
        transport=transport21, clock=RealClock(), sleeper=FakeSleeper(),
        base_url="http://127.0.0.1:1", api_key="k21", model_full_id="fake/model",
        valid_ids=valid21, trigger_timeout_s=5.0,
    )
    ep21 = Episode(
        episode_id="burst_after_trigger_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=4, condition="fixed_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=8,
        victim_input_len=16, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=16, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block21",
        order_in_block=1,
    )
    result21 = await run_regular_episode(
        ctx21, ep21, schedule_fingerprint="sha256:" + "0" * 64,
        server_metadata={}, stabilization_ref={},
    )
    burst_starts_after_trigger = (
        result21["burst_interval"] is not None
        and result21["burst_interval"]["start_ns"] >= result21["trigger"]["trigger_perf_ns"]
    )
    check(
        "(21) burst requests only start once the trigger has fired",
        result21["status"] == REQUEST_STATUS_COMPLETE and burst_starts_after_trigger,
        str(result21.get("validation_errors")),
    )

    ep22 = Episode(**{**vars(ep21), "condition": "no_burst", "episode_id": "no_burst_ep22"})
    result22 = await run_regular_episode(
        ctx21, ep22, schedule_fingerprint="sha256:" + "0" * 64,
        server_metadata={}, stabilization_ref={},
    )
    check(
        "(22) no_burst issues zero burst requests",
        result22["burst_requests"] == [] and result22["burst_interval"] is None,
    )

    # --- 23/24/25/29/30/31/32/33/34: full run_smoke_block scenarios ---------
    import tempfile as _tempfile

    tok_block = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
    secret_key = "self-test-secret-key-should-never-leak"
    run_server_path_fixture = Path("/nonexistent/run_server.sh")

    # (23) partial stabilization prevents every regular episode.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle23, block_id23 = _make_fixture_block_bundle("llama", 700001)
        t23 = _make_success_transport()
        t23.queue_script(
            f"{block_id23}:stabilization:3",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(256)), token_events=[[1]] * 64,
                finish_reason="stop", usage={"prompt_tokens": 256, "completion_tokens": 64},
            ),
        )
        server_adapter23 = FakeServerProcessAdapter()
        summary23 = await run_smoke_block(
            bundle=bundle23, block_id=block_id23, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18101, resume=False, api_key=secret_key,
            transport=t23, tokenizer=tok_block, server_adapter=server_adapter23,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_written = list((tmp_path / "out" / "episodes").glob("*.json")) if (tmp_path / "out" / "episodes").exists() else []
        check(
            "(23) partial stabilization prevents every regular episode from running",
            summary23["overall_status"] == "stabilization_failed" and not episodes_written,
            str(summary23.get("overall_status")),
        )
        check("(31, block on stab-fail) episode dir stays empty under episodes/", not episodes_written)
        check(
            "(32) stabilization output is written under stabilization/",
            (tmp_path / "out" / "stabilization" / f"{block_id23}.json").exists(),
        )
        check(
            "(34a) API key never appears in the smoke summary",
            secret_key not in json.dumps(summary23),
        )

    # (24) drift alone does not block the episodes (abort_on_stability_drift=False).
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle24, block_id24 = _make_fixture_block_bundle("llama", 700002)
        t24 = _make_success_transport()
        valid_ids24 = compute_valid_token_ids(tok_block)
        # First half: 1 token/event (fast). Second half: dramatically more
        # elapsed wall time via extra keepalives -- still functionally
        # complete, but the two halves' timing looks very different.
        for i in range(10, 20):
            p_seed24 = stabilization_prompt_seed(bundle24.json_obj["seed"], "llama", block_id24, i)
            prompt_ids24 = generate_token_id_prompt(p_seed24, valid_ids24, STABILIZATION_INPUT_LEN)
            t24.queue_script(
                f"{block_id24}:stabilization:{i}",
                FakeStreamScript(
                    prompt_token_ids_echo=prompt_ids24, token_events=[[1]] * 64,
                    usage={"prompt_tokens": 256, "completion_tokens": 64},
                    extra_keepalives=25,
                ),
            )
        server_adapter24 = FakeServerProcessAdapter()
        summary24 = await run_smoke_block(
            bundle=bundle24, block_id=block_id24, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18102, resume=False, api_key=secret_key,
            transport=t24, tokenizer=tok_block, server_adapter=server_adapter24,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        stab24 = json.loads((tmp_path / "out" / "stabilization" / f"{block_id24}.json").read_text())
        check(
            "(24) functional/stabilization pass even with large timing drift "
            "between halves (abort_on_stability_drift=False)",
            stab24["functional_passed"] is True and stab24["stabilization_passed"] is True,
            str([r["validation_errors"] for r in stab24["request_results"] if r["status"] != "complete"]),
        )
        check(
            "(24b) drift is documented, not gating: episodes still ran",
            summary24["overall_status"] == "block_complete",
        )


    # (25) a partial episode prevents the next episode of the block.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle25, block_id25 = _make_fixture_block_bundle("llama", 700003)
        block_eps25 = find_block(bundle25, block_id25)
        second_ep25 = block_eps25[1]
        t25 = _make_success_transport()
        t25.queue_script(
            f"{second_ep25.episode_id}:victim:0",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(second_ep25.victim_input_len)),
                token_events=[[1]] * (second_ep25.victim_output_len - 1),
                usage={
                    "prompt_tokens": second_ep25.victim_input_len,
                    "completion_tokens": second_ep25.victim_output_len - 1,
                },
            ),
        )
        server_adapter25 = FakeServerProcessAdapter()
        summary25 = await run_smoke_block(
            bundle=bundle25, block_id=block_id25, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18103, resume=False, api_key=secret_key,
            transport=t25, tokenizer=tok_block, server_adapter=server_adapter25,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        st25 = summary25["episode_statuses"]
        check(
            "(25) a partial episode prevents the next episode(s) of the "
            "block from starting",
            st25[block_eps25[0].episode_id] == CLASSIFICATION_VALID_COMPLETE
            and st25[block_eps25[1].episode_id] == CLASSIFICATION_PARTIAL
            and st25[block_eps25[2].episode_id] == CLASSIFICATION_MISSING
            and st25[block_eps25[3].episode_id] == CLASSIFICATION_MISSING
            and not episode_result_path(tmp_path / "out", block_eps25[2].episode_id).exists(),
            str(st25),
        )
        # Per section 22, a 'partial' file is exactly as non-resumable as
        # 'invalid'/'corrupted' -- it must gate resume with a clear abort,
        # not be silently rerun.
        server_adapter25_resume = FakeServerProcessAdapter()
        raised25b = False
        try:
            await run_smoke_block(
                bundle=bundle25, block_id=block_id25, output_dir=tmp_path / "out",
                host="127.0.0.1", port=18104, resume=True, api_key=secret_key,
                transport=_make_success_transport(), tokenizer=tok_block,
                server_adapter=server_adapter25_resume, sleeper=FakeSleeper(), clock=RealClock(),
                run_server_path=run_server_path_fixture,
            )
        except ServerLifecycleError:
            raised25b = True
        check(
            "(25b) a leftover 'partial' episode file also gates --resume "
            "with a clear abort (never silently rerun)",
            raised25b and len(server_adapter25_resume.started) == 0,
        )

    # (29) --resume begins at the first genuinely missing episode, restarts
    # the server, and leaves already-valid_complete episodes untouched.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle29, block_id29 = _make_fixture_block_bundle("llama", 700005)
        block_eps29 = find_block(bundle29, block_id29)
        server_adapter29a = FakeServerProcessAdapter()
        summary29a = await run_smoke_block(
            bundle=bundle29, block_id=block_id29, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18107, resume=False, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter29a, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        check("(29 setup) initial full run completes", summary29a["overall_status"] == "block_complete")

        kept_path = episode_result_path(tmp_path / "out", block_eps29[0].episode_id)
        kept_text = kept_path.read_text()
        for ep in block_eps29[1:]:
            episode_result_path(tmp_path / "out", ep.episode_id).unlink()

        server_adapter29b = FakeServerProcessAdapter()
        summary29b = await run_smoke_block(
            bundle=bundle29, block_id=block_id29, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18108, resume=True, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter29b, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        check(
            "(29) --resume begins at the first missing episode, restarts "
            "the server (stabilization is mandatory again), and leaves "
            "the already-valid_complete episode's file byte-for-byte untouched",
            summary29b["overall_status"] == "block_complete"
            and len(server_adapter29b.started) == 1
            and kept_path.read_text() == kept_text
            and all(
                v == CLASSIFICATION_VALID_COMPLETE for v in summary29b["episode_statuses"].values()
            ),
            str(summary29b["episode_statuses"]),
        )

    # (30) a foreign/invalid result file must never be silently overwritten,
    # and must gate resume before any server is started.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle30, block_id30 = _make_fixture_block_bundle("llama", 700006)
        block_eps30 = find_block(bundle30, block_id30)
        bad_path = episode_result_path(tmp_path / "out", block_eps30[0].episode_id)
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_payload = json.dumps({"runner_version": "not-the-real-runner"})
        bad_path.write_text(bad_payload, encoding="utf-8")
        server_adapter30 = FakeServerProcessAdapter()
        raised30 = False
        try:
            await run_smoke_block(
                bundle=bundle30, block_id=block_id30, output_dir=tmp_path / "out",
                host="127.0.0.1", port=18109, resume=True, api_key=secret_key,
                transport=_make_success_transport(), tokenizer=tok_block,
                server_adapter=server_adapter30, sleeper=FakeSleeper(), clock=RealClock(),
                run_server_path=run_server_path_fixture,
            )
        except ServerLifecycleError:
            raised30 = True
        check(
            "(30) a foreign/invalid episode result file is rejected, not "
            "silently overwritten, and no server is started",
            raised30 and len(server_adapter30.started) == 0 and bad_path.read_text() == bad_payload,
        )

    # (31/32/33/34 full happy path) episodes/ + stabilization/ dirs, no temp
    # files after success, API key never leaks anywhere on disk.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle31, block_id31 = _make_fixture_block_bundle("llama", 700004)
        block_eps31 = find_block(bundle31, block_id31)
        server_adapter31 = FakeServerProcessAdapter()
        summary31 = await run_smoke_block(
            bundle=bundle31, block_id=block_id31, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18106, resume=False, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter31, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        episodes_ok = all(
            episode_result_path(tmp_path / "out", ep.episode_id).exists() for ep in block_eps31
        )
        check("(31) episode files live under episodes/", episodes_ok and summary31["overall_status"] == "block_complete")
        check(
            "(32, happy path) stabilization file lives under stabilization/",
            stabilization_result_path(tmp_path / "out", block_id31).exists(),
        )
        leftovers = list((tmp_path / "out").rglob("*.tmp.*"))
        check("(33) the atomic writer leaves no temp file behind after success", not leftovers, str(leftovers))
        all_text = "".join(p.read_text() for p in (tmp_path / "out").rglob("*.json"))
        check(
            "(34) the API key never appears in any serialized result on disk",
            secret_key not in all_text and secret_key not in json.dumps(summary31),
        )
        check(
            "(28) stop_server only signals the server's own process "
            "group (SIGTERM path), never a global kill",
            server_adapter31.started[0].terminated and not server_adapter31.started[0].killed,
        )

    # --- 26/27. Server command shape -----------------------------------------
    cmd = build_server_command(Path("/x/run_server.sh"), "llama", 12, "127.0.0.1", 8123)
    check(
        "(26) the server command is exactly `bash run_server.sh <model> "
        "<offload_gb> <host> <port>`",
        cmd == ["bash", "/x/run_server.sh", "llama", "12", "127.0.0.1", "8123"],
    )
    check(
        "(27) the server command never contains an API key",
        all("secret" not in part.lower() and "key" not in part.lower() for part in cmd),
    )

    # =========================================================================
    # Patch: Stage-2-Realpfad-Abschluss -- sections 1, 2, 3, 5, 6, 7
    # =========================================================================

    # --- Section 1: readiness polls through transient connection errors ----
    # (P1-1) two ConnectionRefusedError, then a successful health/models check.
    t_p1a = FakeTransport()
    t_p1a.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    t_p1a.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    t_p1a.set_get_response(HEALTH_ENDPOINT, 200, {})
    t_p1a.set_get_response(MODELS_ENDPOINT, 200, {"data": [{"id": "fake/model"}]})
    handle_p1a = FakeServerHandle(["bash", "x"])
    readiness_p1a = await wait_for_server_ready(
        t_p1a, handle_p1a, "http://x", "k", "fake/model", FakeSleeper(),
        timeout_s=5.0, poll_interval_s=0.001,
    )
    check(
        "(P1-1) readiness polls through two transient ConnectionRefusedError "
        "and then succeeds",
        readiness_p1a["detected_model"] == "fake/model" and readiness_p1a["poll_count"] >= 3,
        str(readiness_p1a),
    )

    # (P1-2) health 200 immediately; models 503 then 200.
    t_p1b = FakeTransport()
    t_p1b.set_get_response(HEALTH_ENDPOINT, 200, {})
    t_p1b.queue_get_status(MODELS_ENDPOINT, 503, {})
    t_p1b.set_get_response(MODELS_ENDPOINT, 200, {"data": [{"id": "fake/model"}]})
    handle_p1b = FakeServerHandle(["bash", "x"])
    readiness_p1b = await wait_for_server_ready(
        t_p1b, handle_p1b, "http://x", "k", "fake/model", FakeSleeper(),
        timeout_s=5.0, poll_interval_s=0.001,
    )
    check(
        "(P1-2) readiness polls through a transient /v1/models 503 and then succeeds",
        readiness_p1b["detected_model"] == "fake/model",
    )

    # (P1-3) server process dies mid-poll -> a clear error.
    class _DyingSleeper:
        def __init__(self, handle: FakeServerHandle) -> None:
            self.handle = handle

        async def sleep(self, seconds: float) -> None:
            self.handle.alive = False
            await asyncio.sleep(0)

    t_p1c = FakeTransport()
    t_p1c.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    handle_p1c = FakeServerHandle(["bash", "x"])
    raised_p1c = False
    try:
        await wait_for_server_ready(
            t_p1c, handle_p1c, "http://x", "k", "fake/model", _DyingSleeper(handle_p1c),
            timeout_s=5.0, poll_interval_s=0.001,
        )
    except ServerLifecycleError:
        raised_p1c = True
    check("(P1-3) the server process dying mid-poll raises a clear ServerLifecycleError", raised_p1c)

    # (P1-4) only transient errors until the deadline -> a clean timeout.
    class _AlwaysFailTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            raise ConnectionRefusedError("refused")

    handle_p1d = FakeServerHandle(["bash", "x"])
    raised_p1d = False
    try:
        await wait_for_server_ready(
            _AlwaysFailTransport(), handle_p1d, "http://x", "k", "fake/model", FakeSleeper(),
            timeout_s=0.05, poll_interval_s=0.001,
        )
    except ServerLifecycleError:
        raised_p1d = True
    check("(P1-4) only-transient errors until the deadline -> a clean ServerLifecycleError timeout", raised_p1d)

    # (P1-5) cancellation is never swallowed.
    class _HangingTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            await asyncio.Event().wait()

    handle_p1e = FakeServerHandle(["bash", "x"])
    readiness_task = asyncio.create_task(
        wait_for_server_ready(
            _HangingTransport(), handle_p1e, "http://x", "k", "fake/model", FakeSleeper(),
            timeout_s=30.0, poll_interval_s=1.0,
        )
    )
    await asyncio.sleep(0.02)
    readiness_task.cancel()
    cancelled_p1e = False
    try:
        await readiness_task
    except asyncio.CancelledError:
        cancelled_p1e = True
    check("(P1-5) readiness cancellation (asyncio.CancelledError) is never swallowed", cancelled_p1e)

    # --- Section 2: post-stabilization health gate --------------------------
    t_health_ok = FakeTransport()
    t_health_ok.set_get_response(HEALTH_ENDPOINT, 200, {})
    result_health_ok = await check_post_stabilization_health(t_health_ok, "http://x")
    check(
        "(P2-1) post-stabilization health check: HTTP 200 -> ok=True",
        result_health_ok["ok"] is True and result_health_ok["http_status"] == 200,
    )

    t_health_503 = FakeTransport()
    t_health_503.set_get_response(HEALTH_ENDPOINT, 503, {})
    result_health_503 = await check_post_stabilization_health(t_health_503, "http://x")
    check(
        "(P2-2) post-stabilization health check: HTTP 503 -> ok=False",
        result_health_503["ok"] is False and result_health_503["http_status"] == 503,
    )

    class _RaisingHealthTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            raise ConnectionRefusedError("refused")

    result_health_exc = await check_post_stabilization_health(_RaisingHealthTransport(), "http://x")
    check(
        "(P2-3) post-stabilization health check: connection exception -> "
        "ok=False, no crash",
        result_health_exc["ok"] is False and result_health_exc["error_type"] == "ConnectionRefusedError",
    )

    class _CountingHealthTransport:
        """Wraps a base FakeTransport: /health returns 200 for the first
        `pass_count` calls (covering readiness's own polling), then
        `fail_status`/`fail_exc` for every call after that (covering the
        post-stabilization gate specifically)."""

        def __init__(self, base: "FakeTransport", pass_count: int, fail_status: int | None = None, fail_exc: BaseException | None = None) -> None:
            self.base = base
            self.pass_count = pass_count
            self.fail_status = fail_status
            self.fail_exc = fail_exc
            self.health_calls = 0

        async def get_json(self, url: str, headers: dict, timeout_s: float):
            if url.endswith(HEALTH_ENDPOINT):
                self.health_calls += 1
                if self.health_calls > self.pass_count:
                    if self.fail_exc is not None:
                        raise self.fail_exc
                    return self.fail_status, {}
                return 200, {}
            return await self.base.get_json(url, headers, timeout_s)

        async def stream_completion(self, *a, **kw):
            async for x in self.base.stream_completion(*a, **kw):
                yield x

    # (P2-4) health 200 after stabilization -> cooldown happens, episodes run.
    bundle_p2a, block_id_p2a = _make_fixture_block_bundle("llama", 800101)
    t_p2a = _CountingHealthTransport(_make_success_transport(), pass_count=1, fail_status=200)
    server_adapter_p2a = FakeServerProcessAdapter()
    sleeper_p2a = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2a = await run_smoke_block(
            bundle=bundle_p2a, block_id=block_id_p2a, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18301, resume=False, api_key="k",
            transport=t_p2a, tokenizer=tok_block, server_adapter=server_adapter_p2a,
            sleeper=sleeper_p2a, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        check(
            "(P2-4) health 200 after stabilization -> cooldown happens and episodes run",
            summary_p2a["overall_status"] == "block_complete"
            and summary_p2a["readiness"]["post_stabilization_health"]["ok"] is True
            and COOLDOWN_S in sleeper_p2a.calls,
            str(summary_p2a.get("overall_status")),
        )

    # (P2-5) health 503 after stabilization -> no episode runs.
    bundle_p2b, block_id_p2b = _make_fixture_block_bundle("llama", 800102)
    t_p2b = _CountingHealthTransport(_make_success_transport(), pass_count=1, fail_status=503)
    server_adapter_p2b = FakeServerProcessAdapter()
    sleeper_p2b = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2b = await run_smoke_block(
            bundle=bundle_p2b, block_id=block_id_p2b, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18302, resume=False, api_key="k",
            transport=t_p2b, tokenizer=tok_block, server_adapter=server_adapter_p2b,
            sleeper=sleeper_p2b, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_dir_p2b = tmp_path / "out" / "episodes"
        episodes_written_p2b = list(episodes_dir_p2b.glob("*.json")) if episodes_dir_p2b.exists() else []
        check(
            "(P2-5) health 503 after stabilization -> "
            "post_stabilization_health_failed, no episode runs, no cooldown",
            summary_p2b["overall_status"] == "post_stabilization_health_failed"
            and not episodes_written_p2b
            and COOLDOWN_S not in sleeper_p2b.calls,
        )

    # (P2-6) connection exception at the post-stabilization health check.
    bundle_p2c, block_id_p2c = _make_fixture_block_bundle("llama", 800103)
    t_p2c = _CountingHealthTransport(
        _make_success_transport(), pass_count=1, fail_exc=ConnectionRefusedError("refused")
    )
    server_adapter_p2c = FakeServerProcessAdapter()
    sleeper_p2c = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2c = await run_smoke_block(
            bundle=bundle_p2c, block_id=block_id_p2c, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18303, resume=False, api_key="k",
            transport=t_p2c, tokenizer=tok_block, server_adapter=server_adapter_p2c,
            sleeper=sleeper_p2c, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_dir_p2c = tmp_path / "out" / "episodes"
        episodes_written_p2c = list(episodes_dir_p2c.glob("*.json")) if episodes_dir_p2c.exists() else []
        check(
            "(P2-6) a connection exception at the post-stabilization health "
            "check does not crash and prevents every episode",
            summary_p2c["overall_status"] == "post_stabilization_health_failed" and not episodes_written_p2c,
        )

    # --- Section 3: robust, verified server stop ----------------------------
    def _always_free(host: str, port: int) -> bool:
        return True

    def _always_occupied(host: str, port: int) -> bool:
        return False

    handle_p3a = FakeServerHandle(["bash", "x"], already_dead=True)
    stop_p3a = await stop_server(handle_p3a, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-1) an already-dead process gets no unnecessary signal, "
        "stop_success=True",
        stop_p3a["term_sent"] is False and not handle_p3a.terminated and stop_p3a["stop_success"] is True,
    )

    handle_p3b = FakeServerHandle(["bash", "x"], raise_on_terminate=ProcessLookupError())
    stop_p3b = await stop_server(handle_p3b, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-2) ProcessLookupError on SIGTERM is handled cleanly (no crash, "
        "stop_success=True, no stop_error)",
        stop_p3b["stop_success"] is True and stop_p3b["stop_error"] is None,
    )

    handle_p3c = FakeServerHandle(["bash", "x"])
    stop_p3c = await stop_server(handle_p3c, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-3) process dies + port confirmed free -> stop_success=True",
        stop_p3c["stop_success"] is True and stop_p3c["alive_after_stop"] is False,
    )

    handle_p3d = FakeServerHandle(["bash", "x"], dies_on_terminate=False, dies_on_kill=False)
    stop_p3d = await stop_server(
        handle_p3d, "127.0.0.1", 1, FakeSleeper(),
        timeout_s=0.01, kill_confirm_timeout_s=0.01, port_free_check=_always_free,
    )
    check(
        "(P3-4) a process that survives SIGKILL -> stop_success=False, "
        "forced_kill=True, alive_after_stop=True",
        stop_p3d["stop_success"] is False
        and stop_p3d["forced_kill"] is True
        and stop_p3d["alive_after_stop"] is True,
    )

    handle_p3e = FakeServerHandle(["bash", "x"])
    stop_p3e = await stop_server(
        handle_p3e, "127.0.0.1", 1, FakeSleeper(),
        port_free_check=_always_occupied, port_poll_timeout_s=0.01,
    )
    check(
        "(P3-5) process dies but the port stays occupied -> stop_success=False",
        stop_p3e["stop_success"] is False
        and stop_p3e["alive_after_stop"] is False
        and stop_p3e["port_free_after_stop"] is False,
    )

    class _StuckServerAdapter:
        def __init__(self) -> None:
            self.started: list[FakeServerHandle] = []

        def start(self, cmd: list[str], log_path: Path) -> FakeServerHandle:
            h = FakeServerHandle(cmd, dies_on_terminate=False, dies_on_kill=False)
            self.started.append(h)
            return h

    bundle_p3f, block_id_p3f = _make_fixture_block_bundle("llama", 800201)
    adapter_p3f = _StuckServerAdapter()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p3f = await run_smoke_block(
            bundle=bundle_p3f, block_id=block_id_p3f, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18310, resume=False, api_key="k",
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=adapter_p3f, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
            stop_timeout_s=0.01, stop_kill_confirm_timeout_s=0.01, stop_port_poll_timeout_s=0.01,
        )
        check(
            "(P3-6) a block that otherwise finished cleanly is downgraded to "
            "'server_stop_failed' when the server process never actually stops",
            summary_p3f["overall_status"] == "server_stop_failed"
            and summary_p3f["server_stop"]["stop_success"] is False,
            str(summary_p3f.get("overall_status")),
        )

    handle_p3g = FakeServerHandle(["bash", "x"])
    await stop_server(handle_p3g, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-7) stop_server only ever signals the handle's own PGID "
        "(terminate_group()/kill_group() on that exact handle)",
        handle_p3g.terminated and not handle_p3g.killed,
    )

    # --- Section 5: SSE/JSON protocol errors devalue a request --------------
    async def _exec_raw(extra_raw_events: list[str], token_events=None) -> dict:
        expected_prompt, expected_completion = 10, 4
        t = FakeTransport()
        t.queue_script(
            "t_proto",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(expected_prompt)),
                token_events=token_events if token_events is not None else [[1], [2], [3], [4]],
                usage={"prompt_tokens": expected_prompt, "completion_tokens": expected_completion},
                extra_raw_events_before_finish=extra_raw_events,
            ),
        )
        return await execute_completion_request(
            transport=t, clock=RealClock(), url="http://x/v1/completions",
            api_key="k", model_full_id="m", prompt_token_ids=list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t_proto", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    r_p5_1 = await _exec_raw(["{not valid json"])
    check(
        "(P5-1) an invalid-JSON SSE event, followed by an otherwise complete "
        "stream, is never 'complete'",
        r_p5_1["status"] != REQUEST_STATUS_COMPLETE
        and any("JSON parse error" in e for e in r_p5_1["validation_errors"]),
        str(r_p5_1["validation_errors"]),
    )

    r_p5_2 = await _exec_raw(
        [json.dumps({"choices": [{"index": 0, "token_ids": ["1"], "finish_reason": None}]})]
    )
    check(
        "(P5-2) token_ids containing non-int elements is a protocol error -> not 'complete'",
        r_p5_2["status"] != REQUEST_STATUS_COMPLETE
        and any("token_ids" in e for e in r_p5_2["validation_errors"]),
        str(r_p5_2["validation_errors"]),
    )

    r_p5_3 = await _exec_raw(
        [json.dumps({"choices": [{"index": 0, "token_ids": [], "finish_reason": None}], "usage": []})]
    )
    check(
        "(P5-3) usage as a list instead of a dict is a protocol error and "
        "never crashes -> not 'complete'",
        r_p5_3["status"] != REQUEST_STATUS_COMPLETE
        and any("usage" in e for e in r_p5_3["validation_errors"]),
        str(r_p5_3["validation_errors"]),
    )

    r_p5_4 = await _exec_raw([json.dumps({"prompt_token_ids": [999] * 10, "choices": []})])
    check(
        "(P5-4) contradicting prompt_token_ids across multiple events -> not 'complete'",
        r_p5_4["status"] != REQUEST_STATUS_COMPLETE
        and any("contradicting prompt_token_ids" in e for e in r_p5_4["validation_errors"]),
        str(r_p5_4["validation_errors"]),
    )

    r_p5_5 = await _exec_raw([])
    check(
        "(P5-5) a fully well-formed protocol stays 'complete'",
        r_p5_5["status"] == REQUEST_STATUS_COMPLETE, str(r_p5_5["validation_errors"]),
    )

    # --- Section 6: trigger failures preserve full raw abort data ----------
    tok_trig = FakeTokenizerAdapter(vocab_size=500, special_token_ids={0, 1, 2})
    valid_trig = compute_valid_token_ids(tok_trig)

    t_p6a = FakeTransport()
    t_p6a.default_script_factory = lambda payload: FakeStreamScript(
        hang=True, prompt_token_ids_echo=None, token_events=[]
    )
    ctx_p6a = RunContext(
        transport=t_p6a, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id="m", valid_ids=valid_trig, trigger_timeout_s=0.05,
    )
    ep_p6a = Episode(
        episode_id="trig_timeout_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=3, condition="no_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=6,
        victim_input_len=8, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=8, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p6a",
        order_in_block=1,
    )
    result_p6a = await run_regular_episode(
        ctx_p6a, ep_p6a, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    check(
        "(P6-1) trigger timeout: exactly N victim records are stored and "
        "the episode is marked failed",
        len(result_p6a["victim_requests"]) == ep_p6a.victim_request_count
        and result_p6a["trigger"]["status"] == "timeout"
        and result_p6a["status"] == "failed",
        str(len(result_p6a["victim_requests"])),
    )
    check("(P6-2) after a trigger timeout, no fake stream is still active", t_p6a.active_stream_count == 0)
    check(
        "(P6-2b) every stored victim record after a trigger timeout has a "
        "well-formed identity, even the ones that never started",
        all(
            r.get("request_id") == f"trig_timeout_ep:victim:{i}" and r.get("role") == "victim"
            for i, r in enumerate(result_p6a["victim_requests"])
        ),
    )

    t_p6b = FakeTransport()
    t_p6b.queue_script(
        "trig_partial_ep:victim:0",
        FakeStreamScript(
            prompt_token_ids_echo=list(range(8)), token_events=[[1]],
            usage={"prompt_tokens": 8, "completion_tokens": 1},
        ),
    )
    t_p6b.queue_script(
        "trig_partial_ep:victim:1", FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])
    )
    t_p6b.queue_script(
        "trig_partial_ep:victim:2", FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])
    )
    ctx_p6b = RunContext(
        transport=t_p6b, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id="m", valid_ids=valid_trig, trigger_timeout_s=2.0,
    )
    ep_p6b = Episode(
        episode_id="trig_partial_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=3, condition="fixed_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=3,
        victim_input_len=8, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=8, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p6b",
        order_in_block=1,
    )
    result_p6b = await run_regular_episode(
        ctx_p6b, ep_p6b, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    check(
        "(P6-3) a first-wave request that delivers one token and then ends "
        "incomplete triggers pretrigger_failure even while a sibling is "
        "still waiting",
        result_p6b["trigger"]["status"] == "pretrigger_failure",
        str(result_p6b["trigger"]),
    )
    check("(P6-4) zero burst requests are started after a pretrigger_failure", result_p6b["burst_requests"] == [])
    check(
        "(P6-5) raw SSE events from the request that did start are preserved",
        len(result_p6b["victim_requests"]) == 3
        and any(r.get("raw_sse_events") for r in result_p6b["victim_requests"]),
        str([len(r.get("raw_sse_events") or []) for r in result_p6b["victim_requests"]]),
    )

    # --- Section 7: exact first wave -----------------------------------------
    t_p7 = _make_success_transport()
    ctx_p7 = RunContext(
        transport=t_p7, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id=MODEL_FULL_ID["llama"], valid_ids=compute_valid_token_ids(tok_block),
        trigger_timeout_s=5.0,
    )
    ep_p7 = Episode(
        episode_id="firstwave_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=4, condition="no_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=333, burst_workload_seed=444, victim_request_count=10,
        victim_input_len=16, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=16, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p7",
        order_in_block=1,
    )
    result_p7 = await run_regular_episode(
        ctx_p7, ep_p7, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    first_started = t_p7.call_order[: ep_p7.concurrency]
    expected_first = [f"{ep_p7.episode_id}:victim:{i}" for i in range(ep_p7.concurrency)]
    check(
        "(P7-1) the first `concurrency` requests to actually start are "
        "exactly victim indices 0..concurrency-1",
        first_started == expected_first, str(t_p7.call_order),
    )
    check(
        "(P7-2) never more than `concurrency` victim streams are active at once",
        t_p7.max_active_stream_count <= ep_p7.concurrency, str(t_p7.max_active_stream_count),
    )
    check("(P7-setup) the episode itself still completes successfully", result_p7["status"] == REQUEST_STATUS_COMPLETE)

    # =========================================================================
    # Patch: real-vLLM prompt-token-id mapping (choices[0] vs top-level)
    # =========================================================================

    async def _exec_full_raw(
        raw_events: list[str], expected_prompt: int = 5, expected_completion: int = 3,
        sent_prompt: list[int] | None = None,
    ) -> dict:
        t = FakeTransport()
        t.queue_script(
            "t_map",
            FakeStreamScript(
                prompt_token_ids_echo=None, token_events=[], include_done=False,
                extra_raw_events_before_finish=raw_events,
            ),
        )
        return await execute_completion_request(
            transport=t, clock=RealClock(), url="http://x/v1/completions",
            api_key="k", model_full_id="m",
            prompt_token_ids=sent_prompt if sent_prompt is not None else list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t_map", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    prompt5 = [10, 11, 12, 13, 14]

    # --- extract_prompt_token_ids() unit checks (isolated, no I/O) ----------
    check(
        "(helper-a) extract_prompt_token_ids: top-level only -> recognized",
        extract_prompt_token_ids({"prompt_token_ids": prompt5}, {}, 0) == (prompt5, []),
    )
    check(
        "(helper-b) extract_prompt_token_ids: choices[0] only -> recognized",
        extract_prompt_token_ids({}, {"prompt_token_ids": prompt5}, 0) == (prompt5, []),
    )
    check(
        "(helper-c) extract_prompt_token_ids: both positions null -> (None, [])",
        extract_prompt_token_ids({"prompt_token_ids": None}, {"prompt_token_ids": None}, 0) == (None, []),
    )
    _bad_top, _bad_errs = extract_prompt_token_ids({"prompt_token_ids": "nope"}, {}, 0)
    check(
        "(helper-d) extract_prompt_token_ids never crashes on malformed input",
        _bad_top is None and len(_bad_errs) == 1,
    )

    # (1) prompt-token-ids reported only at the top level.
    events_1 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r1 = await _exec_full_raw(events_1, sent_prompt=prompt5)
    check(
        "(1) prompt-token-ids reported only at the top level are recognized",
        r1["status"] == REQUEST_STATUS_COMPLETE and r1["prompt_token_ids_returned"] == prompt5,
        str(r1["validation_errors"]),
    )

    # (2) prompt-token-ids reported only inside choices[0] (real vLLM 0.17.1 shape).
    events_2 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length", "prompt_token_ids": None}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r2 = await _exec_full_raw(events_2, sent_prompt=prompt5)
    check(
        "(2) prompt-token-ids reported only inside choices[0] are recognized",
        r2["status"] == REQUEST_STATUS_COMPLETE and r2["prompt_token_ids_returned"] == prompt5,
        str(r2["validation_errors"]),
    )

    # (3) identical top-level and choices[0] values in the same event.
    events_3 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r3 = await _exec_full_raw(events_3, sent_prompt=prompt5)
    check(
        "(3) identical top-level and choices[0] prompt_token_ids within the "
        "same event are both recognized, not treated as a conflict",
        r3["status"] == REQUEST_STATUS_COMPLETE and r3["prompt_token_ids_returned"] == prompt5,
        str(r3["validation_errors"]),
    )

    # (4) contradicting top-level vs choices[0] within the same event.
    events_4 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": [999, 999, 999, 999, 999]}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r4 = await _exec_full_raw(events_4, sent_prompt=prompt5)
    check(
        "(4) contradicting top-level vs choices[0] prompt_token_ids within "
        "the same event -> not 'complete'",
        r4["status"] != REQUEST_STATUS_COMPLETE,
        str(r4["validation_errors"]),
    )

    # (5) choices[0].prompt_token_ids of the wrong type.
    events_5 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": "not-a-list"}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r5 = await _exec_full_raw(events_5, sent_prompt=prompt5)
    check(
        "(5) choices[0].prompt_token_ids of the wrong type -> not 'complete'",
        r5["status"] != REQUEST_STATUS_COMPLETE,
        str(r5["validation_errors"]),
    )

    # (6) only the first event carries choices[0].prompt_token_ids, later events send null.
    events_6 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length", "prompt_token_ids": None}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r6 = await _exec_full_raw(events_6, sent_prompt=prompt5)
    check(
        "(6) only the first event carries choices[0].prompt_token_ids, "
        "later events send null -> 'complete'",
        r6["status"] == REQUEST_STATUS_COMPLETE,
        str(r6["validation_errors"]),
    )

    # (7) two events with an identical choices[0].prompt_token_ids list.
    events_7 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r7 = await _exec_full_raw(events_7, sent_prompt=prompt5)
    check(
        "(7) two events with an identical choices[0].prompt_token_ids list -> 'complete'",
        r7["status"] == REQUEST_STATUS_COMPLETE,
        str(r7["validation_errors"]),
    )

    # (8) two events with different choices[0].prompt_token_ids lists.
    events_8 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": [1, 2, 3, 4, 5]}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r8 = await _exec_full_raw(events_8, sent_prompt=prompt5)
    check(
        "(8) two events with different choices[0].prompt_token_ids lists -> not 'complete'",
        r8["status"] != REQUEST_STATUS_COMPLETE,
        str(r8["validation_errors"]),
    )

    # (9) the real vLLM 0.17.1 event shape at realistic 256/64 dimensions:
    # choice-level prompt ids on event 0 only, one output token per event,
    # a trailing usage-only event with empty choices, then [DONE].
    prompt256 = list(range(2000, 2256))
    output64 = list(range(3000, 3064))
    events_9: list[str] = []
    for i, tok in enumerate(output64):
        choice = {
            "index": 0, "text": "", "token_ids": [tok],
            "finish_reason": "length" if i == len(output64) - 1 else None,
        }
        choice["prompt_token_ids"] = prompt256 if i == 0 else None
        events_9.append(json.dumps({"choices": [choice]}))
    events_9.append(json.dumps({"choices": [], "usage": {"prompt_tokens": 256, "completion_tokens": 64}}))
    events_9.append("[DONE]")
    r9 = await _exec_full_raw(events_9, expected_prompt=256, expected_completion=64, sent_prompt=prompt256)
    check(
        "(9) the real vLLM 0.17.1 event shape (choice-level prompt ids on "
        "event 0, null afterward, trailing usage-only event) -> 'complete'",
        r9["status"] == REQUEST_STATUS_COMPLETE,
        str(r9["validation_errors"]),
    )

    # (10) prompt_token_ids_returned holds exactly the 256 server-reported ids.
    check(
        "(10) prompt_token_ids_returned holds exactly the 256 server-reported ids",
        r9["prompt_token_ids_returned"] == prompt256,
    )

    # (11) resume-depth validation accepts the resulting real-shaped record.
    ep_map = _build_fixture_episodes("llama", 900001)[0]
    tok_map = FakeTokenizerAdapter(vocab_size=5000, special_token_ids={0, 1, 2})
    valid_map = compute_valid_token_ids(tok_map)
    p_seed_map = victim_prompt_seed(ep_map, 0)
    g_seed_map = victim_generation_seed(ep_map, 0)
    prompt_map = generate_token_id_prompt(p_seed_map, valid_map, ep_map.victim_input_len)
    output_map = list(range(4000, 4000 + ep_map.victim_output_len))
    events_11: list[str] = []
    for i, tok in enumerate(output_map):
        choice = {
            "index": 0, "text": "", "token_ids": [tok],
            "finish_reason": "length" if i == len(output_map) - 1 else None,
        }
        choice["prompt_token_ids"] = prompt_map if i == 0 else None
        events_11.append(json.dumps({"choices": [choice]}))
    events_11.append(
        json.dumps(
            {"choices": [], "usage": {"prompt_tokens": ep_map.victim_input_len, "completion_tokens": ep_map.victim_output_len}}
        )
    )
    events_11.append("[DONE]")

    t_map11 = FakeTransport()
    t_map11.queue_script(
        f"{ep_map.episode_id}:victim:0",
        FakeStreamScript(
            prompt_token_ids_echo=None, token_events=[], include_done=False,
            extra_raw_events_before_finish=events_11,
        ),
    )
    r11 = await execute_completion_request(
        transport=t_map11, clock=RealClock(), url="http://x/v1/completions",
        api_key="k", model_full_id="m", prompt_token_ids=prompt_map,
        max_tokens=ep_map.victim_output_len, min_tokens=ep_map.victim_output_len, temperature=0.0,
        request_seed=g_seed_map, request_id=f"{ep_map.episode_id}:victim:0", role="victim", request_index=0,
        prompt_seed=p_seed_map, generation_seed=g_seed_map,
        expected_prompt_tokens=ep_map.victim_input_len, expected_completion_tokens=ep_map.victim_output_len,
        http_timeout_s=5.0,
    )
    depth_errors_11 = validate_complete_request_record(r11, episode=ep_map, role="victim", request_index=0)
    check(
        "(11) resume-depth validation accepts the resulting real-shaped "
        "complete request record",
        r11["status"] == REQUEST_STATUS_COMPLETE and depth_errors_11 == [],
        str((r11.get("status"), depth_errors_11)),
    )

    return results


def run_fake_block_integration_test() -> tuple[bool, list[str]]:
    """Section 26: a full simulated block run -- server start/readiness
    simulated, 20 stabilization requests, simulated cooldown, four
    complete regular episodes, simulated server stop -- with every JSON
    output validated. No sleeping, no GPU, no real network/server."""
    notes: list[str] = []
    ok = True

    def note(msg: str) -> None:
        notes.append(msg)

    try:
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture_seed = 999001
            # "llama" is reused deliberately so MODEL_FULL_ID resolves; this
            # is still an isolated, self-built fixture bundle in a fresh
            # temp dir, not the real official schedule.
            bundle, block_id = _make_fixture_block_bundle("llama", fixture_seed)
            episodes = bundle.episodes[:BLOCK_SIZE]

            tok = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
            transport = _make_success_transport()
            server_adapter = FakeServerProcessAdapter()
            sleeper = FakeSleeper()
            clock = RealClock()
            fake_api_key = "fake-integration-secret-9f8e7d"

            summary = asyncio.run(
                run_smoke_block(
                    bundle=bundle, block_id=block_id, output_dir=tmp_path / "out",
                    host="127.0.0.1", port=18200, resume=False, api_key=fake_api_key,
                    transport=transport, tokenizer=tok, server_adapter=server_adapter,
                    sleeper=sleeper, clock=clock, run_server_path=tmp_path / "run_server.sh",
                )
            )

            if summary.get("overall_status") != "block_complete":
                ok = False
                note(f"expected overall_status='block_complete', got {summary.get('overall_status')!r}")

            if len(server_adapter.started) != 1:
                ok = False
                note(f"expected exactly one simulated server start, got {len(server_adapter.started)}")
            elif not server_adapter.started[0].terminated:
                ok = False
                note("simulated server was never terminated by stop_server()")

            stab_path = stabilization_result_path(tmp_path / "out", block_id)
            if not stab_path.exists():
                ok = False
                note("stabilization output file is missing")
            else:
                stab_obj = json.loads(stab_path.read_text(encoding="utf-8"))
                if stab_obj.get("status") != REQUEST_STATUS_COMPLETE:
                    ok = False
                    note("stabilization status != complete")
                if len(stab_obj.get("request_results", [])) != STABILIZATION_REQUEST_COUNT:
                    ok = False
                    note("stabilization did not run exactly 20 requests")
                if stab_obj.get("record_type") != RECORD_TYPE_STABILIZATION:
                    ok = False
                    note("stabilization record_type is wrong")

            for ep in episodes:
                p = episode_result_path(tmp_path / "out", ep.episode_id)
                if not p.exists():
                    ok = False
                    note(f"episode output file missing for {ep.episode_id}")
                    continue
                obj = json.loads(p.read_text(encoding="utf-8"))
                if obj.get("status") != REQUEST_STATUS_COMPLETE:
                    ok = False
                    note(f"episode {ep.episode_id} status != complete")
                if obj.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong result_schema_version")
                if obj.get("runner_version") != RUNNER_VERSION:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong runner_version")
                if len(obj.get("victim_requests", [])) != ep.victim_request_count:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong victim_requests count")
                expected_burst = ep.burst_parallel_requests if ep.condition == "fixed_burst" else 0
                if len(obj.get("burst_requests", [])) != expected_burst:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong burst_requests count")
                if fake_api_key in json.dumps(obj):
                    ok = False
                    note(f"API key leaked into episode {ep.episode_id} result file")

            leftovers = list((tmp_path / "out").rglob("*.tmp.*"))
            if leftovers:
                ok = False
                note(f"atomic writer left temp file(s) behind: {leftovers}")

            if fake_api_key in json.dumps(summary):
                ok = False
                note("API key leaked into smoke_run_summary.json")

            if not (tmp_path / "out" / "smoke_run_summary.json").exists():
                ok = False
                note("smoke_run_summary.json was not written")

    except Exception as exc:  # noqa: BLE001 -- a failing integration test must report, not crash --self-test
        ok = False
        note(f"fake block integration test raised an unexpected exception: {exc!r}")

    return ok, notes


def run_self_test() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # --- derive_seed determinism ------------------------------------------
    a = derive_seed("20260711", "x")
    b = derive_seed("20260711", "x")
    c = derive_seed("20260711", "y")
    check("derive_seed is deterministic", a == b)
    check("derive_seed differs for different input", a != c)
    check(
        "derive_seed returns a non-negative int below 2**31-1",
        isinstance(a, int) and 0 <= a < 2**31 - 1,
    )

    # --- valid fixture is accepted ------------------------------------------
    fixture_seed = 12345
    good = _build_fixture_episodes("testmodel", fixture_seed)
    errors = _check_model_structure(
        "testmodel", good, fixture_seed,
        repeats=1, episodes_per_model=8, blocks_per_model=2,
        expected_state_sequence=["low", "high"],
    )
    check("valid fixture schedule is accepted", not errors, str(errors[:3]))

    # --- corrupted fixtures are rejected -------------------------------------
    def fixture_errors(mutate) -> list[str]:
        eps = _build_fixture_episodes("testmodel", fixture_seed)
        mutate(eps)
        return _check_model_structure(
            "testmodel", eps, fixture_seed,
            repeats=1, episodes_per_model=8, blocks_per_model=2,
            expected_state_sequence=["low", "high"],
        )

    check(
        "reordering two episodes within a block is rejected",
        bool(fixture_errors(lambda eps: eps.__setitem__(slice(0, 2), [eps[1], eps[0]]))),
    )
    check(
        "wrong repeat value is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "repeat", 2))),
    )
    check(
        "broken block contiguity (interleaving) is rejected",
        bool(fixture_errors(lambda eps: eps.__setitem__(1, eps.pop(5)))),
    )
    check(
        "wrong episode_seed is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "episode_seed", eps[0].episode_seed + 1))),
    )
    check(
        "wrong restart_server_before_block sequence is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "restart_server_before_block", 0))),
    )

    # --- fingerprint round-trip ----------------------------------------------
    payload = {"a": 1, "b": [1, 2, 3], "c": {"x": 1.5}}
    fp1 = recompute_fingerprint(payload)
    payload_with_fp = dict(payload)
    payload_with_fp["schedule_fingerprint"] = fp1
    fp2 = recompute_fingerprint(payload_with_fp)
    check("fingerprint recomputation ignores schedule_fingerprint key", fp1 == fp2)
    check("fingerprint has valid format", is_valid_fingerprint_format(fp1))
    tampered = dict(payload)
    tampered["a"] = 2
    fp3 = recompute_fingerprint(tampered)
    check("fingerprint changes when payload changes", fp1 != fp3)

    # --- csv/json consistency check -------------------------------------
    sample_json_row = {f: getattr(good[0], f) for f in EPISODE_FIELDS}
    sample_csv_row = {f: str(getattr(good[0], f)) for f in EPISODE_FIELDS}
    normalized, norm_errors = normalize_csv_row(sample_csv_row, 0)
    check("csv row normalizes back to matching types", not norm_errors)
    check(
        "normalized csv row equals json row",
        all(normalized[f] == sample_json_row[f] for f in EPISODE_FIELDS),
    )
    sample_csv_row_bad = dict(sample_csv_row)
    sample_csv_row_bad["victim_output_len"] = "999"
    normalized_bad, _ = normalize_csv_row(sample_csv_row_bad, 0)
    check(
        "csv/json mismatch is detectable after normalization",
        normalized_bad["victim_output_len"] != sample_json_row["victim_output_len"],
    )

    # --- strict type checking (bool must not pass as int) -------------------
    bad_bool_episode = dict(sample_json_row)
    bad_bool_episode["victim_request_count"] = True
    schema_errors = check_json_episode_schema(bad_bool_episode, 0)
    check(
        "bool is rejected where int is expected",
        any("victim_request_count" in e for e in schema_errors),
    )
    extra_field_episode = dict(sample_json_row)
    extra_field_episode["warmup_requests"] = 1
    schema_errors2 = check_json_episode_schema(extra_field_episode, 0)
    check(
        "unexpected extra field (e.g. warmup_requests) is rejected",
        any("warmup_requests" in e for e in schema_errors2),
    )
    missing_field_episode = dict(sample_json_row)
    del missing_field_episode["block_id"]
    schema_errors3 = check_json_episode_schema(missing_field_episode, 0)
    check(
        "missing field is rejected",
        any("block_id" in e for e in schema_errors3),
    )

    # --- result-file classification -----------------------------------------
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ep0 = good[0]
        ep1 = good[1]
        expected_fp = "sha256:" + "0" * 64

        _fixture_tok = FakeTokenizerAdapter(vocab_size=5000, special_token_ids={0, 1, 2})
        _fixture_valid_ids = compute_valid_token_ids(_fixture_tok)

        def make_valid_request_record(ep: Episode, role: str, index: int) -> dict:
            if role == "victim":
                p_seed = victim_prompt_seed(ep, index)
                g_seed = victim_generation_seed(ep, index)
                input_len, output_len = ep.victim_input_len, ep.victim_output_len
            else:
                p_seed = burst_prompt_seed(ep, index)
                g_seed = burst_generation_seed(ep, index)
                input_len, output_len = ep.burst_input_len, ep.burst_output_len
            prompt_ids = generate_token_id_prompt(p_seed, _fixture_valid_ids, input_len)
            output_ids = [7000 + k for k in range(output_len)]
            return {
                "request_id": f"{ep.episode_id}:{role}:{index}",
                "role": role,
                "request_index": index,
                "prompt_seed": p_seed,
                "generation_seed": g_seed,
                "prompt_token_ids_sent": prompt_ids,
                "prompt_token_ids_returned": list(prompt_ids),
                "prompt_sha256": prompt_sha256(prompt_ids),
                "expected_prompt_tokens": input_len,
                "expected_completion_tokens": output_len,
                "usage": {"prompt_tokens": input_len, "completion_tokens": output_len},
                "output_token_ids": output_ids,
                "output_text": "",
                "finish_reason": "length",
                "raw_sse_events": [],
                "done_received": True,
                "request_start_utc": "1970-01-01T00:00:00Z",
                "request_end_utc": "1970-01-01T00:00:01Z",
                "request_start_ns": 1000,
                "first_token_receive_ns": 1100,
                "last_token_receive_ns": 1200,
                "stream_end_ns": 1300,
                "ttft_ms": 0.1,
                "client_observed_tpot_ms": 0.01,
                "e2el_ms": 0.3,
                "itl_available": True,
                "itl_ms": [],
                "token_batch_sizes": None,
                "token_batch_interarrival_ms": None,
                "chunk_interarrival_ms": None,
                "http_status": 200,
                "timed_out": False,
                "cancelled": False,
                "error_type": None,
                "error_message": None,
                "validation_errors": [],
                "status": REQUEST_STATUS_COMPLETE,
            }

        def make_valid_result(ep: Episode) -> dict:
            burst_count = ep.burst_parallel_requests if ep.condition == "fixed_burst" else 0
            return {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "run_mode": "smoke",
                "schedule_fingerprint": expected_fp,
                "episode_id": ep.episode_id,
                "schedule_row": asdict(ep),
                "record_type": RECORD_TYPE_REGULAR_EPISODE,
                "status": "complete",
                "trigger": {
                    "status": "ok", "trigger_utc": "1970-01-01T00:00:00Z",
                    "trigger_perf_ns": 1000, "waited_ms": 0.0,
                },
                "burst_interval": (
                    {"start_ns": 1100, "end_ns": 1300} if ep.condition == "fixed_burst" else None
                ),
                "victim_requests": [
                    make_valid_request_record(ep, "victim", i) for i in range(ep.victim_request_count)
                ],
                "burst_requests": [
                    make_valid_request_record(ep, "burst", j) for j in range(burst_count)
                ],
            }

        valid_result = make_valid_result(ep0)

        missing_path = tmp_path / "missing.json"
        cls, _ = classify_result_file(missing_path, ep0, expected_fp, "smoke")
        check("missing result file classified as 'missing'", cls == CLASSIFICATION_MISSING)

        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps(valid_result), encoding="utf-8")
        cls, notes = classify_result_file(valid_path, ep0, expected_fp, "smoke")
        check(
            "well-formed complete result classified as 'valid_complete' (test 9)",
            cls == CLASSIFICATION_VALID_COMPLETE, str(notes),
        )

        corrupted_path = tmp_path / "corrupted.json"
        corrupted_path.write_text("{not valid json", encoding="utf-8")
        cls, _ = classify_result_file(corrupted_path, ep0, expected_fp, "smoke")
        check("malformed JSON classified as 'corrupted'", cls == CLASSIFICATION_CORRUPTED)

        partial_result = dict(valid_result)
        partial_result["status"] = "in_progress"
        partial_path = tmp_path / "partial.json"
        partial_path.write_text(json.dumps(partial_result), encoding="utf-8")
        cls, _ = classify_result_file(partial_path, ep0, expected_fp, "smoke")
        check("status != complete classified as 'partial'", cls == CLASSIFICATION_PARTIAL)

        bad_count_result = dict(valid_result)
        bad_count_result["victim_requests"] = [{}] * (ep0.victim_request_count - 1)
        bad_count_path = tmp_path / "badcount.json"
        bad_count_path.write_text(json.dumps(bad_count_result), encoding="utf-8")
        cls, _ = classify_result_file(bad_count_path, ep0, expected_fp, "smoke")
        check("wrong victim_requests count classified as 'invalid' (test 7)", cls == CLASSIFICATION_INVALID)

        missing_key_result = dict(valid_result)
        del missing_key_result["schedule_row"]
        mk_path = tmp_path / "missingkey.json"
        mk_path.write_text(json.dumps(missing_key_result), encoding="utf-8")
        cls, _ = classify_result_file(mk_path, ep0, expected_fp, "smoke")
        check("missing required result key classified as 'invalid'", cls == CLASSIFICATION_INVALID)

        # --- new resume-validation tests (this patch) -----------------------

        # Test 1: file at episode A's path contains a fully valid, complete
        # result -- but for episode B. Must be 'invalid' when validated
        # against the expected episode A, never silently accepted.
        wrong_episode_result = make_valid_result(ep1)
        wrong_episode_path = tmp_path / f"{ep0.episode_id}.json"
        wrong_episode_path.write_text(json.dumps(wrong_episode_result), encoding="utf-8")
        cls, notes = classify_result_file(wrong_episode_path, ep0, expected_fp, "smoke")
        check(
            "valid result for a different episode is rejected as 'invalid' "
            "when checked against the expected episode (test 1)",
            cls == CLASSIFICATION_INVALID, str(notes),
        )

        # Test 2: wrong runner_version.
        wrong_runner_result = dict(valid_result)
        wrong_runner_result["runner_version"] = "some-other-runner-9.9"
        wr_path = tmp_path / "wrongrunner.json"
        wr_path.write_text(json.dumps(wrong_runner_result), encoding="utf-8")
        cls, _ = classify_result_file(wr_path, ep0, expected_fp, "smoke")
        check("wrong runner_version classified as 'invalid' (test 2)", cls == CLASSIFICATION_INVALID)

        # Test 3: episode_id is a list -- must not crash, must be 'invalid'.
        list_episode_id_result = dict(valid_result)
        list_episode_id_result["episode_id"] = []
        lid_path = tmp_path / "listepid.json"
        lid_path.write_text(json.dumps(list_episode_id_result), encoding="utf-8")
        try:
            cls, _ = classify_result_file(lid_path, ep0, expected_fp, "smoke")
            crashed = False
        except Exception:
            crashed = True
            cls = None
        check(
            "episode_id as a list is classified as 'invalid' without crashing (test 3)",
            (not crashed) and cls == CLASSIFICATION_INVALID,
        )

        # Test 4: result_schema_version = true (bool) must NOT be accepted
        # as version 1, and must not crash.
        bool_version_result = dict(valid_result)
        bool_version_result["result_schema_version"] = True
        bv_path = tmp_path / "boolversion.json"
        bv_path.write_text(json.dumps(bool_version_result), encoding="utf-8")
        cls, _ = classify_result_file(bv_path, ep0, expected_fp, "smoke")
        check(
            "result_schema_version=true (bool) is rejected as 'invalid', "
            "not accepted as version 1 (test 4)",
            cls == CLASSIFICATION_INVALID,
        )

        # Test 5: result_schema_version = 1 as a real int is still accepted.
        real_int_version_result = dict(valid_result)
        real_int_version_result["result_schema_version"] = RESULT_SCHEMA_VERSION
        riv_path = tmp_path / "realintversion.json"
        riv_path.write_text(json.dumps(real_int_version_result), encoding="utf-8")
        cls, notes = classify_result_file(riv_path, ep0, expected_fp, "smoke")
        check(
            "result_schema_version as a real int stays accepted (test 5)",
            cls == CLASSIFICATION_VALID_COMPLETE, str(notes),
        )

        # Test 6: schedule_row is not a dict.
        list_row_result = dict(valid_result)
        list_row_result["schedule_row"] = ["not", "a", "dict"]
        lr_path = tmp_path / "listrow.json"
        lr_path.write_text(json.dumps(list_row_result), encoding="utf-8")
        cls, _ = classify_result_file(lr_path, ep0, expected_fp, "smoke")
        check("schedule_row as a non-dict is classified as 'invalid' (test 6)", cls == CLASSIFICATION_INVALID)

        # Test 7 (type-level, complementing the count-mismatch test above):
        # victim_requests is not an array at all.
        non_array_victim_result = dict(valid_result)
        non_array_victim_result["victim_requests"] = "not-a-list"
        nav_path = tmp_path / "nonarrayvictim.json"
        nav_path.write_text(json.dumps(non_array_victim_result), encoding="utf-8")
        cls, _ = classify_result_file(nav_path, ep0, expected_fp, "smoke")
        check("victim_requests as a non-array is classified as 'invalid' (test 7)", cls == CLASSIFICATION_INVALID)

        # Test 8: burst_requests is not an array at all.
        non_array_burst_result = dict(valid_result)
        non_array_burst_result["burst_requests"] = 42
        nab_path = tmp_path / "nonarrayburst.json"
        nab_path.write_text(json.dumps(non_array_burst_result), encoding="utf-8")
        cls, _ = classify_result_file(nab_path, ep0, expected_fp, "smoke")
        check("burst_requests as a non-array is classified as 'invalid' (test 8)", cls == CLASSIFICATION_INVALID)

        # Test 10: scan_existing_results() must not skip any schedule
        # episode just because a file with a foreign episode_id exists at
        # a different episode's expected path -- each expected filename is
        # validated strictly against its own specific episode.
        fake_bundle = LoadedBundle(
            schedule_dir=tmp_path,
            json_obj={},
            csv_fieldnames=[],
            csv_rows=[],
            audit_text="",
            episodes=[ep0, ep1],
            fingerprint=expected_fp,
        )
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir()
        (scan_dir / EPISODES_SUBDIR).mkdir()
        # ep0's own expected file actually holds ep1's (valid) result --
        # must show up as 'invalid' for ep0, and ep1 itself must still be
        # correctly reported as 'missing' (not silently matched/skipped).
        (scan_dir / EPISODES_SUBDIR / f"{ep0.episode_id}.json").write_text(
            json.dumps(make_valid_result(ep1)), encoding="utf-8"
        )
        classifications = scan_existing_results(scan_dir, fake_bundle, "smoke")
        check(
            "scan_existing_results: episode with a foreign result file is "
            "'invalid', not silently matched (test 10)",
            classifications.get(ep0.episode_id) == CLASSIFICATION_INVALID,
            str(classifications),
        )
        check(
            "scan_existing_results: the actual owner episode is still "
            "reported 'missing', not skipped (test 10)",
            classifications.get(ep1.episode_id) == CLASSIFICATION_MISSING,
            str(classifications),
        )

        # --- Stage-2 patch, section 4: deep per-request resume validation ---
        ep_burst = next(e for e in good if e.condition == "fixed_burst")

        def write_and_classify(mutated: dict, ep: Episode = ep0) -> tuple[str, list[str]]:
            p = tmp_path / f"depth_{len(list(tmp_path.glob('depth_*.json')))}.json"
            p.write_text(json.dumps(mutated), encoding="utf-8")
            return classify_result_file(p, ep, expected_fp, "smoke")

        # (4-1) empty request dicts are no longer accepted.
        empty_dicts_result = dict(make_valid_result(ep0))
        empty_dicts_result["victim_requests"] = [{}] * ep0.victim_request_count
        cls, notes = write_and_classify(empty_dicts_result)
        check("(4-1) empty request dicts -> invalid", cls == CLASSIFICATION_INVALID, str(notes[:2]))

        # (4-2) a request with status='incomplete' is rejected even though
        # every list has the right length.
        incomplete_result = dict(make_valid_result(ep0))
        incomplete_result["victim_requests"] = list(incomplete_result["victim_requests"])
        incomplete_result["victim_requests"][5] = dict(incomplete_result["victim_requests"][5])
        incomplete_result["victim_requests"][5]["status"] = "incomplete"
        cls, _ = write_and_classify(incomplete_result)
        check("(4-2) a request with status='incomplete' -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-3) a tampered prompt_sha256.
        bad_hash_result = dict(make_valid_result(ep0))
        bad_hash_result["victim_requests"] = list(bad_hash_result["victim_requests"])
        bad_hash_result["victim_requests"][0] = dict(bad_hash_result["victim_requests"][0])
        bad_hash_result["victim_requests"][0]["prompt_sha256"] = "0" * 64
        cls, _ = write_and_classify(bad_hash_result)
        check("(4-3) wrong prompt_sha256 -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-4) prompt_token_ids_sent != prompt_token_ids_returned.
        mismatched_prompt_result = dict(make_valid_result(ep0))
        mismatched_prompt_result["victim_requests"] = list(mismatched_prompt_result["victim_requests"])
        mismatched_prompt_result["victim_requests"][0] = dict(mismatched_prompt_result["victim_requests"][0])
        mismatched_prompt_result["victim_requests"][0]["prompt_token_ids_returned"] = [999] * ep0.victim_input_len
        cls, _ = write_and_classify(mismatched_prompt_result)
        check("(4-4) prompt_token_ids_sent != prompt_token_ids_returned -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-5) wrong usage counters.
        bad_usage_result = dict(make_valid_result(ep0))
        bad_usage_result["victim_requests"] = list(bad_usage_result["victim_requests"])
        bad_usage_result["victim_requests"][0] = dict(bad_usage_result["victim_requests"][0])
        bad_usage_result["victim_requests"][0]["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
        cls, _ = write_and_classify(bad_usage_result)
        check("(4-5) wrong usage counters -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-6) wrong output length.
        bad_outlen_result = dict(make_valid_result(ep0))
        bad_outlen_result["victim_requests"] = list(bad_outlen_result["victim_requests"])
        bad_outlen_result["victim_requests"][0] = dict(bad_outlen_result["victim_requests"][0])
        bad_outlen_result["victim_requests"][0]["output_token_ids"] = [1, 2, 3]
        cls, _ = write_and_classify(bad_outlen_result)
        check("(4-6) wrong output_token_ids length -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-7) wrong role/index on a request record.
        bad_role_result = dict(make_valid_result(ep0))
        bad_role_result["victim_requests"] = list(bad_role_result["victim_requests"])
        bad_role_result["victim_requests"][0] = dict(bad_role_result["victim_requests"][0])
        bad_role_result["victim_requests"][0]["role"] = "burst"
        cls, _ = write_and_classify(bad_role_result)
        check("(4-7a) wrong role -> invalid", cls == CLASSIFICATION_INVALID)

        bad_index_result = dict(make_valid_result(ep0))
        bad_index_result["victim_requests"] = list(bad_index_result["victim_requests"])
        bad_index_result["victim_requests"][0] = dict(bad_index_result["victim_requests"][0])
        bad_index_result["victim_requests"][0]["request_index"] = 17
        cls, _ = write_and_classify(bad_index_result)
        check("(4-7b) wrong request_index -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-8) duplicate index.
        dup_index_result = dict(make_valid_result(ep0))
        dup_index_result["victim_requests"] = list(dup_index_result["victim_requests"])
        dup_index_result["victim_requests"][1] = dict(dup_index_result["victim_requests"][0])
        cls, _ = write_and_classify(dup_index_result)
        check("(4-8) duplicate request_index -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-9) wrong deterministic seed.
        bad_seed_result = dict(make_valid_result(ep0))
        bad_seed_result["victim_requests"] = list(bad_seed_result["victim_requests"])
        bad_seed_result["victim_requests"][0] = dict(bad_seed_result["victim_requests"][0])
        bad_seed_result["victim_requests"][0]["prompt_seed"] = 424242424
        cls, _ = write_and_classify(bad_seed_result)
        check("(4-9) wrong deterministic prompt_seed -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-10) timestamps out of order.
        bad_order_result = dict(make_valid_result(ep0))
        bad_order_result["victim_requests"] = list(bad_order_result["victim_requests"])
        bad_order_result["victim_requests"][0] = dict(bad_order_result["victim_requests"][0])
        bad_order_result["victim_requests"][0]["last_token_receive_ns"] = 1
        cls, _ = write_and_classify(bad_order_result)
        check("(4-10) out-of-order request timestamps -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-11) a fully correct real Stage-2 schema instance is accepted,
        # including a fixed_burst episode with a real burst_interval.
        good_no_burst = write_and_classify(dict(make_valid_result(ep0)), ep0)
        check("(4-11a) fully correct no_burst episode -> valid_complete", good_no_burst[0] == CLASSIFICATION_VALID_COMPLETE, str(good_no_burst[1]))
        good_burst = write_and_classify(dict(make_valid_result(ep_burst)), ep_burst)
        check(
            "(4-11b) fully correct fixed_burst episode (4 burst requests + "
            "burst_interval) -> valid_complete",
            good_burst[0] == CLASSIFICATION_VALID_COMPLETE, str(good_burst[1]),
        )

        # trigger.status must be 'ok'; burst_interval must match condition.
        bad_trigger_result = dict(make_valid_result(ep0))
        bad_trigger_result["trigger"] = {"status": "timeout"}
        cls, _ = write_and_classify(bad_trigger_result)
        check("(4-extra) trigger.status != 'ok' -> invalid", cls == CLASSIFICATION_INVALID)

        no_burst_with_interval_result = dict(make_valid_result(ep0))
        no_burst_with_interval_result["burst_interval"] = {"start_ns": 1, "end_ns": 2}
        cls, _ = write_and_classify(no_burst_with_interval_result)
        check(
            "(4-extra) no_burst episode with a non-null burst_interval -> invalid",
            cls == CLASSIFICATION_INVALID,
        )

        fixed_burst_missing_interval_result = dict(make_valid_result(ep_burst))
        fixed_burst_missing_interval_result["burst_interval"] = None
        cls, _ = write_and_classify(fixed_burst_missing_interval_result, ep_burst)
        check(
            "(4-extra) fixed_burst episode with a null burst_interval -> invalid",
            cls == CLASSIFICATION_INVALID,
        )

        # --- output-dir marker conflict --------------------------------
        write_run_mode_marker(tmp_path, "official")
        conflict_raised = False
        try:
            check_output_dir_not_shared(tmp_path, "smoke")
        except OutputDirConflictError:
            conflict_raised = True
        check("output-dir marker conflict (official vs smoke) is rejected", conflict_raised)

        no_conflict_raised = False
        try:
            check_output_dir_not_shared(tmp_path, "official")
        except OutputDirConflictError:
            no_conflict_raised = True
        check("output-dir marker matching the same mode is accepted", not no_conflict_raised)

    # --- CLI mode mutual exclusivity / --resume guard -----------------------
    def parse_expect_systemexit(argv: list[str]) -> bool:
        try:
            parse_args(argv)
        except SystemExit:
            return True
        return False

    check("no mode flag is rejected", parse_expect_systemexit([]))
    check(
        "two mode flags together are rejected",
        parse_expect_systemexit(["--self-test", "--dry-run"]),
    )
    check(
        "--resume without --official-run/--smoke-test is rejected",
        parse_expect_systemexit(["--dry-run", "--resume"]),
    )
    check(
        "--resume with --official-run is accepted",
        not parse_expect_systemexit(["--official-run", "--resume"]),
    )

    # --- VLLM_API_KEY is read from the environment ONLY inside
    # read_api_key_from_env(), and that function is only ever invoked
    # from the real --smoke-test execution path -- never from
    # --self-test, --dry-run, or --official-run. (Stage 1 never read it
    # at all; Stage 2 legitimately needs it for the real smoke test, so
    # this check's scope narrows accordingly instead of disappearing.)
    own_source = SCRIPT_PATH.read_text(encoding="utf-8")
    try:
        read_api_key_source = inspect.getsource(read_api_key_from_env)
    except (OSError, TypeError):
        read_api_key_source = ""
    env_access_patterns = [
        r'os\.environ\.get\(\s*["\']VLLM_API_KEY',
        r'os\.environ\[\s*["\']VLLM_API_KEY',
        r'os\.getenv\(\s*["\']VLLM_API_KEY',
        r'\.get\(\s*["\']VLLM_API_KEY',
    ]
    source_without_that_function = own_source.replace(read_api_key_source, "", 1)
    found_env_access_elsewhere = any(
        re.search(p, source_without_that_function) for p in env_access_patterns
    )
    check(
        "VLLM_API_KEY is read from the environment only inside "
        "read_api_key_from_env(), nowhere else in this module",
        bool(read_api_key_source) and not found_env_access_elsewhere,
    )
    main_source = inspect.getsource(main)
    official_branch_source = main_source.split("if args.official_run:")[1].split(
        "assert args.smoke_test"
    )[0]
    check(
        "(35) --official-run never calls read_api_key_from_env(), never "
        "starts a server, and never runs a smoke block",
        "read_api_key_from_env(" not in official_branch_source
        and "run_smoke_block(" not in official_branch_source
        and "server_adapter" not in official_branch_source,
    )

    # --- 36/37/38: CLI validation for the new Stage-2 flags -----------------
    check(
        "(36) '--smoke-test' without '--smoke-block' is rejected",
        parse_expect_systemexit(["--smoke-test"]),
    )
    check(
        "(36b) '--smoke-block' without '--smoke-test' is rejected",
        parse_expect_systemexit(["--dry-run", "--smoke-block", "x"]),
    )
    fixture_bundle_for_block_check, _bid = _make_fixture_block_bundle("llama", 42)
    invalid_block_raised = False
    try:
        find_and_validate_smoke_block(fixture_bundle_for_block_check, "does_not_exist")
    except ValueError:
        invalid_block_raised = True
    check("(37) an invalid/unknown --smoke-block is rejected", invalid_block_raised)
    check(
        "(38) --port outside 1-65535 is rejected",
        parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "0"])
        and parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "70000"]),
    )
    check(
        "(38b) --port within 1-65535 is accepted",
        not parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "8000"]),
    )

    # --- Stage 2: async request/trigger/stabilization/smoke-block checks ---
    results.extend(asyncio.run(_stage2_async_checks()))

    # --- Section 26: fake full-block integration test (no sleep, no GPU) ---
    fake_block_ok, fake_block_notes = run_fake_block_integration_test()
    check(
        "fake full-block integration test (simulated server, stabilization "
        "+ 4 episodes, all JSON outputs validated)",
        fake_block_ok, "; ".join(fake_block_notes),
    )

    # --- summary --------------------------------------------------------
    print("Self-test results")
    print("=" * 60)
    all_passed = True
    for name, passed, detail in results:
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        line = f"[{status}] {name}"
        if detail and not passed:
            line += f" -- {detail}"
        print(line)
    print("=" * 60)
    print(f"{sum(1 for _, p, _ in results if p)}/{len(results)} checks passed")
    print("SELF-TEST: PASS" if all_passed else "SELF-TEST: FAIL")
    return 0 if all_passed else 1


# ============================================================================
# main
# ============================================================================

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

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
        mode = "official"
        output_dir = (
            args.output_dir if args.output_dir is not None else default_output_dir(mode)
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            check_output_dir_not_shared(output_dir, mode)
            write_run_mode_marker(output_dir, mode)
        except OutputDirConflictError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(
                f"ERROR: could not prepare output-dir '{output_dir}': {exc}",
                file=sys.stderr,
            )
            return 1

        print(f"schedule_fingerprint: {bundle.fingerprint}")
        print(f"run_mode: {mode}")
        print(f"output_dir: {output_dir}")

        if args.resume:
            classifications = scan_existing_results(output_dir, bundle, mode)
            counts: dict[str, int] = {}
            for c in classifications.values():
                counts[c] = counts.get(c, 0) + 1
            print()
            print("Resume scan (file-level depth validation):")
            for key in (
                CLASSIFICATION_MISSING,
                CLASSIFICATION_VALID_COMPLETE,
                CLASSIFICATION_PARTIAL,
                CLASSIFICATION_INVALID,
                CLASSIFICATION_CORRUPTED,
            ):
                print(f"  {key}: {counts.get(key, 0)}")

        # Section 24: --official-run stays fully blocked in stage 2. No
        # server is started and no result file is written above this
        # line -- output-dir preparation and the read-only resume scan
        # are the only side effects, exactly as in stage 1.
        print()
        print("Official execution is disabled until Stage 3.", file=sys.stderr)
        return 1

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
