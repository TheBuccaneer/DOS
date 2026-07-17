#!/usr/bin/env python3
"""
run_prefill_trigger_sweep.py -- frozen exploratory Prefill-Trigger-Sweep runner.

Generalizes the frozen Prefill-Screen's fixed "first token of every
active-wave request" barrier into a configurable
`trigger_after_decode_tokens` barrier (1 / 16 / 32 actually-received
output tokens per active-wave request), to test whether the selective
active-wave stall occurs only immediately after decode starts, or also
later in decode. Two fully independent, eingefrorene per-model
campaigns (llama, qwen), each:
  - states: cpu_offload_gb 0 (low) and 12 (high)
  - victim concurrency: 4 (fixed; the active wave is request_index 0..3)
  - trigger positions: 1, 16, 32 received output tokens
  - conditions: no_burst and prefill_burst
  - repeats: 2
  - prefill burst: 4 parallel requests, 2048 input tokens, 16 output tokens
  - chunked prefill budget: max_num_batched_tokens = 2048
  => 24 regular episodes / 12 blocks per model.

Each of the twelve per-model state/trigger blocks follows the same
hardened protocol mechanically inherited from the frozen Prefill-Screen
infrastructure: server start, readiness validation, one excluded
stabilization run, health gate, fixed cooldown, two regular episodes
(no_burst + prefill_burst) in schedule order, atomic outputs, and
verified process-group shutdown with port-release polling. No
simplified rewrite of that lifecycle/resume/integrity logic was made.

The generator is not imported at runtime. It implements its own,
independent canonical-fingerprint computation. The operative bundle
for a given --model-key consists of exactly:
  prefill_trigger_sweep_schedule.json
  prefill_trigger_sweep_schedule.csv
  prefill_trigger_sweep_schedule_audit.txt
one such bundle per model, in separate directories
(new/runs/prefill_trigger_sweep/<model_key>/). --model-key selects
ONLY the matching bundle/model registry entry; it can never override
the model recorded inside a foreign bundle -- see
`check_official_contract`.

CLI modes:
  --self-test     Run focused no-GPU/no-network contract and lifecycle tests.
  --dry-run       Validate the complete bundle and print the execution plan.
  --smoke-test    Execute exactly one selected two-episode block.
  --official-run  Execute all twelve blocks / 24 regular episodes.

Examples:
  python3 run_prefill_trigger_sweep.py --self-test
  python3 run_prefill_trigger_sweep.py --model-key llama --dry-run \
      --schedule-dir /path/to/new/runs/prefill_trigger_sweep/llama
  python3 run_prefill_trigger_sweep.py --model-key qwen --dry-run \
      --schedule-dir /path/to/new/runs/prefill_trigger_sweep/qwen
  VLLM_API_KEY=... GPU_DEVICE=0 python3 run_prefill_trigger_sweep.py \
      --model-key llama --official-run \
      --schedule-dir /path/to/new/runs/prefill_trigger_sweep/llama \
      --output-dir /path/to/new/runs/prefill_trigger_sweep/llama/results/official
  VLLM_API_KEY=... GPU_DEVICE=0 python3 run_prefill_trigger_sweep.py \
      --model-key llama --smoke-test \
      --smoke-block llama_block01_low_trigger1 \
      --schedule-dir /path/to/new/runs/prefill_trigger_sweep/llama \
      --output-dir /path/to/new/runs/prefill_trigger_sweep/llama/results/smoke

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
from datetime import datetime, timedelta, timezone
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
# Frozen Schema v2 episode schema (26 fields, exact set in both directions)
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


# ============================================================================
# Frozen official Prefill-Trigger-Sweep contract
#
# This is a fully separate, explorative screening design -- derived
# mechanically from the frozen Phase A contract, but with its own
# schedule seed, model set, repeat count, burst shape, and fingerprint.
# It does not read, write, or share any file with new/runs/phase_a/.
# ============================================================================

SCHEMA_VERSION = 2
DESIGN_VERSION = "prefill-trigger-sweep-v1"

# Model registry: the ONLY place model-dependent values live. Both
# models share exactly one runner code path; --model-key only selects
# which entry (and which frozen bundle/fingerprint) applies.
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

# Per-model frozen fingerprints. Filled in only after each model's
# schedule was independently generated and its schedule_fingerprint
# re-verified (Section 5/15 of the project prompt). A placeholder of
# None means "not yet frozen for real execution" and MUST cause a hard
# failure in check_official_contract if that model is actually used
# outside --self-test/--dry-run.
OFFICIAL_FINGERPRINTS: dict[str, str | None] = {
    "llama": "sha256:c1465aaf29bb35d483f7abd6a5d0dbfd76390c3e69c011eadb587e0efd7f1a2a",
    "qwen": "sha256:20159b3c519c8589fe4fa4b824706833711744574f0befb87b98e12d2e8b9550",
}

OFFICIAL_SEED_BY_MODEL: dict[str, int] = {
    "llama": 20260717,
    "qwen": 20260717,
}
OFFICIAL_EPISODE_COUNT_PER_MODEL = 24
OFFICIAL_BLOCK_COUNT_PER_MODEL = 12
OFFICIAL_TRIGGER_POSITIONS = [1, 16, 32]
OFFICIAL_CONCURRENCY = 4
OFFICIAL_MAX_NUM_BATCHED_TOKENS = 2048
OFFICIAL_REPEATS = 2

# Prefill-heavy, bounded burst condition, mechanically inherited from
# Prefill-Screen. Kept as a single named constant so every burst-shape/
# -count/-interval check below derives from one place.
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
OFFICIAL_BURST_CONFIGURATION = {
    "burst_parallel_requests": 4,
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
}

# Frozen trigger rotation by repeat (Section 8 of the project prompt).
TRIGGER_ROTATION_BY_REPEAT: dict[int, list[int]] = {
    1: [1, 16, 32],
    2: [16, 32, 1],
}
# 12 blocks total per model: 2 repeats x 3 triggers x 2 states, in that
# rotation order (state low precedes high within each trigger).
EXPECTED_STATE_TRIGGER_SEQUENCE: list[tuple[str, int]] = [
    (state_label, trigger)
    for repeat in (1, 2)
    for trigger in TRIGGER_ROTATION_BY_REPEAT[repeat]
    for state_label in ("low", "high")
]

BLOCK_SIZE = len(OFFICIAL_CONDITIONS)  # 2 (one no_burst + one prefill_burst)
BLOCKS_PER_MODEL = OFFICIAL_BLOCK_COUNT_PER_MODEL  # 12
EPISODES_PER_MODEL = BLOCK_SIZE * BLOCKS_PER_MODEL  # 24
TOTAL_RESTART_MARKERS_PER_MODEL = BLOCKS_PER_MODEL  # 12

REQUIRED_BUNDLE_FILENAMES = (
    "prefill_trigger_sweep_schedule.json",
    "prefill_trigger_sweep_schedule.csv",
    "prefill_trigger_sweep_schedule_audit.txt",
)

RUN_MODE_MARKER_FILENAME = ".prefill_trigger_sweep_run_mode"


# ============================================================================
# Portable paths (derived from this file's own location, not hardcoded)
# Expected location: <PROJECT_ROOT>/new/scripts/prefill_trigger_sweep/run_prefill_trigger_sweep.py
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SCHEDULE_DIR_ROOT = PROJECT_ROOT / "new" / "runs" / "prefill_trigger_sweep"


def default_schedule_dir(model_key: str) -> Path:
    return DEFAULT_SCHEDULE_DIR_ROOT / model_key


def default_output_dir(model_key: str, mode: str) -> Path:
    """Return a model-separated result directory.

    Llama and Qwen are independent frozen campaigns and must never share
    the same implicit output directory.
    """
    return (
        PROJECT_ROOT
        / "new"
        / "runs"
        / "prefill_trigger_sweep"
        / model_key
        / "results"
        / mode
    )


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

    if len(csv_rows) != OFFICIAL_EPISODE_COUNT_PER_MODEL:
        errors.append(
            f"csv has {len(csv_rows)} row(s), expected exactly "
            f"{OFFICIAL_EPISODE_COUNT_PER_MODEL}"
        )
    if len(json_episodes) != OFFICIAL_EPISODE_COUNT_PER_MODEL:
        errors.append(
            f"json has {len(json_episodes)} episode(s), expected exactly "
            f"{OFFICIAL_EPISODE_COUNT_PER_MODEL}"
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
        r"^design_version:\s*prefill-trigger-sweep-v1\s*$", audit_text, re.MULTILINE
    ):
        errors.append(
            "audit report does not contain 'design_version: prefill-trigger-sweep-v1'"
        )
    if not re.search(r"^total episodes:\s*24\s*$", audit_text, re.MULTILINE):
        errors.append("audit report does not contain 'total episodes: 24'")
    if not re.search(r"^OVERALL:\s*PASS\s*$", audit_text, re.MULTILINE):
        errors.append("audit report does not contain 'OVERALL: PASS'")

    return errors


# ============================================================================
# Official top-level contract check
#
# A bundle belongs to exactly one model. --model-key selects which
# MODEL_REGISTRY/OFFICIAL_FINGERPRINTS entry this bundle is checked
# against -- it can never make a foreign bundle (e.g. a qwen bundle
# loaded while --model-key=llama) pass, because model_key/model_id are
# checked against the CLI-selected model, not merely against
# "some registered model".
# ============================================================================

def check_official_contract(json_obj: dict, model_key: str) -> list[str]:
    errors: list[str] = []

    if model_key not in MODEL_REGISTRY:
        return [f"unknown --model-key {model_key!r}; known: {sorted(MODEL_REGISTRY)}"]

    expected_model_id = MODEL_REGISTRY[model_key]["model_id"]
    expected_seed = OFFICIAL_SEED_BY_MODEL[model_key]
    expected_fp = OFFICIAL_FINGERPRINTS.get(model_key)

    def check_eq(key: str, expected: object) -> None:
        actual = json_obj.get(key, "<MISSING>")
        if actual != expected:
            errors.append(f"{key} = {actual!r}, expected {expected!r}")

    check_eq("schema_version", SCHEMA_VERSION)
    check_eq("design_version", DESIGN_VERSION)
    check_eq("seed", expected_seed)
    check_eq("repeats", OFFICIAL_REPEATS)
    check_eq("model_key", model_key)
    check_eq("model_id", expected_model_id)
    check_eq("concurrency", OFFICIAL_CONCURRENCY)
    check_eq("trigger_positions", OFFICIAL_TRIGGER_POSITIONS)
    check_eq("conditions", OFFICIAL_CONDITIONS)
    check_eq("states", OFFICIAL_STATES)
    check_eq("max_num_batched_tokens", OFFICIAL_MAX_NUM_BATCHED_TOKENS)
    check_eq("victim_configuration", OFFICIAL_VICTIM_CONFIGURATION)
    check_eq("burst_configuration", OFFICIAL_BURST_CONFIGURATION)
    check_eq("stabilization_configuration", OFFICIAL_STABILIZATION_CONFIGURATION)
    check_eq("episode_count", OFFICIAL_EPISODE_COUNT_PER_MODEL)

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
        if expected_fp is None:
            errors.append(
                f"model_key={model_key!r} has no frozen OFFICIAL_FINGERPRINTS "
                f"entry yet; this bundle cannot be used for --official-run or "
                f"--smoke-test"
            )
        elif fp != expected_fp:
            errors.append(
                f"schedule_fingerprint {fp!r} does not match the frozen "
                f"Prefill-Trigger-Sweep fingerprint for model_key={model_key!r} "
                f"({expected_fp!r})"
            )

    return errors


# ============================================================================
# Structural schedule validation (independent of the fingerprint).
#
# Unlike Prefill-Screen (which merged multiple models into one
# schedule), each Prefill-Trigger-Sweep bundle belongs to exactly one
# model -- so structural validation operates on a single, homogeneous
# episode list.
# ============================================================================

def check_structural_schedule(
    episodes: list[Episode], schedule_seed: object, model_key: str
) -> list[str]:
    errors: list[str] = []
    ctx_model = f"model_key={model_key}"

    if model_key not in MODEL_REGISTRY:
        return [f"unknown --model-key {model_key!r}"]
    expected_model_id = MODEL_REGISTRY[model_key]["model_id"]

    if len(episodes) != OFFICIAL_EPISODE_COUNT_PER_MODEL:
        errors.append(
            f"{ctx_model}: expected {OFFICIAL_EPISODE_COUNT_PER_MODEL} episodes, "
            f"found {len(episodes)}"
        )

    all_episode_ids = [ep.episode_id for ep in episodes]
    if len(all_episode_ids) != len(set(all_episode_ids)):
        dup = sorted({e for e in all_episode_ids if all_episode_ids.count(e) > 1})
        errors.append(f"{ctx_model}: duplicate episode_id(s): {dup}")
    all_episode_seeds = [ep.episode_seed for ep in episodes]
    if len(all_episode_seeds) != len(set(all_episode_seeds)):
        errors.append(f"{ctx_model}: duplicate episode_seed(s) found")

    # --- per-episode exact value + deterministic re-derivation checks -----
    for ep in episodes:
        ctx = f"{ctx_model}, episode={ep.episode_id}"

        if ep.model_key != model_key:
            errors.append(f"{ctx}: episode.model_key {ep.model_key!r} != {model_key!r}")
        if ep.model_id != expected_model_id:
            errors.append(f"{ctx}: episode.model_id {ep.model_id!r} != {expected_model_id!r}")

        expected_state_label = STATE_LABEL_BY_OFFLOAD.get(ep.offload_gb)
        if expected_state_label is None:
            errors.append(f"{ctx}: invalid offload_gb {ep.offload_gb!r}")
        elif ep.state_label != expected_state_label:
            errors.append(
                f"{ctx}: state_label {ep.state_label!r} != expected "
                f"{expected_state_label!r} for offload_gb {ep.offload_gb}"
            )

        if ep.concurrency != OFFICIAL_CONCURRENCY:
            errors.append(f"{ctx}: concurrency {ep.concurrency!r} != {OFFICIAL_CONCURRENCY}")
        if ep.trigger_after_decode_tokens not in OFFICIAL_TRIGGER_POSITIONS:
            errors.append(
                f"{ctx}: trigger_after_decode_tokens {ep.trigger_after_decode_tokens!r} "
                f"not in {OFFICIAL_TRIGGER_POSITIONS}"
            )
        if ep.condition not in OFFICIAL_CONDITIONS:
            errors.append(f"{ctx}: invalid condition {ep.condition!r}")
        if ep.max_num_batched_tokens != OFFICIAL_MAX_NUM_BATCHED_TOKENS:
            errors.append(f"{ctx}: max_num_batched_tokens != {OFFICIAL_MAX_NUM_BATCHED_TOKENS}")
        if ep.random_seed != schedule_seed:
            errors.append(f"{ctx}: random_seed {ep.random_seed!r} != schedule seed {schedule_seed!r}")

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
            errors.append(f"{ctx}: burst_input_len != {OFFICIAL_BURST_CONFIGURATION['burst_input_len']}")
        if ep.burst_output_len != OFFICIAL_BURST_CONFIGURATION["burst_output_len"]:
            errors.append(f"{ctx}: burst_output_len != {OFFICIAL_BURST_CONFIGURATION['burst_output_len']}")
        if ep.burst_temperature != OFFICIAL_BURST_CONFIGURATION["burst_temperature"]:
            errors.append(f"{ctx}: burst_temperature != 0.0")

        if ep.order_in_block == 1:
            if ep.restart_server_before_block != 1:
                errors.append(f"{ctx}: order_in_block=1 requires restart_server_before_block==1")
        else:
            if ep.restart_server_before_block != 0:
                errors.append(f"{ctx}: order_in_block={ep.order_in_block} requires restart_server_before_block==0")

        expected_episode_id = (
            f"{model_key}_off{ep.offload_gb}_conc{OFFICIAL_CONCURRENCY}_"
            f"trigger{ep.trigger_after_decode_tokens}_{ep.condition}_rep{ep.repeat}"
        )
        if ep.episode_id != expected_episode_id:
            errors.append(f"{ctx}: episode_id != expected derivation {expected_episode_id!r}")

        expected_episode_seed = derive_seed(str(schedule_seed), ep.episode_id)
        if ep.episode_seed != expected_episode_seed:
            errors.append(f"{ctx}: episode_seed does not match derive_seed(seed, episode_id)")

        expected_victim_seed = derive_seed(str(schedule_seed), model_key, "victim", str(ep.repeat))
        if ep.victim_workload_seed != expected_victim_seed:
            errors.append(f"{ctx}: victim_workload_seed does not match the expected derivation")

        expected_burst_seed = derive_seed(str(schedule_seed), model_key, "burst", str(ep.repeat))
        if ep.burst_workload_seed != expected_burst_seed:
            errors.append(f"{ctx}: burst_workload_seed does not match the expected derivation")

        if ep.victim_workload_seed == ep.burst_workload_seed:
            errors.append(f"{ctx}: episode {ep.episode_id} has identical victim/burst workload seeds")

    # --- exact repeat set per (offload, trigger, condition) cell ------------
    repeats_by_cell: dict[tuple[int, int, str], set[int]] = {}
    for ep in episodes:
        key = (ep.offload_gb, ep.trigger_after_decode_tokens, ep.condition)
        repeats_by_cell.setdefault(key, set()).add(ep.repeat)
    expected_repeat_set = set(range(1, OFFICIAL_REPEATS + 1))
    expected_cells = {
        (offload_gb, trigger, condition)
        for offload_gb in STATE_LABEL_BY_OFFLOAD
        for trigger in OFFICIAL_TRIGGER_POSITIONS
        for condition in OFFICIAL_CONDITIONS
    }
    for key in sorted(expected_cells):
        actual = repeats_by_cell.get(key, set())
        if actual != expected_repeat_set:
            errors.append(
                f"{ctx_model}: cell {key} has repeat value(s) {sorted(actual)}, "
                f"expected exactly {sorted(expected_repeat_set)}"
            )

    # --- contiguous block / execution-order checks + block ID/state/
    #     trigger rotation --------------------------------------------------
    blocks: dict[str, list[Episode]] = {}
    block_ids_in_order: list[str] = []
    for ep in episodes:
        if ep.block_id not in blocks:
            blocks[ep.block_id] = []
            block_ids_in_order.append(ep.block_id)
        blocks[ep.block_id].append(ep)

    seen_block_ids: set[str] = set()
    idx = 0
    n = len(episodes)
    block_position = 0
    while idx < n:
        bid = episodes[idx].block_id
        if bid in seen_block_ids:
            errors.append(f"{ctx_model}: block_id {bid!r} reappears at a non-contiguous position")
        seen_block_ids.add(bid)
        run_end = idx
        while run_end < n and episodes[run_end].block_id == bid:
            run_end += 1
        run_episodes = episodes[idx:run_end]
        block_position += 1

        if len(run_episodes) != BLOCK_SIZE:
            errors.append(
                f"{ctx_model}: contiguous block {bid!r} has {len(run_episodes)} "
                f"immediately consecutive episode(s), expected exactly {BLOCK_SIZE}"
            )
        else:
            order_sequence = [ep.order_in_block for ep in run_episodes]
            if order_sequence != [1, 2]:
                errors.append(f"{ctx_model}: block {bid!r} order_in_block sequence {order_sequence} != [1, 2]")
            restart_sequence = [ep.restart_server_before_block for ep in run_episodes]
            if restart_sequence != [1, 0]:
                errors.append(f"{ctx_model}: block {bid!r} restart sequence {restart_sequence} != [1, 0]")

            conditions_here = sorted(ep.condition for ep in run_episodes)
            if conditions_here != sorted(OFFICIAL_CONDITIONS):
                errors.append(f"{ctx_model}: block {bid!r} conditions {conditions_here} != {sorted(OFFICIAL_CONDITIONS)}")

            state_labels = {ep.state_label for ep in run_episodes}
            offloads = {ep.offload_gb for ep in run_episodes}
            triggers = {ep.trigger_after_decode_tokens for ep in run_episodes}
            repeats_in_block = {ep.repeat for ep in run_episodes}
            if len(state_labels) != 1 or len(offloads) != 1:
                errors.append(f"{ctx_model}: block {bid!r} mixes state/offload values")
            if len(triggers) != 1:
                errors.append(f"{ctx_model}: block {bid!r} mixes trigger_after_decode_tokens values")
            if len(repeats_in_block) != 1:
                errors.append(f"{ctx_model}: block {bid!r} mixes repeat values")

            if block_position <= len(EXPECTED_STATE_TRIGGER_SEQUENCE) and len(state_labels) == 1 and len(triggers) == 1:
                expected_state, expected_trigger = EXPECTED_STATE_TRIGGER_SEQUENCE[block_position - 1]
                actual_state = next(iter(state_labels))
                actual_trigger = next(iter(triggers))
                if (actual_state, actual_trigger) != (expected_state, expected_trigger):
                    errors.append(
                        f"{ctx_model}: block at position {block_position} has "
                        f"(state, trigger)=({actual_state!r}, {actual_trigger}), expected "
                        f"({expected_state!r}, {expected_trigger})"
                    )
                expected_block_id = f"{model_key}_block{block_position:02d}_{expected_state}_trigger{expected_trigger}"
                if bid != expected_block_id:
                    errors.append(
                        f"{ctx_model}: block at position {block_position} has block_id "
                        f"{bid!r}, expected {expected_block_id!r}"
                    )
        idx = run_end

    if block_position != OFFICIAL_BLOCK_COUNT_PER_MODEL:
        errors.append(f"{ctx_model}: expected {OFFICIAL_BLOCK_COUNT_PER_MODEL} blocks, found {block_position}")

    total_restart_markers = sum(1 for ep in episodes if ep.restart_server_before_block == 1)
    if total_restart_markers != TOTAL_RESTART_MARKERS_PER_MODEL:
        errors.append(
            f"{ctx_model}: expected {TOTAL_RESTART_MARKERS_PER_MODEL} restart markers, "
            f"found {total_restart_markers}"
        )

    # --- condition-first balance: 6/6 global, 1/1 per (state, trigger) -----
    first_conditions_by_block = {bid: blocks[bid][0].condition for bid in block_ids_in_order}
    no_burst_first_count = sum(1 for c in first_conditions_by_block.values() if c == "no_burst")
    burst_first_count = sum(1 for c in first_conditions_by_block.values() if c == BURST_CONDITION)
    if no_burst_first_count != 6:
        errors.append(f"{ctx_model}: expected exactly 6 no_burst-first blocks, found {no_burst_first_count}")
    if burst_first_count != 6:
        errors.append(f"{ctx_model}: expected exactly 6 prefill_burst-first blocks, found {burst_first_count}")

    cell_first: dict[tuple[str, int], list[str]] = {}
    for bid in block_ids_in_order:
        eps = blocks[bid]
        key = (eps[0].state_label, eps[0].trigger_after_decode_tokens)
        cell_first.setdefault(key, []).append(eps[0].condition)
    for key, firsts in sorted(cell_first.items()):
        if sorted(firsts) != sorted(["no_burst", BURST_CONDITION]):
            errors.append(
                f"{ctx_model}: state/trigger cell {key} does not have exactly one "
                f"no_burst-first and one prefill_burst-first repeat: {firsts}"
            )
        first_by_repeat: dict[int, str] = {}
        for bid in block_ids_in_order:
            first_ep = blocks[bid][0]
            if (first_ep.state_label, first_ep.trigger_after_decode_tokens) != key:
                continue
            if first_ep.repeat in first_by_repeat:
                errors.append(
                    f"{ctx_model}: duplicate block for state/trigger/repeat "
                    f"{key + (first_ep.repeat,)}"
                )
            else:
                first_by_repeat[first_ep.repeat] = first_ep.condition
        for ep in [
            e for bid in block_ids_in_order for e in blocks[bid]
            if (e.state_label, e.trigger_after_decode_tokens) == key
        ]:
            expected_cf = first_by_repeat.get(ep.repeat)
            if expected_cf is None:
                errors.append(
                    f"{ctx_model}: no block-first condition found for episode "
                    f"{ep.episode_id} (repeat={ep.repeat})"
                )
            elif ep.condition_first_in_block != expected_cf:
                errors.append(
                    f"{ctx_model}: episode {ep.episode_id} "
                    f"condition_first_in_block != actual block-first condition"
                )

    # --- workload seed constancy within repeat (across state/trigger/
    #     condition) + victim/burst independence between repeats -----------
    victim_seeds_by_repeat: dict[int, set[int]] = {}
    burst_seeds_by_repeat: dict[int, set[int]] = {}
    for ep in episodes:
        victim_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.victim_workload_seed)
        burst_seeds_by_repeat.setdefault(ep.repeat, set()).add(ep.burst_workload_seed)
    for repeat, seeds in victim_seeds_by_repeat.items():
        if len(seeds) != 1:
            errors.append(f"{ctx_model}: victim_workload_seed not constant within repeat={repeat}")
    for repeat, seeds in burst_seeds_by_repeat.items():
        if len(seeds) != 1:
            errors.append(f"{ctx_model}: burst_workload_seed not constant within repeat={repeat}")

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


def load_and_validate_bundle(
    schedule_dir: Path, model_key: str
) -> tuple[LoadedBundle | None, list[str]]:
    if model_key not in MODEL_REGISTRY:
        return None, [f"unknown --model-key {model_key!r}; known: {sorted(MODEL_REGISTRY)}"]

    try:
        paths = find_bundle_paths(schedule_dir)
        json_obj = load_json_bundle(paths["prefill_trigger_sweep_schedule.json"])
        csv_fieldnames, csv_rows = load_csv_bundle(paths["prefill_trigger_sweep_schedule.csv"])
        audit_text = load_audit_text(paths["prefill_trigger_sweep_schedule_audit.txt"])
    except BundleLoadError as exc:
        return None, [str(exc)]

    errors: list[str] = []
    # --model-key/CLI value alone decides which contract this bundle is
    # checked against; a foreign bundle's own json_obj['model_key'] can
    # never override that -- see check_official_contract.
    errors.extend(check_official_contract(json_obj, model_key))

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

    errors.extend(check_structural_schedule(episodes, json_obj.get("seed"), model_key))

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
                "model_key": first.model_key,
                "model_id": first.model_id,
                "offload_gb": first.offload_gb,
                "state_label": first.state_label,
                "trigger_after_decode_tokens": first.trigger_after_decode_tokens,
                "repeat": first.repeat,
                "condition_first_in_block": first.condition_first_in_block,
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
            f"  model_key: {block['model_key']}, model_id: {block['model_id']}, "
            f"offload_gb: {block['offload_gb']}, state: {block['state_label']}, "
            f"trigger_after_decode_tokens: {block['trigger_after_decode_tokens']}, "
            f"repeat: {block['repeat']}, condition_first: {block['condition_first_in_block']}"
        )
        for ep in block["episodes"]:
            print(
                f"    order {ep.order_in_block}: {ep.episode_id} "
                f"(concurrency={ep.concurrency}, condition={ep.condition}, "
                f"fingerprint_field trigger={ep.trigger_after_decode_tokens})"
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

RESULT_SCHEMA_VERSION = 5
RUNNER_VERSION = "run_prefill_trigger_sweep-v2-timing1"

# Section: task/semaphore/dispatch/wave/trigger-exposure timing
# instrumentation (additive; observational only -- see
# TIMING_INSTRUMENTATION_NAME docstring below for scope).
TIMING_INSTRUMENTATION_VERSION = 2
TIMING_INSTRUMENTATION_NAME = "task_semaphore_dispatch_wave_v2"

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
    "trigger",
    "burst_interval",
    "timing_instrumentation_version",
    "timing_instrumentation_name",
    "victim_phase_start_ns",
    "queue_timing_summary",
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
    "trigger": dict,
    "timing_instrumentation_version": int,
    "timing_instrumentation_name": str,
    "victim_phase_start_ns": int,
    "queue_timing_summary": dict,
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

    # The raw normalized SSE trace is the scientific source of truth for
    # token timing.  Resume validation therefore reconstructs the emitted
    # output stream instead of trusting only the pre-computed aggregates.
    raw_events = record.get("raw_sse_events", "<MISSING>")
    reconstructed_output: list[int] = []
    positive_event_ns: list[int] = []
    if not isinstance(raw_events, list) or not raw_events:
        errors.append(f"{role}[{request_index}]: raw_sse_events is missing, not a list, or empty")
    else:
        previous_receive_ns: int | None = None
        for event_pos, event in enumerate(raw_events):
            if not isinstance(event, dict):
                errors.append(f"{role}[{request_index}]: raw_sse_events[{event_pos}] is not a dict")
                continue
            if event.get("event_index") != event_pos:
                errors.append(f"{role}[{request_index}]: raw_sse_events[{event_pos}].event_index is not contiguous")
            receive_ns = event.get("receive_perf_counter_ns")
            if type(receive_ns) is not int:
                errors.append(f"{role}[{request_index}]: raw_sse_events[{event_pos}].receive_perf_counter_ns is not int")
            else:
                if previous_receive_ns is not None and receive_ns < previous_receive_ns:
                    errors.append(f"{role}[{request_index}]: raw_sse_events receive timestamps are not monotonic")
                previous_receive_ns = receive_ns
                if "request_start_ns" in ns_values and receive_ns < ns_values["request_start_ns"]:
                    errors.append(f"{role}[{request_index}]: raw SSE event precedes request_start_ns")
                if "stream_end_ns" in ns_values and receive_ns > ns_values["stream_end_ns"]:
                    errors.append(f"{role}[{request_index}]: raw SSE event follows stream_end_ns")
            token_ids = event.get("token_ids")
            if not isinstance(token_ids, list) or not all(type(x) is int for x in token_ids):
                errors.append(f"{role}[{request_index}]: raw_sse_events[{event_pos}].token_ids is not list[int]")
                continue
            reconstructed_output.extend(token_ids)
            if token_ids and type(receive_ns) is int:
                positive_event_ns.append(receive_ns)

        if output_ids is not None and reconstructed_output != output_ids:
            errors.append(f"{role}[{request_index}]: raw SSE token_ids do not reconstruct output_token_ids")
        if positive_event_ns:
            if ns_values.get("first_token_receive_ns") != positive_event_ns[0]:
                errors.append(f"{role}[{request_index}]: first_token_receive_ns does not match raw SSE trace")
            if ns_values.get("last_token_receive_ns") != positive_event_ns[-1]:
                errors.append(f"{role}[{request_index}]: last_token_receive_ns does not match raw SSE trace")
        elif expected_completion_tokens > 0:
            errors.append(f"{role}[{request_index}]: raw SSE trace contains no output-token event")

    return errors


def validate_complete_request_record(
    record: object, *, episode: Episode, role: str, request_index: int,
    trigger_perf_ns: object = None,
) -> list[str]:
    """Public entry point for Section 4: validates a single victim/burst
    request record against the deterministic derivation and functional
    completeness rules for `episode`/`role`/`request_index`.

    `trigger_perf_ns` is new, optional, and defaults to None -- existing
    callers are completely unaffected. When a caller supplies the
    episode's trigger.trigger_perf_ns for a `role == "victim"` record,
    this additionally runs validate_victim_timing_instrumentation() (the
    task/semaphore/dispatch/wave/trigger-exposure checks) and appends
    any errors it finds.
    """
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

    errors = _validate_request_record_fields(
        record,
        episode_id=episode.episode_id,
        role=role,
        request_index=request_index,
        expected_prompt_seed=expected_prompt_seed,
        expected_generation_seed=expected_generation_seed,
        expected_prompt_tokens=expected_prompt_tokens,
        expected_completion_tokens=expected_completion_tokens,
    )
    if role == "victim" and trigger_perf_ns is not None:
        errors.extend(
            validate_victim_timing_instrumentation(
                record, episode=episode, request_index=request_index, trigger_perf_ns=trigger_perf_ns,
            )
        )
    return errors


def validate_victim_timing_instrumentation(
    record: object, *, episode: Episode, request_index: int, trigger_perf_ns: object,
) -> list[str]:
    """Section: validates the new task/semaphore/dispatch/wave/trigger
    instrumentation on a single victim request record. Additive and
    standalone -- never called for burst/stabilization records, and
    never invoked unless a caller explicitly opts in (see
    validate_complete_request_record above). Never raises; always
    returns a (possibly empty) list of textual error descriptions."""
    if not isinstance(record, dict):
        return [f"victim[{request_index}]: record is not a dict"]

    errors: list[str] = []
    ns_fields = (
        "task_created_ns", "victim_phase_start_ns", "semaphore_acquired_ns",
        "request_dispatch_ns", "request_terminal_ns",
    )
    ns_values: dict[str, int] = {}
    for f in ns_fields:
        v = record.get(f, "<MISSING>")
        if type(v) is not int:
            errors.append(f"victim[{request_index}]: {f!r} is missing or not an int")
        else:
            ns_values[f] = v

    if len(ns_values) == len(ns_fields):
        if not (
            ns_values["victim_phase_start_ns"] <= ns_values["task_created_ns"]
            <= ns_values["semaphore_acquired_ns"] <= ns_values["request_dispatch_ns"]
            <= ns_values["request_terminal_ns"]
        ):
            errors.append(
                f"victim[{request_index}]: timestamps are not monotonic "
                f"(victim_phase_start_ns <= task_created_ns <= "
                f"semaphore_acquired_ns <= request_dispatch_ns <= "
                f"request_terminal_ns): {ns_values}"
            )

        expected = _compute_queue_timing_fields(
            task_created_ns=ns_values["task_created_ns"],
            victim_phase_start_ns=ns_values["victim_phase_start_ns"],
            semaphore_acquired_ns=ns_values["semaphore_acquired_ns"],
            request_dispatch_ns=ns_values["request_dispatch_ns"],
            request_terminal_ns=ns_values["request_terminal_ns"],
        )
        for key, expected_value in expected.items():
            stored_value = record.get(key, "<MISSING>")
            if expected_value is None:
                if stored_value is not None and stored_value != "<MISSING>":
                    errors.append(f"victim[{request_index}]: {key!r} should be null but is {stored_value!r}")
            elif not _is_finite_number(stored_value) or abs(float(stored_value) - expected_value) > 1e-6:
                errors.append(
                    f"victim[{request_index}]: {key!r} stored as {stored_value!r}, "
                    f"recomputed as {expected_value!r}"
                )

    concurrency = episode.concurrency
    try:
        expected_wave_id, expected_wave_position = _compute_wave(request_index, concurrency)
    except ValueError as exc:
        errors.append(f"victim[{request_index}]: cannot compute expected wave: {exc}")
        expected_wave_id = expected_wave_position = None

    wave_id = record.get("wave_id", "<MISSING>")
    wave_position = record.get("wave_position", "<MISSING>")
    if expected_wave_id is not None:
        if wave_id != expected_wave_id:
            errors.append(f"victim[{request_index}]: wave_id {wave_id!r} != expected {expected_wave_id!r}")
        if wave_position != expected_wave_position:
            errors.append(
                f"victim[{request_index}]: wave_position {wave_position!r} != "
                f"expected {expected_wave_position!r}"
            )
        if expected_wave_id == 0 and request_index >= concurrency:
            errors.append(f"victim[{request_index}]: wave_id==0 but request_index >= concurrency")
        if expected_wave_id >= 1 and request_index < concurrency:
            errors.append(f"victim[{request_index}]: wave_id>=1 but request_index < concurrency")

    if type(trigger_perf_ns) is int and len(ns_values) == len(ns_fields):
        expected_exposure = _compute_trigger_exposure(
            task_created_ns=ns_values["task_created_ns"],
            semaphore_acquired_ns=ns_values["semaphore_acquired_ns"],
            request_dispatch_ns=ns_values["request_dispatch_ns"],
            request_terminal_ns=ns_values["request_terminal_ns"],
            trigger_perf_ns=trigger_perf_ns,
        )
        for key, expected_value in expected_exposure.items():
            stored_value = record.get(key, "<MISSING>")
            if stored_value != expected_value:
                errors.append(
                    f"victim[{request_index}]: {key!r} stored as {stored_value!r}, "
                    f"recomputed as {expected_value!r}"
                )

    return errors


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def validate_trigger_record(
    trigger: object,
    *,
    episode: Episode,
    burst_interval: object,
) -> list[str]:
    """Deep-validate the scientific trigger metadata used for resume.

    A file is not resume-safe merely because trigger.status == "ok".  The
    threshold, four active-wave crossings, exact global trigger instant,
    crossing skew, per-request counts, and burst ordering must all agree.
    """
    errors: list[str] = []
    if not isinstance(trigger, dict):
        return ["trigger is not a dict"]

    if trigger.get("status") != "ok":
        errors.append("trigger.status != 'ok'")
    if trigger.get("trigger_after_decode_tokens") != episode.trigger_after_decode_tokens:
        errors.append("trigger_after_decode_tokens does not match schedule")

    trigger_ns = trigger.get("trigger_perf_ns")
    observed_ns = trigger.get("trigger_observed_perf_ns")
    if type(trigger_ns) is not int:
        errors.append("trigger_perf_ns is not an int")
    if type(observed_ns) is not int:
        errors.append("trigger_observed_perf_ns is not an int")
    elif type(trigger_ns) is int and observed_ns < trigger_ns:
        errors.append("trigger_observed_perf_ns precedes trigger_perf_ns")

    for utc_key in ("trigger_utc", "trigger_wall_time_utc", "trigger_observed_utc"):
        utc_value = trigger.get(utc_key)
        if not isinstance(utc_value, str) or not utc_value.strip():
            errors.append(f"{utc_key} is missing or empty")
        else:
            try:
                parsed_utc = datetime.fromisoformat(utc_value.replace("Z", "+00:00"))
                if parsed_utc.utcoffset() is None:
                    errors.append(f"{utc_key} is missing an explicit UTC offset")
            except ValueError:
                errors.append(f"{utc_key} is not valid ISO-8601")
    if trigger.get("trigger_utc") != trigger.get("trigger_wall_time_utc"):
        errors.append("trigger_utc != trigger_wall_time_utc")
    for key in ("waited_ms", "trigger_wait_duration_ms", "trigger_crossing_skew_ms", "trigger_dispatch_delay_ms"):
        value = trigger.get(key)
        if not _is_finite_number(value) or float(value) < 0:
            errors.append(f"{key} is not a finite non-negative number")

    waited_ms = trigger.get("waited_ms")
    wait_duration_ms = trigger.get("trigger_wait_duration_ms")
    if _is_finite_number(waited_ms) and _is_finite_number(wait_duration_ms) and not math.isclose(
        float(waited_ms), float(wait_duration_ms), rel_tol=0.0, abs_tol=1e-9
    ):
        errors.append("waited_ms != trigger_wait_duration_ms")

    dispatch_ms = trigger.get("trigger_dispatch_delay_ms")
    if type(trigger_ns) is int and type(observed_ns) is int and _is_finite_number(dispatch_ms):
        expected_dispatch_ms = (observed_ns - trigger_ns) / 1e6
        if not math.isclose(float(dispatch_ms), expected_dispatch_ms, rel_tol=0.0, abs_tol=1e-9):
            errors.append("trigger_dispatch_delay_ms is inconsistent with monotonic timestamps")

    try:
        logical_dt = datetime.fromisoformat(str(trigger.get("trigger_wall_time_utc")).replace("Z", "+00:00"))
        observed_dt = datetime.fromisoformat(str(trigger.get("trigger_observed_utc")).replace("Z", "+00:00"))
        wall_delay_ms = (observed_dt - logical_dt).total_seconds() * 1000.0
        if wall_delay_ms < 0:
            errors.append("trigger_observed_utc precedes trigger_wall_time_utc")
        elif _is_finite_number(dispatch_ms) and not math.isclose(
            wall_delay_ms, float(dispatch_ms), rel_tol=0.0, abs_tol=0.002
        ):
            errors.append("wall-clock trigger delay is inconsistent with monotonic dispatch delay")
    except (TypeError, ValueError):
        # ISO-format errors are already reported above.
        pass

    expected_active = min(episode.concurrency, episode.victim_request_count)
    if trigger.get("active_wave_request_count") != expected_active:
        errors.append(
            f"active_wave_request_count != {expected_active}"
        )

    details = trigger.get("active_wave_requests")
    if not isinstance(details, list) or len(details) != expected_active:
        errors.append(
            f"active_wave_requests must contain exactly {expected_active} entries"
        )
        return errors

    seen: set[int] = set()
    crossing_values: list[int] = []
    for pos, detail in enumerate(details):
        if not isinstance(detail, dict):
            errors.append(f"active_wave_requests[{pos}] is not a dict")
            continue
        idx = detail.get("request_index")
        if type(idx) is not int or idx < 0 or idx >= expected_active:
            errors.append(f"active_wave_requests[{pos}].request_index invalid")
            continue
        if idx in seen:
            errors.append(f"duplicate active-wave request_index {idx}")
        seen.add(idx)

        crossing_ns = detail.get("threshold_crossing_ns")
        crossing_count = detail.get("received_token_count_at_crossing")
        global_count = detail.get("received_token_count_at_global_trigger")
        status_at_trigger = detail.get("request_status_at_global_trigger")

        if type(crossing_ns) is not int:
            errors.append(f"active request {idx}: threshold_crossing_ns is not int")
        else:
            crossing_values.append(crossing_ns)
            if type(trigger_ns) is int and crossing_ns > trigger_ns:
                errors.append(f"active request {idx}: crossing occurs after global trigger")
        if type(crossing_count) is not int or crossing_count < episode.trigger_after_decode_tokens:
            errors.append(f"active request {idx}: crossing token count below threshold")
        if type(global_count) is not int:
            errors.append(f"active request {idx}: global-trigger token count is not int")
        elif type(crossing_count) is int and global_count < crossing_count:
            errors.append(f"active request {idx}: global-trigger count below crossing count")
        if status_at_trigger not in ("running", REQUEST_STATUS_COMPLETE):
            errors.append(f"active request {idx}: invalid request_status_at_global_trigger")

    if seen != set(range(expected_active)):
        errors.append("active-wave request indices are not exactly 0..concurrency-1")

    if len(crossing_values) == expected_active and type(trigger_ns) is int:
        expected_trigger_ns = max(crossing_values)
        if trigger_ns != expected_trigger_ns:
            errors.append("trigger_perf_ns is not the last active-wave crossing")
        expected_skew_ms = (max(crossing_values) - min(crossing_values)) / 1e6
        actual_skew = trigger.get("trigger_crossing_skew_ms")
        if _is_finite_number(actual_skew) and not math.isclose(
            float(actual_skew), expected_skew_ms, rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append("trigger_crossing_skew_ms is inconsistent with crossing timestamps")

    if episode.condition == "no_burst":
        if burst_interval is not None:
            errors.append("burst_interval must be null for no_burst")
    else:
        if not isinstance(burst_interval, dict):
            errors.append("burst_interval is not a dict for prefill_burst")
        else:
            start_ns = burst_interval.get("start_ns")
            end_ns = burst_interval.get("end_ns")
            if type(start_ns) is not int or type(end_ns) is not int or end_ns < start_ns:
                errors.append("burst_interval has invalid start/end")
            elif type(trigger_ns) is int and start_ns < trigger_ns:
                errors.append("burst starts before the global trigger")

    return errors


def validate_trigger_against_victim_requests(
    trigger: object,
    victim_requests: object,
    *,
    episode: Episode,
) -> list[str]:
    """Cross-check derived trigger metadata against normalized token traces."""
    errors: list[str] = []
    if not isinstance(trigger, dict) or not isinstance(victim_requests, list):
        return ["cannot cross-validate trigger against victim_requests"]

    trigger_ns = trigger.get("trigger_perf_ns")
    threshold = episode.trigger_after_decode_tokens
    expected_active = min(episode.concurrency, episode.victim_request_count)
    details = trigger.get("active_wave_requests")
    if type(trigger_ns) is not int or not isinstance(details, list):
        return ["trigger/raw-trace cross-validation prerequisites are malformed"]

    detail_by_index = {
        d.get("request_index"): d
        for d in details
        if isinstance(d, dict) and type(d.get("request_index")) is int
    }
    record_by_index = {
        r.get("request_index"): r
        for r in victim_requests
        if isinstance(r, dict) and type(r.get("request_index")) is int
    }

    for idx in range(expected_active):
        detail = detail_by_index.get(idx)
        record = record_by_index.get(idx)
        if detail is None or record is None:
            errors.append(f"active request {idx}: missing trigger detail or victim record")
            continue
        raw_events = record.get("raw_sse_events")
        if not isinstance(raw_events, list):
            errors.append(f"active request {idx}: raw_sse_events is not a list")
            continue

        cumulative = 0
        first_crossing_ns: int | None = None
        first_crossing_count: int | None = None
        count_at_global_trigger = 0
        trace_usable = True
        for event_pos, event in enumerate(raw_events):
            if not isinstance(event, dict):
                trace_usable = False
                errors.append(f"active request {idx}: raw event {event_pos} is not a dict")
                continue
            receive_ns = event.get("receive_perf_counter_ns")
            token_ids = event.get("token_ids")
            if type(receive_ns) is not int or not isinstance(token_ids, list) or not all(type(x) is int for x in token_ids):
                trace_usable = False
                errors.append(f"active request {idx}: raw event {event_pos} cannot be used for trigger reconstruction")
                continue
            cumulative += len(token_ids)
            if receive_ns <= trigger_ns:
                count_at_global_trigger = cumulative
            if first_crossing_ns is None and cumulative >= threshold:
                first_crossing_ns = receive_ns
                first_crossing_count = cumulative

        if not trace_usable:
            continue
        if first_crossing_ns is None:
            errors.append(f"active request {idx}: raw trace never reaches trigger threshold")
            continue
        if detail.get("threshold_crossing_ns") != first_crossing_ns:
            errors.append(f"active request {idx}: threshold_crossing_ns does not match raw SSE trace")
        if detail.get("received_token_count_at_crossing") != first_crossing_count:
            errors.append(f"active request {idx}: crossing token count does not match raw SSE trace")
        if detail.get("received_token_count_at_global_trigger") != count_at_global_trigger:
            errors.append(f"active request {idx}: global-trigger token count does not match raw SSE trace")

        stream_end_ns = record.get("stream_end_ns")
        expected_status = record.get("status") if type(stream_end_ns) is int and stream_end_ns <= trigger_ns else "running"
        if detail.get("request_status_at_global_trigger") != expected_status:
            errors.append(f"active request {idx}: status at global trigger does not match request timeline")

    return errors


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

    # --- new mandatory timing-instrumentation contract (schema 5) --------
    if obj["timing_instrumentation_version"] != TIMING_INSTRUMENTATION_VERSION:
        notes.append(
            f"timing_instrumentation_version {obj['timing_instrumentation_version']!r} "
            f"!= {TIMING_INSTRUMENTATION_VERSION!r}"
        )
    if obj["timing_instrumentation_name"] != TIMING_INSTRUMENTATION_NAME:
        notes.append(
            f"timing_instrumentation_name {obj['timing_instrumentation_name']!r} "
            f"!= {TIMING_INSTRUMENTATION_NAME!r}"
        )

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

    trigger = obj["trigger"]
    trigger_perf_ns = trigger.get("trigger_perf_ns") if isinstance(trigger, dict) else None

    victim_requests = obj["victim_requests"]
    if len(victim_requests) != expected_episode.victim_request_count:
        notes.append(
            f"expected exactly {expected_episode.victim_request_count} "
            f"victim_requests"
        )
    else:
        for i, record in enumerate(victim_requests):
            notes.extend(
                validate_complete_request_record(
                    record, episode=expected_episode, role="victim", request_index=i,
                    trigger_perf_ns=trigger_perf_ns,
                )
            )

    # --- victim_phase_start_ns: top-level int, identical across every
    # complete victim record (Section 3 of the timing-instrumentation
    # contract). Checked independently of the per-record validator above,
    # since that validator only checks a record against itself, not
    # cross-record/top-level consistency.
    episode_phase_start = obj.get("victim_phase_start_ns")
    if isinstance(victim_requests, list):
        record_phase_starts = {
            r.get("victim_phase_start_ns")
            for r in victim_requests
            if isinstance(r, dict) and r.get("status") == REQUEST_STATUS_COMPLETE
        }
        if record_phase_starts and record_phase_starts != {episode_phase_start}:
            notes.append(
                f"victim_phase_start_ns is not identical across all complete "
                f"victim requests and the episode top level: "
                f"top-level={episode_phase_start!r}, per-request={sorted(record_phase_starts, key=str)!r}"
            )

    expected_burst_count = (
        expected_episode.burst_parallel_requests
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

    # --- queue_timing_summary: must be exactly reproducible from the
    # victim_requests actually stored in this same file (Section 4).
    # Never repaired/recomputed for the caller -- a mismatch is only ever
    # reported as an invalidating note.
    if isinstance(victim_requests, list) and all(isinstance(r, dict) for r in victim_requests):
        expected_summary = _build_queue_timing_summary(victim_requests)
        stored_summary = obj.get("queue_timing_summary")
        summary_errors = _diff_queue_timing_summary(stored_summary, expected_summary)
        notes.extend(summary_errors)

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

    burst_interval = obj["burst_interval"]
    notes.extend(
        validate_trigger_record(
            trigger, episode=expected_episode, burst_interval=burst_interval
        )
    )
    notes.extend(
        validate_trigger_against_victim_requests(
            trigger, victim_requests, episode=expected_episode
        )
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


def require_output_dir_mode_marker(output_dir: Path, mode: str) -> None:
    marker_path = output_dir / RUN_MODE_MARKER_FILENAME
    if not marker_path.is_file():
        raise OutputDirConflictError(
            f"--resume requires an existing run-mode marker at {marker_path}"
        )
    check_output_dir_not_shared(output_dir, mode)


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


def _logical_utc_from_observation(observed_utc: str, delay_ns: int) -> str:
    """Map a later wall-clock observation back to the logical monotonic
    trigger instant.  ISO parsing failure is a hard data-quality issue in
    real execution, so callers validate the returned value before resume.
    """
    try:
        normalized = observed_utc.replace("Z", "+00:00")
        observed_dt = datetime.fromisoformat(normalized)
        logical_dt = observed_dt - timedelta(microseconds=max(0, delay_ns) / 1_000)
        return logical_dt.isoformat()
    except (TypeError, ValueError, OverflowError):
        return observed_utc


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
        self._tok = AutoTokenizer.from_pretrained(model_full_id, local_files_only=True)
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


def _redact_secret(text: str, secret: str | None) -> str:
    """Best-effort redaction before exception text reaches JSON output."""
    if not secret:
        return text
    return text.replace(f"Bearer {secret}", "Bearer <redacted>").replace(secret, "<redacted>")


def _compute_queue_timing_fields(
    *,
    task_created_ns: int | None,
    victim_phase_start_ns: int | None,
    semaphore_acquired_ns: int | None,
    request_dispatch_ns: int | None,
    request_terminal_ns: int | None,
) -> dict:
    """Purely observational derived queue/dispatch timing (see project
    prompt: task_created_ns / victim_phase_start_ns / semaphore_acquired_ns
    / request_dispatch_ns / request_terminal_ns). Each derived value is
    computed ONLY when every timestamp it needs is present, an int, and
    non-negative (monotonic) -- otherwise it is `None`, never a silently
    stored negative duration.
    """

    def _delta_ms(later: object, earlier: object) -> float | None:
        if type(later) is not int or type(earlier) is not int:
            return None
        if later < earlier:
            return None
        return (later - earlier) / 1e6

    return {
        "local_queue_wait_ms": _delta_ms(semaphore_acquired_ns, task_created_ns),
        "admission_to_dispatch_ms": _delta_ms(request_dispatch_ns, semaphore_acquired_ns),
        "task_creation_offset_ms": _delta_ms(task_created_ns, victim_phase_start_ns),
        "dispatch_from_phase_start_ms": _delta_ms(request_dispatch_ns, victim_phase_start_ns),
        "total_e2el_from_task_creation_ms": _delta_ms(request_terminal_ns, task_created_ns),
        "total_e2el_from_phase_start_ms": _delta_ms(request_terminal_ns, victim_phase_start_ns),
    }


def _compute_wave(request_index: object, concurrency: object) -> tuple[int, int]:
    """wave_id/wave_position from request_index and the episode's actual
    concurrency ONLY. Raises ValueError on a missing/invalid
    request_index or concurrency -- callers must turn this into a
    visible validation error, never a silent 'later wave'/large wave_id
    fallback."""
    if type(request_index) is not int or request_index < 0:
        raise ValueError(f"invalid request_index for wave computation: {request_index!r}")
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError(f"invalid concurrency for wave computation: {concurrency!r}")
    return request_index // concurrency, request_index % concurrency


def _compute_trigger_exposure(
    *,
    task_created_ns: int | None,
    semaphore_acquired_ns: int | None,
    request_dispatch_ns: int | None,
    request_terminal_ns: int | None,
    trigger_perf_ns: int | None,
) -> dict:
    """Section: trigger-related request classification. Only meaningful
    once trigger_perf_ns is finally known -- callers must invoke this
    AFTER the global trigger instant is established, never before. Every
    comparison is None-safe: a missing timestamp simply makes the
    corresponding boolean False and is reflected in
    trigger_exposure_group as 'unknown', never a crash.
    """
    if type(trigger_perf_ns) is not int:
        return {
            "was_created_at_trigger": None,
            "was_admitted_at_trigger": None,
            "was_dispatched_at_trigger": None,
            "was_running_at_trigger": None,
            "trigger_exposure_group": "unknown",
        }

    was_created_at_trigger = type(task_created_ns) is int and task_created_ns <= trigger_perf_ns
    was_admitted_at_trigger = (
        type(semaphore_acquired_ns) is int and semaphore_acquired_ns <= trigger_perf_ns
    )
    was_dispatched_at_trigger = (
        type(request_dispatch_ns) is int and request_dispatch_ns <= trigger_perf_ns
    )
    was_running_at_trigger = was_dispatched_at_trigger and (
        type(request_terminal_ns) is not int or request_terminal_ns > trigger_perf_ns
    )

    if type(request_dispatch_ns) is int and type(request_terminal_ns) is int and (
        request_dispatch_ns <= trigger_perf_ns < request_terminal_ns
    ):
        group = "running_at_trigger"
    elif type(request_terminal_ns) is int and request_terminal_ns <= trigger_perf_ns:
        group = "completed_before_trigger"
    elif (
        type(semaphore_acquired_ns) is int
        and semaphore_acquired_ns <= trigger_perf_ns
        and type(request_dispatch_ns) is int
        and request_dispatch_ns > trigger_perf_ns
    ):
        group = "admitted_not_dispatched_at_trigger"
    elif (
        type(task_created_ns) is int
        and task_created_ns <= trigger_perf_ns
        and type(semaphore_acquired_ns) is int
        and semaphore_acquired_ns > trigger_perf_ns
    ):
        group = "queued_at_trigger"
    elif type(task_created_ns) is int and task_created_ns > trigger_perf_ns:
        group = "created_after_trigger"
    else:
        group = "unknown"

    return {
        "was_created_at_trigger": was_created_at_trigger,
        "was_admitted_at_trigger": was_admitted_at_trigger,
        "was_dispatched_at_trigger": was_dispatched_at_trigger,
        "was_running_at_trigger": was_running_at_trigger,
        "trigger_exposure_group": group,
    }



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
    on_output_tokens: Callable[[int, int], None] | None = None,
    task_created_ns: int | None = None,
    victim_phase_start_ns: int | None = None,
    semaphore_acquired_ns: int | None = None,
    wave_id: int | None = None,
    wave_position: int | None = None,
) -> dict:
    """
    `on_output_tokens`, if given, is invoked synchronously every time one
    or more real output token_ids are appended to this request's stream
    -- i.e. once per non-empty SSE token batch, NOT once per SSE event
    (a single SSE event/batch can contain multiple output tokens; see
    Section 11 of the project prompt). It receives
    (cumulative_output_token_count, receive_perf_counter_ns) so callers
    can implement an arbitrary `received_output_token_count >=
    trigger_after_decode_tokens` threshold rather than a fixed
    first-token trigger. Passing None reproduces the original
    Prefill-Screen behavior of not tracking any threshold.

    `task_created_ns` / `victim_phase_start_ns` / `semaphore_acquired_ns`
    / `wave_id` / `wave_position`, if given, are purely observational
    values ESTABLISHED BY THE CALLER (task_created_ns immediately before
    this request's own asyncio.create_task(...); victim_phase_start_ns
    once per episode before the first victim task is created;
    semaphore_acquired_ns immediately after this request's semaphore
    acquisition) and are only carried through into the returned record
    plus the derived queue/dispatch timing fields -- they never affect
    scheduling, the semaphore, or request execution itself. All five
    default to None so burst/stabilization callers (which do not supply
    them) are completely unaffected.
    """
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
    # request_dispatch_ns: set to None here and only actually captured
    # inside _consume(), immediately before the real transport call --
    # i.e. only if the transport path is genuinely entered. A failure
    # before _consume() runs (e.g. this coroutine being cancelled pre-
    # dispatch) correctly leaves it None. request_start_ns's own
    # existing semantics/value are left completely untouched.
    request_dispatch_ns: int | None = None

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
        nonlocal request_dispatch_ns
        http_status = 200
        # Immediately before the real transport call -- no further
        # synchronous preparation happens between this line and the
        # actual dispatch, and no additional await is introduced.
        request_dispatch_ns = clock.perf_counter_ns()
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
                    last_token_receive_ns = now_ns
                    token_event_receipts.append((now_ns, len(token_ids_here)))
                    # Fires on every batch with the TRUE cumulative
                    # output-token count (len(output_token_ids)), not the
                    # SSE-event count -- a single batch can carry more
                    # than one token_id, and a threshold crossing inside
                    # a multi-token batch must still be detected exactly
                    # once, at the batch that pushes the cumulative count
                    # to or past the threshold (Section 11).
                    if on_output_tokens is not None:
                        on_output_tokens(len(output_token_ids), now_ns)
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
        error_message = _redact_secret(str(exc), api_key)

    stream_end_ns = clock.perf_counter_ns()
    # request_terminal_ns: the final request-end instant, covering
    # success/timeout/cancel/http-error/generic-exception alike, since
    # this line runs unconditionally after the try/except above in every
    # case -- functionally equivalent to a finally-path capture. Kept as
    # an explicit alias rather than a second, independently-timed call.
    request_terminal_ns = stream_end_ns
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
    queue_timing_fields = _compute_queue_timing_fields(
        task_created_ns=task_created_ns,
        victim_phase_start_ns=victim_phase_start_ns,
        semaphore_acquired_ns=semaphore_acquired_ns,
        request_dispatch_ns=request_dispatch_ns,
        request_terminal_ns=request_terminal_ns,
    )
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
        # --- additive task/semaphore/dispatch/wave/trigger instrumentation ---
        "task_created_ns": task_created_ns,
        "victim_phase_start_ns": victim_phase_start_ns,
        "semaphore_acquired_ns": semaphore_acquired_ns,
        "request_dispatch_ns": request_dispatch_ns,
        "request_terminal_ns": request_terminal_ns,
        **queue_timing_fields,
        "wave_id": wave_id,
        "wave_position": wave_position,
        # Trigger-exposure fields are only meaningful once the episode's
        # global trigger_perf_ns is known, which is always AFTER this
        # single request has already returned -- run_regular_episode
        # fills these in (victim requests only) once trigger_ns is
        # finalized. Present here as explicit None placeholders so every
        # record (including burst/stabilization, which never get
        # enriched) has a uniform, discoverable shape.
        "was_created_at_trigger": None,
        "was_admitted_at_trigger": None,
        "was_dispatched_at_trigger": None,
        "was_running_at_trigger": None,
        "trigger_exposure_group": None,
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
            # New timing/wave/trigger fields: default to None/unknown here.
            # Known-safe task-creation-time metadata (task_created_ns,
            # victim_phase_start_ns, wave_id, wave_position) is filled in
            # separately by _apply_known_task_metadata() -- see below --
            # since it is only available to the caller in
            # run_regular_episode, not to this generic helper.
            r.setdefault("task_created_ns", None)
            r.setdefault("victim_phase_start_ns", None)
            r.setdefault("semaphore_acquired_ns", None)
            r.setdefault("request_dispatch_ns", None)
            r.setdefault("request_terminal_ns", None)
            r.setdefault("local_queue_wait_ms", None)
            r.setdefault("admission_to_dispatch_ms", None)
            r.setdefault("task_creation_offset_ms", None)
            r.setdefault("dispatch_from_phase_start_ms", None)
            r.setdefault("total_e2el_from_task_creation_ms", None)
            r.setdefault("total_e2el_from_phase_start_ms", None)
            r.setdefault("wave_id", None)
            r.setdefault("wave_position", None)
            r.setdefault("was_created_at_trigger", None)
            r.setdefault("was_admitted_at_trigger", None)
            r.setdefault("was_dispatched_at_trigger", None)
            r.setdefault("was_running_at_trigger", None)
            r.setdefault("trigger_exposure_group", None)
        enriched.append(r)
    return enriched


def _apply_known_task_metadata(records: list[dict], task_metadata: dict[int, dict]) -> None:
    """Section 7 (error-path task metadata): mutates each minimal/partial
    victim record in place, filling in ONLY the metadata that was
    genuinely captured outside the request coroutine at task-creation
    time (request_index, task_created_ns, victim_phase_start_ns, wave_id,
    wave_position) -- via setdefault, so a record that already has its
    own real values (because the task got far enough to build one) is
    never overwritten. Deliberately never invents
    semaphore_acquired_ns/request_dispatch_ns/request_terminal_ns or any
    derived ms field for a task that never reached that stage; those
    stay None/whatever _enrich_minimal_records already defaulted them to."""
    for record in records:
        idx = record.get("request_index")
        meta = task_metadata.get(idx)
        if meta is None:
            continue
        for key, value in meta.items():
            if record.get(key) is None:
                record[key] = value


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
    ctx: RunContext,
    episode: Episode,
    i: int,
    on_output_tokens: Callable[[int, int], None] | None = None,
    *,
    task_created_ns: int | None = None,
    victim_phase_start_ns: int | None = None,
    semaphore_acquired_ns: int | None = None,
    wave_id: int | None = None,
    wave_position: int | None = None,
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
        on_output_tokens=on_output_tokens,
        task_created_ns=task_created_ns,
        victim_phase_start_ns=victim_phase_start_ns,
        semaphore_acquired_ns=semaphore_acquired_ns,
        wave_id=wave_id,
        wave_position=wave_position,
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
# Trigger watcher -- generalized token-count barrier (Sections 11-13, 16-18)
#
# Generalizes Prefill-Screen's fixed "first output token of every
# active-wave request" barrier into a configurable
# `trigger_after_decode_tokens` threshold (1 counts actually-received
# output tokens, not SSE events -- a single SSE batch can carry more
# than one token_id). trigger_after_decode_tokens=1 is semantically
# identical to the old first-token trigger.
# ============================================================================

@dataclass
class _ActiveWaveCrossing:
    request_index: int
    threshold_crossing_ns: int | None = None
    received_token_count_at_crossing: int | None = None
    # One entry per non-empty output-token batch.  This makes the count at
    # the exact global barrier reconstructable even when a faster request
    # receives additional batches while waiting for the slowest request.
    token_count_history: list[tuple[int, int]] = field(default_factory=list)


def make_threshold_callback(
    request_index: int,
    threshold: int,
    crossing: _ActiveWaveCrossing,
    event: asyncio.Event,
) -> Callable[[int, int], None]:
    """Builds the on_output_tokens callback for one active-wave request.
    Sets `event` and records the crossing exactly once, at the first
    token BATCH (not SSE event) whose cumulative count reaches or
    exceeds `threshold` -- correctly handling a single batch that jumps
    the cumulative count past the threshold in one step."""

    def _cb(cumulative_count: int, receive_ns: int) -> None:
        # Keep every cumulative count with its receive timestamp.  Do this
        # even after the individual request crossed, because the global
        # barrier may occur later when the slowest active-wave request
        # reaches the threshold.
        crossing.token_count_history.append((receive_ns, cumulative_count))
        if crossing.threshold_crossing_ns is not None:
            return
        if cumulative_count >= threshold:
            crossing.threshold_crossing_ns = receive_ns
            crossing.received_token_count_at_crossing = cumulative_count
            if not event.is_set():
                event.set()

    return _cb


async def _watch_trigger(
    active_wave_indices: Iterable[int],
    threshold_events: dict[int, asyncio.Event],
    victim_tasks: dict[int, asyncio.Task],
    timeout_s: float,
) -> str:
    """Returns 'ok' once every active-wave request has crossed its
    individual trigger_after_decode_tokens threshold; 'timeout' if
    timeout_s elapses first; or 'pretrigger_failure' if an active-wave
    request finishes without ever crossing its threshold, OR finishes
    with a non-'complete' status even after having crossed it (a
    request that crosses its threshold and then fails/becomes
    incomplete before the GLOBAL trigger -- i.e. before every
    active-wave request has crossed -- must also abort the trigger; see
    Section 11's 'kein Burst mit unvollständiger Barriere'). No implicit
    fallback to a timeout- or first-token-only barrier is ever
    substituted here."""
    indices = tuple(sorted(active_wave_indices))
    start = time.monotonic()
    all_events_task = asyncio.ensure_future(
        asyncio.gather(*(threshold_events[i].wait() for i in indices))
    )
    pending = {victim_tasks[i]: i for i in indices}
    try:
        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                return "timeout"
            waitables = {all_events_task, *pending.keys()}
            done, _pending = await asyncio.wait(
                waitables, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            # Inspect completed active-wave requests BEFORE accepting a
            # simultaneously completed all-events task.  asyncio.wait may
            # return both in the same `done` set; returning "ok" first would
            # incorrectly permit a burst after an active request failed.
            for t in [t for t in list(pending) if t in done]:
                idx = pending.pop(t)
                if not threshold_events[idx].is_set():
                    return "pretrigger_failure"
                try:
                    result = t.result()
                except BaseException:
                    return "pretrigger_failure"
                if not isinstance(result, dict) or result.get("status") != REQUEST_STATUS_COMPLETE:
                    return "pretrigger_failure"
            if all_events_task in done:
                # A task can become done between asyncio.wait returning and
                # this check.  Validate every already-finished active-wave
                # task once more before declaring the barrier complete.
                for idx in indices:
                    task = victim_tasks[idx]
                    if not task.done():
                        continue
                    if not threshold_events[idx].is_set():
                        return "pretrigger_failure"
                    try:
                        result = task.result()
                    except BaseException:
                        return "pretrigger_failure"
                    if not isinstance(result, dict) or result.get("status") != REQUEST_STATUS_COMPLETE:
                        return "pretrigger_failure"
                return "ok"
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
    run_mode: str,
    victim_phase_start_ns: int,
    queue_timing_summary: dict | None = None,
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
        "timestamps": {"trigger_utc": trigger.get("trigger_utc")},
        "trigger": trigger,
        "burst_interval": burst_interval,
        "victim_requests": victim_results,
        "burst_requests": burst_results,
        "aggregate_metrics": _aggregate_metrics(victim_results, burst_results, burst_interval),
        "status": status,
        "validation_errors": validation_errors,
        # --- additive timing/queue/wave/trigger instrumentation ---
        "timing_instrumentation_version": TIMING_INSTRUMENTATION_VERSION,
        "timing_instrumentation_name": TIMING_INSTRUMENTATION_NAME,
        "victim_phase_start_ns": victim_phase_start_ns,
        "queue_timing_summary": queue_timing_summary,
    }


# ============================================================================
# Regular episode runner (Sections 16-20)
# ============================================================================

def _enrich_trigger_exposure_fields(victim_results: list[dict], trigger_perf_ns: object) -> None:
    """Mutates each victim record in place, filling in the
    was_*_at_trigger booleans and trigger_exposure_group -- only doable
    AFTER trigger_perf_ns is finally known, i.e. after _watch_trigger()
    has returned. Never touches burst/stabilization records."""
    trigger_ns = trigger_perf_ns if type(trigger_perf_ns) is int else None
    for record in victim_results:
        if not isinstance(record, dict):
            continue
        record.update(
            _compute_trigger_exposure(
                task_created_ns=record.get("task_created_ns"),
                semaphore_acquired_ns=record.get("semaphore_acquired_ns"),
                request_dispatch_ns=record.get("request_dispatch_ns"),
                request_terminal_ns=record.get("request_terminal_ns"),
                trigger_perf_ns=trigger_ns,
            )
        )


def _build_queue_timing_summary(victim_results: list[dict]) -> dict:
    """Purely descriptive, episode-level roll-up of the new queue/
    dispatch/wave timing fields -- never a substitute for the raw
    per-request data, which remains fully stored in victim_requests."""
    by_wave: dict[int, list[dict]] = {}
    for r in victim_results:
        if not isinstance(r, dict):
            continue
        wave = r.get("wave_id")
        if type(wave) is int:
            by_wave.setdefault(wave, []).append(r)

    request_count_by_wave = {str(w): len(rs) for w, rs in sorted(by_wave.items())}

    exposure_counts: dict[str, int] = {}
    for r in victim_results:
        if isinstance(r, dict):
            group = r.get("trigger_exposure_group") or "unknown"
            exposure_counts[group] = exposure_counts.get(group, 0) + 1

    def _by_wave(field: str, agg) -> dict[str, float | None]:
        out = {}
        for w, rs in sorted(by_wave.items()):
            values = [r[field] for r in rs if type(r.get(field)) in (int, float)]
            out[str(w)] = agg(values) if values else None
        return out

    return {
        "request_count_by_wave": request_count_by_wave,
        "request_count_by_trigger_exposure_group": exposure_counts,
        "median_task_creation_offset_ms_by_wave": _by_wave("task_creation_offset_ms", _median),
        "median_local_queue_wait_ms_by_wave": _by_wave("local_queue_wait_ms", _median),
        "max_local_queue_wait_ms_by_wave": _by_wave("local_queue_wait_ms", max),
        "median_dispatch_from_phase_start_ms_by_wave": _by_wave("dispatch_from_phase_start_ms", _median),
        "median_total_e2el_from_task_creation_ms_by_wave": _by_wave("total_e2el_from_task_creation_ms", _median),
        "median_total_e2el_from_phase_start_ms_by_wave": _by_wave("total_e2el_from_phase_start_ms", _median),
    }


_QUEUE_TIMING_SUMMARY_FLOAT_TOLERANCE = 1e-6


def _diff_queue_timing_summary(stored: object, expected: dict, *, path: str = "queue_timing_summary") -> list[str]:
    """Structured recursive comparison between a stored queue_timing_summary
    and one freshly recomputed (via _build_queue_timing_summary) from the
    SAME file's own victim_requests. Since both sides are produced by the
    identical deterministic code, any difference beyond a small,
    documented float tolerance means the stored summary is missing,
    manipulated, or otherwise inconsistent -- never repaired here, only
    reported."""
    if stored is None:
        return [f"{path} is missing"]
    if not isinstance(stored, dict):
        return [f"{path} is not a dict"]

    errors: list[str] = []
    if set(stored.keys()) != set(expected.keys()):
        errors.append(
            f"{path} has key set {sorted(stored.keys())}, expected "
            f"{sorted(expected.keys())}"
        )

    for key, expected_value in expected.items():
        if key not in stored:
            continue  # already reported via the key-set check above
        stored_value = stored[key]
        sub_path = f"{path}.{key}"
        if isinstance(expected_value, dict):
            if not isinstance(stored_value, dict):
                errors.append(f"{sub_path} is not a dict")
                continue
            if set(stored_value.keys()) != set(expected_value.keys()):
                errors.append(
                    f"{sub_path} has key set {sorted(stored_value.keys())}, "
                    f"expected {sorted(expected_value.keys())}"
                )
            for inner_key, inner_expected in expected_value.items():
                if inner_key not in stored_value:
                    continue
                inner_stored = stored_value[inner_key]
                inner_path = f"{sub_path}.{inner_key}"
                if inner_expected is None:
                    if inner_stored is not None:
                        errors.append(f"{inner_path} is {inner_stored!r}, expected null")
                elif type(inner_expected) is int:
                    if type(inner_stored) is not int or inner_stored != inner_expected:
                        errors.append(
                            f"{inner_path} = {inner_stored!r}, "
                            f"recomputed as integer {inner_expected!r}"
                        )
                elif type(inner_expected) is float:
                    if (
                        type(inner_stored) not in (int, float)
                        or not math.isfinite(float(inner_stored))
                        or abs(float(inner_stored) - inner_expected)
                           > _QUEUE_TIMING_SUMMARY_FLOAT_TOLERANCE
                    ):
                        errors.append(
                            f"{inner_path} = {inner_stored!r}, "
                            f"recomputed as {inner_expected!r}"
                        )
                elif inner_stored != inner_expected:
                    errors.append(f"{inner_path} = {inner_stored!r}, recomputed as {inner_expected!r}")
        elif stored_value != expected_value:
            errors.append(f"{sub_path} = {stored_value!r}, recomputed as {expected_value!r}")

    return errors


async def run_regular_episode(
    ctx: RunContext,
    episode: Episode,
    *,
    schedule_fingerprint: str,
    server_metadata: dict,
    stabilization_ref: dict,
    run_mode: str,
) -> dict:
    concurrency = episode.concurrency
    n = episode.victim_request_count
    threshold = episode.trigger_after_decode_tokens
    active_wave_size = min(concurrency, n)
    active_wave = tuple(range(active_wave_size))
    threshold_events = {i: asyncio.Event() for i in active_wave}
    active_started_events = {i: asyncio.Event() for i in active_wave}
    crossings: dict[int, _ActiveWaveCrossing] = {
        i: _ActiveWaveCrossing(request_index=i) for i in active_wave
    }
    sem = asyncio.Semaphore(concurrency)
    victim_tasks: dict[int, asyncio.Task] = {}
    burst_tasks: list[asyncio.Task] = []

    async def _victim(i: int, task_created_ns: int, wave_id: int, wave_position: int) -> dict:
        cb = (
            make_threshold_callback(i, threshold, crossings[i], threshold_events[i])
            if i in active_wave
            else None
        )
        async with sem:
            # Immediately after successful semaphore acquisition, before
            # any further request preparation/dispatch -- purely
            # observational, does not affect scheduling or fairness.
            semaphore_acquired_ns = ctx.clock.perf_counter_ns()
            if i in active_started_events:
                active_started_events[i].set()
            return await _run_victim_request(
                ctx, episode, i, on_output_tokens=cb,
                task_created_ns=task_created_ns,
                victim_phase_start_ns=victim_phase_start_ns,
                semaphore_acquired_ns=semaphore_acquired_ns,
                wave_id=wave_id,
                wave_position=wave_position,
            )

    try:
        # victim_phase_start_ns: set exactly once per episode, immediately
        # before the first victim task is created -- the shared origin
        # for task_creation_offset_ms/dispatch_from_phase_start_ms/
        # total_e2el_from_phase_start_ms across ALL victim requests,
        # active and later wave alike.
        victim_phase_start_ns = ctx.clock.perf_counter_ns()
        # Section 7: known-safe metadata captured OUTSIDE each request's
        # own coroutine, so it survives even if the task is cancelled/
        # fails before it can build its own full record. Never includes
        # semaphore_acquired_ns/request_dispatch_ns/request_terminal_ns --
        # those are only ever real if genuinely observed.
        task_metadata: dict[int, dict] = {}

        # Explicit active wave: do not rely solely on implicit
        # semaphore-acquisition fairness -- start exactly victim indices
        # 0..concurrency-1 first, yield control so they actually get a
        # chance to acquire their semaphore slots, and only then create/
        # activate the remaining deterministic queue (indices
        # concurrency..n-1), which will block on the same semaphore until
        # an active-wave slot frees up. This does not change the
        # scientific schedule order or any seed derivation -- only the
        # startup sequencing guarantee.
        for i in range(active_wave_size):
            wave_id, wave_position = _compute_wave(i, concurrency)
            task_created_ns = ctx.clock.perf_counter_ns()
            task_metadata[i] = {
                "request_index": i, "task_created_ns": task_created_ns,
                "victim_phase_start_ns": victim_phase_start_ns,
                "wave_id": wave_id, "wave_position": wave_position,
            }
            victim_tasks[i] = asyncio.create_task(_victim(i, task_created_ns, wave_id, wave_position))
        # A real barrier, not a timing assumption: every active-wave task
        # must have acquired one of the four semaphore slots before any
        # later-wave task is even created.
        await asyncio.gather(*(active_started_events[i].wait() for i in active_wave))

        for i in range(active_wave_size, n):
            wave_id, wave_position = _compute_wave(i, concurrency)
            task_created_ns = ctx.clock.perf_counter_ns()
            task_metadata[i] = {
                "request_index": i, "task_created_ns": task_created_ns,
                "victim_phase_start_ns": victim_phase_start_ns,
                "wave_id": wave_id, "wave_position": wave_position,
            }
            victim_tasks[i] = asyncio.create_task(_victim(i, task_created_ns, wave_id, wave_position))
        await asyncio.sleep(0)

        trigger_start_ns = ctx.clock.perf_counter_ns()
        trigger_status = await _watch_trigger(
            active_wave, threshold_events, victim_tasks, ctx.trigger_timeout_s
        )
        trigger_observed_ns = ctx.clock.perf_counter_ns()
        trigger_observed_utc = ctx.clock.utcnow_iso()

        crossing_ns_values = [
            crossings[i].threshold_crossing_ns
            for i in sorted(active_wave)
            if crossings[i].threshold_crossing_ns is not None
        ]
        # The scientific trigger is the instant at which the LAST active-
        # wave request crossed.  The previous implementation used the later
        # coroutine-resumption timestamp, which silently shifted burst/control
        # window alignment by event-loop scheduling delay.
        trigger_ns = (
            max(crossing_ns_values)
            if trigger_status == "ok" and len(crossing_ns_values) == len(active_wave)
            else trigger_observed_ns
        )
        dispatch_delay_ns = max(0, trigger_observed_ns - trigger_ns)
        trigger_wall_time_utc = _logical_utc_from_observation(
            trigger_observed_utc, dispatch_delay_ns
        )
        crossing_skew_ms = (
            (max(crossing_ns_values) - min(crossing_ns_values)) / 1e6
            if len(crossing_ns_values) == len(active_wave) and crossing_ns_values
            else None
        )

        # Section 13: per-active-wave-request crossing detail.
        # request_status_at_global_trigger / received_token_count_at_
        # global_trigger are filled in below, once victim_results are
        # available (this dict is mutated again after the gather()).
        active_wave_detail = [
            {
                "request_index": i,
                "threshold_crossing_ns": crossings[i].threshold_crossing_ns,
                "received_token_count_at_crossing": crossings[i].received_token_count_at_crossing,
                "received_token_count_at_global_trigger": None,
                "request_status_at_global_trigger": None,
            }
            for i in sorted(active_wave)
        ]

        trigger = {
            "status": trigger_status,
            # trigger_utc is retained for backward readability and now
            # denotes the exact logical barrier instant, not the later
            # event-loop observation.
            "trigger_utc": trigger_wall_time_utc,
            "trigger_wall_time_utc": trigger_wall_time_utc,
            "trigger_observed_utc": trigger_observed_utc,
            "trigger_perf_ns": trigger_ns,
            "trigger_observed_perf_ns": trigger_observed_ns,
            "trigger_dispatch_delay_ms": dispatch_delay_ns / 1e6,
            "waited_ms": (trigger_observed_ns - trigger_start_ns) / 1e6,
            "trigger_after_decode_tokens": threshold,
            "trigger_wait_duration_ms": (trigger_observed_ns - trigger_start_ns) / 1e6,
            "trigger_crossing_skew_ms": crossing_skew_ms,
            "active_wave_request_count": len(active_wave),
            "active_wave_requests": active_wave_detail,
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
            _apply_known_task_metadata(victim_results, task_metadata)
            _enrich_trigger_exposure_fields(victim_results, trigger_ns)
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
                run_mode=run_mode,
                victim_phase_start_ns=victim_phase_start_ns,
                queue_timing_summary=_build_queue_timing_summary(victim_results),
            )

        if episode.condition == BURST_CONDITION:
            for j in range(episode.burst_parallel_requests):
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
    _enrich_trigger_exposure_fields(victim_results, trigger_ns)

    # Fill the exact logical snapshot at trigger_perf_ns.  Counts are
    # reconstructed from each request's batch history at/before the global
    # barrier.  Status is reconstructed from stream_end_ns: a request that
    # had not ended by the barrier was still running at that instant.
    victim_by_index = {r.get("request_index"): r for r in victim_results}
    for detail in trigger["active_wave_requests"]:
        i = detail["request_index"]
        history = crossings[i].token_count_history
        counts_at_barrier = [
            count for receive_ns, count in history if receive_ns <= trigger_ns
        ]
        detail["received_token_count_at_global_trigger"] = (
            max(counts_at_barrier) if counts_at_barrier else None
        )
        vr = victim_by_index.get(i)
        if vr is None:
            detail["request_status_at_global_trigger"] = None
        elif (
            type(vr.get("stream_end_ns")) is int
            and vr["stream_end_ns"] <= trigger_ns
        ):
            detail["request_status_at_global_trigger"] = vr.get("status")
        else:
            detail["request_status_at_global_trigger"] = "running"

    all_complete = all(r.get("status") == REQUEST_STATUS_COMPLETE for r in victim_results) and all(
        r.get("status") == REQUEST_STATUS_COMPLETE for r in burst_results
    )

    validation_errors: list[str] = []
    if not all_complete:
        validation_errors.append("one or more requests after trigger were not complete")

    # --- mandatory timing-instrumentation validation (Section 1) --------
    # Runs once, after all requests have finished and trigger_exposure
    # fields are already filled in -- never inside the time-critical
    # streaming path. Applies identically to no_burst and prefill_burst.
    # Every victim request is checked; none are silently dropped from
    # evaluation, and every finding is recorded with its request_index.
    timing_valid = True
    for record in victim_results:
        if record.get("status") != REQUEST_STATUS_COMPLETE:
            # Already reflected in all_complete/validation_errors above --
            # a request that legitimately failed before some stage is
            # expected to have partial timestamps; that is not a new,
            # separate timing-instrumentation error.
            continue
        idx = record.get("request_index")
        record_errors = validate_victim_timing_instrumentation(
            record, episode=episode, request_index=idx, trigger_perf_ns=trigger_ns,
        )
        if record_errors:
            timing_valid = False
            validation_errors.extend(f"victim[{idx}] timing: {e}" for e in record_errors)

    burst_interval = None
    if burst_results:
        starts = [r["request_start_ns"] for r in burst_results if r.get("request_start_ns") is not None]
        ends = [r["stream_end_ns"] for r in burst_results if r.get("stream_end_ns") is not None]
        if starts and ends:
            burst_interval = {"start_ns": min(starts), "end_ns": max(ends)}

    status = REQUEST_STATUS_COMPLETE if (all_complete and timing_valid) else "failed"

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
        run_mode=run_mode,
        victim_phase_start_ns=victim_phase_start_ns,
        queue_timing_summary=_build_queue_timing_summary(victim_results),
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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
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
        return {"available": False, "error_type": type(exc).__name__, "error": _redact_secret(str(exc), api_key)}
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
    try:
        text = run_server_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ServerLifecycleError(f"could not read run_server.sh: {exc}") from exc
    required_fragments = (
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        'MAX_NUM_BATCHED_TOKENS="2048"',
        "--cpu-offload-gb",
        "llama)",
        "qwen)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ServerLifecycleError(
            f"run_server.sh is missing required frozen-contract fragment(s): {missing}"
        )
    if "--api-key" in text:
        raise ServerLifecycleError(
            "run_server.sh must not place the API key on the process command line"
        )


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
# kernel, git commit, and explicit SHA-256 hashes of the six frozen
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
    interpreter/library versions, and hashes the six frozen files. Never
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
            "run_prefill_trigger_sweep.py": SCRIPT_PATH,
            "run_prefill_trigger_sweep.sh": SCRIPT_PATH.parent / "run_prefill_trigger_sweep.sh",
            "run_server.sh": SCRIPT_PATH.parent / "run_server.sh",
            "prefill_trigger_sweep_schedule.json": schedule_dir / "prefill_trigger_sweep_schedule.json",
            "prefill_trigger_sweep_schedule.csv": schedule_dir / "prefill_trigger_sweep_schedule.csv",
            "prefill_trigger_sweep_schedule_audit.txt": schedule_dir / "prefill_trigger_sweep_schedule_audit.txt",
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
                "run_prefill_trigger_sweep.py": "a" * 64,
                "run_prefill_trigger_sweep.sh": "b" * 64,
                "run_server.sh": "c" * 64,
                "prefill_trigger_sweep_schedule.json": "d" * 64,
                "prefill_trigger_sweep_schedule.csv": "e" * 64,
                "prefill_trigger_sweep_schedule_audit.txt": "0" * 64,
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
        "run_prefill_trigger_sweep.py",
        "run_prefill_trigger_sweep.sh",
        "run_server.sh",
        "prefill_trigger_sweep_schedule.json",
        "prefill_trigger_sweep_schedule.csv",
        "prefill_trigger_sweep_schedule_audit.txt",
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
            f"file_hashes must contain exactly the six expected files: "
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
    _eq("trigger_after_decode_tokens", find_block(bundle, block_id)[0].trigger_after_decode_tokens)
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
    model_key = block_episodes[0].model_key
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
    precondition logic for exactly one block's 2 episodes. Identical for
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
    model_key = block_episodes[0].model_key
    offload_gb = block_episodes[0].offload_gb
    state_label = block_episodes[0].state_label
    model_full_id = MODEL_REGISTRY[model_key]["model_id"]

    result: dict[str, Any] = {
        "server_start": None,
        "readiness": None,
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

        if should_abort is not None and should_abort():
            # Checked again immediately before the actual server start --
            # a signal may have arrived while resolving the port/command.
            result["overall_status"] = "interrupted"
            return result

        handle = server_adapter.start(cmd, log_path)
        result["server_start"] = {
            "cmd": cmd, "pid": handle.pid, "pgid": handle.pgid, "start_utc": handle.start_utc,
        }

        base_url = f"http://{host}:{port}"
        readiness_info = await wait_for_server_ready(
            transport, handle, base_url, api_key, model_full_id, sleeper
        )
        readiness_info["capability_summary"] = await fetch_capability_summary(
            transport, base_url, api_key
        )
        result["readiness"] = readiness_info

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
                run_mode=run_mode,
            )
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
        result["error"] = _redact_secret(str(exc), api_key)
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
            if not stop_result.get("stop_success"):
                # A failed stop is operationally dominant even when the
                # scientific block had already failed: the next block must
                # never proceed while an old server may still be alive.
                result["status_before_server_stop_failure"] = result.get("overall_status")
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

    if resume:
        if not output_dir.exists():
            raise ServerLifecycleError(
                f"--smoke-test --resume requires an existing output directory; {output_dir} does not exist"
            )
        require_output_dir_mode_marker(output_dir, RUN_MODE_SMOKE)
    else:
        if output_dir.exists():
            existing_entries = sorted(p.name for p in output_dir.iterdir())
            if existing_entries:
                raise ServerLifecycleError(
                    f"--smoke-test without --resume requires a new or completely empty output directory; "
                    f"found entries in {output_dir}: {existing_entries}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
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
        # Genuine resume no-op: do not rewrite a summary timestamp or any
        # other byte once both scientific episode files are already valid.
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
        # integrity_manifest.json, .prefill_trigger_sweep_run_mode) or unknown ones --
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
            # Both episodes are valid_complete but this block's
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

        model_key = block_episodes[0].model_key
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
            "repeat": block_episodes[0].repeat,
            "trigger_after_decode_tokens": block_episodes[0].trigger_after_decode_tokens,
            "run_mode": run_mode,
            "schedule_fingerprint": bundle.fingerprint,
            "start_utc": block_start_utc,
            "end_utc": block_end_utc,
            "planned_episode_ids": [ep.episode_id for ep in block_episodes],
            "executed_episode_ids": protocol_result.get("executed_episode_ids") or [],
            "skipped_episode_ids": [ep.episode_id for ep in block_episodes if ep.episode_id not in executed_ids],
            "episode_statuses": dict(episode_statuses),
            "stabilization_status": protocol_result.get("stabilization_status"),
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
        prog="run_prefill_trigger_sweep.py",
        description=(
            "Prefill-Trigger-Sweep runner (frozen exploratory campaign: schedule-bundle validation, "
            "resume-depth checks, self-test, dry-run, real execution of a "
            "single selectable smoke-test block, and the complete "
            "official 12-block/24-episode screening campaign). --official-run is "
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
        help="Really run the complete frozen screening campaign (12 server blocks, "
        "24 regular episodes) in exact schedule order: strict environment "
        "gate, per-block server start/readiness/stabilization/health/"
        "cooldown/episodes/verified stop, atomic manifest/summary/"
        "block-summary output, and a final self-verifying integrity "
        "manifest on genuine completion.",
    )
    mode_group.add_argument(
        "--smoke-test", action="store_true",
        help="Really execute exactly one --smoke-block: server start, "
        "readiness, one stabilization run, cooldown, its two regular "
        "episodes, and a targeted server stop.",
    )
    parser.add_argument(
        "--model-key", type=str, choices=sorted(MODEL_REGISTRY), default=None,
        help="Which model's frozen bundle/registry entry to use. Required "
        "for --dry-run, --official-run, and --smoke-test; not used by "
        "--self-test (which exercises both models internally).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Only valid with --official-run or --smoke-test: resume from "
        "existing result files after a strict environment-fingerprint and "
        "resume-depth validation; never silently overwrites anything.",
    )
    parser.add_argument(
        "--schedule-dir", type=Path, default=None,
        help=f"Bundle directory containing {list(REQUIRED_BUNDLE_FILENAMES)} "
        f"(default: new/runs/prefill_trigger_sweep/<model-key>).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Result output directory for --official-run/--smoke-test "
        "(default: new/runs/prefill_trigger_sweep/<model-key>/results/<mode>). "
        "Without --resume this directory must not exist "
        "or must be completely empty.",
    )
    parser.add_argument(
        "--smoke-block", type=str, default=None,
        help="Required with --smoke-test (and disallowed otherwise): the "
        "exact block_id to execute, e.g. 'llama_block01_low_trigger1'.",
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
    if not args.self_test and args.model_key is None:
        parser.error("--model-key is required (except with --self-test)")
    if args.model_key is not None and args.schedule_dir is None:
        args.schedule_dir = default_schedule_dir(args.model_key)
    return args




# ============================================================================
# Self-test
# ============================================================================
#

# ============================================================================
# Self-test (--self-test): focused no-GPU/no-network checks
# ============================================================================

def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    print("Prefill-Trigger-Sweep self-test (no GPU, no network)")
    print("=" * 60)

    # 1/2/3: token-batch vs SSE-event counting.
    ev = asyncio.Event()
    crossing = _ActiveWaveCrossing(request_index=0)
    cb = make_threshold_callback(0, 1, crossing, ev)
    cb(1, 1000)
    check("trigger=1 fires on first single-token batch", ev.is_set() and crossing.received_token_count_at_crossing == 1)

    ev2 = asyncio.Event()
    crossing2 = _ActiveWaveCrossing(request_index=0)
    cb2 = make_threshold_callback(0, 16, crossing2, ev2)
    for cum in (1, 5, 10, 16):
        cb2(cum, cum * 100)
    check("trigger=16 fires exactly at cumulative count 16", ev2.is_set() and crossing2.received_token_count_at_crossing == 16)

    ev3 = asyncio.Event()
    crossing3 = _ActiveWaveCrossing(request_index=0)
    cb3 = make_threshold_callback(0, 32, crossing3, ev3)
    cb3(20, 100)  # one SSE event, multiple tokens -> jumps straight past nothing yet
    check("trigger=32 not yet fired at cumulative 20 (single multi-token batch)", not ev3.is_set())
    cb3(35, 200)  # a single batch that jumps the cumulative count past 32 in one step
    check(
        "trigger=32 fires on the batch that crosses 32, not per-SSE-event",
        ev3.is_set() and crossing3.received_token_count_at_crossing == 35,
    )

    # 4: trigger=1 matches old first-token semantics (fires on the very
    # first non-empty batch regardless of its size).
    ev4 = asyncio.Event()
    crossing4 = _ActiveWaveCrossing(request_index=0)
    cb4 = make_threshold_callback(0, 1, crossing4, ev4)
    cb4(3, 500)
    check("trigger=1 fires on the first batch even if it has >1 tokens", ev4.is_set())

    # 8: crossing-skew computation.
    c_a, c_b = _ActiveWaveCrossing(0, threshold_crossing_ns=1_000_000), _ActiveWaveCrossing(1, threshold_crossing_ns=1_250_000)
    skew_ms = (max(c_a.threshold_crossing_ns, c_b.threshold_crossing_ns) - min(c_a.threshold_crossing_ns, c_b.threshold_crossing_ns)) / 1e6
    check("trigger_crossing_skew_ms computed correctly", abs(skew_ms - 0.25) < 1e-9)

    # 9/10/11: watch_trigger behavior via a minimal asyncio harness.
    async def _watch_ok_case() -> str:
        events = {0: asyncio.Event(), 1: asyncio.Event()}

        async def _immediate_complete(i: int) -> dict:
            events[i].set()
            await asyncio.sleep(0)
            return {"status": REQUEST_STATUS_COMPLETE, "request_index": i}

        tasks = {i: asyncio.create_task(_immediate_complete(i)) for i in events}
        await asyncio.sleep(0)
        status = await _watch_trigger({0, 1}, events, tasks, timeout_s=5.0)
        await asyncio.gather(*tasks.values())
        return status

    async def _watch_pretrigger_failure_case() -> str:
        events = {0: asyncio.Event(), 1: asyncio.Event()}

        async def _fails_before_threshold(i: int) -> dict:
            if i == 1:
                return {"status": REQUEST_STATUS_FAILED, "request_index": i}
            events[i].set()
            await asyncio.sleep(0.01)
            return {"status": REQUEST_STATUS_COMPLETE, "request_index": i}

        tasks = {i: asyncio.create_task(_fails_before_threshold(i)) for i in events}
        status = await _watch_trigger({0, 1}, events, tasks, timeout_s=5.0)
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        return status

    check("global trigger waits for + resolves once all active-wave requests cross", asyncio.run(_watch_ok_case()) == "ok")
    check("request failing before its own threshold aborts the barrier (pretrigger_failure)", asyncio.run(_watch_pretrigger_failure_case()) == "pretrigger_failure")

    # 14/15/16/17: --model-key <-> bundle matching.
    llama_payload = build_canonical_payload_for_selftest("llama")
    qwen_payload = build_canonical_payload_for_selftest("qwen")
    llama_errors_same = check_official_contract(llama_payload, "llama")
    llama_errors_cross = check_official_contract(llama_payload, "qwen")
    qwen_errors_same = check_official_contract(qwen_payload, "qwen")
    check(
        "llama contract check under --model-key llama has no model/seed/schema mismatch",
        not any("model_key" in e or "model_id" in e or "seed =" in e for e in llama_errors_same),
    )
    check("llama bundle rejected under --model-key qwen (more errors than same-model)", len(llama_errors_cross) > len(llama_errors_same))
    check(
        "qwen contract check under --model-key qwen has no model/seed/schema mismatch",
        not any("model_key" in e or "model_id" in e or "seed =" in e for e in qwen_errors_same),
    )
    check("unknown --model-key hard-fails", bool(check_official_contract(llama_payload, "does-not-exist")))

    # 18/19: fingerprints and seed domains differ between models.
    check("llama and qwen have distinct frozen fingerprints", OFFICIAL_FINGERPRINTS["llama"] != OFFICIAL_FINGERPRINTS["qwen"])
    llama_eps = [parse_episode_from_json(e) for e in llama_payload["episodes"]]
    qwen_eps = [parse_episode_from_json(e) for e in qwen_payload["episodes"]]
    llama_seeds = {e.victim_workload_seed for e in llama_eps} | {e.burst_workload_seed for e in llama_eps}
    qwen_seeds = {e.victim_workload_seed for e in qwen_eps} | {e.burst_workload_seed for e in qwen_eps}
    check("victim/burst workload seeds are disjoint between llama and qwen", llama_seeds.isdisjoint(qwen_seeds))

    # Structural checks reused directly (also covers items 5-7, 12-13 at
    # the schedule level: trigger set, block/condition-first structure).
    check("llama schedule passes full structural validation", not check_structural_schedule(llama_eps, llama_payload["seed"], "llama"))
    check("qwen schedule passes full structural validation", not check_structural_schedule(qwen_eps, qwen_payload["seed"], "qwen"))

    # 23: no API key ever appears in a serialized bundle payload.
    check(
        "no 'api_key'-like field present in the canonical schedule payload",
        "api_key" not in json.dumps(llama_payload).lower() and "vllm_api_key" not in json.dumps(llama_payload).lower(),
    )

    print("=" * 60)
    if failures:
        print(f"FAIL: {len(failures)} self-test(s) failed: {failures}")
        return 1
    print("PASS: all self-tests passed.")
    print(
        "NOTE: this focused self-test suite covers the new trigger-barrier "
        "semantics and dual-model bundle contract (offline, no GPU/network). "
        "It is not a substitute for a full line-by-line review of the "
        "lifecycle/resume/integrity machinery mechanically inherited from "
        "Prefill-Screen before the first real --smoke-test."
    )
    return 0


def build_canonical_payload_for_selftest(model_key: str) -> dict:
    """Regenerates an in-memory canonical payload for --self-test using
    this module's own frozen constants (NOT the generator, which is
    never imported here) -- just enough structure for
    check_official_contract/check_structural_schedule to exercise
    against, without touching disk."""
    seed = OFFICIAL_SEED_BY_MODEL[model_key]
    model_id = MODEL_REGISTRY[model_key]["model_id"]
    trigger_rotation = TRIGGER_ROTATION_BY_REPEAT
    episodes: list[dict] = []
    block_number = 0
    for repeat in (1, 2):
        for trigger in trigger_rotation[repeat]:
            for offload_gb, state_label in ((0, "low"), (12, "high")):
                block_number += 1
                block_id = f"{model_key}_block{block_number:02d}_{state_label}_trigger{trigger}"
                cell_base = derive_seed(str(seed), model_key, state_label, str(trigger), "condition_order") % 2
                r1_no_burst_first = cell_base == 0
                no_burst_first = r1_no_burst_first if repeat == 1 else not r1_no_burst_first
                order = ["no_burst", BURST_CONDITION] if no_burst_first else [BURST_CONDITION, "no_burst"]
                victim_seed = derive_seed(str(seed), model_key, "victim", str(repeat))
                burst_seed = derive_seed(str(seed), model_key, "burst", str(repeat))
                for pos, condition in enumerate(order, start=1):
                    episode_id = f"{model_key}_off{offload_gb}_conc4_trigger{trigger}_{condition}_rep{repeat}"
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "model_key": model_key,
                            "model_id": model_id,
                            "offload_gb": offload_gb,
                            "state_label": state_label,
                            "concurrency": 4,
                            "trigger_after_decode_tokens": trigger,
                            "condition": condition,
                            "repeat": repeat,
                            "random_seed": seed,
                            "episode_seed": derive_seed(str(seed), episode_id),
                            "victim_workload_seed": victim_seed,
                            "burst_workload_seed": burst_seed,
                            "victim_request_count": 20,
                            "victim_input_len": 256,
                            "victim_output_len": 64,
                            "victim_temperature": 0.0,
                            "burst_parallel_requests": 4,
                            "burst_input_len": 2048,
                            "burst_output_len": 16,
                            "burst_temperature": 0.0,
                            "max_num_batched_tokens": 2048,
                            "condition_first_in_block": order[0],
                            "restart_server_before_block": 1 if pos == 1 else 0,
                            "block_id": block_id,
                            "order_in_block": pos,
                        }
                    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "repeats": OFFICIAL_REPEATS,
        "model_key": model_key,
        "model_id": model_id,
        "states": OFFICIAL_STATES,
        "concurrency": OFFICIAL_CONCURRENCY,
        "trigger_positions": OFFICIAL_TRIGGER_POSITIONS,
        "trigger_rotation_by_repeat": {str(r): list(t) for r, t in TRIGGER_ROTATION_BY_REPEAT.items()},
        "conditions": OFFICIAL_CONDITIONS,
        "max_num_batched_tokens": OFFICIAL_MAX_NUM_BATCHED_TOKENS,
        "victim_configuration": OFFICIAL_VICTIM_CONFIGURATION,
        "burst_configuration": OFFICIAL_BURST_CONFIGURATION,
        "stabilization_configuration": OFFICIAL_STABILIZATION_CONFIGURATION,
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    fp = recompute_fingerprint(payload)
    payload["schedule_fingerprint"] = fp
    return payload




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
        return run_self_test()

    assert args.model_key is not None  # enforced in parse_args
    bundle, errors = load_and_validate_bundle(args.schedule_dir, args.model_key)
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
            args.output_dir if args.output_dir is not None else default_output_dir(args.model_key, mode)
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
            return HFTokenizerAdapter(MODEL_REGISTRY[model_key]["model_id"])

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
        args.output_dir if args.output_dir is not None else default_output_dir(args.model_key, mode)
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

    block_model = find_block(bundle, args.smoke_block)[0].model_key
    try:
        transport: HTTPTransport = HttpxTransport()
        tokenizer: TokenizerAdapter = HFTokenizerAdapter(MODEL_REGISTRY[block_model]["model_id"])
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
