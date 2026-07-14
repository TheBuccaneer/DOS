#!/usr/bin/env python3
"""
run_phase_a.py -- Phase A runner, Stage 1 of 3.

Stage 1 scope: schedule-bundle validation (structural + deterministic +
canonical fingerprint), CLI, resume-depth validation of already-existing
episode result files. Stage 1 explicitly does NOT implement: real server
start/stop, HTTP/streaming requests, stabilization execution, GPU
measurement, real episode execution, or signal handling of running
requests. --official-run and --smoke-test fully validate everything up
to (but not including) real execution, then abort with a clear message.

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
    --self-test     Exercises this runner's own validation logic against
                     small synthetic fixtures it builds itself. Does not
                     require --schedule-dir. Fully implemented.
    --dry-run       Loads and fully validates the given --schedule-dir
                     bundle, prints the resulting execution plan, and
                     exits. Opens no network connection, starts no
                     server, loads no tokenizer, writes no result files.
                     Fully implemented.
    --official-run  Same full bundle validation as --dry-run, plus
                     output-dir preparation and (if --resume) a resume
                     scan of already-existing result files, then aborts
                     with "Real execution is not implemented in stage 1."
    --smoke-test    Same as --official-run, but targets a separate
                     results/smoke output directory. official-run and
                     smoke-test must never share an output directory
                     (enforced via a small marker file).

--resume is only valid together with --official-run or --smoke-test.

The API key (VLLM_API_KEY) is deliberately never read anywhere in this
stage -- there is nothing in stage 1 that needs it, and this guarantees
it cannot leak into --dry-run or --self-test output.

Usage:
    python3 run_phase_a.py --self-test
    python3 run_phase_a.py --dry-run --schedule-dir /path/to/runs/phase_a
    python3 run_phase_a.py --official-run --schedule-dir /path/to/runs/phase_a \\
        --output-dir /path/to/runs/phase_a/results/official
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


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

RESULT_SCHEMA_VERSION = 1
RUNNER_VERSION = "run_phase_a-stage1"
RECORD_TYPE_REGULAR_EPISODE = "regular_episode"

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

    Note: request-level (prompt/output token count) depth validation is
    added in Stage 2; `victim_requests`/`burst_requests` are checked here
    only for the correct *count*, per condition.
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

    if notes:
        return CLASSIFICATION_INVALID, notes

    return CLASSIFICATION_VALID_COMPLETE, []


def scan_existing_results(
    output_dir: Path, bundle: LoadedBundle, run_mode: str
) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for ep in bundle.episodes:
        result_path = output_dir / f"{ep.episode_id}.json"
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
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase_a.py",
        description=(
            "Phase A runner (stage 1: schedule-bundle validation, "
            "resume-depth checks, self-test, dry-run). Real execution is "
            "not yet implemented in this stage."
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
        help="Validate everything for a real official run. Real "
        "execution is not implemented in stage 1.",
    )
    mode_group.add_argument(
        "--smoke-test", action="store_true",
        help="Validate everything for a real smoke-test run. Real "
        "execution is not implemented in stage 1.",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.resume and not (args.official_run or args.smoke_test):
        parser.error(
            "--resume is only valid together with --official-run or "
            "--smoke-test"
        )
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

        def make_valid_result(ep: Episode) -> dict:
            return {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "run_mode": "smoke",
                "schedule_fingerprint": expected_fp,
                "episode_id": ep.episode_id,
                "schedule_row": asdict(ep),
                "record_type": RECORD_TYPE_REGULAR_EPISODE,
                "status": "complete",
                "victim_requests": [{}] * ep.victim_request_count,
                "burst_requests": [{}] * (
                    ep.burst_parallel_requests if ep.condition == "fixed_burst" else 0
                ),
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
        # ep0's own expected file actually holds ep1's (valid) result --
        # must show up as 'invalid' for ep0, and ep1 itself must still be
        # correctly reported as 'missing' (not silently matched/skipped).
        (scan_dir / f"{ep0.episode_id}.json").write_text(
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

    # --- no API key is ever READ from the environment in this module -------
    # (Searches for actual os.environ/os.getenv access patterns, not just
    # the bare string "VLLM_API_KEY" -- which legitimately appears in this
    # module's docstring and in this very check's own source text.)
    own_source = SCRIPT_PATH.read_text(encoding="utf-8")
    env_access_patterns = [
        r'os\.environ\.get\(\s*["\']VLLM_API_KEY',
        r'os\.environ\[\s*["\']VLLM_API_KEY',
        r'os\.getenv\(\s*["\']VLLM_API_KEY',
    ]
    found_env_access = any(re.search(p, own_source) for p in env_access_patterns)
    check(
        "VLLM_API_KEY is never read from the environment anywhere in this module",
        not found_env_access,
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

    mode = "official" if args.official_run else "smoke"
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

    print()
    print("Real execution is not implemented in stage 1.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
