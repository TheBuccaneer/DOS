#!/usr/bin/env python3
"""
analyze_phase_a.py

Scientific analysis for the official Phase-A LLM-serving availability dataset.

Primary estimand (per model and victim concurrency):
    episode-level victim median TPOT

For each state s in {low, high}:
    M_no(s)    = median across no_burst episodes of the episode metric
    M_burst(s) = median across fixed_burst episodes of the episode metric

For latency-like metrics:
    log_effect(s) = log(M_burst(s) / M_no(s))

For goodput (higher is better), the direction is aligned so positive means
availability degradation:
    log_effect(s) = log(M_no(s) / M_burst(s))

The normalized state x burst interaction is:
    normalized_log_DiD = log_effect(high) - log_effect(low)

and:
    exp(normalized_log_DiD)

is the ratio of state-specific degradation factors. Values > 1 mean that the
high-offload state is relatively more sensitive to the same bounded burst;
values < 1 mean that the low-offload state is relatively more sensitive.

Bootstrap:
    - resamples EPISODES, never individual requests;
    - independently resamples the four episode groups
      low/no_burst, low/fixed_burst, high/no_burst, high/fixed_burst;
    - uses percentile 95% confidence intervals;
    - is deterministic for a fixed --seed.

The script accepts either:
    1. the official result directory, or
    2. a ZIP containing exactly one official result directory.

Outputs:
    episode_metrics.csv
    cell_summary.csv
    log_did_results.csv
    additive_did_results.csv
    model_level_log_did.csv
    leave_one_repeat_out.csv
    analysis_report.json

Dependencies:
    Python >= 3.10
    numpy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "ERROR: numpy is required. Install it in the analysis environment, "
        "for example: python3 -m pip install numpy"
    ) from exc


SCRIPT_VERSION = "phase-a-analysis-v2"
EXPECTED_EPISODES = 80
EXPECTED_REPEATS_PER_CELL = 5
EXPECTED_STATES = ("low", "high")
EXPECTED_CONDITIONS = ("no_burst", "fixed_burst")
EXPECTED_CONCURRENCIES = (4, 8)
EXPECTED_OFFLOAD = {"low": 0, "high": 12}
DEFAULT_BOOTSTRAP_ITERATIONS = 100_000
DEFAULT_BOOTSTRAP_SEED = 20260715
DEFAULT_PRACTICAL_RATIO = 1.5
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-8

# Positive log-effect always means degradation.
LOG_METRICS: dict[str, str] = {
    "victim_median_tpot_ms": "latency",
    "victim_median_itl_ms": "latency",
    "victim_p95_itl_ms": "latency",
    "victim_itl_iqr_ms": "latency",
    "victim_median_ttft_ms": "latency",
    "victim_p95_ttft_ms": "latency",
    "victim_median_e2el_ms": "latency",
    "victim_p95_e2el_ms": "latency",
    "victim_goodput_tokens_per_s": "goodput",
}

ITL_METRICS = {
    "victim_median_itl_ms",
    "victim_p95_itl_ms",
    "victim_itl_iqr_ms",
}

ADDITIVE_METRICS: dict[str, str] = {
    "victim_success_rate": "higher_is_better",
    "victim_timeout_rate": "higher_is_worse",
    "victim_failure_rate": "higher_is_worse",
}


class AnalysisError(RuntimeError):
    """Raised when the dataset violates an analysis precondition."""


@dataclass(frozen=True)
class DatasetContext:
    root: Path
    temporary_root: Path | None


@dataclass(frozen=True)
class BootstrapLogResult:
    point_log_did: float
    did_samples: np.ndarray
    low_effect_samples: np.ndarray
    high_effect_samples: np.ndarray


@dataclass(frozen=True)
class BootstrapAdditiveResult:
    point_did: float
    did_samples: np.ndarray
    low_effect_samples: np.ndarray
    high_effect_samples: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the official Phase-A result directory/ZIP using "
            "episode-level log-normalized Difference-in-Differences."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Official result directory or ZIP archive containing it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase_a_analysis"),
        help="Directory for CSV/JSON outputs (default: phase_a_analysis).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help=f"Bootstrap iterations (default: {DEFAULT_BOOTSTRAP_ITERATIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Deterministic bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--practical-ratio",
        type=float,
        default=DEFAULT_PRACTICAL_RATIO,
        help=(
            "Exploratory practical threshold for the ratio of degradation "
            f"factors (default: {DEFAULT_PRACTICAL_RATIO})."
        ),
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        default=None,
        help="Optional TTFT SLO threshold; adds per-episode violation rates.",
    )
    parser.add_argument(
        "--e2el-slo-ms",
        type=float,
        default=None,
        help="Optional E2EL SLO threshold; adds per-episode violation rates.",
    )
    args = parser.parse_args(argv)

    if args.bootstrap < 1:
        parser.error("--bootstrap must be >= 1")
    if args.practical_ratio <= 1.0:
        parser.error("--practical-ratio must be > 1")
    for name in ("ttft_slo_ms", "e2el_slo_ms"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise AnalysisError(f"Expected JSON object in {path}")
    return obj


def _safe_zip_member(member: zipfile.ZipInfo) -> None:
    name = member.filename
    if not name or "\x00" in name:
        raise AnalysisError(f"Unsafe ZIP member name: {name!r}")
    normalized = Path(name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise AnalysisError(f"Unsafe ZIP path: {name!r}")
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise AnalysisError(f"ZIP symlink is not allowed: {name!r}")


def _locate_dataset_root(base: Path) -> Path:
    if (base / "official_run_summary.json").is_file():
        return base
    candidates = list(base.rglob("official_run_summary.json"))
    if len(candidates) != 1:
        raise AnalysisError(
            "Expected exactly one official_run_summary.json; found "
            f"{len(candidates)} under {base}"
        )
    return candidates[0].parent


@contextmanager
def open_dataset(input_path: Path) -> Iterator[DatasetContext]:
    input_path = input_path.resolve()
    if input_path.is_dir():
        yield DatasetContext(root=_locate_dataset_root(input_path), temporary_root=None)
        return
    if not input_path.is_file():
        raise AnalysisError(f"Input does not exist: {input_path}")
    if input_path.suffix.lower() != ".zip":
        raise AnalysisError("Input must be a directory or a .zip archive")

    tmp = Path(tempfile.mkdtemp(prefix="phase_a_analysis_"))
    try:
        with zipfile.ZipFile(input_path) as archive:
            for member in archive.infolist():
                _safe_zip_member(member)
            archive.extractall(tmp)
        yield DatasetContext(root=_locate_dataset_root(tmp), temporary_root=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_integrity(root: Path) -> dict[str, Any]:
    manifest_path = root / "integrity_manifest.json"
    manifest = load_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise AnalysisError("integrity_manifest.json: 'files' is not a list")

    manifest_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AnalysisError(f"integrity files[{index}] is not an object")
        rel = entry.get("relative_path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(rel, str) or not rel:
            raise AnalysisError(f"integrity files[{index}] has invalid relative_path")
        if rel in manifest_by_path:
            raise AnalysisError(f"Duplicate integrity path: {rel}")
        if type(size) is not int or size < 0:
            raise AnalysisError(f"Invalid size for integrity path: {rel}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise AnalysisError(f"Invalid SHA-256 for integrity path: {rel}")
        manifest_by_path[rel] = entry

    actual_by_path = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "integrity_manifest.json"
    }
    if set(manifest_by_path) != set(actual_by_path):
        missing = sorted(set(manifest_by_path) - set(actual_by_path))
        extra = sorted(set(actual_by_path) - set(manifest_by_path))
        raise AnalysisError(
            f"Integrity file-set mismatch; missing={missing}, extra={extra}"
        )

    for rel, entry in manifest_by_path.items():
        path = actual_by_path[rel]
        if path.stat().st_size != entry["size_bytes"]:
            raise AnalysisError(f"Integrity size mismatch: {rel}")
        if sha256_file(path) != entry["sha256"]:
            raise AnalysisError(f"Integrity SHA-256 mismatch: {rel}")

    if manifest.get("file_count") != len(entries):
        raise AnalysisError("Integrity file_count does not match entries")

    summary = load_json(root / "official_run_summary.json")
    manifest_schedule = manifest.get("schedule_fingerprint")
    summary_schedule = summary.get("schedule_fingerprint")
    manifest_environment = manifest.get("environment_fingerprint")
    summary_environment = summary.get("environment_fingerprint")
    if not isinstance(manifest_schedule, str) or not manifest_schedule:
        raise AnalysisError("Integrity manifest has no valid schedule_fingerprint")
    if not isinstance(summary_schedule, str) or not summary_schedule:
        raise AnalysisError("Official summary has no valid schedule_fingerprint")
    if manifest_schedule != summary_schedule:
        raise AnalysisError(
            "Schedule fingerprint mismatch between integrity manifest and official summary"
        )
    if not isinstance(manifest_environment, str) or not manifest_environment:
        raise AnalysisError("Integrity manifest has no valid environment_fingerprint")
    if not isinstance(summary_environment, str) or not summary_environment:
        raise AnalysisError("Official summary has no valid environment_fingerprint")
    if manifest_environment != summary_environment:
        raise AnalysisError(
            "Environment fingerprint mismatch between integrity manifest and official summary"
        )

    return {
        "verified": True,
        "file_count": len(entries),
        "schedule_fingerprint": manifest_schedule,
        "environment_fingerprint": manifest_environment,
        "summary_fingerprints_match_manifest": True,
    }


def percentile(values: Iterable[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise AnalysisError("Cannot calculate percentile of an empty sequence")
    return float(np.quantile(arr, p, method="linear"))


def median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise AnalysisError("Cannot calculate median of an empty sequence")
    return float(np.median(arr))


def numeric_close(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
    except (TypeError, ValueError):
        return actual == expected


def compare_stored_aggregate(
    episode_id: str,
    aggregate: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for path, expected_value in expected.items():
        current: Any = aggregate
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise AnalysisError(f"{episode_id}: aggregate is missing {path}")
            current = current[part]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not numeric_close(current, expected_value):
                raise AnalysisError(
                    f"{episode_id}: aggregate mismatch for {path}: "
                    f"stored={current!r}, recomputed={expected_value!r}"
                )
        elif current != expected_value:
            raise AnalysisError(
                f"{episode_id}: aggregate mismatch for {path}: "
                f"stored={current!r}, recomputed={expected_value!r}"
            )


def _request_values(requests: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for request in requests:
        value = request.get(key)
        if value is None:
            raise AnalysisError(
                f"Request {request.get('request_id', '<unknown>')} has null {key}"
            )
        values.append(float(value))
    return values


def derive_episode_metrics(
    episode: dict[str, Any],
    *,
    expected_schedule_fingerprint: str,
    expected_environment_fingerprint: str,
    ttft_slo_ms: float | None,
    e2el_slo_ms: float | None,
) -> dict[str, Any]:
    episode_id = episode.get("episode_id")
    if not isinstance(episode_id, str):
        raise AnalysisError("Episode without a valid episode_id")
    if episode.get("record_type") != "regular_episode":
        raise AnalysisError(f"{episode_id}: record_type is not regular_episode")
    if episode.get("run_mode") != "official":
        raise AnalysisError(f"{episode_id}: run_mode is not official")
    if episode.get("status") != "complete":
        raise AnalysisError(f"{episode_id}: episode status is not complete")
    if episode.get("validation_errors") not in ([], None):
        raise AnalysisError(f"{episode_id}: episode validation_errors is not empty")

    episode_schedule_fingerprint = episode.get("schedule_fingerprint")
    if episode_schedule_fingerprint != expected_schedule_fingerprint:
        raise AnalysisError(
            f"{episode_id}: schedule_fingerprint does not match the official campaign"
        )
    episode_environment_fingerprint = episode.get("environment_fingerprint")
    if (
        episode_environment_fingerprint is not None
        and episode_environment_fingerprint != expected_environment_fingerprint
    ):
        raise AnalysisError(
            f"{episode_id}: environment_fingerprint does not match the official campaign"
        )

    schedule = episode.get("schedule_row")
    if not isinstance(schedule, dict):
        raise AnalysisError(f"{episode_id}: schedule_row is not an object")
    if schedule.get("episode_id") != episode_id:
        raise AnalysisError(f"{episode_id}: schedule_row episode_id mismatch")

    model = schedule.get("model")
    state = schedule.get("state_label")
    offload = schedule.get("offload_gb")
    concurrency = schedule.get("concurrency")
    condition = schedule.get("condition")
    repeat = schedule.get("repeat")
    if state not in EXPECTED_STATES:
        raise AnalysisError(f"{episode_id}: unexpected state {state!r}")
    if offload != EXPECTED_OFFLOAD[state]:
        raise AnalysisError(f"{episode_id}: offload/state mismatch")
    if concurrency not in EXPECTED_CONCURRENCIES:
        raise AnalysisError(f"{episode_id}: unexpected concurrency {concurrency!r}")
    if condition not in EXPECTED_CONDITIONS:
        raise AnalysisError(f"{episode_id}: unexpected condition {condition!r}")
    if type(repeat) is not int or not 1 <= repeat <= EXPECTED_REPEATS_PER_CELL:
        raise AnalysisError(f"{episode_id}: invalid repeat {repeat!r}")

    victim_requests = episode.get("victim_requests")
    burst_requests = episode.get("burst_requests")
    if not isinstance(victim_requests, list) or len(victim_requests) != 20:
        raise AnalysisError(f"{episode_id}: expected exactly 20 victim requests")
    expected_burst = 4 if condition == "fixed_burst" else 0
    if not isinstance(burst_requests, list) or len(burst_requests) != expected_burst:
        raise AnalysisError(
            f"{episode_id}: expected exactly {expected_burst} burst requests"
        )

    for request in victim_requests + burst_requests:
        if not isinstance(request, dict):
            raise AnalysisError(f"{episode_id}: request record is not an object")
        if request.get("status") != "complete":
            raise AnalysisError(
                f"{episode_id}: request {request.get('request_id')} is not complete"
            )
        if request.get("validation_errors") not in ([], None):
            raise AnalysisError(
                f"{episode_id}: request {request.get('request_id')} has validation errors"
            )

    tpots = _request_values(victim_requests, "client_observed_tpot_ms")
    ttfts = _request_values(victim_requests, "ttft_ms")
    e2els = _request_values(victim_requests, "e2el_ms")

    itls: list[float] = []
    itl_available_request_count = 0
    for request in victim_requests:
        values = request.get("itl_ms")
        if request.get("itl_available") is True and isinstance(values, list):
            itl_available_request_count += 1
            itls.extend(float(value) for value in values)
    victim_itl_complete = itl_available_request_count == len(victim_requests) and bool(itls)
    if not victim_itl_complete:
        # ITL is all-or-nothing at episode level. Partial request coverage is
        # recorded, but no potentially selective partial ITL statistic is used.
        itls = []

    complete_output_tokens = sum(len(request.get("output_token_ids") or []) for request in victim_requests)
    start_ns = min(int(request["request_start_ns"]) for request in victim_requests)
    end_ns = max(int(request["stream_end_ns"]) for request in victim_requests)
    span_s = (end_ns - start_ns) / 1e9
    if span_s <= 0:
        raise AnalysisError(f"{episode_id}: non-positive victim span")
    goodput = complete_output_tokens / span_s

    success_count = sum(request.get("status") == "complete" for request in victim_requests)
    timeout_count = sum(request.get("timed_out") is True for request in victim_requests)
    failure_count = sum(request.get("status") != "complete" for request in victim_requests)

    row: dict[str, Any] = {
        "episode_id": episode_id,
        "block_id": schedule.get("block_id"),
        "model": model,
        "state_label": state,
        "offload_gb": offload,
        "concurrency": concurrency,
        "condition": condition,
        "repeat": repeat,
        "order_in_block": schedule.get("order_in_block"),
        "victim_workload_seed": schedule.get("victim_workload_seed"),
        "burst_workload_seed": schedule.get("burst_workload_seed"),
        "victim_median_tpot_ms": median(tpots),
        "victim_p95_tpot_ms": percentile(tpots, 0.95),
        "victim_itl_complete": victim_itl_complete,
        "victim_itl_available_request_count": itl_available_request_count,
        "victim_median_itl_ms": median(itls) if victim_itl_complete else None,
        "victim_p95_itl_ms": percentile(itls, 0.95) if victim_itl_complete else None,
        "victim_itl_iqr_ms": (
            percentile(itls, 0.75) - percentile(itls, 0.25)
            if victim_itl_complete
            else None
        ),
        "victim_median_ttft_ms": median(ttfts),
        "victim_p95_ttft_ms": percentile(ttfts, 0.95),
        "victim_median_e2el_ms": median(e2els),
        "victim_p95_e2el_ms": percentile(e2els, 0.95),
        "victim_goodput_tokens_per_s": goodput,
        "victim_success_rate": success_count / len(victim_requests),
        "victim_timeout_rate": timeout_count / len(victim_requests),
        "victim_failure_rate": failure_count / len(victim_requests),
        "victim_complete_output_tokens": complete_output_tokens,
        "burst_request_count": len(burst_requests),
        "burst_output_tokens": sum(len(request.get("output_token_ids") or []) for request in burst_requests),
        "trigger_waited_ms": float((episode.get("trigger") or {}).get("waited_ms", math.nan)),
    }

    if ttft_slo_ms is not None:
        row["victim_ttft_slo_violation_rate"] = sum(value > ttft_slo_ms for value in ttfts) / len(ttfts)
    if e2el_slo_ms is not None:
        row["victim_e2el_slo_violation_rate"] = sum(value > e2el_slo_ms for value in e2els) / len(e2els)

    aggregate = episode.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        raise AnalysisError(f"{episode_id}: aggregate_metrics is missing")
    expected_aggregate = {
        "victim_client_observed_tpot_ms.median": row["victim_median_tpot_ms"],
        "victim_client_observed_tpot_ms.p95": row["victim_p95_tpot_ms"],
        "victim_ttft_ms.median": row["victim_median_ttft_ms"],
        "victim_ttft_ms.p95": row["victim_p95_ttft_ms"],
        "victim_e2el_ms.median": row["victim_median_e2el_ms"],
        "victim_e2el_ms.p95": row["victim_p95_e2el_ms"],
        "victim_complete_output_tokens": complete_output_tokens,
        "victim_goodput_tokens_per_s": goodput,
        "victim_complete_request_count": success_count,
        "victim_incomplete_request_count": len(victim_requests) - success_count,
        "burst_complete_request_count": len(burst_requests),
        "burst_incomplete_request_count": 0,
        "burst_output_tokens": row["burst_output_tokens"],
    }
    if victim_itl_complete:
        expected_aggregate.update(
            {
                "victim_itl_ms.median": row["victim_median_itl_ms"],
                "victim_itl_ms.p95": row["victim_p95_itl_ms"],
                "victim_itl_ms.n": len(itls),
            }
        )
    compare_stored_aggregate(episode_id, aggregate, expected_aggregate)
    return row


def load_episode_rows(
    root: Path,
    *,
    ttft_slo_ms: float | None,
    e2el_slo_ms: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = load_json(root / "official_run_summary.json")
    if summary.get("overall_status") != "complete":
        raise AnalysisError("official_run_summary overall_status is not complete")
    if summary.get("planned_episodes") != EXPECTED_EPISODES:
        raise AnalysisError("official_run_summary planned_episodes is not 80")
    if summary.get("valid_complete_episodes") != EXPECTED_EPISODES:
        raise AnalysisError("official_run_summary valid_complete_episodes is not 80")
    if summary.get("missing_episodes") != 0:
        raise AnalysisError("official_run_summary reports missing episodes")

    episode_paths = sorted((root / "episodes").glob("*.json"))
    if len(episode_paths) != EXPECTED_EPISODES:
        raise AnalysisError(
            f"Expected {EXPECTED_EPISODES} episode files, found {len(episode_paths)}"
        )
    schedule_fingerprint = summary.get("schedule_fingerprint")
    environment_fingerprint = summary.get("environment_fingerprint")
    if not isinstance(schedule_fingerprint, str) or not schedule_fingerprint:
        raise AnalysisError("official_run_summary has no valid schedule_fingerprint")
    if not isinstance(environment_fingerprint, str) or not environment_fingerprint:
        raise AnalysisError("official_run_summary has no valid environment_fingerprint")

    rows = [
        derive_episode_metrics(
            load_json(path),
            expected_schedule_fingerprint=schedule_fingerprint,
            expected_environment_fingerprint=environment_fingerprint,
            ttft_slo_ms=ttft_slo_ms,
            e2el_slo_ms=e2el_slo_ms,
        )
        for path in episode_paths
    ]

    ids = [row["episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AnalysisError("Duplicate episode_id in dataset")

    cell_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    repeats_by_cell: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    models: set[str] = set()
    for row in rows:
        models.add(str(row["model"]))
        key = (
            row["model"],
            row["state_label"],
            row["concurrency"],
            row["condition"],
        )
        cell_counts[key] += 1
        repeats_by_cell[key].add(int(row["repeat"]))

    expected_cell_count = len(models) * len(EXPECTED_STATES) * len(EXPECTED_CONCURRENCIES) * len(EXPECTED_CONDITIONS)
    if len(cell_counts) != expected_cell_count:
        raise AnalysisError(
            f"Expected {expected_cell_count} design cells, found {len(cell_counts)}"
        )
    for key, count in sorted(cell_counts.items()):
        if count != EXPECTED_REPEATS_PER_CELL:
            raise AnalysisError(f"Cell {key} has {count} episodes, expected 5")
        if repeats_by_cell[key] != set(range(1, EXPECTED_REPEATS_PER_CELL + 1)):
            raise AnalysisError(f"Cell {key} has invalid repeat set")

    # Confirm matched victim seeds across state and condition for each model/conc/repeat.
    matched: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        key = (row["model"], row["concurrency"], row["repeat"])
        matched[key].add(int(row["victim_workload_seed"]))
    bad_matches = {key: values for key, values in matched.items() if len(values) != 1}
    if bad_matches:
        raise AnalysisError(f"Victim workload seeds are not matched: {bad_matches}")

    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matching_metric_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int,
    state: str,
    condition: str,
    metric: str,
    excluded_repeat: int | None = None,
) -> list[dict[str, Any]]:
    matching = [
        row
        for row in rows
        if row["model"] == model
        and row["concurrency"] == concurrency
        and row["state_label"] == state
        and row["condition"] == condition
        and (excluded_repeat is None or row["repeat"] != excluded_repeat)
    ]
    expected = EXPECTED_REPEATS_PER_CELL - (1 if excluded_repeat is not None else 0)
    if len(matching) != expected:
        raise AnalysisError(
            f"Expected {expected} episodes for {model}/{concurrency}/{state}/"
            f"{condition}/{metric}, found {len(matching)}"
        )
    return matching


def group_values(
    rows: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int,
    state: str,
    condition: str,
    metric: str,
    excluded_repeat: int | None = None,
) -> np.ndarray:
    matching = matching_metric_rows(
        rows,
        model=model,
        concurrency=concurrency,
        state=state,
        condition=condition,
        metric=metric,
        excluded_repeat=excluded_repeat,
    )
    missing = [row["episode_id"] for row in matching if row.get(metric) is None]
    if missing:
        raise AnalysisError(
            f"Missing episode-level values for {model}/{concurrency}/{state}/"
            f"{condition}/{metric}: {missing}"
        )
    arr = np.asarray([float(row[metric]) for row in matching], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise AnalysisError(f"Non-finite values for metric {metric}")
    return arr


def complete_metric_counts(
    rows: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int,
    metric: str,
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for state in EXPECTED_STATES:
        for condition in EXPECTED_CONDITIONS:
            matching = matching_metric_rows(
                rows,
                model=model,
                concurrency=concurrency,
                state=state,
                condition=condition,
                metric=metric,
            )
            counts[(state, condition)] = sum(row.get(metric) is not None for row in matching)
    return counts


def deterministic_rng(base_seed: int, *parts: object) -> np.random.Generator:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(derived)


def bootstrap_medians(values: np.ndarray, iterations: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    return np.median(values[indices], axis=1)


def aligned_log_effect(no_burst: np.ndarray, fixed_burst: np.ndarray, metric_kind: str) -> float:
    m_no = float(np.median(no_burst))
    m_burst = float(np.median(fixed_burst))
    if m_no <= 0 or m_burst <= 0:
        raise AnalysisError("Log-ratio metric contains a non-positive median")
    if metric_kind == "latency":
        return math.log(m_burst) - math.log(m_no)
    if metric_kind == "goodput":
        return math.log(m_no) - math.log(m_burst)
    raise AnalysisError(f"Unknown metric kind: {metric_kind}")


def bootstrap_log_did(
    groups: dict[tuple[str, str], np.ndarray],
    *,
    metric_kind: str,
    iterations: int,
    rng: np.random.Generator,
) -> BootstrapLogResult:
    low_no = groups[("low", "no_burst")]
    low_burst = groups[("low", "fixed_burst")]
    high_no = groups[("high", "no_burst")]
    high_burst = groups[("high", "fixed_burst")]

    point_low = aligned_log_effect(low_no, low_burst, metric_kind)
    point_high = aligned_log_effect(high_no, high_burst, metric_kind)
    point_did = point_high - point_low

    med_low_no = bootstrap_medians(low_no, iterations, rng)
    med_low_burst = bootstrap_medians(low_burst, iterations, rng)
    med_high_no = bootstrap_medians(high_no, iterations, rng)
    med_high_burst = bootstrap_medians(high_burst, iterations, rng)

    if np.any(med_low_no <= 0) or np.any(med_low_burst <= 0):
        raise AnalysisError("Bootstrap encountered non-positive low-state medians")
    if np.any(med_high_no <= 0) or np.any(med_high_burst <= 0):
        raise AnalysisError("Bootstrap encountered non-positive high-state medians")

    if metric_kind == "latency":
        low_effect = np.log(med_low_burst) - np.log(med_low_no)
        high_effect = np.log(med_high_burst) - np.log(med_high_no)
    else:
        low_effect = np.log(med_low_no) - np.log(med_low_burst)
        high_effect = np.log(med_high_no) - np.log(med_high_burst)
    return BootstrapLogResult(
        point_log_did=point_did,
        did_samples=high_effect - low_effect,
        low_effect_samples=low_effect,
        high_effect_samples=high_effect,
    )


def aligned_additive_effect(no_burst: np.ndarray, fixed_burst: np.ndarray, direction: str) -> float:
    raw = float(np.median(fixed_burst) - np.median(no_burst))
    if direction == "higher_is_worse":
        return raw
    if direction == "higher_is_better":
        return -raw
    raise AnalysisError(f"Unknown additive direction: {direction}")


def bootstrap_additive_did(
    groups: dict[tuple[str, str], np.ndarray],
    *,
    direction: str,
    iterations: int,
    rng: np.random.Generator,
) -> BootstrapAdditiveResult:
    low_no = groups[("low", "no_burst")]
    low_burst = groups[("low", "fixed_burst")]
    high_no = groups[("high", "no_burst")]
    high_burst = groups[("high", "fixed_burst")]

    point_low = aligned_additive_effect(low_no, low_burst, direction)
    point_high = aligned_additive_effect(high_no, high_burst, direction)

    med_low_no = bootstrap_medians(low_no, iterations, rng)
    med_low_burst = bootstrap_medians(low_burst, iterations, rng)
    med_high_no = bootstrap_medians(high_no, iterations, rng)
    med_high_burst = bootstrap_medians(high_burst, iterations, rng)

    if direction == "higher_is_worse":
        low_effect = med_low_burst - med_low_no
        high_effect = med_high_burst - med_high_no
    else:
        low_effect = med_low_no - med_low_burst
        high_effect = med_high_no - med_high_burst

    return BootstrapAdditiveResult(
        point_did=point_high - point_low,
        did_samples=high_effect - low_effect,
        low_effect_samples=low_effect,
        high_effect_samples=high_effect,
    )


def ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def effect_direction(ratio: float, tolerance: float = 1e-12) -> str:
    if ratio > 1.0 + tolerance:
        return "high_more_sensitive"
    if ratio < 1.0 - tolerance:
        return "low_more_sensitive"
    return "no_point_difference"


def build_cell_summary(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        for concurrency in EXPECTED_CONCURRENCIES:
            for state in EXPECTED_STATES:
                for condition in EXPECTED_CONDITIONS:
                    matching = [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["concurrency"] == concurrency
                        and row["state_label"] == state
                        and row["condition"] == condition
                    ]
                    for metric in metrics:
                        values = np.asarray(
                            [float(row[metric]) for row in matching if row.get(metric) is not None],
                            dtype=float,
                        )
                        complete = len(values) == len(matching)
                        output.append(
                            {
                                "model": model,
                                "concurrency": concurrency,
                                "state_label": state,
                                "condition": condition,
                                "metric": metric,
                                "analysis_status": (
                                    "ok" if complete else "insufficient_complete_itl_episodes"
                                ),
                                "n_episodes": len(values),
                                "n_episodes_total": len(matching),
                                "median": float(np.median(values)) if len(values) else None,
                                "mean": float(np.mean(values)) if len(values) else None,
                                "sample_sd": (
                                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                                ),
                                "iqr": (
                                    float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                                    if len(values)
                                    else None
                                ),
                                "min": float(np.min(values)) if len(values) else None,
                                "max": float(np.max(values)) if len(values) else None,
                            }
                        )
    return output


def analyze_log_metrics(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    practical_ratio: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], np.ndarray]]:
    output: list[dict[str, Any]] = []
    distributions: dict[tuple[str, int, str], np.ndarray] = {}
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        for concurrency in EXPECTED_CONCURRENCIES:
            for metric, metric_kind in LOG_METRICS.items():
                if metric in ITL_METRICS:
                    counts = complete_metric_counts(
                        rows,
                        model=model,
                        concurrency=concurrency,
                        metric=metric,
                    )
                    if any(count != EXPECTED_REPEATS_PER_CELL for count in counts.values()):
                        output.append(
                            {
                                "model": model,
                                "concurrency": concurrency,
                                "metric": metric,
                                "metric_kind": metric_kind,
                                "primary_metric": False,
                                "analysis_status": "insufficient_complete_itl_episodes",
                                "n_episodes_per_cell": None,
                                "low_no_burst_complete_episodes": counts[("low", "no_burst")],
                                "low_fixed_burst_complete_episodes": counts[("low", "fixed_burst")],
                                "high_no_burst_complete_episodes": counts[("high", "no_burst")],
                                "high_fixed_burst_complete_episodes": counts[("high", "fixed_burst")],
                            }
                        )
                        continue

                groups = {
                    (state, condition): group_values(
                        rows,
                        model=model,
                        concurrency=concurrency,
                        state=state,
                        condition=condition,
                        metric=metric,
                    )
                    for state in EXPECTED_STATES
                    for condition in EXPECTED_CONDITIONS
                }
                rng = deterministic_rng(seed, "log", model, concurrency, metric)
                result = bootstrap_log_did(
                    groups,
                    metric_kind=metric_kind,
                    iterations=iterations,
                    rng=rng,
                )
                distributions[(model, concurrency, metric)] = result.did_samples

                low_no = float(np.median(groups[("low", "no_burst")]))
                low_burst = float(np.median(groups[("low", "fixed_burst")]))
                high_no = float(np.median(groups[("high", "no_burst")]))
                high_burst = float(np.median(groups[("high", "fixed_burst")]))
                low_effect = math.exp(aligned_log_effect(groups[("low", "no_burst")], groups[("low", "fixed_burst")], metric_kind))
                high_effect = math.exp(aligned_log_effect(groups[("high", "no_burst")], groups[("high", "fixed_burst")], metric_kind))
                ratio = math.exp(result.point_log_did)

                log_low_ci = ci(result.low_effect_samples)
                log_high_ci = ci(result.high_effect_samples)
                log_did_ci = ci(result.did_samples)
                low_effect_ci = tuple(math.exp(value) for value in log_low_ci)
                high_effect_ci = tuple(math.exp(value) for value in log_high_ci)
                ratio_ci = tuple(math.exp(value) for value in log_did_ci)
                inverse_threshold = 1.0 / practical_ratio

                output.append(
                    {
                        "model": model,
                        "concurrency": concurrency,
                        "metric": metric,
                        "metric_kind": metric_kind,
                        "primary_metric": metric == "victim_median_tpot_ms",
                        "analysis_status": "ok",
                        "n_episodes_per_cell": EXPECTED_REPEATS_PER_CELL,
                        "low_no_burst_complete_episodes": EXPECTED_REPEATS_PER_CELL,
                        "low_fixed_burst_complete_episodes": EXPECTED_REPEATS_PER_CELL,
                        "high_no_burst_complete_episodes": EXPECTED_REPEATS_PER_CELL,
                        "high_fixed_burst_complete_episodes": EXPECTED_REPEATS_PER_CELL,
                        "low_no_burst_median": low_no,
                        "low_fixed_burst_median": low_burst,
                        "low_degradation_factor": low_effect,
                        "low_degradation_factor_ci_low": low_effect_ci[0],
                        "low_degradation_factor_ci_high": low_effect_ci[1],
                        "high_no_burst_median": high_no,
                        "high_fixed_burst_median": high_burst,
                        "high_degradation_factor": high_effect,
                        "high_degradation_factor_ci_low": high_effect_ci[0],
                        "high_degradation_factor_ci_high": high_effect_ci[1],
                        "normalized_log_did": result.point_log_did,
                        "normalized_log_did_ci_low": log_did_ci[0],
                        "normalized_log_did_ci_high": log_did_ci[1],
                        "degradation_factor_ratio_high_over_low": ratio,
                        "ratio_ci_low": ratio_ci[0],
                        "ratio_ci_high": ratio_ci[1],
                        "direction": effect_direction(ratio),
                        "ci_excludes_no_interaction": ratio_ci[1] < 1.0 or ratio_ci[0] > 1.0,
                        "bootstrap_probability_high_more_sensitive": float(np.mean(result.did_samples > 0.0)),
                        "practical_ratio_threshold": practical_ratio,
                        "point_practically_relevant": ratio >= practical_ratio or ratio <= inverse_threshold,
                        "ci_entirely_beyond_practical_threshold": ratio_ci[0] >= practical_ratio or ratio_ci[1] <= inverse_threshold,
                        "bootstrap_iterations": iterations,
                    }
                )
    return output, distributions


def analyze_additive_metrics(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    extra_metrics: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = {**ADDITIVE_METRICS, **extra_metrics}
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        for concurrency in EXPECTED_CONCURRENCIES:
            for metric, direction in metrics.items():
                groups = {
                    (state, condition): group_values(
                        rows,
                        model=model,
                        concurrency=concurrency,
                        state=state,
                        condition=condition,
                        metric=metric,
                    )
                    for state in EXPECTED_STATES
                    for condition in EXPECTED_CONDITIONS
                }
                rng = deterministic_rng(seed, "additive", model, concurrency, metric)
                result = bootstrap_additive_did(
                    groups,
                    direction=direction,
                    iterations=iterations,
                    rng=rng,
                )
                did_ci = ci(result.did_samples)
                low_ci = ci(result.low_effect_samples)
                high_ci = ci(result.high_effect_samples)
                output.append(
                    {
                        "model": model,
                        "concurrency": concurrency,
                        "metric": metric,
                        "direction_alignment": direction,
                        "n_episodes_per_cell": EXPECTED_REPEATS_PER_CELL,
                        "low_no_burst_median": float(np.median(groups[("low", "no_burst")])),
                        "low_fixed_burst_median": float(np.median(groups[("low", "fixed_burst")])),
                        "low_aligned_degradation": aligned_additive_effect(groups[("low", "no_burst")], groups[("low", "fixed_burst")], direction),
                        "low_aligned_degradation_ci_low": low_ci[0],
                        "low_aligned_degradation_ci_high": low_ci[1],
                        "high_no_burst_median": float(np.median(groups[("high", "no_burst")])),
                        "high_fixed_burst_median": float(np.median(groups[("high", "fixed_burst")])),
                        "high_aligned_degradation": aligned_additive_effect(groups[("high", "no_burst")], groups[("high", "fixed_burst")], direction),
                        "high_aligned_degradation_ci_low": high_ci[0],
                        "high_aligned_degradation_ci_high": high_ci[1],
                        "additive_did_high_minus_low": result.point_did,
                        "additive_did_ci_low": did_ci[0],
                        "additive_did_ci_high": did_ci[1],
                        "ci_excludes_no_interaction": did_ci[1] < 0.0 or did_ci[0] > 0.0,
                        "bootstrap_probability_high_more_sensitive": float(np.mean(result.did_samples > 0.0)),
                        "bootstrap_iterations": iterations,
                    }
                )
    return output


def build_model_level_log_did(
    log_rows: list[dict[str, Any]],
    distributions: dict[tuple[str, int, str], np.ndarray],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in log_rows})
    for model in models:
        for metric in LOG_METRICS:
            by_conc = {
                int(row["concurrency"]): row
                for row in log_rows
                if row["model"] == model and row["metric"] == metric
            }
            if set(by_conc) != set(EXPECTED_CONCURRENCIES):
                raise AnalysisError(f"Missing concurrency result for {model}/{metric}")
            if any(row.get("analysis_status") != "ok" for row in by_conc.values()):
                output.append(
                    {
                        "model": model,
                        "metric": metric,
                        "aggregation": "equal_weight_mean_of_concurrency_specific_log_did",
                        "analysis_status": "insufficient_complete_itl_episodes",
                        "direction_consistent_across_concurrency": False,
                        "reportable_under_prespecified_rule": False,
                        "note": "Not computed because at least one concurrency lacks five complete ITL episodes in every cell.",
                    }
                )
                continue

            points = [float(by_conc[c]["normalized_log_did"]) for c in EXPECTED_CONCURRENCIES]
            directions = [math.copysign(1.0, point) if point != 0 else 0.0 for point in points]
            consistent = directions[0] == directions[1] and directions[0] != 0.0
            pooled_samples = np.mean(
                np.vstack([distributions[(model, c, metric)] for c in EXPECTED_CONCURRENCIES]),
                axis=0,
            )
            pooled_point = float(np.mean(points))
            pooled_ci = ci(pooled_samples)
            ratio = math.exp(pooled_point)
            ratio_ci = tuple(math.exp(value) for value in pooled_ci)
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "aggregation": "equal_weight_mean_of_concurrency_specific_log_did",
                    "analysis_status": "ok",
                    "conc4_normalized_log_did": points[0],
                    "conc8_normalized_log_did": points[1],
                    "direction_consistent_across_concurrency": consistent,
                    "reportable_under_prespecified_rule": consistent,
                    "aggregated_normalized_log_did": pooled_point,
                    "aggregated_log_did_ci_low": pooled_ci[0],
                    "aggregated_log_did_ci_high": pooled_ci[1],
                    "aggregated_degradation_factor_ratio": ratio,
                    "aggregated_ratio_ci_low": ratio_ci[0],
                    "aggregated_ratio_ci_high": ratio_ci[1],
                    "direction": effect_direction(ratio),
                    "note": (
                        "Report only when direction_consistent_across_concurrency=true; "
                        "otherwise retain concurrency-specific results."
                    ),
                }
            )
    return output


def build_leave_one_repeat_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        for concurrency in EXPECTED_CONCURRENCIES:
            for metric, metric_kind in LOG_METRICS.items():
                if metric in ITL_METRICS:
                    counts = complete_metric_counts(
                        rows,
                        model=model,
                        concurrency=concurrency,
                        metric=metric,
                    )
                    if any(count != EXPECTED_REPEATS_PER_CELL for count in counts.values()):
                        for repeat in range(1, EXPECTED_REPEATS_PER_CELL + 1):
                            output.append(
                                {
                                    "model": model,
                                    "concurrency": concurrency,
                                    "metric": metric,
                                    "excluded_repeat": repeat,
                                    "analysis_status": "insufficient_complete_itl_episodes",
                                    "full_normalized_log_did": None,
                                    "leave_one_repeat_out_log_did": None,
                                    "leave_one_repeat_out_ratio": None,
                                    "same_direction_as_full": None,
                                }
                            )
                        continue

                full_groups = {
                    (state, condition): group_values(
                        rows,
                        model=model,
                        concurrency=concurrency,
                        state=state,
                        condition=condition,
                        metric=metric,
                    )
                    for state in EXPECTED_STATES
                    for condition in EXPECTED_CONDITIONS
                }
                full_did = aligned_log_effect(full_groups[("high", "no_burst")], full_groups[("high", "fixed_burst")], metric_kind) - aligned_log_effect(full_groups[("low", "no_burst")], full_groups[("low", "fixed_burst")], metric_kind)
                for repeat in range(1, EXPECTED_REPEATS_PER_CELL + 1):
                    groups = {
                        (state, condition): group_values(
                            rows,
                            model=model,
                            concurrency=concurrency,
                            state=state,
                            condition=condition,
                            metric=metric,
                            excluded_repeat=repeat,
                        )
                        for state in EXPECTED_STATES
                        for condition in EXPECTED_CONDITIONS
                    }
                    did = aligned_log_effect(groups[("high", "no_burst")], groups[("high", "fixed_burst")], metric_kind) - aligned_log_effect(groups[("low", "no_burst")], groups[("low", "fixed_burst")], metric_kind)
                    output.append(
                        {
                            "model": model,
                            "concurrency": concurrency,
                            "metric": metric,
                            "excluded_repeat": repeat,
                            "analysis_status": "ok",
                            "full_normalized_log_did": full_did,
                            "leave_one_repeat_out_log_did": did,
                            "leave_one_repeat_out_ratio": math.exp(did),
                            "same_direction_as_full": (did > 0) == (full_did > 0) if did != 0 and full_did != 0 else did == full_did,
                        }
                    )
    return output


def primary_console_summary(log_rows: list[dict[str, Any]]) -> None:
    print("\nPrimary metric: episode-level victim median TPOT")
    print("=" * 78)
    primary = [row for row in log_rows if row["primary_metric"]]
    for row in sorted(primary, key=lambda x: (x["model"], x["concurrency"])):
        print(
            f"{row['model']:>6} conc={row['concurrency']}: "
            f"low factor={row['low_degradation_factor']:.4f}x, "
            f"high factor={row['high_degradation_factor']:.4f}x, "
            f"ratio={row['degradation_factor_ratio_high_over_low']:.4f} "
            f"[95% CI {row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}] "
            f"-> {row['direction']}"
        )
    print("=" * 78)


def build_report(
    *,
    root: Path,
    summary: dict[str, Any],
    integrity: dict[str, Any],
    rows: list[dict[str, Any]],
    log_results: list[dict[str, Any]],
    additive_results: list[dict[str, Any]],
    model_results: list[dict[str, Any]],
    loo_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    primary = [row for row in log_results if row["primary_metric"]]
    loo_primary = [row for row in loo_results if row["metric"] == "victim_median_tpot_ms"]
    loo_summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in loo_primary}):
        for concurrency in EXPECTED_CONCURRENCIES:
            subset = [
                row
                for row in loo_primary
                if row["model"] == model and row["concurrency"] == concurrency
            ]
            values = [float(row["leave_one_repeat_out_ratio"]) for row in subset]
            loo_summary.append(
                {
                    "model": model,
                    "concurrency": concurrency,
                    "metric": "victim_median_tpot_ms",
                    "min_leave_one_repeat_out_ratio": min(values),
                    "max_leave_one_repeat_out_ratio": max(values),
                    "all_same_direction_as_full": all(bool(row["same_direction_as_full"]) for row in subset),
                }
            )

    return {
        "analysis_script_version": SCRIPT_VERSION,
        "input_root_name": root.name,
        "analysis_parameters": {
            "bootstrap_iterations": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_ci": "percentile_95_percent",
            "bootstrap_unit": "episode",
            "groups_resampled_independently": True,
            "request_level_pseudoreplication": False,
            "practical_ratio_threshold_exploratory": args.practical_ratio,
            "ttft_slo_ms": args.ttft_slo_ms,
            "e2el_slo_ms": args.e2el_slo_ms,
        },
        "estimand": {
            "primary_metric": "episode-level victim median TPOT",
            "low_log_effect": "log(median_low_fixed_burst / median_low_no_burst)",
            "high_log_effect": "log(median_high_fixed_burst / median_high_no_burst)",
            "normalized_log_did": "high_log_effect - low_log_effect",
            "ratio": "exp(normalized_log_did)",
            "interpretation": {
                "ratio_gt_1": "high state relatively more sensitive",
                "ratio_lt_1": "low state relatively more sensitive",
                "ratio_eq_1": "no normalized interaction",
            },
            "goodput_alignment": "uses log(no_burst / fixed_burst) so positive means degradation",
        },
        "dataset_validation": {
            "official_summary_complete": summary.get("overall_status") == "complete",
            "episode_count": len(rows),
            "integrity_verified": integrity.get("verified") is True,
            "integrity": integrity,
            "schedule_fingerprint": summary.get("schedule_fingerprint"),
            "environment_fingerprint": summary.get("environment_fingerprint"),
            "models": sorted({row["model"] for row in rows}),
            "concurrencies": sorted({row["concurrency"] for row in rows}),
            "states": sorted({row["state_label"] for row in rows}),
            "conditions": sorted({row["condition"] for row in rows}),
            "repeats_per_cell": EXPECTED_REPEATS_PER_CELL,
        },
        "primary_results": primary,
        "primary_leave_one_repeat_out_summary": loo_summary,
        "model_level_primary_results": [
            row for row in model_results if row["metric"] == "victim_median_tpot_ms"
        ],
        "secondary_log_result_count": len(log_results) - len(primary),
        "additive_result_count": len(additive_results),
        "important_interpretation_limits": [
            "The five episodes per cell are the inferential units; the 20 requests inside an episode are not independent replications.",
            "Percentile bootstrap intervals with n=5 per cell can be wide and discrete.",
            "Concurrency-specific results are primary. Model-level aggregation is reportable only when both concurrency-specific effects have the same direction.",
            "The practical factor threshold is exploratory unless separately frozen before a confirmatory run.",
            "SLO rates are calculated only when explicit thresholds are provided on the command line.",
            "ITL metrics are analyzed only when all 20 victim requests have reliable ITL in all five episodes of every required cell; incomplete ITL never blocks the primary TPOT analysis.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AnalysisError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_dataset(args.input) as context:
        root = context.root
        integrity = verify_integrity(root)
        rows, summary = load_episode_rows(
            root,
            ttft_slo_ms=args.ttft_slo_ms,
            e2el_slo_ms=args.e2el_slo_ms,
        )

        extra_additive: dict[str, str] = {}
        if args.ttft_slo_ms is not None:
            extra_additive["victim_ttft_slo_violation_rate"] = "higher_is_worse"
        if args.e2el_slo_ms is not None:
            extra_additive["victim_e2el_slo_violation_rate"] = "higher_is_worse"

        all_cell_metrics = list(LOG_METRICS) + list(ADDITIVE_METRICS) + list(extra_additive)
        cell_summary = build_cell_summary(rows, all_cell_metrics)
        log_results, distributions = analyze_log_metrics(
            rows,
            iterations=args.bootstrap,
            seed=args.seed,
            practical_ratio=args.practical_ratio,
        )
        additive_results = analyze_additive_metrics(
            rows,
            iterations=args.bootstrap,
            seed=args.seed,
            extra_metrics=extra_additive,
        )
        model_results = build_model_level_log_did(log_results, distributions)
        loo_results = build_leave_one_repeat_out(rows)
        report = build_report(
            root=root,
            summary=summary,
            integrity=integrity,
            rows=rows,
            log_results=log_results,
            additive_results=additive_results,
            model_results=model_results,
            loo_results=loo_results,
            args=args,
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["concurrency"],
            row["state_label"],
            row["condition"],
            row["repeat"],
        ),
    )
    write_csv(output_dir / "episode_metrics.csv", rows_sorted)
    write_csv(output_dir / "cell_summary.csv", cell_summary)
    write_csv(output_dir / "log_did_results.csv", log_results)
    write_csv(output_dir / "additive_did_results.csv", additive_results)
    write_csv(output_dir / "model_level_log_did.csv", model_results)
    write_csv(output_dir / "leave_one_repeat_out.csv", loo_results)
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    primary_console_summary(log_results)
    print(f"PASS: analysis completed; outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
