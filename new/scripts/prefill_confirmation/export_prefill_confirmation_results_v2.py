#!/usr/bin/env python3
"""
export_prefill_confirmation_results_v2.py

Creates a compact, analysis-ready Schema-5 export from the completed Llama and Qwen
Prefill-Confirmation campaigns. Standard library only; no pandas/scipy needed.

The exporter:
  * auto-detects the newest completed official run for each model, or accepts
    explicit --llama-dir / --qwen-dir paths;
  * verifies official summaries, file counts, result statuses, pair structure,
    block summaries, stabilization records, and every hash listed in the
    integrity manifest;
  * extracts only scientific metrics and timing metadata -- never raw SSE
    payloads, prompt token IDs, or generated token IDs;
  * writes compact CSV/JSON/TXT artifacts and a ZIP suitable for upload.

Usage:
  cd ~/projects/DOS
  python3 new/scripts/prefill_confirmation/export_prefill_confirmation_results.py

Optional:
  python3 export_prefill_confirmation_results.py \
      --project-root ~/projects/DOS \
      --llama-dir PATH/TO/LLAMA/OFFICIAL \
      --qwen-dir PATH/TO/QWEN/OFFICIAL \
      --output-dir new/runs/prefill_confirmation/analysis_export

Exit code is non-zero if a structural or integrity problem is detected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_MODELS = ("llama", "qwen")
EXPECTED_EPISODES_PER_MODEL = 96
EXPECTED_BLOCKS_PER_MODEL = 48
EXPECTED_STABILIZATIONS_PER_MODEL = 48
EXPECTED_CONDITIONS = {"no_burst", "prefill_burst"}
EXPECTED_OFFLOADS = {0, 8, 12}
EXPECTED_CONCURRENCIES = {4, 8}
EXPECTED_REPEATS = set(range(1, 9))
PAIR_METRICS = (
    "tpot_median_ms", "tpot_p95_ms", "tpot_p99_ms",
    "ttft_median_ms", "ttft_p95_ms", "ttft_p99_ms",
    "e2el_median_ms", "e2el_p95_ms", "e2el_p99_ms",
    "itl_median_ms", "itl_p95_ms", "itl_p99_ms",
    "throughput_tokens_per_s",
    "active_wave_tpot_median_ms", "later_wave_tpot_median_ms",
    "active_wave_ttft_median_ms", "later_wave_ttft_median_ms",
    "active_wave_e2el_median_ms", "later_wave_e2el_median_ms",
    "queue_wait_median_ms", "active_wave_queue_wait_median_ms",
    "later_wave_queue_wait_median_ms",
    "admission_to_dispatch_median_ms",
    "e2el_from_task_creation_median_ms",
    "active_wave_e2el_from_task_creation_median_ms",
    "later_wave_e2el_from_task_creation_median_ms",
    "e2el_from_phase_start_median_ms",
    "active_wave_e2el_from_phase_start_median_ms",
    "later_wave_e2el_from_phase_start_median_ms",
    "overlap_request_count", "overlap_median_ms", "overlap_max_ms",
)

# Effect fields used for cell summaries and repeat-matched within-model
# interactions. The primary TPOT analysis is the paired log ratio.
ANALYSIS_EFFECT_FIELDS = (
    ("log_tpot_ratio", "log_ratio_tpot_median_ms"),
    ("delta_tpot_median_ms", "delta_tpot_median_ms"),
    ("delta_itl_p95_ms", "delta_itl_p95_ms"),
    ("delta_ttft_median_ms", "delta_ttft_median_ms"),
    ("delta_e2el_median_ms", "delta_e2el_median_ms"),
    ("delta_later_wave_tpot_median_ms", "delta_later_wave_tpot_median_ms"),
    ("delta_active_wave_tpot_median_ms", "delta_active_wave_tpot_median_ms"),
    ("delta_later_wave_queue_wait_median_ms", "delta_later_wave_queue_wait_median_ms"),
    ("delta_active_wave_queue_wait_median_ms", "delta_active_wave_queue_wait_median_ms"),
    ("delta_later_wave_e2el_from_task_creation_median_ms",
     "delta_later_wave_e2el_from_task_creation_median_ms"),
    ("delta_active_wave_e2el_from_task_creation_median_ms",
     "delta_active_wave_e2el_from_task_creation_median_ms"),
)

CSV_FLOAT_DIGITS = 9


class ExportError(RuntimeError):
    pass


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportError(f"Missing required file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ExportError(f"Top-level JSON object expected in {path}")
    return obj


def safe_float(value: Any) -> float | None:
    if type(value) in (int, float) and math.isfinite(float(value)):
        return float(value)
    return None


def nested(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def percentile(values: Iterable[float], p: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else None


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return round(value, CSV_FLOAT_DIGITS)
    if value is None:
        return ""
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_latest_official(project_root: Path, model: str) -> Path:
    results_root = project_root / "new" / "runs" / "prefill_confirmation" / model / "results"
    if not results_root.is_dir():
        raise ExportError(f"Results directory not found for {model}: {results_root}")

    candidates: list[Path] = []
    for path in results_root.iterdir():
        if not path.is_dir():
            continue
        summary_path = path / "official_run_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = read_json(summary_path)
        except ExportError:
            continue
        if summary.get("run_mode") == "official" and summary.get("overall_status") == "complete":
            candidates.append(path)

    if not candidates:
        raise ExportError(
            f"No completed official run found for {model} under {results_root}. "
            f"Pass --{model}-dir explicitly if it has a different name."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def verify_integrity_manifest(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return ["integrity_manifest.files is not a list"]

    seen: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"integrity entry {idx} is not an object")
            continue
        rel = entry.get("relative_path")
        size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
            errors.append(f"integrity entry {idx} has unsafe relative_path={rel!r}")
            continue
        if rel in seen:
            errors.append(f"duplicate integrity path: {rel}")
            continue
        seen.add(rel)
        path = run_dir / rel
        if not path.is_file():
            errors.append(f"integrity-listed file missing: {rel}")
            continue
        actual_size = path.stat().st_size
        if type(size) is not int or actual_size != size:
            errors.append(f"size mismatch for {rel}: manifest={size!r}, disk={actual_size}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"invalid SHA-256 field for {rel}")
        else:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                errors.append(f"SHA-256 mismatch for {rel}")

    declared_count = manifest.get("file_count")
    if type(declared_count) is int and declared_count != len(entries):
        errors.append(
            f"integrity file_count mismatch: declared={declared_count}, entries={len(entries)}"
        )
    return errors


def flatten_episode(model: str, path: Path, obj: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = obj.get("schedule_row")
    if not isinstance(row, dict):
        raise ExportError(f"{path}: schedule_row missing or not an object")
    agg = obj.get("aggregate_metrics")
    if not isinstance(agg, dict):
        raise ExportError(f"{path}: aggregate_metrics missing or not an object")
    victims = obj.get("victim_requests")
    if not isinstance(victims, list):
        raise ExportError(f"{path}: victim_requests missing or not a list")

    block_id = row.get("block_id")
    episode_id = row.get("episode_id")
    condition = row.get("condition")
    offload = row.get("offload_gb")
    concurrency = row.get("concurrency")
    repeat = row.get("repeat")

    request_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = defaultdict(list)
    overlap_values: list[float] = []
    exposure_counts: Counter[str] = Counter()

    overlap_meta = agg.get("victim_overlap")
    if not isinstance(overlap_meta, list):
        overlap_meta = []

    def add(prefix: str, metric: str, value: float | None) -> None:
        if value is not None:
            buckets[f"{prefix}{metric}"].append(value)

    for index, req in enumerate(victims):
        if not isinstance(req, dict):
            raise ExportError(f"{path}: victim_requests[{index}] is not an object")
        wave_id = req.get("wave_id")
        wave_prefix = "active_" if wave_id == 0 else "later_" if type(wave_id) is int and wave_id >= 1 else "unknown_"
        exposure = str(req.get("trigger_exposure_group") or "unknown")
        exposure_counts[exposure] += 1

        ttft = safe_float(req.get("ttft_ms"))
        tpot = safe_float(req.get("client_observed_tpot_ms"))
        e2el = safe_float(req.get("e2el_ms"))
        queue_wait = safe_float(req.get("local_queue_wait_ms"))
        admission_dispatch = safe_float(req.get("admission_to_dispatch_ms"))
        e2el_task = safe_float(req.get("total_e2el_from_task_creation_ms"))
        e2el_phase = safe_float(req.get("total_e2el_from_phase_start_ms"))

        for metric, value in (
            ("tpot", tpot), ("ttft", ttft), ("e2el", e2el),
            ("queue_wait", queue_wait), ("e2el_task", e2el_task),
            ("e2el_phase", e2el_phase),
        ):
            add("all_", metric, value)
            add(wave_prefix, metric, value)
        add("all_", "admission_dispatch", admission_dispatch)

        itls = req.get("itl_ms")
        itl_values = (
            [float(v) for v in itls if type(v) in (int, float) and math.isfinite(float(v))]
            if isinstance(itls, list)
            else []
        )

        overlap = overlap_meta[index] if index < len(overlap_meta) else {}
        if not isinstance(overlap, dict):
            overlap = {}
        overlap_ms = safe_float(overlap.get("overlap_ms")) or 0.0
        if overlap_ms > 0:
            overlap_values.append(overlap_ms)

        request_rows.append(
            {
                "model": model,
                "episode_id": episode_id,
                "block_id": block_id,
                "condition": condition,
                "offload_gb": offload,
                "concurrency": concurrency,
                "repeat": repeat,
                "request_index": req.get("request_index"),
                "wave_id": wave_id,
                "wave_position": req.get("wave_position"),
                "trigger_exposure_group": exposure,
                "status": req.get("status"),
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "e2el_ms": e2el,
                "itl_median_ms": median(itl_values),
                "itl_p95_ms": percentile(itl_values, 0.95),
                "itl_p99_ms": percentile(itl_values, 0.99),
                "local_queue_wait_ms": queue_wait,
                "admission_to_dispatch_ms": admission_dispatch,
                "task_creation_offset_ms": safe_float(req.get("task_creation_offset_ms")),
                "dispatch_from_phase_start_ms": safe_float(req.get("dispatch_from_phase_start_ms")),
                "total_e2el_from_task_creation_ms": e2el_task,
                "total_e2el_from_phase_start_ms": e2el_phase,
                "was_created_at_trigger": req.get("was_created_at_trigger"),
                "was_admitted_at_trigger": req.get("was_admitted_at_trigger"),
                "was_dispatched_at_trigger": req.get("was_dispatched_at_trigger"),
                "was_running_at_trigger": req.get("was_running_at_trigger"),
                "overlaps_burst": bool(overlap.get("overlaps_burst")),
                "overlap_ms": overlap_ms,
            }
        )

    trigger = obj.get("trigger") if isinstance(obj.get("trigger"), dict) else {}
    episode_row: dict[str, Any] = {
        "model": model,
        "episode_id": episode_id,
        "block_id": block_id,
        "condition": condition,
        "condition_first_in_block": row.get("condition_first_in_block"),
        "order_in_block": row.get("order_in_block"),
        "offload_gb": offload,
        "state_label": row.get("state_label"),
        "concurrency": concurrency,
        "repeat": repeat,
        "trigger_after_decode_tokens": row.get("trigger_after_decode_tokens"),
        "status": obj.get("status"),
        "run_mode": obj.get("run_mode"),
        "schedule_fingerprint": obj.get("schedule_fingerprint"),
        "tpot_median_ms": safe_float(nested(agg, "victim_client_observed_tpot_ms", "median")),
        "tpot_p95_ms": safe_float(nested(agg, "victim_client_observed_tpot_ms", "p95")),
        "tpot_p99_ms": safe_float(nested(agg, "victim_client_observed_tpot_ms", "p99")),
        "ttft_median_ms": safe_float(nested(agg, "victim_ttft_ms", "median")),
        "ttft_p95_ms": safe_float(nested(agg, "victim_ttft_ms", "p95")),
        "ttft_p99_ms": safe_float(nested(agg, "victim_ttft_ms", "p99")),
        "e2el_median_ms": safe_float(nested(agg, "victim_e2el_ms", "median")),
        "e2el_p95_ms": safe_float(nested(agg, "victim_e2el_ms", "p95")),
        "e2el_p99_ms": safe_float(nested(agg, "victim_e2el_ms", "p99")),
        "itl_median_ms": safe_float(nested(agg, "victim_itl_ms", "median")),
        "itl_p95_ms": safe_float(nested(agg, "victim_itl_ms", "p95")),
        "itl_p99_ms": safe_float(nested(agg, "victim_itl_ms", "p99")),
        "itl_n": nested(agg, "victim_itl_ms", "n"),
        "throughput_tokens_per_s": safe_float(agg.get("victim_throughput_tokens_per_s")),
        "victim_complete_request_count": agg.get("victim_complete_request_count"),
        "victim_incomplete_request_count": agg.get("victim_incomplete_request_count"),
        "burst_duration_ms": safe_float(agg.get("burst_duration_ms")),
        "burst_complete_request_count": agg.get("burst_complete_request_count"),
        "burst_incomplete_request_count": agg.get("burst_incomplete_request_count"),
        "trigger_crossing_skew_ms": safe_float(trigger.get("trigger_crossing_skew_ms")),
        "trigger_dispatch_delay_ms": safe_float(trigger.get("trigger_dispatch_delay_ms")),
        "trigger_wait_duration_ms": safe_float(trigger.get("trigger_wait_duration_ms")),
        "active_wave_tpot_median_ms": median(buckets["active_tpot"]),
        "later_wave_tpot_median_ms": median(buckets["later_tpot"]),
        "active_wave_ttft_median_ms": median(buckets["active_ttft"]),
        "later_wave_ttft_median_ms": median(buckets["later_ttft"]),
        "active_wave_e2el_median_ms": median(buckets["active_e2el"]),
        "later_wave_e2el_median_ms": median(buckets["later_e2el"]),
        "queue_wait_median_ms": median(buckets["all_queue_wait"]),
        "active_wave_queue_wait_median_ms": median(buckets["active_queue_wait"]),
        "later_wave_queue_wait_median_ms": median(buckets["later_queue_wait"]),
        "admission_to_dispatch_median_ms": median(buckets["all_admission_dispatch"]),
        "e2el_from_task_creation_median_ms": median(buckets["all_e2el_task"]),
        "active_wave_e2el_from_task_creation_median_ms": median(buckets["active_e2el_task"]),
        "later_wave_e2el_from_task_creation_median_ms": median(buckets["later_e2el_task"]),
        "e2el_from_phase_start_median_ms": median(buckets["all_e2el_phase"]),
        "active_wave_e2el_from_phase_start_median_ms": median(buckets["active_e2el_phase"]),
        "later_wave_e2el_from_phase_start_median_ms": median(buckets["later_e2el_phase"]),
        "overlap_request_count": len(overlap_values),
        "overlap_median_ms": median(overlap_values),
        "overlap_max_ms": max(overlap_values) if overlap_values else 0.0,
        "trigger_exposure_counts_json": json.dumps(dict(sorted(exposure_counts.items())), sort_keys=True),
    }
    return episode_row, request_rows

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def exact_sign_test_two_sided(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v)) and float(v) != 0.0]
    n = len(vals)
    if n == 0:
        return None
    positives = sum(v > 0 for v in vals)
    k = min(positives, n - positives)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(
    values: list[float],
    stat: Callable[[Iterable[float]], float | None],
    rng: random.Random,
    iterations: int,
) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None, None
    n = len(vals)
    samples: list[float] = []
    for _ in range(iterations):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        result = stat(sample)
        if result is not None:
            samples.append(float(result))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def summarize_values(
    values: list[float],
    *,
    rng: random.Random,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    mean_lo, mean_hi = bootstrap_ci(vals, mean, rng, bootstrap_iterations)
    med_lo, med_hi = bootstrap_ci(vals, median, rng, bootstrap_iterations)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "positive_count": sum(v > 0 for v in vals),
        "negative_count": sum(v < 0 for v in vals),
        "zero_count": sum(v == 0 for v in vals),
        "sign_test_p_two_sided": exact_sign_test_two_sided(vals),
        "bootstrap_mean_ci_low": mean_lo,
        "bootstrap_mean_ci_high": mean_hi,
        "bootstrap_median_ci_low": med_lo,
        "bootstrap_median_ci_high": med_hi,
    }


def build_pairs(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        by_block[(str(row["model"]), str(row["block_id"]))][str(row["condition"])] = row

    pairs: list[dict[str, Any]] = []
    for (model, block_id), conditions in sorted(by_block.items()):
        if set(conditions) != EXPECTED_CONDITIONS:
            raise ExportError(
                f"Block {model}/{block_id} does not contain exactly no_burst and prefill_burst"
            )
        base = conditions["no_burst"]
        burst = conditions["prefill_burst"]
        pair: dict[str, Any] = {
            "model": model,
            "block_id": block_id,
            "offload_gb": base["offload_gb"],
            "state_label": base["state_label"],
            "concurrency": base["concurrency"],
            "repeat": base["repeat"],
            "condition_first_in_block": base["condition_first_in_block"],
        }
        for metric in PAIR_METRICS:
            b = safe_float(base.get(metric))
            x = safe_float(burst.get(metric))
            pair[f"no_burst_{metric}"] = b
            pair[f"prefill_burst_{metric}"] = x
            pair[f"delta_{metric}"] = (x - b) if b is not None and x is not None else None
            pair[f"ratio_{metric}"] = (x / b) if b not in (None, 0.0) and x is not None else None
            pair[f"pct_{metric}"] = (
                100.0 * (x - b) / b if b not in (None, 0.0) and x is not None else None
            )
        tpot_ratio = safe_float(pair.get("ratio_tpot_median_ms"))
        pair["log_ratio_tpot_median_ms"] = (
            math.log(tpot_ratio) if tpot_ratio is not None and tpot_ratio > 0 else None
        )
        pairs.append(pair)
    return pairs

def build_group_summaries(
    pairs: list[dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(str(pair["model"]), int(pair["offload_gb"]), int(pair["concurrency"]))].append(pair)

    rows: list[dict[str, Any]] = []
    for (model, offload, concurrency), items in sorted(groups.items()):
        for metric, field in ANALYSIS_EFFECT_FIELDS:
            values = [
                float(item[field])
                for item in items
                if safe_float(item.get(field)) is not None
            ]
            summary = summarize_values(values, rng=rng, bootstrap_iterations=bootstrap_iterations)
            rows.append(
                {
                    "contrast_type": "burst_effect",
                    "model": model,
                    "offload_gb": offload,
                    "concurrency": concurrency,
                    "metric": metric,
                    "source_field": field,
                    **summary,
                }
            )
    return rows

def summarize_contrasts(
    contrast_rows: list[dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in contrast_rows:
        key = (
            row["contrast_type"],
            row.get("model", ""),
            row.get("offload_gb", ""),
            row.get("concurrency", ""),
            row.get("metric", ""),
            row.get("comparison", ""),
        )
        value = safe_float(row.get("value"))
        if value is not None:
            grouped[key].append(value)
        meta[key] = {
            "contrast_type": row["contrast_type"],
            "model": row.get("model", ""),
            "offload_gb": row.get("offload_gb", ""),
            "concurrency": row.get("concurrency", ""),
            "metric": row.get("metric", ""),
            "comparison": row.get("comparison", ""),
        }

    out: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append(
            {
                **meta[key],
                **summarize_values(
                    values, rng=rng, bootstrap_iterations=bootstrap_iterations
                ),
            }
        )
    return out


def build_interaction_contrasts(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for pair in pairs:
        indexed[(str(pair["model"]), int(pair["offload_gb"]), int(pair["concurrency"]), int(pair["repeat"]))] = pair

    detail: list[dict[str, Any]] = []

    # State × burst: within-model, repeat-matched contrast. For the primary
    # metric this is a log-ratio difference, i.e. log(D_high / D_low).
    for model in EXPECTED_MODELS:
        for concurrency in sorted(EXPECTED_CONCURRENCIES):
            for offload in (8, 12):
                for repeat in sorted(EXPECTED_REPEATS):
                    hi = indexed[(model, offload, concurrency, repeat)]
                    low = indexed[(model, 0, concurrency, repeat)]
                    for metric, field in ANALYSIS_EFFECT_FIELDS:
                        a = safe_float(hi.get(field))
                        b = safe_float(low.get(field))
                        detail.append(
                            {
                                "contrast_type": "state_x_burst",
                                "model": model,
                                "offload_gb": offload,
                                "concurrency": concurrency,
                                "repeat": repeat,
                                "metric": metric,
                                "source_field": field,
                                "comparison": f"offload{offload}_minus_offload0",
                                "value": (a - b) if a is not None and b is not None else None,
                            }
                        )

    # Concurrency × burst: within-model, repeat-matched contrast.
    for model in EXPECTED_MODELS:
        for offload in sorted(EXPECTED_OFFLOADS):
            for repeat in sorted(EXPECTED_REPEATS):
                c8 = indexed[(model, offload, 8, repeat)]
                c4 = indexed[(model, offload, 4, repeat)]
                for metric, field in ANALYSIS_EFFECT_FIELDS:
                    a = safe_float(c8.get(field))
                    b = safe_float(c4.get(field))
                    detail.append(
                        {
                            "contrast_type": "concurrency_x_burst",
                            "model": model,
                            "offload_gb": offload,
                            "concurrency": 8,
                            "repeat": repeat,
                            "metric": metric,
                            "source_field": field,
                            "comparison": "conc8_minus_conc4",
                            "value": (a - b) if a is not None and b is not None else None,
                        }
                    )

    # Deliberately no repeat-matched Llama-vs-Qwen contrast: repeat numbers
    # are not paired across tokenizers/models.
    return detail, []

def build_exposure_episode_summaries(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        key = (
            row["model"], row["episode_id"], row["block_id"], row["condition"],
            row["offload_gb"], row["concurrency"], row["repeat"],
            row["trigger_exposure_group"],
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    metrics = (
        "tpot_ms", "ttft_ms", "e2el_ms", "local_queue_wait_ms",
        "admission_to_dispatch_ms", "total_e2el_from_task_creation_ms",
        "total_e2el_from_phase_start_ms", "overlap_ms",
    )
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        model, episode_id, block_id, condition, offload, concurrency, repeat, exposure = key
        row: dict[str, Any] = {
            "model": model, "episode_id": episode_id, "block_id": block_id,
            "condition": condition, "offload_gb": offload,
            "concurrency": concurrency, "repeat": repeat,
            "trigger_exposure_group": exposure, "request_count": len(items),
        }
        for metric in metrics:
            values = [v for item in items if (v := safe_float(item.get(metric))) is not None]
            row[f"{metric}_median"] = median(values)
            row[f"{metric}_p95"] = percentile(values, 0.95)
        out.append(row)
    return out


def build_exposure_pairs(exposure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in exposure_rows:
        key = (
            row["model"], row["block_id"], row["offload_gb"], row["concurrency"],
            row["repeat"], row["trigger_exposure_group"],
        )
        grouped[key][str(row["condition"])] = row

    out: list[dict[str, Any]] = []
    metric_fields = [
        key for key in exposure_rows[0].keys()
        if key.endswith("_median") or key.endswith("_p95")
    ] if exposure_rows else []
    for key, conditions in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        model, block_id, offload, concurrency, repeat, exposure = key
        base = conditions.get("no_burst", {})
        burst = conditions.get("prefill_burst", {})
        row: dict[str, Any] = {
            "model": model, "block_id": block_id, "offload_gb": offload,
            "concurrency": concurrency, "repeat": repeat,
            "trigger_exposure_group": exposure,
            "no_burst_request_count": base.get("request_count", 0),
            "prefill_burst_request_count": burst.get("request_count", 0),
        }
        for metric in metric_fields:
            b = safe_float(base.get(metric))
            x = safe_float(burst.get(metric))
            row[f"no_burst_{metric}"] = b
            row[f"prefill_burst_{metric}"] = x
            row[f"delta_{metric}"] = (x - b) if b is not None and x is not None else None
        out.append(row)
    return out


def bootstrap_unpaired_difference(
    a: list[float], b: list[float], *, rng: random.Random, iterations: int
) -> tuple[float | None, float | None]:
    if not a or not b:
        return None, None
    samples: list[float] = []
    for _ in range(iterations):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        samples.append(statistics.fmean(sa) - statistics.fmean(sb))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def build_unpaired_model_comparisons(
    pairs: list[dict[str, Any]], *, rng: random.Random, bootstrap_iterations: int
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        cells[(str(pair["model"]), int(pair["offload_gb"]), int(pair["concurrency"]))].append(pair)

    out: list[dict[str, Any]] = []
    for offload in sorted(EXPECTED_OFFLOADS):
        for concurrency in sorted(EXPECTED_CONCURRENCIES):
            for metric, field in ANALYSIS_EFFECT_FIELDS:
                q = [float(x[field]) for x in cells[("qwen", offload, concurrency)] if safe_float(x.get(field)) is not None]
                l = [float(x[field]) for x in cells[("llama", offload, concurrency)] if safe_float(x.get(field)) is not None]
                lo, hi = bootstrap_unpaired_difference(q, l, rng=rng, iterations=bootstrap_iterations)
                out.append({
                    "comparison": "qwen_minus_llama_unpaired",
                    "offload_gb": offload, "concurrency": concurrency,
                    "metric": metric, "source_field": field,
                    "n_qwen": len(q), "n_llama": len(l),
                    "qwen_mean": mean(q), "llama_mean": mean(l),
                    "mean_difference": (mean(q) - mean(l)) if q and l else None,
                    "bootstrap_mean_difference_ci_low": lo,
                    "bootstrap_mean_difference_ci_high": hi,
                })
    return out

def collect_stabilizations(model: str, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "stabilization").glob("*.json")):
        obj = read_json(path)
        first = obj.get("first_half_medians") if isinstance(obj.get("first_half_medians"), dict) else {}
        second = obj.get("second_half_medians") if isinstance(obj.get("second_half_medians"), dict) else {}
        rel = obj.get("relative_change") if isinstance(obj.get("relative_change"), dict) else {}
        block_id = obj.get("block_id")
        rows.append(
            {
                "model": model,
                "block_id": block_id,
                "offload_gb": obj.get("offload_gb"),
                "state_label": obj.get("state_label"),
                "status": obj.get("status"),
                "functional_passed": obj.get("functional_passed"),
                "first_median_ttft_ms": safe_float(first.get("median_ttft_ms")),
                "second_median_ttft_ms": safe_float(second.get("median_ttft_ms")),
                "relative_ttft_change": safe_float(rel.get("median_ttft_ms")),
                "first_median_tpot_ms": safe_float(first.get("median_tpot_ms")),
                "second_median_tpot_ms": safe_float(second.get("median_tpot_ms")),
                "relative_tpot_change": safe_float(rel.get("median_tpot_ms")),
                "first_median_e2el_ms": safe_float(first.get("median_e2el_ms")),
                "second_median_e2el_ms": safe_float(second.get("median_e2el_ms")),
                "relative_e2el_change": safe_float(rel.get("median_e2el_ms")),
            }
        )
    return rows


def validate_campaign(
    model: str,
    run_dir: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[str] = []
    summary = read_json(run_dir / "official_run_summary.json")
    manifest = read_json(run_dir / "official_run_manifest.json")
    integrity = read_json(run_dir / "integrity_manifest.json")

    expected_summary = {
        "run_mode": "official",
        "overall_status": "complete",
        "planned_blocks": EXPECTED_BLOCKS_PER_MODEL,
        "completed_blocks": EXPECTED_BLOCKS_PER_MODEL,
        "pending_blocks": 0,
        "planned_episodes": EXPECTED_EPISODES_PER_MODEL,
        "valid_complete_episodes": EXPECTED_EPISODES_PER_MODEL,
        "missing_episodes": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"{model}: official summary {key}={summary.get(key)!r}, expected {expected!r}")

    errors.extend(f"{model}: {e}" for e in verify_integrity_manifest(run_dir, integrity))

    episode_paths = sorted((run_dir / "episodes").glob("*.json"))
    block_paths = sorted((run_dir / "block_summaries").glob("*.json"))
    stabilization_paths = sorted((run_dir / "stabilization").glob("*.json"))

    if len(episode_paths) != EXPECTED_EPISODES_PER_MODEL:
        errors.append(f"{model}: found {len(episode_paths)} episode files, expected 96")
    if len(block_paths) != EXPECTED_BLOCKS_PER_MODEL:
        errors.append(f"{model}: found {len(block_paths)} block summaries, expected 48")
    if len(stabilization_paths) != EXPECTED_STABILIZATIONS_PER_MODEL:
        errors.append(f"{model}: found {len(stabilization_paths)} stabilization files, expected 48")

    episode_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    block_conditions: dict[str, set[str]] = defaultdict(set)
    cells: Counter[tuple[int, int, int, str]] = Counter()
    fingerprints: set[str] = set()

    for path in episode_paths:
        obj = read_json(path)
        row = obj.get("schedule_row")
        if not isinstance(row, dict):
            errors.append(f"{model}: {path.name}: invalid schedule_row")
            continue
        if obj.get("status") != "complete":
            errors.append(f"{model}: {path.name}: status={obj.get('status')!r}, expected 'complete'")
        if obj.get("run_mode") != "official":
            errors.append(f"{model}: {path.name}: run_mode={obj.get('run_mode')!r}")
        if row.get("model_key") != model:
            errors.append(f"{model}: {path.name}: schedule_row.model_key={row.get('model_key')!r}")
        if row.get("offload_gb") not in EXPECTED_OFFLOADS:
            errors.append(f"{model}: {path.name}: unexpected offload {row.get('offload_gb')!r}")
        if row.get("concurrency") not in EXPECTED_CONCURRENCIES:
            errors.append(f"{model}: {path.name}: unexpected concurrency {row.get('concurrency')!r}")
        if row.get("repeat") not in EXPECTED_REPEATS:
            errors.append(f"{model}: {path.name}: unexpected repeat {row.get('repeat')!r}")
        if row.get("condition") not in EXPECTED_CONDITIONS:
            errors.append(f"{model}: {path.name}: unexpected condition {row.get('condition')!r}")

        block_id = str(row.get("block_id"))
        condition = str(row.get("condition"))
        block_conditions[block_id].add(condition)
        cells[
            (
                int(row.get("offload_gb")),
                int(row.get("concurrency")),
                int(row.get("repeat")),
                condition,
            )
        ] += 1
        if isinstance(obj.get("schedule_fingerprint"), str):
            fingerprints.add(obj["schedule_fingerprint"])

        try:
            ep_row, req_rows = flatten_episode(model, path, obj)
        except ExportError as exc:
            errors.append(str(exc))
            continue
        episode_rows.append(ep_row)
        request_rows.extend(req_rows)

    expected_cells = {
        (offload, conc, repeat, condition)
        for offload in EXPECTED_OFFLOADS
        for conc in EXPECTED_CONCURRENCIES
        for repeat in EXPECTED_REPEATS
        for condition in EXPECTED_CONDITIONS
    }
    if set(cells) != expected_cells or any(count != 1 for count in cells.values()):
        errors.append(f"{model}: episode grid is not exactly 3×2×8×2 with one episode per cell")
    if len(block_conditions) != EXPECTED_BLOCKS_PER_MODEL:
        errors.append(f"{model}: found {len(block_conditions)} distinct blocks, expected 48")
    for block_id, conditions in sorted(block_conditions.items()):
        if conditions != EXPECTED_CONDITIONS:
            errors.append(f"{model}: block {block_id} conditions={sorted(conditions)}")
    if len(fingerprints) != 1:
        errors.append(f"{model}: expected one schedule fingerprint, found {sorted(fingerprints)}")
    elif summary.get("schedule_fingerprint") not in fingerprints:
        errors.append(f"{model}: episode fingerprint differs from official summary")

    block_rows: list[dict[str, Any]] = []
    for path in block_paths:
        obj = read_json(path)
        row = {
            "model": model,
            "block_id": obj.get("block_id"),
            "offload_gb": obj.get("offload_gb"),
            "state_label": obj.get("state_label"),
            "repeat": obj.get("repeat"),
            "overall_status": obj.get("overall_status"),
            "stabilization_status": obj.get("stabilization_status"),
            "server_stop_success": nested(obj, "server_stop", "stop_success"),
            "port_free_after_stop": nested(obj, "server_stop", "port_free_after_stop"),
            "forced_kill": nested(obj, "server_stop", "forced_kill"),
            "start_utc": obj.get("start_utc"),
            "end_utc": obj.get("end_utc"),
        }
        if row["overall_status"] != "block_complete":
            errors.append(f"{model}: block {path.name} overall_status={row['overall_status']!r}")
        if row["stabilization_status"] != "complete":
            errors.append(f"{model}: block {path.name} stabilization_status={row['stabilization_status']!r}")
        if row["server_stop_success"] is not True or row["port_free_after_stop"] is not True:
            errors.append(f"{model}: block {path.name} server stop/port release failed")
        block_rows.append(row)

    stabilization_rows = collect_stabilizations(model, run_dir)
    for row in stabilization_rows:
        if row["status"] != "complete" or row["functional_passed"] is not True:
            errors.append(f"{model}: stabilization {row['block_id']} not complete")

    return errors, summary, manifest, integrity, episode_rows, request_rows, block_rows + stabilization_rows


def split_block_and_stabilization_rows(
    mixed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    stabs: list[dict[str, Any]] = []
    for row in mixed_rows:
        if "relative_tpot_change" in row:
            stabs.append(row)
        else:
            blocks.append(row)
    return blocks, stabs


def make_report(
    run_dirs: dict[str, Path],
    campaign_errors: list[str],
    episode_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    group_summaries: list[dict[str, Any]],
    interaction_summaries: list[dict[str, Any]],
    stabilization_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("Prefill-Confirmation Compact Export v2 (Schema-5 aware)")
    lines.append("=" * 76)
    lines.append(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    for model in EXPECTED_MODELS:
        lines.append(f"{model}: {run_dirs[model]}")
    lines.append("")
    lines.append("VALIDATION")
    lines.append("-" * 76)
    if campaign_errors:
        lines.append(f"FAIL: {len(campaign_errors)} issue(s)")
        lines.extend(f"  - {error}" for error in campaign_errors)
    else:
        lines.append("PASS: both initial-8 campaigns complete, structurally valid, and hashes match")
    lines.append("")
    lines.append(f"Episode rows:       {len(episode_rows)}")
    lines.append(f"Victim rows:        {len(request_rows)}")
    lines.append(f"Paired blocks:      {len(pairs)}")
    lines.append(f"Stabilization rows: {len(stabilization_rows)}")
    lines.append("")

    lines.append("PRIMARY PAIRED EFFECT: log(TPOT_burst / TPOT_no_burst)")
    lines.append("-" * 76)
    lookup = [row for row in group_summaries if row["metric"] == "log_tpot_ratio"]
    for row in sorted(lookup, key=lambda r: (r["model"], r["offload_gb"], r["concurrency"])):
        pct = (math.exp(row["mean"]) - 1.0) * 100.0 if row["mean"] is not None else None
        lo = (math.exp(row["bootstrap_mean_ci_low"]) - 1.0) * 100.0 if row["bootstrap_mean_ci_low"] is not None else None
        hi = (math.exp(row["bootstrap_mean_ci_high"]) - 1.0) * 100.0 if row["bootstrap_mean_ci_high"] is not None else None
        lines.append(
            f"{row['model']:5s} offload={row['offload_gb']:2d} conc={row['concurrency']}: "
            f"n={row['n']} geometric slowdown={fmt(pct)}% "
            f"mean95%=[{fmt(lo)}%, {fmt(hi)}%] positive={row['positive_count']}/{row['n']}"
        )
    lines.append("")

    lines.append("PRIMARY STATE × BURST: difference of paired log TPOT ratios")
    lines.append("-" * 76)
    state_rows = [
        row for row in interaction_summaries
        if row["contrast_type"] == "state_x_burst" and row["metric"] == "log_tpot_ratio"
    ]
    for row in sorted(state_rows, key=lambda r: (r["model"], str(r["comparison"]), r["concurrency"])):
        pct = (math.exp(row["mean"]) - 1.0) * 100.0 if row["mean"] is not None else None
        lo = (math.exp(row["bootstrap_mean_ci_low"]) - 1.0) * 100.0 if row["bootstrap_mean_ci_low"] is not None else None
        hi = (math.exp(row["bootstrap_mean_ci_high"]) - 1.0) * 100.0 if row["bootstrap_mean_ci_high"] is not None else None
        lines.append(
            f"{row['model']:5s} {row['comparison']} conc={row['concurrency']}: "
            f"n={row['n']} relative interaction={fmt(pct)}% "
            f"mean95%=[{fmt(lo)}%, {fmt(hi)}%] positive={row['positive_count']}/{row['n']}"
        )
    lines.append("")

    lines.append("QUEUE/WAVE SECONDARIES INCLUDED")
    lines.append("-" * 76)
    lines.append("later/active-wave queue wait, E2EL-from-task-creation, exposure-group summaries")
    lines.append("Llama-vs-Qwen comparisons are exported as UNPAIRED bootstrap contrasts.")
    lines.append("")

    rel_tpot = [safe_float(row.get("relative_tpot_change")) for row in stabilization_rows if safe_float(row.get("relative_tpot_change")) is not None]
    rel_ttft = [safe_float(row.get("relative_ttft_change")) for row in stabilization_rows if safe_float(row.get("relative_ttft_change")) is not None]
    lines.append("STABILIZATION DRIFT (second half relative to first half)")
    lines.append("-" * 76)
    lines.append(
        f"TPOT: n={len(rel_tpot)}, median={fmt(median(rel_tpot))}, "
        f"p95(abs)={fmt(percentile([abs(v) for v in rel_tpot], 0.95))}, "
        f"max(abs)={fmt(max((abs(v) for v in rel_tpot), default=0.0))}"
    )
    lines.append(
        f"TTFT: n={len(rel_ttft)}, median={fmt(median(rel_ttft))}, "
        f"p95(abs)={fmt(percentile([abs(v) for v in rel_ttft], 0.95))}, "
        f"max(abs)={fmt(max((abs(v) for v in rel_ttft), default=0.0))}"
    )
    lines.append("")
    lines.append("Upload the generated ZIP for the full scientific interpretation.")
    return "\n".join(lines) + "\n"

def copy_metadata_files(output_dir: Path, model: str, run_dir: Path) -> None:
    target = output_dir / "metadata" / model
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "official_run_summary.json",
        "official_run_manifest.json",
        "integrity_manifest.json",
    ):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, target / name)


def make_zip(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == zip_path.resolve():
                continue
            zf.write(path, path.relative_to(output_dir).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="DOS project root (default: current working directory)",
    )
    parser.add_argument("--llama-dir", type=Path, default=None)
    parser.add_argument("--qwen-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory; default is under new/runs/prefill_confirmation/",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
        help="Deterministic bootstrap iterations per summary (default: 5000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if args.bootstrap_iterations < 100:
        raise ExportError("--bootstrap-iterations must be at least 100")

    run_dirs = {
        "llama": (
            args.llama_dir.expanduser().resolve()
            if args.llama_dir
            else find_latest_official(project_root, "llama")
        ),
        "qwen": (
            args.qwen_dir.expanduser().resolve()
            if args.qwen_dir
            else find_latest_official(project_root, "qwen")
        ),
    }

    stamp = utc_stamp()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "new" / "runs" / "prefill_confirmation" / f"analysis_export_{stamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExportError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_errors: list[str] = []
    all_episode_rows: list[dict[str, Any]] = []
    all_request_rows: list[dict[str, Any]] = []
    all_block_rows: list[dict[str, Any]] = []
    all_stabilization_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for model in EXPECTED_MODELS:
        print(f"[{model}] reading {run_dirs[model]}")
        (
            errors,
            summary,
            manifest,
            integrity,
            episode_rows,
            request_rows,
            mixed_rows,
        ) = validate_campaign(model, run_dirs[model])
        block_rows, stabilization_rows = split_block_and_stabilization_rows(mixed_rows)
        all_errors.extend(errors)
        all_episode_rows.extend(episode_rows)
        all_request_rows.extend(request_rows)
        all_block_rows.extend(block_rows)
        all_stabilization_rows.extend(stabilization_rows)
        summaries[model] = {
            "run_dir": str(run_dirs[model]),
            "official_summary": summary,
            "environment_fingerprint": manifest.get("environment_fingerprint"),
            "integrity_file_count": integrity.get("file_count"),
            "validation_errors": errors,
        }
        copy_metadata_files(output_dir, model, run_dirs[model])

    pairs = build_pairs(all_episode_rows)
    if len(pairs) != 96:
        all_errors.append(f"Across both models: found {len(pairs)} paired blocks, expected 96")

    rng = random.Random(20260718)
    group_summaries = build_group_summaries(
        pairs, rng=rng, bootstrap_iterations=args.bootstrap_iterations
    )
    interaction_detail, _ = build_interaction_contrasts(pairs)
    interaction_summaries = summarize_contrasts(
        interaction_detail,
        rng=rng,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    exposure_episode_rows = build_exposure_episode_summaries(all_request_rows)
    exposure_pairs = build_exposure_pairs(exposure_episode_rows)
    unpaired_model_comparisons = build_unpaired_model_comparisons(
        pairs, rng=rng, bootstrap_iterations=args.bootstrap_iterations
    )

    write_csv(output_dir / "episodes.csv", all_episode_rows)
    write_csv(output_dir / "victim_requests.csv", all_request_rows)
    write_csv(output_dir / "paired_effects.csv", pairs)
    write_csv(output_dir / "cell_summaries.csv", group_summaries)
    write_csv(output_dir / "interaction_contrasts_detail.csv", interaction_detail)
    write_csv(output_dir / "interaction_summaries.csv", interaction_summaries)
    write_csv(output_dir / "exposure_group_episode_summaries.csv", exposure_episode_rows)
    write_csv(output_dir / "exposure_group_paired_effects.csv", exposure_pairs)
    write_csv(output_dir / "model_comparison_unpaired.csv", unpaired_model_comparisons)
    write_csv(output_dir / "stabilization.csv", all_stabilization_rows)
    write_csv(output_dir / "block_summaries.csv", all_block_rows)

    validation = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_directories": {k: str(v) for k, v in run_dirs.items()},
        "campaigns": summaries,
        "counts": {
            "episodes": len(all_episode_rows),
            "victim_requests": len(all_request_rows),
            "paired_blocks": len(pairs),
            "block_summaries": len(all_block_rows),
            "stabilizations": len(all_stabilization_rows),
            "exposure_group_episode_rows": len(exposure_episode_rows),
            "exposure_group_pairs": len(exposure_pairs),
        },
        "validation_passed": not all_errors,
        "validation_errors": all_errors,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": 20260718,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = make_report(
        run_dirs,
        all_errors,
        all_episode_rows,
        all_request_rows,
        pairs,
        group_summaries,
        interaction_summaries,
        all_stabilization_rows,
    )
    (output_dir / "REPORT.txt").write_text(report, encoding="utf-8")

    readme = """Prefill-Confirmation compact analysis export v2

Files:
- validation.json: campaign completeness, paths, counts, integrity result
- REPORT.txt: quick local overview
- episodes.csv: one row per regular episode (192 total)
- victim_requests.csv: one compact row per victim request (3840 total)
- paired_effects.csv: one no_burst/prefill_burst pair per block (96 total)
- cell_summaries.csv: paired burst effects by model/offload/concurrency
- interaction_contrasts_detail.csv: repeat-matched state/concurrency/model contrasts
- interaction_summaries.csv: repeat-matched within-model interaction summaries
- exposure_group_episode_summaries.csv: Schema-5 exposure-group metrics per episode
- exposure_group_paired_effects.csv: paired burst effects within exposure groups
- model_comparison_unpaired.csv: independent Llama/Qwen bootstrap comparisons
- stabilization.csv: first-half/second-half drift per server start
- block_summaries.csv: lifecycle/stop status per block
- metadata/: original official summaries/manifests/integrity manifests

Excluded deliberately:
- raw SSE event payloads
- prompt token IDs
- generated token IDs
- server logs
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    zip_path = output_dir.parent / f"prefill_confirmation_compact_{stamp}.zip"
    make_zip(output_dir, zip_path)

    print()
    print(report, end="")
    print(f"Export directory: {output_dir}")
    print(f"Compact ZIP:      {zip_path}")
    print(f"ZIP size:         {zip_path.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"Validation:       {'PASS' if not all_errors else 'FAIL'}")
    if all_errors:
        print("The compact artifacts were still written for diagnosis.")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
