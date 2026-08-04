#!/usr/bin/env python3
"""Read-only campaign-wide Raw-SSE reconstruction and compact summary."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ANALYSIS_VERSION = "llama-campaign-raw-sse-summary-v1"


def load(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: top level is not an object")
    return obj


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def med(values):
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def close(a, b, tol=1e-6) -> bool:
    return finite(a) and finite(b) and abs(float(a) - float(b)) <= tol


def raw_record(record: dict, label: str):
    errors = []
    if record.get("status") != "complete":
        errors.append(f"{label}: status={record.get('status')!r}")
    if record.get("validation_errors") != []:
        errors.append(f"{label}: validation_errors not empty")
    if record.get("http_status") != 200:
        errors.append(f"{label}: http_status={record.get('http_status')!r}")
    if record.get("done_received") is not True:
        errors.append(f"{label}: done_received is not true")
    if record.get("timed_out") is not False or record.get("cancelled") is not False:
        errors.append(f"{label}: timeout/cancellation flag set")

    events = record.get("raw_sse_events")
    if not isinstance(events, list) or not events:
        return errors + [f"{label}: raw_sse_events missing/empty"], {}

    output = []
    positive_ns = []
    batch_sizes = []
    done_count = 0
    prev_ns = None

    for pos, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"{label}: event {pos} not an object")
            continue
        if event.get("event_index") != pos:
            errors.append(f"{label}: non-contiguous event_index at {pos}")
        status = event.get("parse_status")
        if status not in ("ok", "done", "keepalive"):
            errors.append(f"{label}: invalid parse_status {status!r} at {pos}")
        if status == "done":
            done_count += 1
        ns = event.get("receive_perf_counter_ns")
        if type(ns) is not int:
            errors.append(f"{label}: receive timestamp not int at {pos}")
        else:
            if prev_ns is not None and ns < prev_ns:
                errors.append(f"{label}: non-monotonic receive timestamps")
            prev_ns = ns
        token_ids = event.get("token_ids")
        if not isinstance(token_ids, list) or not all(type(x) is int for x in token_ids):
            errors.append(f"{label}: token_ids not list[int] at {pos}")
            continue
        output.extend(token_ids)
        if token_ids and type(ns) is int:
            positive_ns.append(ns)
            batch_sizes.append(len(token_ids))

    if done_count != 1:
        errors.append(f"{label}: DONE count {done_count} != 1")
    if output != record.get("output_token_ids"):
        errors.append(f"{label}: raw token concatenation mismatch")
    expected = record.get("expected_completion_tokens")
    if type(expected) is not int or len(output) != expected:
        errors.append(f"{label}: reconstructed token count {len(output)} != {expected!r}")
    if not positive_ns:
        return errors + [f"{label}: no positive-token event"], {}

    first_ns, last_ns = positive_ns[0], positive_ns[-1]
    if record.get("first_token_receive_ns") != first_ns:
        errors.append(f"{label}: first_token_receive_ns mismatch")
    if record.get("last_token_receive_ns") != last_ns:
        errors.append(f"{label}: last_token_receive_ns mismatch")
    if record.get("role") == "victim" and record.get("first_token_perf_ns") != first_ns:
        errors.append(f"{label}: first_token_perf_ns mismatch")
    if record.get("role") == "burst" and record.get("burst_first_token_perf_ns") != first_ns:
        errors.append(f"{label}: burst_first_token_perf_ns mismatch")

    raw_tpot = None
    if len(output) > 1:
        raw_tpot = (last_ns - first_ns) / (len(output) - 1) / 1e6
        if not close(record.get("client_observed_tpot_ms"), raw_tpot):
            errors.append(f"{label}: TPOT mismatch")

    raw_itl = None
    if all(size == 1 for size in batch_sizes):
        raw_itl = [(positive_ns[i] - positive_ns[i - 1]) / 1e6 for i in range(1, len(positive_ns))]
        stored = record.get("itl_ms")
        if not isinstance(stored, list) or len(stored) != len(raw_itl):
            errors.append(f"{label}: ITL length mismatch")
        elif any(not close(a, b) for a, b in zip(stored, raw_itl)):
            errors.append(f"{label}: ITL value mismatch")

    return errors, {
        "first_ns": first_ns,
        "last_ns": last_ns,
        "tpot_ms": raw_tpot,
        "max_itl_ms": max(raw_itl) if raw_itl else None,
    }


def active_stats(ep: dict, raw_cache: dict):
    active = set((ep.get("trigger") or {}).get("active_cohort_request_indices") or [])
    victims = [r for r in ep.get("victim_requests", []) if r.get("request_index") in active]
    return {
        "active_n": len(victims),
        "median_tpot_ms": med([raw_cache.get(r.get("request_id"), {}).get("tpot_ms") for r in victims]),
        "median_max_itl_ms": med([raw_cache.get(r.get("request_id"), {}).get("max_itl_ms") for r in victims]),
        "median_completion_ms": med([
            (r["stream_end_perf_ns"] - r["task_created_perf_ns"]) / 1e6
            for r in victims
            if type(r.get("stream_end_perf_ns")) is int and type(r.get("task_created_perf_ns")) is int
        ]),
    }


def ratio(a, b):
    return float(a) / float(b) if finite(a) and finite(b) and float(b) != 0 else None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: campaign_finalize.py <copied-campaign-dir> <capture-dir>", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).resolve()
    capture = Path(sys.argv[2]).resolve()

    episode_paths = sorted((source / "episodes").glob("*.json"))
    block_paths = sorted((source / "block_summaries").glob("*.json"))
    stab_paths = sorted((source / "stabilization").glob("*.json"))
    log_paths = sorted((source / "server_logs").glob("*.log"))

    errors = []
    expected_counts = {"episodes": 32, "blocks": 16, "stabilization": 16, "logs": 16}
    actual_counts = {
        "episodes": len(episode_paths),
        "blocks": len(block_paths),
        "stabilization": len(stab_paths),
        "logs": len(log_paths),
    }
    for key, expected in expected_counts.items():
        if actual_counts[key] != expected:
            errors.append(f"{key} count {actual_counts[key]} != {expected}")

    official = load(source / "official_run_summary.json")
    if official.get("overall_status") != "complete":
        errors.append("official overall_status != complete")
    if official.get("valid_complete_episodes") != 32:
        errors.append("valid_complete_episodes != 32")
    if official.get("completed_blocks") != 16:
        errors.append("completed_blocks != 16")
    if official.get("missing_episodes") != 0:
        errors.append("missing_episodes != 0")

    episodes = [load(p) for p in episode_paths]
    pairs = defaultdict(dict)
    raw_cache = {}

    for ep in episodes:
        row = ep.get("schedule_row") or {}
        block = ep.get("block_id")
        condition = row.get("condition")
        if not isinstance(block, str) or condition not in ("no_burst", "prefill_burst"):
            errors.append(f"{ep.get('episode_id')}: invalid block/condition")
            continue
        pairs[block][condition] = ep
        for key in ("victim_requests", "burst_requests"):
            records = ep.get(key)
            if not isinstance(records, list):
                errors.append(f"{ep.get('episode_id')}: {key} not a list")
                continue
            for rec in records:
                if not isinstance(rec, dict):
                    errors.append(f"{ep.get('episode_id')}: invalid request record")
                    continue
                rid = rec.get("request_id")
                rec_errors, raw = raw_record(rec, f"{ep.get('episode_id')}:{rid}")
                errors.extend(rec_errors)
                if isinstance(rid, str):
                    raw_cache[rid] = raw

    rows = []
    class_counts = Counter()
    raw_class_counts = Counter()

    for block in sorted(pairs):
        pair = pairs[block]
        no = pair.get("no_burst")
        burst = pair.get("prefill_burst")
        pair_errors = []
        if no is None or burst is None:
            pair_errors.append("missing pair member")
            no, burst = no or {}, burst or {}

        for label, ep, expected_bursts in (("no_burst", no, 0), ("prefill_burst", burst, 4)):
            if ep.get("status") != "complete" or ep.get("validation_errors") != []:
                pair_errors.append(f"{label}: episode invalid")
            victims = ep.get("victim_requests") or []
            bursts = ep.get("burst_requests") or []
            trigger = ep.get("trigger") or {}
            row = ep.get("schedule_row") or {}
            k = row.get("server_max_num_seqs")
            trigger_ns = trigger.get("trigger_perf_ns")
            active = trigger.get("active_cohort_request_indices") or []
            if len(victims) != 20 or len(bursts) != expected_bursts:
                pair_errors.append(f"{label}: wrong request counts")
            if trigger.get("status") != "ok" or type(trigger_ns) is not int:
                pair_errors.append(f"{label}: invalid trigger")
                continue
            if type(k) is not int or len(active) != k or len(set(active)) != k:
                pair_errors.append(f"{label}: invalid active cohort")
            by_idx = {r.get("request_index"): r for r in victims if isinstance(r, dict)}
            if set(by_idx) != set(range(20)):
                pair_errors.append(f"{label}: victim indices != 0..19")
                continue
            t16 = [by_idx[i].get("token_16_perf_ns") for i in active]
            if not t16 or not all(type(x) is int for x in t16) or trigger_ns != max(t16):
                pair_errors.append(f"{label}: trigger reconstruction failed")
            active_set = set(active)
            for i, rec in by_idx.items():
                dispatch, first = rec.get("request_dispatch_ns"), rec.get("first_token_perf_ns")
                if type(dispatch) is not int or dispatch > trigger_ns:
                    pair_errors.append(f"{label}: victim {i} not dispatched by trigger")
                if i in active_set:
                    if type(first) is not int or first > trigger_ns:
                        pair_errors.append(f"{label}: active victim {i} lacks output")
                else:
                    if type(first) is not int or first <= trigger_ns:
                        pair_errors.append(f"{label}: nonactive victim {i} had output")
                    if rec.get("decode_tokens_received_at_trigger") != 0:
                        pair_errors.append(f"{label}: nonactive victim {i} token count != 0")
            opens = [r.get("stream_open_or_response_headers_perf_ns") for r in victims]
            firsts = [r.get("first_token_perf_ns") for r in victims]
            if not all(type(x) is int for x in opens + firsts) or not all(x < min(firsts) for x in opens):
                pair_errors.append(f"{label}: stream-open gate failed")
            if ep.get("no_client_admission_semaphore_used") is not True:
                pair_errors.append(f"{label}: client semaphore gate failed")

        row = burst.get("schedule_row") or {}
        active = set((burst.get("trigger") or {}).get("active_cohort_request_indices") or [])
        victims = {r.get("request_index"): r for r in burst.get("victim_requests", []) if isinstance(r, dict)}
        bursts = [r for r in burst.get("burst_requests", []) if isinstance(r, dict)]
        cls = raw_cls = "D"
        first_burst = last_active_stream = last_all_stream = last_active_raw = last_all_raw = None

        try:
            if not pair_errors:
                first_burst = min(raw_cache[r["request_id"]]["first_ns"] for r in bursts)
                last_active_stream = max(victims[i]["stream_end_perf_ns"] for i in active)
                last_all_stream = max(r["stream_end_perf_ns"] for r in victims.values())
                last_active_raw = max(raw_cache[victims[i]["request_id"]]["last_ns"] for i in active)
                last_all_raw = max(raw_cache[r["request_id"]]["last_ns"] for r in victims.values())
                cls = "A" if first_burst < last_active_stream else "C" if first_burst > last_all_stream else "B"
                raw_cls = "A" if first_burst < last_active_raw else "C" if first_burst > last_all_raw else "B"
                if cls != raw_cls:
                    pair_errors.append(f"stream-end class {cls} != raw-token class {raw_cls}")
                    cls = "D"
        except (KeyError, TypeError, ValueError) as exc:
            pair_errors.append(f"classification reconstruction failed: {exc}")
            cls = raw_cls = "D"

        no_stats, burst_stats = active_stats(no, raw_cache), active_stats(burst, raw_cache)
        row_out = {
            "block_id": block,
            "offload_gb": row.get("offload_gb"),
            "server_max_num_seqs": row.get("server_max_num_seqs"),
            "repeat": row.get("repeat"),
            "classification": cls,
            "raw_output_classification": raw_cls,
            "active_tpot_ratio": ratio(burst_stats.get("median_tpot_ms"), no_stats.get("median_tpot_ms")),
            "active_max_itl_ratio": ratio(burst_stats.get("median_max_itl_ms"), no_stats.get("median_max_itl_ms")),
            "active_completion_delta_ms": (
                burst_stats["median_completion_ms"] - no_stats["median_completion_ms"]
                if finite(burst_stats.get("median_completion_ms")) and finite(no_stats.get("median_completion_ms"))
                else None
            ),
            "burst_after_active_stream_end_s": (
                (first_burst - last_active_stream) / 1e9 if type(first_burst) is int and type(last_active_stream) is int else None
            ),
            "remaining_until_all_stream_end_s": (
                (last_all_stream - first_burst) / 1e9 if type(first_burst) is int and type(last_all_stream) is int else None
            ),
            "pair_errors": pair_errors,
        }
        rows.append(row_out)
        class_counts[cls] += 1
        raw_class_counts[raw_cls] += 1
        errors.extend(f"{block}: {e}" for e in pair_errors)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["offload_gb"], row["server_max_num_seqs"])].append(row)
    cells = []
    for (offload, k), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        valid_rows = [r for r in group if r["classification"] != "D"]
        cells.append({
            "offload_gb": offload,
            "server_max_num_seqs": k,
            "pair_count": len(group),
            "class_counts": dict(sorted(Counter(r["classification"] for r in group).items())),
            "median_active_tpot_ratio": med([r["active_tpot_ratio"] for r in valid_rows]),
            "median_active_max_itl_ratio": med([r["active_max_itl_ratio"] for r in valid_rows]),
            "median_active_completion_delta_ms": med([r["active_completion_delta_ms"] for r in valid_rows]),
            "median_burst_after_active_stream_end_s": med([r["burst_after_active_stream_end_s"] for r in valid_rows]),
            "median_remaining_until_all_stream_end_s": med([r["remaining_until_all_stream_end_s"] for r in valid_rows]),
        })

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "source_campaign": str(source),
        "technical_pass": not errors,
        "errors": errors,
        "artifact_counts": actual_counts,
        "official_run_summary": official,
        "classification_counts": dict(sorted(class_counts.items())),
        "raw_output_classification_counts": dict(sorted(raw_class_counts.items())),
        "blocks": rows,
        "cells": cells,
        "scope_boundary": "Client-visible output timing only; internal admission and prefill-start times are not observed.",
    }

    json_path = capture / "campaign_analysis_summary.json"
    txt_path = capture / "campaign_analysis_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "Llama Server-WAITING Campaign — Raw-SSE Summary",
        "=" * 58,
        f"technical_pass: {summary['technical_pass']}",
        f"errors: {len(errors)}",
        f"classification_counts: {summary['classification_counts']}",
        f"raw_output_classification_counts: {summary['raw_output_classification_counts']}",
        "",
        "Cells:",
    ]
    for cell in cells:
        lines.append(
            f"offload={cell['offload_gb']} k={cell['server_max_num_seqs']} "
            f"classes={cell['class_counts']} TPOT={cell['median_active_tpot_ratio']} "
            f"ITL={cell['median_active_max_itl_ratio']} completion_ms={cell['median_active_completion_delta_ms']} "
            f"after_active_s={cell['median_burst_after_active_stream_end_s']} "
            f"remaining_all_s={cell['median_remaining_until_all_stream_end_s']}"
        )
    lines += ["", "Scope:", summary["scope_boundary"]]
    if errors:
        lines += ["", "Errors:"] + [f"- {e}" for e in errors]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"analysis_json={json_path}")
    print(f"analysis_txt={txt_path}")
    print(f"technical_pass={summary['technical_pass']}")
    print(f"classification_counts={summary['classification_counts']}")
    print(f"raw_output_classification_counts={summary['raw_output_classification_counts']}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
