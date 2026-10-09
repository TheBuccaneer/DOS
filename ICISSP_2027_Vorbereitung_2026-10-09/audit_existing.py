#!/usr/bin/env python3
"""Read-only integrity and observability inventory of selected historical runs.

Uses only the standard library, sends no requests, imports no experiment runners.
Writes compact JSON to stdout; does not establish validity of the new experiment.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

CAMPAIGNS = (
    "chunk_budget_screen/results/official_20260716T075455Z",
    "server_waiting_confirmation/llama/official_campaign_20260804T141510Z",
    "server_waiting_confirmation/qwen/official_campaign_fixed_20260803T183032Z",
)
METRIC = re.compile(r"^(vllm:[a-zA-Z_:0-9]+)", re.M)


def check_stream(record):
    errors = []
    reconstructed = []
    times = []
    previous = None
    events = record.get("raw_sse_events")
    if not isinstance(events, list) or not events:
        return ["missing_raw_sse"]
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append("invalid_event")
            continue
        ns = event.get("receive_perf_counter_ns")
        ids = event.get("token_ids")
        if event.get("event_index") != index:
            errors.append("event_index")
        if type(ns) is not int or ns < 0:
            errors.append("invalid_timestamp")
        else:
            if previous is not None and ns < previous:
                errors.append("nonmonotonic_timestamp")
            previous = ns
        if not isinstance(ids, list) or not all(type(v) is int for v in ids):
            errors.append("invalid_token_ids")
            continue
        reconstructed.extend(ids)
        if ids and type(ns) is int:
            times.append(ns)
    if reconstructed != record.get("output_token_ids"):
        errors.append("output_reconstruction")
    if len(reconstructed) != record.get("expected_completion_tokens"):
        errors.append("completion_token_count")
    if not times:
        errors.append("no_output_tokens")
    else:
        if times[0] != record.get("first_token_receive_ns"):
            errors.append("first_token_timestamp")
        if times[-1] != record.get("last_token_receive_ns"):
            errors.append("last_token_timestamp")
        start, end = record.get("request_start_ns"), record.get("stream_end_ns")
        if type(start) is not int or type(end) is not int or not start <= times[0] <= times[-1] <= end:
            errors.append("request_time_bounds")
    return errors


def request_identities(episode):
    rows = episode.get("victim_requests", [])
    identities = {}
    for row in rows:
        index = row["request_index"]
        if index in identities:
            raise ValueError("duplicate_request_index")
        identities[index] = {
            key: row.get(key) for key in (
                "prompt_sha256", "prompt_token_ids_sent", "generation_seed",
                "expected_prompt_tokens", "expected_completion_tokens",
            )
        }
    return identities


def audit_campaign(root, relative):
    base = root / "new/runs" / relative
    files = sorted((base / "episodes").glob("*.json"))
    summary = {"path": relative, "episode_count": len(files)}
    errors = []
    if not files:
        errors.append({"path": relative, "error": "no_episodes"})
    groups = defaultdict(list)
    metrics = set()
    statuses, quality = Counter(), Counter()
    total_requests = total_tokens = samples = after = 0
    for file in files:
        episode = json.loads(file.read_text())
        statuses[episode.get("status")] += 1
        if episode.get("status") != "complete":
            errors.append({"file": file.name, "error": "incomplete_episode"})
        groups[episode["block_id"]].append(episode)
        for role in ("victim_requests", "burst_requests"):
            for record in episode.get(role, []):
                total_requests += 1
                total_tokens += len(record.get("output_token_ids", []))
                issues = check_stream(record)
                if issues:
                    errors.append({"file": file.name, "role": role,
                                   "index": record.get("request_index"), "errors": issues})
        trigger = episode.get("trigger", {})
        quality[trigger.get("metrics_quality", {}).get("metrics_quality_status", "absent")] += 1
        trigger_ns = trigger.get("trigger_perf_ns")
        for sample in trigger.get("metrics_raw_samples", []):
            samples += 1
            received = sample.get("response_received_perf_ns")
            if type(trigger_ns) is int and type(received) is int and received > trigger_ns:
                after += 1
            metrics.update(METRIC.findall(sample.get("raw_body") or ""))
    matched = mismatch = cohorts_differ = 0
    orders = Counter()
    for block, pair in groups.items():
        if len(pair) != 2 or {e["schedule_row"]["condition"] for e in pair} != {"no_burst", "prefill_burst"}:
            errors.append({"block": block, "error": "invalid_pair"})
            continue
        ordered = sorted(pair, key=lambda e: e["schedule_row"].get("within_block_order", e["schedule_row"].get("order_in_block")))
        orders[ordered[0]["schedule_row"]["condition"]] += 1
        if request_identities(pair[0]) == request_identities(pair[1]):
            matched += 1
        else:
            mismatch += 1
            errors.append({"block": block, "error": "pair_request_identity_mismatch"})
        cohorts = [e.get("trigger", {}).get("active_cohort_request_indices") for e in pair]
        if all(c is not None for c in cohorts) and set(cohorts[0]) != set(cohorts[1]):
            cohorts_differ += 1
    manifest = base / "official_run_manifest.json"
    obj = json.loads(manifest.read_text()) if manifest.exists() else {}
    summary.update({
        "historical_environment": {k: obj.get(k) for k in (
            "vllm_version", "torch_version", "transformers_version", "python_version",
            "python_executable", "hostname", "git_commit")},
        "request_count": total_requests, "output_token_count": total_tokens,
        "episode_statuses": dict(statuses), "matched_request_identity_pairs": matched,
        "mismatched_request_identity_pairs": mismatch, "pair_first_conditions": dict(orders),
        "pairs_with_different_observed_cohorts": cohorts_differ,
        "stored_metrics_samples": samples, "stored_samples_after_logical_trigger": after,
        "metrics_quality": dict(quality), "metric_names": sorted(metrics),
        "integrity_errors": errors,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    reports = [audit_campaign(root, c) for c in CAMPAIGNS]
    source_paths = (
        "new/scripts/prefill_confirmation/run_prefill_confirmation.py",
        "new/scripts/chunk_budget_screen/run_chunk_budget_screen.py",
        "new/scripts/server_waiting_confirmation/run_server_waiting_confirmation.py",
    )
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "mode": "offline_read_only",
        "new_experiment_gate": "NOT_CONFIRMED",
        "git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths},
        "campaigns": reports,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return int(any(r["integrity_errors"] for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
