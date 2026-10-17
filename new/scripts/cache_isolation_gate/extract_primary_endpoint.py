#!/usr/bin/env python3
"""Read-only extraction of a fixed-cohort token-index endpoint from Raw-SSE.

This tool sends no requests and validates no causal study design. It writes JSON
to stdout. Cohort, token index and observation origin must be supplied explicitly.
"""
import argparse
import json
from pathlib import Path
import sys


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: expected nonnegative integer")
    return value


def receipt_at_token(record, target):
    events = record.get("raw_sse_events")
    if not isinstance(events, list) or not events:
        raise ValueError("missing raw_sse_events")
    tokens = []
    previous = None
    crossing = None
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("invalid SSE event")
        if type(event.get("event_index")) is not int or event.get("event_index") != index:
            raise ValueError("noncontiguous event_index")
        timestamp = integer(event.get("receive_perf_counter_ns"), "SSE timestamp")
        if previous is not None and timestamp < previous:
            raise ValueError("nonmonotonic SSE timestamps")
        previous = timestamp
        ids = event.get("token_ids")
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid SSE token_ids")
        tokens.extend(ids)
        if ids and crossing is None and len(tokens) >= target:
            crossing = timestamp
    if tokens != record.get("output_token_ids"):
        raise ValueError("SSE/output token reconstruction mismatch")
    start = integer(record.get("request_start_ns"), "request_start_ns")
    end = integer(record.get("stream_end_ns"), "stream_end_ns")
    if end < start or any(not start <= e["receive_perf_counter_ns"] <= end for e in events):
        raise ValueError("SSE outside request time bounds")
    return crossing


def field_value(obj, dotted_path):
    for part in dotted_path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise ValueError(f"missing origin field {dotted_path}")
        obj = obj[part]
    return integer(obj, "observation origin")


def extract_episode(episode, cohort, target, origin_field):
    if type(target) is not int or target < 1:
        raise ValueError("token index must be >= 1")
    if not cohort or len(cohort) != len(set(cohort)) or any(type(i) is not int or i < 0 for i in cohort):
        raise ValueError("cohort must contain distinct nonnegative request indices")
    if not isinstance(episode, dict):
        raise ValueError("episode must be an object")
    origin = field_value(episode, origin_field)
    rows = episode.get("victim_requests")
    if not isinstance(rows, list):
        raise ValueError("missing victim_requests")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid victim record")
        index = integer(row.get("request_index"), "request_index")
        if index in by_index:
            raise ValueError("duplicate request_index")
        by_index[index] = row
    result = {
        "episode_id": episode.get("episode_id"),
        "block_id": episode.get("block_id"),
        "condition": episode.get("schedule_row", {}).get("condition"),
        "cohort_indices": list(cohort),
        "token_index_1based": target,
        "origin_field": origin_field,
        "origin_ns": origin,
        "status": "observed",
        "cohort_elapsed_ms": None,
        "design_validity_checked": False,
        "members": [],
        "issues": [],
    }
    if episode.get("status") != "complete":
        result["issues"].append("episode_not_complete")
    receipts = []
    for index in cohort:
        row = by_index.get(index)
        if row is None:
            result["issues"].append(f"request_{index}:missing_request")
            continue
        if row.get("status") != "complete":
            result["issues"].append(f"request_{index}:request_not_complete")
            result["members"].append({"request_index": index, "target_receipt_ns": None})
            continue
        crossing = receipt_at_token(row, target)
        result["members"].append({"request_index": index, "target_receipt_ns": crossing})
        if crossing is None:
            result["issues"].append(f"request_{index}:target_not_reached")
        elif crossing < origin:
            result["issues"].append(f"request_{index}:target_before_origin")
        else:
            receipts.append(crossing)
    if result["issues"]:
        result["status"] = "incomplete_or_invalid"
    else:
        result["cohort_elapsed_ms"] = (max(receipts) - origin) / 1_000_000
    return result


def parse_cohort(text):
    try:
        indices = [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integer request indices") from exc
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("indices must be distinct and nonnegative")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--cohort", type=parse_cohort, required=True)
    parser.add_argument("--token-index", type=int, required=True)
    parser.add_argument("--origin-field", required=True)
    args = parser.parse_args()
    if args.token_index < 1:
        parser.error("--token-index must be >= 1")
    if not args.episodes.is_dir():
        parser.error("--episodes must be an existing directory")
    files = sorted(args.episodes.glob("*.json"))
    if not files:
        parser.error("no JSON episode files found")
    results = []
    failed = False
    for path in files:
        try:
            result = extract_episode(json.loads(path.read_text()), args.cohort, args.token_index, args.origin_field)
            failed |= result["status"] != "observed"
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            result = {"status": "invalid_input", "detail": str(exc), "design_validity_checked": False}
            failed = True
        result["file"] = path.name
        results.append(result)
    print(json.dumps({"mode": "offline_endpoint_extraction", "causal_design_validated": False,
                      "episodes": results}, indent=2, ensure_ascii=False))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
