#!/usr/bin/env bash
# Comprehensive, read-only re-audit of the completed Llama campaign.
# It starts no server, sends no requests, and never modifies the evidence tree.
set -Eeuo pipefail
umask 022

VERSION="llama-full-campaign-reaudit-v1"
RESULT_ROOT="$HOME/projects/DOS/new/runs/server_waiting_confirmation/llama"
TARGET=""

usage() {
    cat <<EOF
$VERSION

Verwendung:
  $0
  $0 /pfad/zu/final_capture_...
  $0 /pfad/zu/official_campaign_...

Das Skript prüft die gesamte Llama-Kampagne erneut:
  - äußeres und inneres Integritätsmanifest
  - alle Pflichtdateien und Artefaktzahlen
  - 32 Episoden, 16 Blöcke, 16 Stabilisierungsläufe, 16 Serverlogs
  - Schedule-/Environment-Bindung und 2/2-Gegenbalancierung
  - alle 704 Requests und alle Raw-SSE-Events
  - Prompt-/Output-Tokenbindung, First/Last Token, ITL und TPOT
  - Trigger, aktive Kohorte und serverseitige Running/Waiting-Metrik
  - A/B/C/D je Block auf Stream-End- und Raw-Token-Ebene
  - wissenschaftliche Zellzusammenfassungen und Ausreißerzerlegung
  - unterschiedliche aktive Request-Indizes werden dokumentiert, aber
    NICHT fälschlich als technischer Fehler behandelt

Ausgaben:
  ~/Downloads/llama_full_reaudit_<ZEITSTEMPEL>/
  ~/Downloads/llama_full_reaudit_<ZEITSTEMPEL>.tar.gz

Die Messdaten werden ausschließlich gelesen.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if test -n "${1:-}"; then
    TARGET="$1"
else
    TARGET="$(find "$RESULT_ROOT" -maxdepth 1 -type d -name 'final_capture_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
    test -n "$TARGET" || { echo "FEHLER: Kein final_capture_* unter $RESULT_ROOT gefunden." >&2; exit 2; }
fi
TARGET="$(readlink -f "$TARGET")"
test -d "$TARGET" || {
    echo "FEHLER: Verzeichnis fehlt: $TARGET" >&2
    exit 2
}

if test -d "$TARGET/source_campaign/episodes"; then
    CAMPAIGN="$TARGET/source_campaign"
elif test -d "$TARGET/episodes"; then
    CAMPAIGN="$TARGET"
else
    echo "FEHLER: Kein source_campaign/episodes oder episodes unter $TARGET." >&2
    exit 2
fi

PYTHON="${VENV_PATH:-$HOME/projects/porto/.venv-py313-repair}/bin/python"
if test ! -x "$PYTHON"; then
    PYTHON="$(command -v python3 || true)"
fi
test -n "$PYTHON" && test -x "$PYTHON" || {
    echo "FEHLER: Kein ausführbares Python gefunden." >&2
    exit 2
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="$HOME/Downloads/llama_full_reaudit_${STAMP}"
REPORT="$OUTDIR/REPORT.txt"
mkdir -p "$OUTDIR"

echo "=== LLAMA FULL CAMPAIGN RE-AUDIT ==="
echo "VERSION=$VERSION"
echo "TARGET=$TARGET"
echo "CAMPAIGN=$CAMPAIGN"
echo "OUTPUT_DIR=$OUTDIR"
echo "READ_ONLY=yes"
echo

set +e
"$PYTHON" - "$TARGET" "$CAMPAIGN" "$OUTDIR" <<'PY' | tee "$REPORT"
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

target = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
outdir = Path(sys.argv[3]).resolve()

EXPECTED = {
    "episodes": (32, "*.json"),
    "block_summaries": (16, "*.json"),
    "stabilization": (16, "*.json"),
    "server_logs": (16, "*.log"),
}
EXPECTED_CELLS = {(0, 4), (0, 8), (12, 4), (12, 8)}
ALLOWED_PARSE = {"ok", "done", "keepalive"}

errors: list[str] = []
warnings: list[str] = []
raw_errors: list[str] = []
raw_warnings: list[str] = []


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 6) -> str:
    return "NA" if not finite(value) else f"{float(value):.{digits}f}"


def median(values) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def value_range(values) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if finite(v)]
    return (min(vals), max(vals)) if vals else (None, None)


def ratio(a: Any, b: Any) -> float | None:
    if finite(a) and finite(b) and float(b) != 0:
        return float(a) / float(b)
    return None


def pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: JSON nicht lesbar: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: Top-Level ist kein Objekt")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_text_manifest(
    manifest_path: Path,
    base_dir: Path,
) -> tuple[str, int, list[str]]:
    if not manifest_path.is_file():
        return "NOT_AVAILABLE", 0, []

    entry_pattern = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
    entries: dict[str, str] = {}
    local_errors: list[str] = []

    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = entry_pattern.match(line)
        if not match:
            local_errors.append(
                f"{manifest_path.name}: ungültige Zeile {line_number}: {line!r}"
            )
            continue
        rel = match.group(2).removeprefix("./")
        entries[rel] = match.group(1).lower()

    for rel, expected in sorted(entries.items()):
        candidate = base_dir / rel
        if not candidate.is_file():
            local_errors.append(f"{manifest_path.name}: fehlt: {rel}")
        elif sha256_file(candidate) != expected:
            local_errors.append(f"{manifest_path.name}: Hashabweichung: {rel}")

    actual = {
        path.relative_to(base_dir).as_posix()
        for path in base_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    listed = set(entries)
    for rel in sorted(actual - listed):
        local_errors.append(f"{manifest_path.name}: nicht gelistet: {rel}")
    for rel in sorted(listed - actual):
        local_errors.append(f"{manifest_path.name}: gelistet, aber fehlt: {rel}")

    return (
        "PASS" if entries and not local_errors else "FAIL",
        len(entries),
        local_errors,
    )


def extract_json_manifest_entries(manifest: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    files = manifest.get("files")

    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = pick(item, "relative_path", "path", "filename", "file")
            digest = pick(item, "sha256", "hash", "digest")
            if isinstance(rel, str) and isinstance(digest, str):
                found[rel.removeprefix("./")] = digest.removeprefix("sha256:").lower()
    elif isinstance(files, dict):
        for rel, item in files.items():
            if isinstance(item, str):
                found[str(rel).removeprefix("./")] = item.removeprefix("sha256:").lower()
            elif isinstance(item, dict):
                digest = pick(item, "sha256", "hash", "digest")
                if isinstance(digest, str):
                    found[str(rel).removeprefix("./")] = digest.removeprefix("sha256:").lower()

    return {
        rel: digest for rel, digest in found.items()
        if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)
    }


def verify_inner_manifest() -> tuple[str, int, list[str], list[str]]:
    path = root / "integrity_manifest.json"
    if not path.is_file():
        return "MISSING", 0, ["integrity_manifest.json fehlt"], []

    try:
        manifest = load_json(path)
    except ValueError as exc:
        return "FAIL", 0, [str(exc)], []

    entries = extract_json_manifest_entries(manifest)
    local_errors: list[str] = []

    for rel, expected in sorted(entries.items()):
        candidate = root / rel
        if not candidate.is_file():
            local_errors.append(f"inner manifest: fehlt: {rel}")
        elif sha256_file(candidate) != expected:
            local_errors.append(f"inner manifest: Hashabweichung: {rel}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "integrity_manifest.json"
    }
    listed = set(entries)
    unlisted = sorted(actual - listed)
    for rel in sorted(listed - actual):
        local_errors.append(f"inner manifest: gelistet, aber fehlt: {rel}")

    if local_errors or not entries:
        status = "FAIL"
    elif not unlisted:
        status = "PASS"
    elif unlisted == ["official_run_summary.json"]:
        status = "KNOWN_FINALIZATION_GAP"
    else:
        status = "UNLISTED_FILES"

    return status, len(entries), local_errors, unlisted


def normalize_offload(value: Any) -> int | float | str:
    if finite(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def inspect_request(
    record: dict[str, Any],
    label: str,
    role: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    local_errors: list[str] = []
    local_warnings: list[str] = []

    if record.get("status") != "complete":
        local_errors.append(f"{label}: status={record.get('status')!r}")
    if record.get("validation_errors") != []:
        local_errors.append(f"{label}: validation_errors nicht leer")
    if record.get("http_status") != 200:
        local_errors.append(f"{label}: http_status={record.get('http_status')!r}")
    if record.get("done_received") is not True:
        local_errors.append(f"{label}: done_received ist nicht true")
    if record.get("timed_out") is not False:
        local_errors.append(f"{label}: timed_out ist nicht false")
    if record.get("cancelled") is not False:
        local_errors.append(f"{label}: cancelled ist nicht false")

    events = record.get("raw_sse_events")
    if not isinstance(events, list) or not events:
        return {}, local_errors + [f"{label}: raw_sse_events fehlen/leer"], local_warnings

    reconstructed_ids: list[int] = []
    token_event_ns: list[int] = []
    batch_sizes: list[int] = []
    done_count = 0
    previous_ns: int | None = None

    for position, event in enumerate(events):
        if not isinstance(event, dict):
            local_errors.append(f"{label}: SSE-Event {position} ist kein Objekt")
            continue

        if event.get("event_index") != position:
            local_errors.append(f"{label}: event_index nicht lückenlos bei {position}")

        parse_status = event.get("parse_status")
        if parse_status not in ALLOWED_PARSE:
            local_errors.append(
                f"{label}: parse_status={parse_status!r} bei Event {position}"
            )
        if parse_status == "done":
            done_count += 1

        receive_ns = event.get("receive_perf_counter_ns")
        if type(receive_ns) is not int:
            local_errors.append(
                f"{label}: receive_perf_counter_ns fehlt bei Event {position}"
            )
        else:
            if previous_ns is not None and receive_ns < previous_ns:
                local_errors.append(f"{label}: SSE-Empfangszeiten nicht monoton")
            previous_ns = receive_ns

        token_ids = event.get("token_ids")
        if not isinstance(token_ids, list) or not all(type(x) is int for x in token_ids):
            local_errors.append(
                f"{label}: token_ids ist nicht list[int] bei Event {position}"
            )
            continue

        reconstructed_ids.extend(token_ids)
        if token_ids and type(receive_ns) is int:
            token_event_ns.append(receive_ns)
            batch_sizes.append(len(token_ids))

    if done_count != 1:
        local_errors.append(f"{label}: DONE-Anzahl {done_count} != 1")
    if reconstructed_ids != record.get("output_token_ids"):
        local_errors.append(f"{label}: Raw-SSE-Token != output_token_ids")

    expected_tokens = record.get("expected_completion_tokens")
    if type(expected_tokens) is not int or len(reconstructed_ids) != expected_tokens:
        local_errors.append(
            f"{label}: Tokenanzahl {len(reconstructed_ids)} != erwartet "
            f"{expected_tokens!r}"
        )

    sent_prompt = record.get("prompt_token_ids_sent")
    returned_prompt = record.get("prompt_token_ids_returned")
    if sent_prompt is not None and returned_prompt is not None and sent_prompt != returned_prompt:
        local_errors.append(f"{label}: Prompt-Tokenbindung stimmt nicht")

    if not token_event_ns:
        return {}, local_errors + [f"{label}: kein positives Token-Event"], local_warnings

    first_ns = token_event_ns[0]
    last_ns = token_event_ns[-1]
    stored_first_receive = record.get("first_token_receive_ns")
    stored_last_receive = record.get("last_token_receive_ns")

    if stored_first_receive != first_ns:
        local_errors.append(f"{label}: first_token_receive_ns stimmt nicht")
    if stored_last_receive != last_ns:
        local_errors.append(f"{label}: last_token_receive_ns stimmt nicht")

    if role == "victim":
        stored_first = pick(record, "first_token_perf_ns", "first_token_ns")
        stream_end_ns = pick(record, "stream_end_perf_ns", "stream_end_ns")
        stream_open_ns = pick(
            record,
            "stream_open_or_response_headers_perf_ns",
            "stream_open_perf_ns",
        )
    else:
        stored_first = pick(record, "burst_first_token_perf_ns", "first_token_perf_ns")
        stream_end_ns = pick(record, "burst_end_perf_ns", "stream_end_perf_ns")
        stream_open_ns = pick(
            record,
            "burst_stream_open_or_response_headers_perf_ns",
            "stream_open_or_response_headers_perf_ns",
        )

    if stored_first != first_ns:
        local_errors.append(f"{label}: gespeicherter First-Token stimmt nicht")

    raw_itl_ms: list[float] | None = None
    max_itl_ms = None
    if all(size == 1 for size in batch_sizes):
        raw_itl_ms = [
            (token_event_ns[i] - token_event_ns[i - 1]) / 1e6
            for i in range(1, len(token_event_ns))
        ]
        max_itl_ms = max(raw_itl_ms) if raw_itl_ms else None
        stored_itl = record.get("itl_ms")
        if not isinstance(stored_itl, list) or len(stored_itl) != len(raw_itl_ms):
            local_errors.append(f"{label}: ITL-Länge stimmt nicht")
        elif any(
            not finite(stored) or abs(float(stored) - raw) > 1e-6
            for stored, raw in zip(stored_itl, raw_itl_ms)
        ):
            local_errors.append(f"{label}: ITL stimmt nicht mit Raw-SSE überein")
    else:
        local_errors.append(
            f"{label}: Token-Batching verhindert vertragliche ITL-Rekonstruktion"
        )

    raw_tpot_ms = None
    if len(reconstructed_ids) > 1:
        raw_tpot_ms = (
            (last_ns - first_ns) / (len(reconstructed_ids) - 1) / 1e6
        )
        stored_tpot = record.get("client_observed_tpot_ms")
        if not finite(stored_tpot) or abs(float(stored_tpot) - raw_tpot_ms) > 1e-6:
            local_errors.append(f"{label}: TPOT stimmt nicht mit Raw-SSE überein")

    task_created_ns = pick(record, "task_created_perf_ns", "task_created_ns")
    dispatch_ns = (
        pick(record, "request_dispatch_ns", "request_dispatch_perf_ns")
        if role == "victim"
        else pick(record, "burst_dispatch_start_perf_ns", "request_dispatch_ns")
    )

    completion_ms = (
        (stream_end_ns - task_created_ns) / 1e6
        if type(stream_end_ns) is int and type(task_created_ns) is int
        else None
    )
    ttft_task_ms = (
        (first_ns - task_created_ns) / 1e6
        if type(task_created_ns) is int else None
    )
    ttft_open_ms = (
        (first_ns - stream_open_ns) / 1e6
        if type(stream_open_ns) is int else None
    )
    token_span_ms = (last_ns - first_ns) / 1e6
    post_token_tail_ms = (
        (stream_end_ns - last_ns) / 1e6
        if type(stream_end_ns) is int else None
    )

    return {
        "request_id": record.get("request_id"),
        "request_index": record.get("request_index"),
        "role": role,
        "event_count": len(events),
        "token_count": len(reconstructed_ids),
        "first_token_ns": first_ns,
        "last_token_ns": last_ns,
        "stream_end_ns": stream_end_ns,
        "stream_open_ns": stream_open_ns,
        "task_created_ns": task_created_ns,
        "dispatch_ns": dispatch_ns,
        "token_16_ns": pick(record, "token_16_perf_ns", "token_16_ns"),
        "decode_tokens_at_trigger": record.get("decode_tokens_received_at_trigger"),
        "completion_ms": completion_ms,
        "ttft_task_ms": ttft_task_ms,
        "ttft_open_ms": ttft_open_ms,
        "token_span_ms": token_span_ms,
        "tpot_ms": raw_tpot_ms,
        "max_itl_ms": max_itl_ms,
        "post_token_tail_ms": post_token_tail_ms,
    }, local_errors, local_warnings


def inspect_episode(path: Path) -> dict[str, Any]:
    episode = load_json(path)
    episode_id = str(episode.get("episode_id") or path.name)
    row = episode.get("schedule_row") or {}
    condition = row.get("condition")
    trigger = episode.get("trigger") or {}
    trigger_ns = trigger.get("trigger_perf_ns")
    active_list = trigger.get("active_cohort_request_indices") or []
    active = set(active_list)
    k = row.get("server_max_num_seqs")

    local_errors: list[str] = []
    local_warnings: list[str] = []

    if episode.get("status") != "complete":
        local_errors.append(f"{episode_id}: episode status != complete")
    if episode.get("validation_errors") != []:
        local_errors.append(f"{episode_id}: episode validation_errors nicht leer")
    if condition not in {"no_burst", "prefill_burst"}:
        local_errors.append(f"{episode_id}: ungültige Condition {condition!r}")
    if trigger.get("status") != "ok":
        local_errors.append(f"{episode_id}: trigger.status != ok")
    if type(trigger_ns) is not int:
        local_errors.append(f"{episode_id}: trigger timestamp fehlt")
    if type(k) is not int or len(active_list) != k or len(active) != k:
        local_errors.append(f"{episode_id}: aktive Kohorte hat nicht exakt k Mitglieder")

    victims_raw = episode.get("victim_requests")
    bursts_raw = episode.get("burst_requests")
    if not isinstance(victims_raw, list):
        victims_raw = []
        local_errors.append(f"{episode_id}: victim_requests ist keine Liste")
    if not isinstance(bursts_raw, list):
        bursts_raw = []
        local_errors.append(f"{episode_id}: burst_requests ist keine Liste")

    expected_bursts = 0 if condition == "no_burst" else 4
    if len(victims_raw) != 20:
        local_errors.append(f"{episode_id}: Victim-Anzahl {len(victims_raw)} != 20")
    if len(bursts_raw) != expected_bursts:
        local_errors.append(
            f"{episode_id}: Burst-Anzahl {len(bursts_raw)} != {expected_bursts}"
        )

    victims: dict[int, dict[str, Any]] = {}
    bursts: list[dict[str, Any]] = []
    event_count = 0

    for position, record in enumerate(victims_raw):
        if not isinstance(record, dict):
            local_errors.append(f"{episode_id}: Victim {position} ist kein Objekt")
            continue
        index = record.get("request_index")
        summary, req_errors, req_warnings = inspect_request(
            record, f"{episode_id}:victim:{index}", "victim"
        )
        local_errors.extend(req_errors)
        local_warnings.extend(req_warnings)
        if type(index) is int:
            if index in victims:
                local_errors.append(f"{episode_id}: doppelter Victim-Index {index}")
            victims[index] = summary
        event_count += int(summary.get("event_count") or 0)

    for position, record in enumerate(bursts_raw):
        if not isinstance(record, dict):
            local_errors.append(f"{episode_id}: Burst {position} ist kein Objekt")
            continue
        summary, req_errors, req_warnings = inspect_request(
            record, f"{episode_id}:burst:{position}", "burst"
        )
        local_errors.extend(req_errors)
        local_warnings.extend(req_warnings)
        bursts.append(summary)
        event_count += int(summary.get("event_count") or 0)

    if set(victims) != set(range(20)):
        local_errors.append(f"{episode_id}: Victim-Indizes sind nicht exakt 0..19")

    if type(trigger_ns) is int and victims and active:
        active_token16 = [
            victims[index].get("token_16_ns")
            for index in active if index in victims
        ]
        if (
            len(active_token16) != len(active)
            or not all(type(value) is int for value in active_token16)
            or trigger_ns != max(active_token16)
        ):
            local_errors.append(f"{episode_id}: Trigger != max(aktive token_16)")

        for index, request in victims.items():
            dispatch = request.get("dispatch_ns")
            first = request.get("first_token_ns")
            end = request.get("stream_end_ns")
            tokens_at_trigger = request.get("decode_tokens_at_trigger")

            if type(dispatch) is not int or dispatch > trigger_ns:
                local_errors.append(
                    f"{episode_id}: Victim {index} nicht vor/am Trigger dispatcht"
                )

            if index in active:
                if type(first) is not int or first > trigger_ns:
                    local_errors.append(
                        f"{episode_id}: aktives Victim {index} ohne Output am Trigger"
                    )
                if type(end) is not int or end <= trigger_ns:
                    local_errors.append(
                        f"{episode_id}: aktives Victim {index} am/vor Trigger beendet"
                    )
                if not finite(tokens_at_trigger) or float(tokens_at_trigger) < 16:
                    local_errors.append(
                        f"{episode_id}: aktives Victim {index} hat <16 Tokens am Trigger"
                    )
            else:
                if type(first) is not int or first <= trigger_ns:
                    local_errors.append(
                        f"{episode_id}: Noncohort-Victim {index} hatte Output am Trigger"
                    )
                if tokens_at_trigger != 0:
                    local_errors.append(
                        f"{episode_id}: Noncohort-Victim {index} Tokenzahl != 0"
                    )

        opens = [request.get("stream_open_ns") for request in victims.values()]
        firsts = [request.get("first_token_ns") for request in victims.values()]
        if (
            not all(type(value) is int for value in opens + firsts)
            or not all(value < min(firsts) for value in opens)
        ):
            local_errors.append(
                f"{episode_id}: nicht alle 20 Streams vor frühestem First-Token offen"
            )

        for position, request in enumerate(bursts):
            dispatch = request.get("dispatch_ns")
            opened = request.get("stream_open_ns")
            first = request.get("first_token_ns")
            end = request.get("stream_end_ns")
            if not all(type(value) is int for value in (dispatch, opened, first, end)):
                local_errors.append(
                    f"{episode_id}: Burst {position} Zeitstempel unvollständig"
                )
            elif not (trigger_ns < dispatch <= opened <= first <= end):
                local_errors.append(
                    f"{episode_id}: Burst {position} Zeitfolge ungültig"
                )

    if episode.get("no_client_admission_semaphore_used") is not True:
        local_errors.append(f"{episode_id}: Client-Semaphore-Gate nicht true")

    quality = trigger.get("metrics_quality") or {}
    expected_waiting = 20 - k if type(k) is int else None
    metrics_ok = (
        quality.get("metrics_quality_status") == "corroborated"
        and finite(quality.get("parsed_running"))
        and float(quality["parsed_running"]) == float(k)
        and finite(quality.get("parsed_waiting"))
        and float(quality["parsed_waiting"]) == float(expected_waiting)
    )
    if not metrics_ok:
        local_errors.append(
            f"{episode_id}: serverseitige Running/Waiting-Metrik nicht corroborated"
        )

    active_requests = [
        victims[index] for index in sorted(active)
        if index in victims
    ]

    def active_median(key: str) -> float | None:
        return median([request.get(key) for request in active_requests])

    return {
        "source_file": path.name,
        "episode_id": episode_id,
        "block_id": episode.get("block_id"),
        "condition": condition,
        "offload": normalize_offload(row.get("offload_gb")),
        "k": k,
        "repeat": row.get("repeat"),
        "order_in_block": row.get("order_in_block"),
        "condition_first_in_block": row.get("condition_first_in_block"),
        "schedule_fingerprint": episode.get("schedule_fingerprint"),
        "environment_fingerprint": episode.get("environment_fingerprint"),
        "trigger_ns": trigger_ns,
        "active_indices": sorted(active),
        "victims": victims,
        "bursts": bursts,
        "active_medians": {
            "completion_ms": active_median("completion_ms"),
            "ttft_task_ms": active_median("ttft_task_ms"),
            "ttft_open_ms": active_median("ttft_open_ms"),
            "token_span_ms": active_median("token_span_ms"),
            "tpot_ms": active_median("tpot_ms"),
            "max_itl_ms": active_median("max_itl_ms"),
            "post_token_tail_ms": active_median("post_token_tail_ms"),
        },
        "event_count": event_count,
        "metrics_ok": metrics_ok,
        "errors": local_errors,
        "warnings": local_warnings,
    }


print("=== 1. INTEGRITÄT UND ARTEFAKTE ===")
outer_status, outer_count, outer_errors = verify_text_manifest(
    target / "BUNDLE_SHA256SUMS.txt", target
)
print(f"outer_manifest_status: {outer_status}")
print(f"outer_manifest_entries: {outer_count}")
errors.extend(outer_errors)

inner_status, inner_count, inner_errors, inner_unlisted = verify_inner_manifest()
print(f"inner_manifest_status: {inner_status}")
print(f"inner_manifest_entries: {inner_count}")
print(f"inner_manifest_unlisted: {inner_unlisted}")
errors.extend(inner_errors)

artifact_counts: dict[str, int] = {}
for name, (expected, pattern) in EXPECTED.items():
    count = len(list((root / name).glob(pattern)))
    artifact_counts[name] = count
    print(f"{name}: {count}/{expected}")
    if count != expected:
        errors.append(f"{name}: {count} != {expected}")

for rel in (
    "official_run_summary.json",
    "official_run_manifest.json",
    "integrity_manifest.json",
    ".server_waiting_confirmation_run_mode",
):
    present = (root / rel).is_file()
    print(f"{rel}_present: {present}")
    if not present:
        errors.append(f"Pflichtdatei fehlt: {rel}")

try:
    official_summary = load_json(root / "official_run_summary.json")
except ValueError as exc:
    errors.append(str(exc))
    official_summary = {}

expected_summary = {
    "overall_status": "complete",
    "planned_episodes": 32,
    "valid_complete_episodes": 32,
    "missing_episodes": 0,
    "planned_blocks": 16,
    "completed_blocks": 16,
}
for key, expected in expected_summary.items():
    actual = official_summary.get(key)
    print(f"summary.{key}: {actual!r}")
    if actual != expected:
        errors.append(f"summary.{key}={actual!r} != {expected!r}")

block_complete = 0
for path in sorted((root / "block_summaries").glob("*.json")):
    try:
        block = load_json(path)
    except ValueError as exc:
        errors.append(str(exc))
        continue
    if block.get("overall_status") == "block_complete":
        block_complete += 1
    else:
        errors.append(
            f"{path.name}: overall_status={block.get('overall_status')!r}"
        )
print(f"block_complete: {block_complete}/16")

stabilization_passed = 0
for path in sorted((root / "stabilization").glob("*.json")):
    try:
        stab = load_json(path)
    except ValueError as exc:
        errors.append(str(exc))
        continue
    if stab.get("stabilization_passed") is True:
        stabilization_passed += 1
    else:
        errors.append(
            f"{path.name}: stabilization_passed={stab.get('stabilization_passed')!r}"
        )
print(f"stabilization_passed: {stabilization_passed}/16")

log_findings: list[str] = []
log_pattern = re.compile(
    r"\b(?:ERROR|CRITICAL)\b|Traceback \(most recent call last\)"
)
for path in sorted((root / "server_logs").glob("*.log")):
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if log_pattern.search(line):
            log_findings.append(f"{path.name}:{line_number}:{line[:240]}")
print(f"server_log_error_lines: {len(log_findings)}")
errors.extend(log_findings[:50])

print()
print("=== 2. ALLE EPISODEN UND RAW-SSE ===")

episodes: list[dict[str, Any]] = []
for path in sorted((root / "episodes").glob("*.json")):
    try:
        summary = inspect_episode(path)
        episodes.append(summary)
        raw_errors.extend(summary["errors"])
        raw_warnings.extend(summary["warnings"])
    except Exception as exc:
        raw_errors.append(f"{path.name}: unerwarteter Auswertefehler: {exc}")

total_requests = sum(
    len(episode["victims"]) + len(episode["bursts"])
    for episode in episodes
)
total_events = sum(episode["event_count"] for episode in episodes)
metrics_ok_count = sum(1 for episode in episodes if episode["metrics_ok"])

print(f"episodes_inspected: {len(episodes)}/32")
print(f"requests_reconstructed: {total_requests}")
print(f"raw_sse_events_reconstructed: {total_events}")
print(f"metrics_corroborated: {metrics_ok_count}/32")
print(f"raw_errors: {len(raw_errors)}")
print(f"raw_warnings: {len(raw_warnings)}")

pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
schedule_fingerprints = set()
environment_fingerprints = set()

for episode in episodes:
    block_id = episode.get("block_id")
    condition = episode.get("condition")
    if not isinstance(block_id, str):
        raw_errors.append(f"{episode['episode_id']}: block_id fehlt")
        continue
    if condition in pairs[block_id]:
        raw_errors.append(f"{block_id}: doppelte Condition {condition}")
    else:
        pairs[block_id][condition] = episode
    if isinstance(episode.get("schedule_fingerprint"), str):
        schedule_fingerprints.add(episode["schedule_fingerprint"])
    if isinstance(episode.get("environment_fingerprint"), str):
        environment_fingerprints.add(episode["environment_fingerprint"])

if isinstance(official_summary.get("schedule_fingerprint"), str):
    schedule_fingerprints.add(official_summary["schedule_fingerprint"])
if isinstance(official_summary.get("environment_fingerprint"), str):
    environment_fingerprints.add(official_summary["environment_fingerprint"])

print(f"paired_blocks: {len(pairs)}/16")
print(f"schedule_fingerprints: {sorted(schedule_fingerprints)}")
print(f"environment_fingerprint_count: {len(environment_fingerprints)}")

if len(pairs) != 16:
    raw_errors.append(f"Blockanzahl {len(pairs)} != 16")
if len(schedule_fingerprints) != 1:
    raw_errors.append("Schedule-Fingerprint nicht einheitlich")
if len(environment_fingerprints) != 1:
    raw_errors.append("Environment-Fingerprint nicht einheitlich")

cell_blocks: dict[tuple[Any, Any], set[str]] = defaultdict(set)
repeat_sets: dict[tuple[Any, Any], set[Any]] = defaultdict(set)
order_balance: dict[tuple[Any, Any], Counter[str]] = defaultdict(Counter)

for block_id, pair in sorted(pairs.items()):
    if set(pair) != {"no_burst", "prefill_burst"}:
        raw_errors.append(
            f"{block_id}: Conditions {sorted(pair)} != no_burst+prefill_burst"
        )
        continue
    representative = pair["prefill_burst"]
    cell = (representative["offload"], representative["k"])
    cell_blocks[cell].add(block_id)
    repeat_sets[cell].add(representative["repeat"])
    first = representative.get("condition_first_in_block")
    if first not in {"no_burst", "prefill_burst"}:
        ordered = sorted(
            (
                ep.get("order_in_block"),
                condition,
            )
            for condition, ep in pair.items()
            if type(ep.get("order_in_block")) is int
        )
        first = ordered[0][1] if len(ordered) == 2 else "UNKNOWN"
    order_balance[cell][str(first)] += 1

print(f"cells: {sorted(cell_blocks, key=str)}")
for cell in sorted(cell_blocks, key=str):
    print(
        f"cell {cell}: blocks={len(cell_blocks[cell])}, "
        f"repeats={sorted(repeat_sets[cell], key=str)}, "
        f"first_condition={dict(order_balance[cell])}"
    )
    if len(cell_blocks[cell]) != 4:
        raw_errors.append(f"Zelle {cell}: nicht 4 Blöcke")
    if repeat_sets[cell] != {1, 2, 3, 4}:
        raw_errors.append(f"Zelle {cell}: Repeats nicht 1..4")
    if (
        order_balance[cell].get("no_burst", 0) != 2
        or order_balance[cell].get("prefill_burst", 0) != 2
    ):
        raw_errors.append(f"Zelle {cell}: Reihenfolge nicht 2/2")

if set(cell_blocks) != EXPECTED_CELLS:
    raw_errors.append(f"Zellen {set(cell_blocks)} != {EXPECTED_CELLS}")

print()
print("=== 3. PAARWEISE KLASSIFIKATION UND KOHORTENVERGLEICH ===")

block_rows: list[dict[str, Any]] = []
class_counts: Counter[str] = Counter()
raw_class_counts: Counter[str] = Counter()
cohort_mismatch_blocks: list[str] = []

for block_id in sorted(pairs):
    pair = pairs[block_id]
    no = pair.get("no_burst")
    burst = pair.get("prefill_burst")
    local_errors: list[str] = []

    if no is None or burst is None:
        local_errors.append("Paarmitglied fehlt")
        no = no or {}
        burst = burst or {}

    no_active = set(no.get("active_indices") or [])
    burst_active = set(burst.get("active_indices") or [])
    overlap = no_active & burst_active
    union = no_active | burst_active
    same_active_set = no_active == burst_active
    if not same_active_set:
        cohort_mismatch_blocks.append(block_id)

    first_burst = None
    last_active_stream = None
    last_all_stream = None
    last_active_token = None
    last_all_token = None

    try:
        first_burst = min(
            request["first_token_ns"] for request in burst["bursts"]
        )
        last_active_stream = max(
            burst["victims"][index]["stream_end_ns"]
            for index in burst_active
        )
        last_all_stream = max(
            request["stream_end_ns"] for request in burst["victims"].values()
        )
        last_active_token = max(
            burst["victims"][index]["last_token_ns"]
            for index in burst_active
        )
        last_all_token = max(
            request["last_token_ns"] for request in burst["victims"].values()
        )
    except Exception as exc:
        local_errors.append(f"Klassifikationszeitpunkte nicht rekonstruierbar: {exc}")

    def classify(first, active_end, all_end):
        if not all(type(value) is int for value in (first, active_end, all_end)):
            return "D"
        if first < active_end:
            return "A"
        if first > all_end:
            return "C"
        return "B"

    stream_class = classify(first_burst, last_active_stream, last_all_stream)
    raw_class = classify(first_burst, last_active_token, last_all_token)
    if stream_class != raw_class:
        local_errors.append(
            f"Stream-End-Klasse {stream_class} != Raw-Token-Klasse {raw_class}"
        )
        stream_class = "D"

    no_med = no.get("active_medians") or {}
    burst_med = burst.get("active_medians") or {}
    metrics = (
        "completion_ms",
        "ttft_task_ms",
        "ttft_open_ms",
        "token_span_ms",
        "tpot_ms",
        "max_itl_ms",
        "post_token_tail_ms",
    )
    deltas = {
        metric: (
            float(burst_med[metric]) - float(no_med[metric])
            if finite(burst_med.get(metric)) and finite(no_med.get(metric))
            else None
        )
        for metric in metrics
    }

    decomposition_residual = (
        deltas["completion_ms"]
        - deltas["ttft_task_ms"]
        - deltas["token_span_ms"]
        - deltas["post_token_tail_ms"]
        if all(
            finite(deltas[key])
            for key in (
                "completion_ms",
                "ttft_task_ms",
                "token_span_ms",
                "post_token_tail_ms",
            )
        )
        else None
    )

    first_condition = burst.get("condition_first_in_block")
    if first_condition not in {"no_burst", "prefill_burst"}:
        ordered = sorted(
            (
                ep.get("order_in_block"),
                condition,
            )
            for condition, ep in pair.items()
            if type(ep.get("order_in_block")) is int
        )
        first_condition = ordered[0][1] if len(ordered) == 2 else "UNKNOWN"

    flags: list[str] = []
    if finite(ratio(burst_med.get("tpot_ms"), no_med.get("tpot_ms"))) and abs(
        ratio(burst_med.get("tpot_ms"), no_med.get("tpot_ms")) - 1
    ) >= 0.05:
        flags.append("TPOT_RATIO_ABS_GE_5_PERCENT")
    if finite(ratio(burst_med.get("max_itl_ms"), no_med.get("max_itl_ms"))) and abs(
        ratio(burst_med.get("max_itl_ms"), no_med.get("max_itl_ms")) - 1
    ) >= 0.20:
        flags.append("MAX_ITL_RATIO_ABS_GE_20_PERCENT")
    if finite(deltas["completion_ms"]) and abs(deltas["completion_ms"]) >= 1000:
        flags.append("COMPLETION_DELTA_ABS_GE_1000_MS")

    attribution: list[str] = []
    completion_delta = deltas["completion_ms"]
    if finite(completion_delta) and abs(completion_delta) >= 1000:
        for metric, label in (
            ("ttft_task_ms", "TTFT/Startphase"),
            ("token_span_ms", "Token-Span/Decode-Dauer"),
            ("post_token_tail_ms", "Post-Token-/Stream-End-Tail"),
        ):
            value = deltas[metric]
            if finite(value) and abs(value) >= 0.5 * abs(completion_delta):
                attribution.append(label)
        if not attribution:
            attribution.append("keine einzelne Komponente nach 50%-Regel")

    row = {
        "block_id": block_id,
        "offload": burst.get("offload"),
        "k": burst.get("k"),
        "repeat": burst.get("repeat"),
        "first_condition": first_condition,
        "stream_class": stream_class,
        "raw_class": raw_class,
        "same_active_set": same_active_set,
        "no_active_indices": sorted(no_active),
        "burst_active_indices": sorted(burst_active),
        "active_overlap_count": len(overlap),
        "active_union_count": len(union),
        "active_jaccard": (
            len(overlap) / len(union) if union else None
        ),
        "tpot_ratio": ratio(
            burst_med.get("tpot_ms"), no_med.get("tpot_ms")
        ),
        "max_itl_ratio": ratio(
            burst_med.get("max_itl_ms"), no_med.get("max_itl_ms")
        ),
        "completion_ratio": ratio(
            burst_med.get("completion_ms"), no_med.get("completion_ms")
        ),
        "deltas": deltas,
        "decomposition_residual_ms": decomposition_residual,
        "burst_after_active_s": (
            (first_burst - last_active_stream) / 1e9
            if type(first_burst) is int and type(last_active_stream) is int
            else None
        ),
        "remaining_until_all_s": (
            (last_all_stream - first_burst) / 1e9
            if type(first_burst) is int and type(last_all_stream) is int
            else None
        ),
        "flags": flags,
        "attribution": attribution,
        "errors": local_errors,
    }

    block_rows.append(row)
    class_counts[stream_class] += 1
    raw_class_counts[raw_class] += 1
    raw_errors.extend(f"{block_id}: {item}" for item in local_errors)

print(
    "block                          cls raw sameSet overlap "
    "TPOT-ratio ITL-ratio Completion-delta-ms TTFT-delta-ms "
    "TokenSpan-delta-ms Tail-delta-ms after-active-s remaining-all-s"
)
for row in block_rows:
    d = row["deltas"]
    print(
        f"{row['block_id']:30s} "
        f"{row['stream_class']:>3s} "
        f"{row['raw_class']:>3s} "
        f"{str(row['same_active_set']):>7s} "
        f"{row['active_overlap_count']:>7d} "
        f"{fmt(row['tpot_ratio'],6):>10s} "
        f"{fmt(row['max_itl_ratio'],6):>9s} "
        f"{fmt(d['completion_ms'],2):>19s} "
        f"{fmt(d['ttft_task_ms'],2):>13s} "
        f"{fmt(d['token_span_ms'],2):>18s} "
        f"{fmt(d['post_token_tail_ms'],2):>13s} "
        f"{fmt(row['burst_after_active_s'],2):>14s} "
        f"{fmt(row['remaining_until_all_s'],2):>15s}"
    )

print(f"classification_counts: {dict(sorted(class_counts.items()))}")
print(f"raw_token_classification_counts: {dict(sorted(raw_class_counts.items()))}")
print(
    f"blocks_with_different_active_indices: "
    f"{len(cohort_mismatch_blocks)}/16 {cohort_mismatch_blocks}"
)
print(
    "Hinweis: Unterschiedliche aktive Indizes sind bei dynamischer "
    "Kohortenbildung kein technischer Fehler. Paarmetriken vergleichen "
    "die jeweiligen Kohortenmediane, nicht identische Request-Indizes."
)

print()
print("=== 4. ZELLZUSAMMENFASSUNG ===")

grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
for row in block_rows:
    grouped[(row["offload"], row["k"])].append(row)

cell_rows: list[dict[str, Any]] = []
print(
    "offload k classes TPOT-med[min,max] ITL-med[min,max] "
    "Completion-med-ms TTFT-med-ms TokenSpan-med-ms Tail-med-ms "
    "after-active-med-s remaining-all-med-s"
)

for cell in sorted(grouped, key=str):
    group = grouped[cell]
    valid = [row for row in group if row["stream_class"] != "D"]
    class_counter = Counter(row["stream_class"] for row in group)

    tpot_min, tpot_max = value_range([row["tpot_ratio"] for row in valid])
    itl_min, itl_max = value_range([row["max_itl_ratio"] for row in valid])

    cell_row = {
        "offload": cell[0],
        "k": cell[1],
        "pair_count": len(group),
        "classes": dict(sorted(class_counter.items())),
        "tpot_ratio_median": median([row["tpot_ratio"] for row in valid]),
        "tpot_ratio_min": tpot_min,
        "tpot_ratio_max": tpot_max,
        "max_itl_ratio_median": median(
            [row["max_itl_ratio"] for row in valid]
        ),
        "max_itl_ratio_min": itl_min,
        "max_itl_ratio_max": itl_max,
        "completion_delta_median_ms": median(
            [row["deltas"]["completion_ms"] for row in valid]
        ),
        "ttft_delta_median_ms": median(
            [row["deltas"]["ttft_task_ms"] for row in valid]
        ),
        "token_span_delta_median_ms": median(
            [row["deltas"]["token_span_ms"] for row in valid]
        ),
        "post_token_tail_delta_median_ms": median(
            [row["deltas"]["post_token_tail_ms"] for row in valid]
        ),
        "burst_after_active_median_s": median(
            [row["burst_after_active_s"] for row in valid]
        ),
        "remaining_until_all_median_s": median(
            [row["remaining_until_all_s"] for row in valid]
        ),
        "active_jaccard_median": median(
            [row["active_jaccard"] for row in valid]
        ),
    }
    cell_rows.append(cell_row)

    class_text = "/".join(
        f"{name}{count}" for name, count in sorted(class_counter.items())
    )
    print(
        f"{str(cell[0]):>7s} {str(cell[1]):>1s} "
        f"{class_text:<7s} "
        f"{fmt(cell_row['tpot_ratio_median'],6)}"
        f"[{fmt(tpot_min,6)},{fmt(tpot_max,6)}] "
        f"{fmt(cell_row['max_itl_ratio_median'],6)}"
        f"[{fmt(itl_min,6)},{fmt(itl_max,6)}] "
        f"{fmt(cell_row['completion_delta_median_ms'],2):>17s} "
        f"{fmt(cell_row['ttft_delta_median_ms'],2):>11s} "
        f"{fmt(cell_row['token_span_delta_median_ms'],2):>17s} "
        f"{fmt(cell_row['post_token_tail_delta_median_ms'],2):>11s} "
        f"{fmt(cell_row['burst_after_active_median_s'],2):>18s} "
        f"{fmt(cell_row['remaining_until_all_median_s'],2):>19s}"
    )

print()
print("=== 5. REIHENFOLGE-SENSITIVITÄT ===")
for cell in sorted(grouped, key=str):
    group = grouped[cell]
    print(f"cell offload={cell[0]} k={cell[1]}")
    for first_condition in ("no_burst", "prefill_burst"):
        subset = [
            row for row in group
            if row["first_condition"] == first_condition
            and row["stream_class"] != "D"
        ]
        print(
            f"  first={first_condition}: n={len(subset)}, "
            f"TPOT_med={fmt(median([r['tpot_ratio'] for r in subset]),6)}, "
            f"ITL_med={fmt(median([r['max_itl_ratio'] for r in subset]),6)}, "
            f"Completion_med_ms="
            f"{fmt(median([r['deltas']['completion_ms'] for r in subset]),2)}"
        )

print()
print("=== 6. OFFLOAD- UND K-MODULATION ===")
cell_lookup = {(row["offload"], row["k"]): row for row in cell_rows}
for k in (4, 8):
    low = cell_lookup.get((0, k))
    high = cell_lookup.get((12, k))
    if low and high:
        print(
            f"offload12-minus-offload0, k={k}: "
            f"TPOT_ratio_delta="
            f"{fmt(high['tpot_ratio_median']-low['tpot_ratio_median'],6)}, "
            f"ITL_ratio_delta="
            f"{fmt(high['max_itl_ratio_median']-low['max_itl_ratio_median'],6)}, "
            f"Completion_delta_difference_ms="
            f"{fmt(high['completion_delta_median_ms']-low['completion_delta_median_ms'],2)}, "
            f"after_active_factor="
            f"{fmt(high['burst_after_active_median_s']/low['burst_after_active_median_s'] if finite(low['burst_after_active_median_s']) and low['burst_after_active_median_s'] != 0 else None,2)}"
        )

for offload in (0, 12):
    k4 = cell_lookup.get((offload, 4))
    k8 = cell_lookup.get((offload, 8))
    if k4 and k8:
        print(
            f"k8-minus-k4, offload={offload}: "
            f"TPOT_ratio_delta="
            f"{fmt(k8['tpot_ratio_median']-k4['tpot_ratio_median'],6)}, "
            f"ITL_ratio_delta="
            f"{fmt(k8['max_itl_ratio_median']-k4['max_itl_ratio_median'],6)}, "
            f"after_active_delta_s="
            f"{fmt(k8['burst_after_active_median_s']-k4['burst_after_active_median_s'],2)}"
        )

print()
print("=== 7. AUSREISSER UND URSACHENZERLEGUNG ===")
outliers = [row for row in block_rows if row["flags"]]
if not outliers:
    print("keine nach den festgelegten deskriptiven Schwellen")
else:
    for row in outliers:
        d = row["deltas"]
        print(
            f"{row['block_id']}: flags={row['flags']}, "
            f"completion_delta_ms={fmt(d['completion_ms'],3)}, "
            f"ttft_delta_ms={fmt(d['ttft_task_ms'],3)}, "
            f"token_span_delta_ms={fmt(d['token_span_ms'],3)}, "
            f"tail_delta_ms={fmt(d['post_token_tail_ms'],3)}, "
            f"decomposition_residual_ms="
            f"{fmt(row['decomposition_residual_ms'],3)}, "
            f"attribution={row['attribution']}, "
            f"same_active_set={row['same_active_set']}, "
            f"active_overlap={row['active_overlap_count']}/"
            f"{row['active_union_count']}"
        )

print()
print("=== 8. VORSICHTIGE WISSENSCHAFTLICHE INTERPRETATION ===")
class_sets = {
    (row["offload"], row["k"]): set(row["classes"])
    for row in cell_rows
}
if (
    class_sets.get((0, 4)) == {"C"}
    and class_sets.get((12, 4)) == {"C"}
    and class_sets.get((0, 8)) == {"B"}
    and class_sets.get((12, 8)) == {"B"}
):
    print(
        "- Das Klassenmuster repliziert vollständig: k=4 ergibt C, "
        "k=8 ergibt B, jeweils bei Offload 0 und 12."
    )
    print(
        "- Auf Klassenebene ist die beobachtete Modulation damit an "
        "max_num_seqs gebunden, nicht an den Offload-Zustand."
    )
else:
    print(
        "- Das einfache k=4/C- und k=8/B-Muster repliziert nicht vollständig; "
        "die Einzelzellen sind maßgeblich."
    )

max_tpot_dev = max(
    (
        abs(float(row["tpot_ratio_median"]) - 1.0)
        for row in cell_rows if finite(row["tpot_ratio_median"])
    ),
    default=float("nan"),
)
if math.isfinite(max_tpot_dev):
    print(
        f"- Größte absolute Abweichung eines TPOT-Zellmedians von 1: "
        f"{100*max_tpot_dev:.3f}%."
    )

print(
    "- Unterschiedliche dynamische aktive Kohorten werden nicht als "
    "technischer Fehler behandelt. Zell- und Paareffekte sind Vergleiche "
    "der jeweiligen Kohortenmediane, keine identischen Request-Matches."
)
print(
    "- A/B/C beziehen sich ausschließlich auf client-sichtbaren Output. "
    "Interne Admission und interner Prefill-Start sind nicht beobachtbar."
)
print(
    "- Nahe-1-Mediane sind deskriptiv und tragen ohne weitere "
    "Kampagnenreplikation keinen universellen Nulleffekt-Claim."
)

completeness_pass = (
    not errors
    and outer_status == "PASS"
    and inner_status in {"PASS", "KNOWN_FINALIZATION_GAP"}
)
raw_pass = not raw_errors
no_d = class_counts.get("D", 0) == 0
scientific_ready = completeness_pass and raw_pass and no_d

result = {
    "version": "llama-full-campaign-reaudit-v1",
    "target": str(target),
    "campaign": str(root),
    "integrity": {
        "outer_status": outer_status,
        "outer_entries": outer_count,
        "inner_status": inner_status,
        "inner_entries": inner_count,
        "inner_unlisted": inner_unlisted,
    },
    "artifact_counts": artifact_counts,
    "official_summary": official_summary,
    "technical": {
        "errors": errors,
        "warnings": warnings,
        "raw_errors": raw_errors,
        "raw_warnings": raw_warnings,
        "episodes_inspected": len(episodes),
        "requests_reconstructed": total_requests,
        "raw_sse_events_reconstructed": total_events,
        "metrics_corroborated": metrics_ok_count,
        "schedule_fingerprints": sorted(schedule_fingerprints),
        "environment_fingerprints": sorted(environment_fingerprints),
    },
    "classification_counts": dict(sorted(class_counts.items())),
    "raw_token_classification_counts": dict(sorted(raw_class_counts.items())),
    "blocks_with_different_active_indices": cohort_mismatch_blocks,
    "blocks": block_rows,
    "cells": cell_rows,
    "status": {
        "completeness_pass": completeness_pass,
        "raw_reconstruction_pass": raw_pass,
        "no_d_classifications": no_d,
        "scientific_summary_ready": scientific_ready,
    },
    "scope": (
        "Client-visible output only. No inference about internal admission "
        "or internal prefill start."
    ),
}

(outdir / "full_reaudit.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

with (outdir / "blocks.csv").open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "block_id", "offload", "k", "repeat", "first_condition",
        "stream_class", "raw_class", "same_active_set",
        "active_overlap_count", "active_union_count", "active_jaccard",
        "tpot_ratio", "max_itl_ratio", "completion_ratio",
        "completion_delta_ms", "ttft_delta_ms", "ttft_open_delta_ms",
        "token_span_delta_ms", "tpot_delta_ms", "max_itl_delta_ms",
        "post_token_tail_delta_ms", "decomposition_residual_ms",
        "burst_after_active_s", "remaining_until_all_s",
        "flags", "attribution",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in block_rows:
        d = row["deltas"]
        writer.writerow({
            "block_id": row["block_id"],
            "offload": row["offload"],
            "k": row["k"],
            "repeat": row["repeat"],
            "first_condition": row["first_condition"],
            "stream_class": row["stream_class"],
            "raw_class": row["raw_class"],
            "same_active_set": row["same_active_set"],
            "active_overlap_count": row["active_overlap_count"],
            "active_union_count": row["active_union_count"],
            "active_jaccard": row["active_jaccard"],
            "tpot_ratio": row["tpot_ratio"],
            "max_itl_ratio": row["max_itl_ratio"],
            "completion_ratio": row["completion_ratio"],
            "completion_delta_ms": d["completion_ms"],
            "ttft_delta_ms": d["ttft_task_ms"],
            "ttft_open_delta_ms": d["ttft_open_ms"],
            "token_span_delta_ms": d["token_span_ms"],
            "tpot_delta_ms": d["tpot_ms"],
            "max_itl_delta_ms": d["max_itl_ms"],
            "post_token_tail_delta_ms": d["post_token_tail_ms"],
            "decomposition_residual_ms": row["decomposition_residual_ms"],
            "burst_after_active_s": row["burst_after_active_s"],
            "remaining_until_all_s": row["remaining_until_all_s"],
            "flags": "|".join(row["flags"]),
            "attribution": "|".join(row["attribution"]),
        })

with (outdir / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
    fieldnames = sorted({
        key for row in cell_rows for key in row
    })
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in cell_rows:
        serializable = dict(row)
        serializable["classes"] = json.dumps(row["classes"], sort_keys=True)
        writer.writerow(serializable)

share_lines = [
    "=== DIESEN BLOCK AN CHATGPT SENDEN ===",
    f"COMPLETENESS_PASS={str(completeness_pass).lower()}",
    f"RAW_RECONSTRUCTION_PASS={str(raw_pass).lower()}",
    f"NO_D_CLASSIFICATIONS={str(no_d).lower()}",
    f"SCIENTIFIC_SUMMARY_READY={str(scientific_ready).lower()}",
    f"OUTER_MANIFEST_STATUS={outer_status}",
    f"INNER_MANIFEST_STATUS={inner_status}",
    f"classification_counts={dict(sorted(class_counts.items()))}",
    f"raw_token_classification_counts={dict(sorted(raw_class_counts.items()))}",
    f"requests_reconstructed={total_requests}",
    f"raw_sse_events_reconstructed={total_events}",
    f"metrics_corroborated={metrics_ok_count}/32",
    (
        "blocks_with_different_active_indices="
        f"{len(cohort_mismatch_blocks)}/16 {cohort_mismatch_blocks}"
    ),
]
for row in cell_rows:
    share_lines.append(
        f"cell offload={row['offload']} k={row['k']}: "
        f"classes={row['classes']}, "
        f"TPOT_med={fmt(row['tpot_ratio_median'],6)}, "
        f"ITL_med={fmt(row['max_itl_ratio_median'],6)}, "
        f"Completion_delta_med_ms={fmt(row['completion_delta_median_ms'],2)}, "
        f"TTFT_delta_med_ms={fmt(row['ttft_delta_median_ms'],2)}, "
        f"TokenSpan_delta_med_ms={fmt(row['token_span_delta_median_ms'],2)}, "
        f"Tail_delta_med_ms={fmt(row['post_token_tail_delta_median_ms'],2)}, "
        f"after_active_med_s={fmt(row['burst_after_active_median_s'],2)}, "
        f"remaining_all_med_s={fmt(row['remaining_until_all_median_s'],2)}"
    )
share_lines.append(
    f"outlier_blocks={[row['block_id'] for row in outliers]}"
)
for row in outliers:
    share_lines.append(
        f"outlier {row['block_id']}: "
        f"completion_delta_ms={fmt(row['deltas']['completion_ms'],3)}, "
        f"ttft_delta_ms={fmt(row['deltas']['ttft_task_ms'],3)}, "
        f"token_span_delta_ms={fmt(row['deltas']['token_span_ms'],3)}, "
        f"tail_delta_ms={fmt(row['deltas']['post_token_tail_ms'],3)}, "
        f"attribution={row['attribution']}, "
        f"same_active_set={row['same_active_set']}, "
        f"active_overlap={row['active_overlap_count']}/"
        f"{row['active_union_count']}"
    )
share_lines.append(
    "scope=client-visible output only; no internal admission/prefill claim"
)

(outdir / "SHARE_WITH_CHATGPT.txt").write_text(
    "\n".join(share_lines) + "\n",
    encoding="utf-8",
)

print()
print("=== 9. ENDSTATUS ===")
for line in share_lines:
    print(line)

if errors:
    print()
    print("=== VOLLSTÄNDIGKEITS-/INTEGRITÄTSFEHLER ===")
    for item in errors:
        print(f"- {item}")

if raw_errors:
    print()
    print("=== RAW-/TRIGGER-FEHLER ===")
    for item in raw_errors[:300]:
        print(f"- {item}")
    if len(raw_errors) > 300:
        print(f"- ... weitere {len(raw_errors)-300}")

if warnings or raw_warnings:
    print()
    print("=== WARNUNGEN ===")
    for item in (warnings + raw_warnings)[:300]:
        print(f"- {item}")

raise SystemExit(0 if scientific_ready else 2)
PY
RC=${PIPESTATUS[0]}
set -e

echo
echo "=== REPORT-BUNDLE ==="
if test -f "$OUTDIR/SHARE_WITH_CHATGPT.txt"; then
    cat "$OUTDIR/SHARE_WITH_CHATGPT.txt"
fi

ARCHIVE="${OUTDIR}.tar.gz"
tar -C "$(dirname "$OUTDIR")" -czf "$ARCHIVE" "$(basename "$OUTDIR")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo
echo "REPORT_DIR=$OUTDIR"
echo "REPORT_TXT=$REPORT"
echo "SHARE_FILE=$OUTDIR/SHARE_WITH_CHATGPT.txt"
echo "REPORT_ARCHIVE=$ARCHIVE"
echo "REPORT_ARCHIVE_SHA256=$(awk '{print $1}' "${ARCHIVE}.sha256")"
echo "COMMAND_RC=$RC"
echo "TERMINAL_STILL_ALIVE"
exit "$RC"
