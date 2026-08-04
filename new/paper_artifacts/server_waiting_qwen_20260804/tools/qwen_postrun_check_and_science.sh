#!/usr/bin/env bash
# Read-only completeness check plus scientific Qwen campaign evaluation.
# Starts no server, sends no requests, and never changes the campaign tree.
set -Eeuo pipefail
umask 022

VERSION="qwen-postrun-check-and-science-v1"
PROJECT="${PROJECT:-$HOME/projects/DOS}"
VENV="${VENV_PATH:-$HOME/projects/porto/.venv-py313-repair}"
RESULT_ROOT="$PROJECT/new/runs/server_waiting_confirmation/qwen"

usage() {
    cat <<EOF
$VERSION

Verwendung:
  $0
  $0 /pfad/zu/final_capture_...
  $0 /pfad/zu/official_campaign_fixed_...

Ohne Pfad wird das neueste passende Qwen-Ergebnis automatisch gewählt.

Das Skript:
  1. prüft Vollständigkeit, Status, Paarung, Schedule-Bindung und Manifeste;
  2. rekonstruiert alle Raw-SSE-Token und Zeitmetriken;
  3. berechnet A/B/C/D, TPOT, Max-ITL, Completion und Zeitmargen;
  4. speichert dieselbe Konsolenausgabe als Textdatei in ~/Downloads.

Es startet keine Messung und verändert keine Ergebnisdatei.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

PYTHON="$VENV/bin/python"
if test ! -x "$PYTHON"; then
    PYTHON="$(command -v python3 || true)"
fi
test -n "$PYTHON" && test -x "$PYTHON" || {
    echo "FEHLER: Kein ausführbares Python gefunden." >&2
    exit 2
}

TARGET="${1:-}"
if test -z "$TARGET"; then
    TARGET="$("$PYTHON" - "$RESULT_ROOT" <<'PY_SELECT'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()
candidates = []
if root.is_dir():
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if (
            path.name.startswith("final_capture_")
            or path.name.startswith("official_campaign_fixed_")
            or path.name.startswith("official_campaign_")
        ):
            if (path / "source_campaign").is_dir() or (path / "episodes").is_dir():
                candidates.append(path)

if not candidates:
    raise SystemExit(3)

print(max(candidates, key=lambda p: p.stat().st_mtime_ns))
PY_SELECT
    )" || {
        echo "FEHLER: Kein Qwen-Kampagnenverzeichnis unter $RESULT_ROOT gefunden." >&2
        exit 2
    }
fi

TARGET="$(readlink -f "$TARGET")"
test -d "$TARGET" || {
    echo "FEHLER: Verzeichnis fehlt: $TARGET" >&2
    exit 2
}

if test -d "$TARGET/source_campaign"; then
    CAMPAIGN="$TARGET/source_campaign"
else
    CAMPAIGN="$TARGET"
fi

test -d "$CAMPAIGN/episodes" || {
    echo "FEHLER: Kein episodes/-Verzeichnis in $CAMPAIGN" >&2
    exit 2
}

REPORT_DIR="${REPORT_DIR:-$HOME/Downloads}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="$REPORT_DIR/qwen_postrun_report_${STAMP}.txt"

echo "=== QWEN POST-RUN CHECK ==="
echo "VERSION=$VERSION"
echo "TARGET=$TARGET"
echo "CAMPAIGN=$CAMPAIGN"
echo "REPORT=$REPORT"
echo "READ_ONLY=yes"
echo

set +e
"$PYTHON" - "$TARGET" "$CAMPAIGN" <<'PY_ANALYSIS' | tee "$REPORT"
from __future__ import annotations

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

EXPECTED_COUNTS = {
    "episodes": (32, "*.json"),
    "block_summaries": (16, "*.json"),
    "stabilization": (16, "*.json"),
    "server_logs": (16, "*.log"),
}
EXPECTED_CELLS = {(0, 4), (0, 8), (12, 4), (12, 8)}
ALLOWED_PARSE_STATUS = {"ok", "done", "keepalive"}

completeness_errors: list[str] = []
raw_errors: list[str] = []
warnings: list[str] = []


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: JSON nicht lesbar: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: Top-Level ist kein Objekt")
    return obj


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def median(values) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def value_range(values) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if finite(v)]
    return (min(vals), max(vals)) if vals else (None, None)


def fmt(value: Any, digits: int = 6) -> str:
    return "NA" if not finite(value) else f"{float(value):.{digits}f}"


def normalize_offload(value: Any) -> int | float | str:
    if finite(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_text_sha_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        if not match:
            completeness_errors.append(
                f"{path.name}: ungültige Manifestzeile {line_no}: {line!r}"
            )
            continue
        rel = match.group(2)
        if rel.startswith("./"):
            rel = rel[2:]
        entries[rel] = match.group(1).lower()
    return entries


def verify_outer_manifest() -> tuple[str, int, list[str]]:
    manifest = target / "BUNDLE_SHA256SUMS.txt"
    if not manifest.is_file():
        return "NOT_AVAILABLE", 0, []

    entries = parse_text_sha_manifest(manifest)
    errors: list[str] = []
    for rel, expected in sorted(entries.items()):
        candidate = target / rel
        if not candidate.is_file():
            errors.append(f"outer manifest: fehlt: {rel}")
        elif sha256_file(candidate) != expected:
            errors.append(f"outer manifest: Hashabweichung: {rel}")

    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path != manifest
    }
    listed = set(entries)
    for rel in sorted(actual - listed):
        errors.append(f"outer manifest: nicht gelistet: {rel}")
    for rel in sorted(listed - actual):
        errors.append(f"outer manifest: gelistet, aber nicht vorhanden: {rel}")

    return ("PASS" if entries and not errors else "FAIL"), len(entries), errors


def extract_inner_entries(manifest: dict[str, Any]) -> dict[str, str]:
    entries: dict[str, str] = {}
    files = manifest.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = next(
                (
                    item.get(key)
                    for key in ("relative_path", "path", "filename", "file")
                    if isinstance(item.get(key), str)
                ),
                None,
            )
            digest = next(
                (
                    item.get(key)
                    for key in ("sha256", "hash", "digest")
                    if isinstance(item.get(key), str)
                ),
                None,
            )
            if rel and digest:
                entries[rel.removeprefix("./")] = digest.removeprefix("sha256:").lower()
    elif isinstance(files, dict):
        for rel, value in files.items():
            if isinstance(value, str):
                entries[str(rel).removeprefix("./")] = value.removeprefix("sha256:").lower()
            elif isinstance(value, dict):
                digest = next(
                    (
                        value.get(key)
                        for key in ("sha256", "hash", "digest")
                        if isinstance(value.get(key), str)
                    ),
                    None,
                )
                if digest:
                    entries[str(rel).removeprefix("./")] = digest.removeprefix("sha256:").lower()
    return {rel: digest for rel, digest in entries.items() if len(digest) == 64}


def verify_inner_manifest() -> tuple[str, int, list[str], list[str]]:
    path = root / "integrity_manifest.json"
    if not path.is_file():
        return "MISSING", 0, ["integrity_manifest.json fehlt"], []

    try:
        manifest = load_json(path)
    except ValueError as exc:
        return "FAIL", 0, [str(exc)], []

    entries = extract_inner_entries(manifest)
    errors: list[str] = []
    unlisted: list[str] = []

    for rel, expected in sorted(entries.items()):
        candidate = root / rel
        if not candidate.is_file():
            errors.append(f"inner manifest: fehlt: {rel}")
        elif sha256_file(candidate) != expected:
            errors.append(f"inner manifest: Hashabweichung: {rel}")

    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    listed = set(entries)
    unlisted = sorted(actual - listed)
    for rel in sorted(listed - actual):
        errors.append(f"inner manifest: gelistet, aber nicht vorhanden: {rel}")

    if errors:
        status = "FAIL"
    elif not entries:
        status = "FAIL"
    elif not unlisted:
        status = "PASS"
    elif unlisted == ["official_run_summary.json"]:
        status = "KNOWN_FINALIZATION_GAP"
    else:
        status = "UNLISTED_FILES"

    return status, len(entries), errors, unlisted


def request_raw_check(record: dict[str, Any], label: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []

    if record.get("status") != "complete":
        errors.append(f"{label}: request status={record.get('status')!r}")
    if record.get("validation_errors") != []:
        errors.append(f"{label}: request validation_errors nicht leer")
    if record.get("http_status") != 200:
        errors.append(f"{label}: http_status={record.get('http_status')!r}")
    if record.get("done_received") is not True:
        errors.append(f"{label}: done_received ist nicht true")
    if record.get("timed_out") is not False:
        errors.append(f"{label}: timed_out ist nicht false")
    if record.get("cancelled") is not False:
        errors.append(f"{label}: cancelled ist nicht false")

    events = record.get("raw_sse_events")
    if not isinstance(events, list) or not events:
        return errors + [f"{label}: raw_sse_events fehlen/leer"], {}

    reconstructed: list[int] = []
    positive_event_ns: list[int] = []
    positive_batch_sizes: list[int] = []
    done_count = 0
    previous_ns: int | None = None

    for pos, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"{label}: SSE-Event {pos} ist kein Objekt")
            continue

        if event.get("event_index") != pos:
            errors.append(f"{label}: event_index nicht lückenlos bei {pos}")

        parse_status = event.get("parse_status")
        if parse_status not in ALLOWED_PARSE_STATUS:
            errors.append(f"{label}: parse_status={parse_status!r} bei Event {pos}")
        if parse_status == "done":
            done_count += 1

        receive_ns = event.get("receive_perf_counter_ns")
        if type(receive_ns) is not int:
            errors.append(f"{label}: receive_perf_counter_ns fehlt bei Event {pos}")
        else:
            if previous_ns is not None and receive_ns < previous_ns:
                errors.append(f"{label}: SSE-Empfangszeiten nicht monoton")
            previous_ns = receive_ns

        token_ids = event.get("token_ids")
        if not isinstance(token_ids, list) or not all(type(x) is int for x in token_ids):
            errors.append(f"{label}: token_ids ist nicht list[int] bei Event {pos}")
            continue

        reconstructed.extend(token_ids)
        if token_ids and type(receive_ns) is int:
            positive_event_ns.append(receive_ns)
            positive_batch_sizes.append(len(token_ids))

    if done_count != 1:
        errors.append(f"{label}: DONE-Anzahl {done_count} != 1")
    if reconstructed != record.get("output_token_ids"):
        errors.append(f"{label}: Raw-SSE-Token != output_token_ids")

    expected_tokens = record.get("expected_completion_tokens")
    if type(expected_tokens) is not int or len(reconstructed) != expected_tokens:
        errors.append(
            f"{label}: Tokenanzahl {len(reconstructed)} != erwartet {expected_tokens!r}"
        )

    sent_prompt = record.get("prompt_token_ids_sent")
    returned_prompt = record.get("prompt_token_ids_returned")
    if sent_prompt is not None and returned_prompt is not None and sent_prompt != returned_prompt:
        errors.append(f"{label}: Prompt-Token-Bindung stimmt nicht")

    if not positive_event_ns:
        return errors + [f"{label}: kein positives Token-Event"], {}

    first_ns = positive_event_ns[0]
    last_ns = positive_event_ns[-1]

    if record.get("first_token_receive_ns") != first_ns:
        errors.append(f"{label}: first_token_receive_ns stimmt nicht")
    if record.get("last_token_receive_ns") != last_ns:
        errors.append(f"{label}: last_token_receive_ns stimmt nicht")

    role = record.get("role")
    if role == "victim" and record.get("first_token_perf_ns") != first_ns:
        errors.append(f"{label}: first_token_perf_ns stimmt nicht")
    if role == "burst" and record.get("burst_first_token_perf_ns") != first_ns:
        errors.append(f"{label}: burst_first_token_perf_ns stimmt nicht")

    raw_tpot_ms = None
    if len(reconstructed) > 1:
        raw_tpot_ms = (last_ns - first_ns) / (len(reconstructed) - 1) / 1e6
        stored_tpot = record.get("client_observed_tpot_ms")
        if not finite(stored_tpot) or abs(float(stored_tpot) - raw_tpot_ms) > 1e-6:
            errors.append(f"{label}: TPOT stimmt nicht mit Raw-SSE überein")

    raw_itl_ms: list[float] | None = None
    if all(size == 1 for size in positive_batch_sizes):
        raw_itl_ms = [
            (positive_event_ns[i] - positive_event_ns[i - 1]) / 1e6
            for i in range(1, len(positive_event_ns))
        ]
        stored_itl = record.get("itl_ms")
        if not isinstance(stored_itl, list) or len(stored_itl) != len(raw_itl_ms):
            errors.append(f"{label}: ITL-Länge stimmt nicht")
        elif any(
            not finite(a) or abs(float(a) - b) > 1e-6
            for a, b in zip(stored_itl, raw_itl_ms)
        ):
            errors.append(f"{label}: ITL stimmt nicht mit Raw-SSE überein")
    else:
        errors.append(f"{label}: Token-Batching verhindert exakte ITL-Rekonstruktion")

    return errors, {
        "first_ns": first_ns,
        "last_ns": last_ns,
        "token_count": len(reconstructed),
        "event_count": len(events),
        "tpot_ms": raw_tpot_ms,
        "max_itl_ms": max(raw_itl_ms) if raw_itl_ms else None,
    }


def active_stats(
    episode: dict[str, Any],
    raw_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    active = set((episode.get("trigger") or {}).get("active_cohort_request_indices") or [])
    victims = [
        request
        for request in episode.get("victim_requests", [])
        if isinstance(request, dict) and request.get("request_index") in active
    ]

    return {
        "active_n": len(victims),
        "median_tpot_ms": median(
            [raw_map.get(str(r.get("request_id")), {}).get("tpot_ms") for r in victims]
        ),
        "median_max_itl_ms": median(
            [raw_map.get(str(r.get("request_id")), {}).get("max_itl_ms") for r in victims]
        ),
        "median_completion_ms": median(
            [
                (r["stream_end_perf_ns"] - r["task_created_perf_ns"]) / 1e6
                for r in victims
                if type(r.get("stream_end_perf_ns")) is int
                and type(r.get("task_created_perf_ns")) is int
                and r["stream_end_perf_ns"] >= r["task_created_perf_ns"]
            ]
        ),
    }


def ratio(numerator: Any, denominator: Any) -> float | None:
    if finite(numerator) and finite(denominator) and float(denominator) != 0:
        return float(numerator) / float(denominator)
    return None


print("=== 1. VOLLSTÄNDIGKEIT UND INTEGRITÄT ===")
print(f"target: {target}")
print(f"campaign: {root}")

outer_status, outer_entries, outer_errors = verify_outer_manifest()
print(f"outer_manifest_status: {outer_status}")
print(f"outer_manifest_entries: {outer_entries}")
for error in outer_errors:
    completeness_errors.append(error)

inner_status, inner_entries, inner_errors, inner_unlisted = verify_inner_manifest()
print(f"inner_manifest_status: {inner_status}")
print(f"inner_manifest_entries: {inner_entries}")
print(f"inner_manifest_unlisted: {inner_unlisted}")
completeness_errors.extend(inner_errors)

artifact_counts: dict[str, int] = {}
for name, (expected, pattern) in EXPECTED_COUNTS.items():
    count = len(list((root / name).glob(pattern)))
    artifact_counts[name] = count
    print(f"{name}_count: {count}/{expected}")
    if count != expected:
        completeness_errors.append(f"{name}: {count} != {expected}")

required_files = [
    "official_run_summary.json",
    "official_run_manifest.json",
    "integrity_manifest.json",
    ".server_waiting_confirmation_run_mode",
]
for rel in required_files:
    present = (root / rel).is_file()
    print(f"{rel}_present: {present}")
    if not present:
        completeness_errors.append(f"Pflichtdatei fehlt: {rel}")

try:
    official_summary = load_json(root / "official_run_summary.json")
except ValueError as exc:
    completeness_errors.append(str(exc))
    official_summary = {}

summary_expectations = {
    "overall_status": "complete",
    "planned_episodes": 32,
    "valid_complete_episodes": 32,
    "missing_episodes": 0,
    "planned_blocks": 16,
    "completed_blocks": 16,
}
for key, expected in summary_expectations.items():
    actual = official_summary.get(key)
    print(f"summary.{key}: {actual!r}")
    if actual != expected:
        completeness_errors.append(f"summary.{key}={actual!r} != {expected!r}")

episode_paths = sorted((root / "episodes").glob("*.json"))
block_paths = sorted((root / "block_summaries").glob("*.json"))
stabilization_paths = sorted((root / "stabilization").glob("*.json"))
server_log_paths = sorted((root / "server_logs").glob("*.log"))

episodes: list[dict[str, Any]] = []
for path in episode_paths:
    try:
        episode = load_json(path)
        episode["_source_file"] = path.name
        episodes.append(episode)
    except ValueError as exc:
        completeness_errors.append(str(exc))

block_complete_count = 0
for path in block_paths:
    try:
        block = load_json(path)
    except ValueError as exc:
        completeness_errors.append(str(exc))
        continue
    if block.get("overall_status") == "block_complete":
        block_complete_count += 1
    else:
        completeness_errors.append(
            f"{path.name}: overall_status={block.get('overall_status')!r}"
        )
print(f"block_complete_count: {block_complete_count}/16")

stabilization_pass_count = 0
for path in stabilization_paths:
    try:
        stab = load_json(path)
    except ValueError as exc:
        completeness_errors.append(str(exc))
        continue
    passed = stab.get("stabilization_passed") is True
    if passed:
        stabilization_pass_count += 1
    else:
        completeness_errors.append(
            f"{path.name}: stabilization_passed={stab.get('stabilization_passed')!r}"
        )
print(f"stabilization_pass_count: {stabilization_pass_count}/16")

server_log_findings: list[str] = []
log_pattern = re.compile(r"\b(?:ERROR|CRITICAL)\b|Traceback \(most recent call last\)")
for path in server_log_paths:
    try:
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if log_pattern.search(line):
                server_log_findings.append(f"{path.name}:{line_no}: {line[:240]}")
    except OSError as exc:
        completeness_errors.append(f"{path}: nicht lesbar: {exc}")
print(f"server_log_error_lines: {len(server_log_findings)}")
if server_log_findings:
    completeness_errors.extend(server_log_findings[:20])

episode_complete_count = 0
episode_validation_clean_count = 0
pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
schedule_fingerprints = set()
environment_fingerprints = set()

for episode in episodes:
    episode_id = episode.get("episode_id") or episode.get("_source_file")
    if episode.get("status") == "complete":
        episode_complete_count += 1
    else:
        completeness_errors.append(
            f"{episode_id}: episode status={episode.get('status')!r}"
        )

    if episode.get("validation_errors") == []:
        episode_validation_clean_count += 1
    else:
        completeness_errors.append(f"{episode_id}: validation_errors nicht leer")

    block_id = episode.get("block_id")
    row = episode.get("schedule_row") or {}
    condition = row.get("condition")
    if not isinstance(block_id, str) or condition not in {"no_burst", "prefill_burst"}:
        completeness_errors.append(f"{episode_id}: Block-/Condition-Identität ungültig")
    elif condition in pairs[block_id]:
        completeness_errors.append(f"{block_id}: doppelte Condition {condition}")
    else:
        pairs[block_id][condition] = episode

    if isinstance(episode.get("schedule_fingerprint"), str):
        schedule_fingerprints.add(episode["schedule_fingerprint"])
    if isinstance(episode.get("environment_fingerprint"), str):
        environment_fingerprints.add(episode["environment_fingerprint"])

print(f"episode_complete_count: {episode_complete_count}/32")
print(f"episode_validation_clean_count: {episode_validation_clean_count}/32")
print(f"paired_blocks_found: {len(pairs)}/16")

if len(pairs) != 16:
    completeness_errors.append(f"Gefundene Blöcke {len(pairs)} != 16")
for block_id, pair in sorted(pairs.items()):
    if set(pair) != {"no_burst", "prefill_burst"}:
        completeness_errors.append(
            f"{block_id}: Conditions {sorted(pair)} != no_burst+prefill_burst"
        )

summary_schedule = official_summary.get("schedule_fingerprint")
if isinstance(summary_schedule, str):
    schedule_fingerprints.add(summary_schedule)
summary_environment = official_summary.get("environment_fingerprint")
if isinstance(summary_environment, str):
    environment_fingerprints.add(summary_environment)

print(f"schedule_fingerprints: {sorted(schedule_fingerprints)}")
print(f"environment_fingerprint_count: {len(environment_fingerprints)}")
if len(schedule_fingerprints) != 1:
    completeness_errors.append("Schedule-Fingerprint ist nicht über alle Artefakte einheitlich")
if len(environment_fingerprints) != 1:
    completeness_errors.append("Environment-Fingerprint ist nicht über alle Episoden einheitlich")

cell_blocks: dict[tuple[Any, Any], set[str]] = defaultdict(set)
order_balance: dict[tuple[Any, Any], Counter[str]] = defaultdict(Counter)
repeat_sets: dict[tuple[Any, Any], set[Any]] = defaultdict(set)

for block_id, pair in pairs.items():
    representative = pair.get("prefill_burst") or pair.get("no_burst") or {}
    row = representative.get("schedule_row") or {}
    cell = (
        normalize_offload(row.get("offload_gb")),
        row.get("server_max_num_seqs"),
    )
    cell_blocks[cell].add(block_id)
    repeat_sets[cell].add(row.get("repeat"))

    first_condition = row.get("condition_first_in_block")
    if first_condition not in {"no_burst", "prefill_burst"}:
        ordered = []
        for condition, episode in pair.items():
            order = (episode.get("schedule_row") or {}).get("order_in_block")
            if type(order) is int:
                ordered.append((order, condition))
        first_condition = min(ordered)[1] if len(ordered) == 2 else "UNKNOWN"
    order_balance[cell][str(first_condition)] += 1

print(f"cells_found: {sorted(cell_blocks, key=str)}")
for cell in sorted(cell_blocks, key=str):
    print(
        f"cell {cell}: blocks={len(cell_blocks[cell])}, "
        f"repeats={sorted(repeat_sets[cell], key=str)}, "
        f"first_condition={dict(order_balance[cell])}"
    )
    if len(cell_blocks[cell]) != 4:
        completeness_errors.append(f"Zelle {cell}: {len(cell_blocks[cell])} Blöcke != 4")
    if repeat_sets[cell] != {1, 2, 3, 4}:
        completeness_errors.append(f"Zelle {cell}: Repeats {repeat_sets[cell]} != 1..4")
    if order_balance[cell].get("no_burst", 0) != 2 or order_balance[cell].get("prefill_burst", 0) != 2:
        completeness_errors.append(f"Zelle {cell}: Reihenfolge nicht 2/2 gegenbalanciert")

if set(cell_blocks) != EXPECTED_CELLS:
    completeness_errors.append(
        f"Zellen {set(cell_blocks)} != erwartet {EXPECTED_CELLS}"
    )

finalization_ok = outer_status == "PASS"
if not finalization_ok:
    completeness_errors.append(
        "Kein vollständig verifiziertes äußeres Final-Capture-Manifest"
    )

print()
print("=== 2. RAW-SSE- UND TRIGGER-REKONSTRUKTION ===")

raw_by_episode: dict[str, dict[str, dict[str, Any]]] = {}
total_requests = 0
total_sse_events = 0
metrics_corroborated_count = 0

for episode in episodes:
    episode_id = str(episode.get("episode_id") or episode.get("_source_file"))
    raw_map: dict[str, dict[str, Any]] = {}
    seen_request_ids: set[str] = set()

    for key in ("victim_requests", "burst_requests"):
        records = episode.get(key)
        if not isinstance(records, list):
            raw_errors.append(f"{episode_id}: {key} ist keine Liste")
            continue
        for index, record in enumerate(records):
            total_requests += 1
            if not isinstance(record, dict):
                raw_errors.append(f"{episode_id}:{key}[{index}]: kein Objekt")
                continue
            request_id = str(record.get("request_id"))
            if request_id in seen_request_ids:
                raw_errors.append(f"{episode_id}: doppelte request_id {request_id}")
            seen_request_ids.add(request_id)

            errors, reconstructed = request_raw_check(
                record, f"{episode_id}:{request_id}"
            )
            raw_errors.extend(errors)
            raw_map[request_id] = reconstructed
            total_sse_events += int(reconstructed.get("event_count") or 0)

    raw_by_episode[episode_id] = raw_map

    trigger = episode.get("trigger") or {}
    quality = trigger.get("metrics_quality") or {}
    if (
        quality.get("metrics_quality_status") == "corroborated"
        and quality.get("parsed_running") == float(
            (episode.get("schedule_row") or {}).get("server_max_num_seqs")
        )
        and quality.get("parsed_waiting") == float(
            20 - (episode.get("schedule_row") or {}).get("server_max_num_seqs")
        )
    ):
        metrics_corroborated_count += 1
    else:
        raw_errors.append(
            f"{episode_id}: serverseitige Running/Waiting-Metrik nicht corroborated"
        )

print(f"requests_reconstructed: {total_requests}")
print(f"raw_sse_events_reconstructed: {total_sse_events}")
print(f"metrics_corroborated_episodes: {metrics_corroborated_count}/32")

rows: list[dict[str, Any]] = []
class_counts: Counter[str] = Counter()
raw_class_counts: Counter[str] = Counter()

for block_id in sorted(pairs):
    pair = pairs[block_id]
    no = pair.get("no_burst")
    burst = pair.get("prefill_burst")
    pair_errors: list[str] = []

    if no is None or burst is None:
        pair_errors.append("Paarmitglied fehlt")
        no = no or {}
        burst = burst or {}

    for condition, episode, expected_bursts in (
        ("no_burst", no, 0),
        ("prefill_burst", burst, 4),
    ):
        episode_id = str(episode.get("episode_id") or condition)
        victims = episode.get("victim_requests") or []
        bursts = episode.get("burst_requests") or []
        row = episode.get("schedule_row") or {}
        trigger = episode.get("trigger") or {}
        trigger_ns = trigger.get("trigger_perf_ns")
        k = row.get("server_max_num_seqs")
        active_list = trigger.get("active_cohort_request_indices") or []
        active = set(active_list)

        if len(victims) != 20:
            pair_errors.append(f"{condition}: Victims {len(victims)} != 20")
        if len(bursts) != expected_bursts:
            pair_errors.append(
                f"{condition}: Burst-Requests {len(bursts)} != {expected_bursts}"
            )
        if trigger.get("status") != "ok" or type(trigger_ns) is not int:
            pair_errors.append(f"{condition}: Trigger ungültig")
            continue
        if type(k) is not int or len(active_list) != k or len(active) != k:
            pair_errors.append(f"{condition}: aktive Kohorte ungültig")

        by_index = {
            request.get("request_index"): request
            for request in victims
            if isinstance(request, dict)
        }
        if set(by_index) != set(range(20)):
            pair_errors.append(f"{condition}: Victim-Indizes nicht exakt 0..19")
            continue

        token16 = [by_index[i].get("token_16_perf_ns") for i in active]
        if (
            not token16
            or not all(type(value) is int for value in token16)
            or trigger_ns != max(token16)
        ):
            pair_errors.append(f"{condition}: Trigger != max(aktive token_16)")

        dispatch_ok = True
        active_output_ok = True
        active_alive_ok = True
        nonactive_output_ok = True
        nonactive_zero_ok = True

        for index, request in by_index.items():
            dispatch = request.get("request_dispatch_ns")
            first = request.get("first_token_perf_ns")
            end = request.get("stream_end_perf_ns")

            if type(dispatch) is not int or dispatch > trigger_ns:
                dispatch_ok = False

            if index in active:
                if type(first) is not int or first > trigger_ns:
                    active_output_ok = False
                if type(end) is not int or end <= trigger_ns:
                    active_alive_ok = False
            else:
                if type(first) is not int or first <= trigger_ns:
                    nonactive_output_ok = False
                if request.get("decode_tokens_received_at_trigger") != 0:
                    nonactive_zero_ok = False

        if not dispatch_ok:
            pair_errors.append(f"{condition}: nicht alle 20 vor Trigger dispatcht")
        if not active_output_ok:
            pair_errors.append(f"{condition}: aktive Kohorte nicht korrekt aktiv")
        if not active_alive_ok:
            pair_errors.append(f"{condition}: aktives Victim am/vor Trigger beendet")
        if not nonactive_output_ok:
            pair_errors.append(f"{condition}: Noncohort hatte Output am Trigger")
        if not nonactive_zero_ok:
            pair_errors.append(f"{condition}: Noncohort-Tokenzahl am Trigger != 0")

        opens = [
            request.get("stream_open_or_response_headers_perf_ns")
            for request in victims
        ]
        firsts = [request.get("first_token_perf_ns") for request in victims]
        if (
            not all(type(value) is int for value in opens + firsts)
            or not all(value < min(firsts) for value in opens)
        ):
            pair_errors.append(f"{condition}: 20-Streams-open-Gate fehlgeschlagen")

        if episode.get("no_client_admission_semaphore_used") is not True:
            pair_errors.append(f"{condition}: Client-Semaphore-Gate fehlgeschlagen")

        if condition == "prefill_burst":
            for request in bursts:
                dispatch = request.get("burst_dispatch_start_perf_ns")
                opened = request.get("burst_stream_open_or_response_headers_perf_ns")
                first = request.get("burst_first_token_perf_ns")
                end = request.get("burst_end_perf_ns")
                if not all(type(value) is int for value in (dispatch, opened, first, end)):
                    pair_errors.append("prefill_burst: Burst-Zeitstempel fehlen")
                elif not (trigger_ns < dispatch <= opened <= first <= end):
                    pair_errors.append("prefill_burst: Burst-Zeitfolge ungültig")

    row = burst.get("schedule_row") or {}
    offload = normalize_offload(row.get("offload_gb"))
    k = row.get("server_max_num_seqs")
    repeat = row.get("repeat")
    burst_episode_id = str(burst.get("episode_id") or "prefill_burst")
    no_episode_id = str(no.get("episode_id") or "no_burst")
    burst_raw = raw_by_episode.get(burst_episode_id, {})
    no_raw = raw_by_episode.get(no_episode_id, {})

    active = set((burst.get("trigger") or {}).get("active_cohort_request_indices") or [])
    victim_by_index = {
        request.get("request_index"): request
        for request in burst.get("victim_requests", [])
        if isinstance(request, dict)
    }
    burst_requests = [
        request
        for request in burst.get("burst_requests", [])
        if isinstance(request, dict)
    ]

    classification = "D"
    raw_classification = "D"
    first_burst = None
    last_active_stream = None
    last_all_stream = None
    last_active_raw = None
    last_all_raw = None

    try:
        if not pair_errors:
            first_burst = min(
                burst_raw[str(request.get("request_id"))]["first_ns"]
                for request in burst_requests
            )
            last_active_stream = max(
                victim_by_index[index]["stream_end_perf_ns"] for index in active
            )
            last_all_stream = max(
                request["stream_end_perf_ns"] for request in victim_by_index.values()
            )
            last_active_raw = max(
                burst_raw[str(victim_by_index[index].get("request_id"))]["last_ns"]
                for index in active
            )
            last_all_raw = max(
                burst_raw[str(request.get("request_id"))]["last_ns"]
                for request in victim_by_index.values()
            )

            classification = (
                "A"
                if first_burst < last_active_stream
                else "C"
                if first_burst > last_all_stream
                else "B"
            )
            raw_classification = (
                "A"
                if first_burst < last_active_raw
                else "C"
                if first_burst > last_all_raw
                else "B"
            )
            if classification != raw_classification:
                pair_errors.append(
                    f"Stream-End-Klasse {classification} != Raw-Token-Klasse {raw_classification}"
                )
                classification = "D"
    except (KeyError, TypeError, ValueError) as exc:
        pair_errors.append(f"Klassifikation nicht rekonstruierbar: {exc}")
        classification = raw_classification = "D"

    no_stats = active_stats(no, no_raw)
    burst_stats = active_stats(burst, burst_raw)

    tpot_ratio = ratio(
        burst_stats.get("median_tpot_ms"),
        no_stats.get("median_tpot_ms"),
    )
    itl_ratio = ratio(
        burst_stats.get("median_max_itl_ms"),
        no_stats.get("median_max_itl_ms"),
    )
    completion_delta = (
        float(burst_stats["median_completion_ms"])
        - float(no_stats["median_completion_ms"])
        if finite(burst_stats.get("median_completion_ms"))
        and finite(no_stats.get("median_completion_ms"))
        else None
    )

    first_condition = row.get("condition_first_in_block")
    if first_condition not in {"no_burst", "prefill_burst"}:
        first_condition = "UNKNOWN"

    row_result = {
        "block_id": block_id,
        "offload": offload,
        "k": k,
        "repeat": repeat,
        "first_condition": first_condition,
        "class": classification,
        "raw_class": raw_classification,
        "tpot_ratio": tpot_ratio,
        "itl_ratio": itl_ratio,
        "completion_delta_ms": completion_delta,
        "after_active_s": (
            (first_burst - last_active_stream) / 1e9
            if type(first_burst) is int and type(last_active_stream) is int
            else None
        ),
        "remaining_until_all_s": (
            (last_all_stream - first_burst) / 1e9
            if type(first_burst) is int and type(last_all_stream) is int
            else None
        ),
        "pair_errors": pair_errors,
    }
    rows.append(row_result)
    class_counts[classification] += 1
    raw_class_counts[raw_classification] += 1
    raw_errors.extend(f"{block_id}: {error}" for error in pair_errors)

print(f"raw_reconstruction_errors: {len(raw_errors)}")
print(f"classification_counts: {dict(sorted(class_counts.items()))}")
print(f"raw_token_classification_counts: {dict(sorted(raw_class_counts.items()))}")

print()
print("=== 3. WISSENSCHAFTLICHE DESKRIPTION JE PAAR ===")
print(
    "block                          cls  first   "
    "TPOT-ratio  ITL-ratio  Completion-delta-ms  "
    "Burst-nach-Aktiv-s  Rest-bis-alle-s"
)
for row in rows:
    print(
        f"{row['block_id']:30s} "
        f"{row['class']:>3s} "
        f"{row['first_condition'][:7]:>7s} "
        f"{fmt(row['tpot_ratio'], 6):>11s} "
        f"{fmt(row['itl_ratio'], 6):>10s} "
        f"{fmt(row['completion_delta_ms'], 2):>20s} "
        f"{fmt(row['after_active_s'], 2):>19s} "
        f"{fmt(row['remaining_until_all_s'], 2):>16s}"
    )

grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
for row in rows:
    grouped[(row["offload"], row["k"])].append(row)

cell_summaries: list[dict[str, Any]] = []
print()
print("=== 4. WISSENSCHAFTLICHE DESKRIPTION JE ZELLE ===")
print(
    "offload  k  Klassen   TPOT-med [min,max]       "
    "ITL-med [min,max]        Completion-med-ms  "
    "nach-Aktiv-med-s  Rest-bis-alle-med-s"
)

for cell in sorted(grouped, key=str):
    group = grouped[cell]
    valid = [row for row in group if row["class"] != "D"]
    classes = Counter(row["class"] for row in group)
    tpot_min, tpot_max = value_range([row["tpot_ratio"] for row in valid])
    itl_min, itl_max = value_range([row["itl_ratio"] for row in valid])

    summary = {
        "offload": cell[0],
        "k": cell[1],
        "classes": dict(sorted(classes.items())),
        "tpot_med": median([row["tpot_ratio"] for row in valid]),
        "tpot_min": tpot_min,
        "tpot_max": tpot_max,
        "itl_med": median([row["itl_ratio"] for row in valid]),
        "itl_min": itl_min,
        "itl_max": itl_max,
        "completion_med": median([row["completion_delta_ms"] for row in valid]),
        "after_active_med": median([row["after_active_s"] for row in valid]),
        "remaining_med": median([row["remaining_until_all_s"] for row in valid]),
    }
    cell_summaries.append(summary)

    class_text = "/".join(f"{key}{value}" for key, value in sorted(classes.items()))
    print(
        f"{str(cell[0]):>7s} {str(cell[1]):>2s} "
        f"{class_text:<9s} "
        f"{fmt(summary['tpot_med'], 6)} "
        f"[{fmt(tpot_min, 6)},{fmt(tpot_max, 6)}]  "
        f"{fmt(summary['itl_med'], 6)} "
        f"[{fmt(itl_min, 6)},{fmt(itl_max, 6)}]  "
        f"{fmt(summary['completion_med'], 2):>17s} "
        f"{fmt(summary['after_active_med'], 2):>18s} "
        f"{fmt(summary['remaining_med'], 2):>20s}"
    )

cell_lookup = {(row["offload"], row["k"]): row for row in cell_summaries}
print()
print("=== 5. OFFLOAD-MODULATION: OFF12 MINUS OFF0 ===")
for k in (4, 8):
    low = cell_lookup.get((0, k))
    high = cell_lookup.get((12, k))
    if low and high:
        print(
            f"k={k}: "
            f"TPOT-ratio-delta={fmt(high['tpot_med'] - low['tpot_med'], 6)}, "
            f"ITL-ratio-delta={fmt(high['itl_med'] - low['itl_med'], 6)}, "
            f"Completion-delta-differenz-ms="
            f"{fmt(high['completion_med'] - low['completion_med'], 2)}"
        )
    else:
        print(f"k={k}: NA")

outliers = []
for row in rows:
    reasons = []
    if finite(row["tpot_ratio"]) and abs(float(row["tpot_ratio"]) - 1.0) >= 0.05:
        reasons.append("TPOT-Abweichung >=5%")
    if finite(row["itl_ratio"]) and abs(float(row["itl_ratio"]) - 1.0) >= 0.20:
        reasons.append("Max-ITL-Abweichung >=20%")
    if finite(row["completion_delta_ms"]) and abs(float(row["completion_delta_ms"])) >= 1000:
        reasons.append("|Completion-Delta| >=1000 ms")
    if reasons:
        outliers.append((row["block_id"], reasons))

print()
print("=== 6. AUFFÄLLIGE EINZELPAARE ===")
if outliers:
    for block_id, reasons in outliers:
        print(f"{block_id}: {', '.join(reasons)}")
else:
    print("keine nach den festgelegten deskriptiven Schwellen")

completeness_pass = not completeness_errors
raw_pass = not raw_errors
no_d = class_counts.get("D", 0) == 0
scientific_ready = completeness_pass and raw_pass and no_d

print()
print("=== 7. AUTOMATISCHE, VORSICHTIGE INTERPRETATION ===")

classes_by_cell = {
    (summary["offload"], summary["k"]): set(summary["classes"])
    for summary in cell_summaries
}
if (
    classes_by_cell.get((0, 4)) == {"C"}
    and classes_by_cell.get((12, 4)) == {"C"}
    and classes_by_cell.get((0, 8)) == {"B"}
    and classes_by_cell.get((12, 8)) == {"B"}
):
    print(
        "- Deskriptiv repliziert sich ein klares k-Regime: "
        "k=4 ergibt C, k=8 ergibt B, jeweils bei Offload 0 und 12."
    )
    print(
        "- Auf Klassenebene zeigt dieser Lauf damit eine Modulation durch "
        "max_num_seqs, nicht durch den Offload-Zustand."
    )
else:
    print(
        "- Das Klassenmuster entspricht nicht vollständig dem einfachen "
        "k=4/C- und k=8/B-Regime; die Einzelzellen sind maßgeblich."
    )

max_tpot_cell_deviation = max(
    (
        abs(float(summary["tpot_med"]) - 1.0)
        for summary in cell_summaries
        if finite(summary["tpot_med"])
    ),
    default=float("nan"),
)
if math.isfinite(max_tpot_cell_deviation):
    print(
        f"- Größte absolute Abweichung eines Zellmedians der TPOT-Ratio von 1: "
        f"{100 * max_tpot_cell_deviation:.3f}%."
    )
    print(
        "- Das ist eine deskriptive Effektgröße; ohne kampagnenübergreifende "
        "Streuung darf daraus kein allgemeiner Nullwirkungssatz folgen."
    )

print(
    "- A/B/C beziehen sich ausschließlich auf client-sichtbaren Output. "
    "Interne Admission und interner Prefill-Start werden nicht beobachtet."
)
print(
    "- Einzelne starke Max-ITL- oder Completion-Ausreißer müssen trotz "
    "stabiler Zellmediane separat berichtet und im Raw Trace geprüft werden."
)

print()
print("=== 8. ENDSTATUS ===")
print(f"COMPLETENESS_PASS={str(completeness_pass).lower()}")
print(f"RAW_RECONSTRUCTION_PASS={str(raw_pass).lower()}")
print(f"NO_D_CLASSIFICATIONS={str(no_d).lower()}")
print(f"SCIENTIFIC_SUMMARY_READY={str(scientific_ready).lower()}")
print(f"OUTER_MANIFEST_STATUS={outer_status}")
print(f"INNER_MANIFEST_STATUS={inner_status}")
print(f"COMPLETENESS_ERRORS={len(completeness_errors)}")
print(f"RAW_ERRORS={len(raw_errors)}")
print(f"WARNINGS={len(warnings)}")

if completeness_errors:
    print()
    print("=== VOLLSTÄNDIGKEITSFEHLER ===")
    for error in completeness_errors:
        print(f"- {error}")

if raw_errors:
    print()
    print("=== RAW-/TRIGGER-FEHLER ===")
    for error in raw_errors[:200]:
        print(f"- {error}")
    if len(raw_errors) > 200:
        print(f"- ... weitere {len(raw_errors) - 200} Fehler")

print()
print("=== DIESEN BLOCK AN CHATGPT SENDEN ===")
print(f"COMPLETENESS_PASS={str(completeness_pass).lower()}")
print(f"RAW_RECONSTRUCTION_PASS={str(raw_pass).lower()}")
print(f"SCIENTIFIC_SUMMARY_READY={str(scientific_ready).lower()}")
print(f"classification_counts={dict(sorted(class_counts.items()))}")
for summary in cell_summaries:
    print(
        f"cell offload={summary['offload']} k={summary['k']}: "
        f"classes={summary['classes']}, "
        f"TPOT_med={fmt(summary['tpot_med'], 6)}, "
        f"ITL_med={fmt(summary['itl_med'], 6)}, "
        f"Completion_med_ms={fmt(summary['completion_med'], 2)}, "
        f"after_active_med_s={fmt(summary['after_active_med'], 2)}, "
        f"remaining_all_med_s={fmt(summary['remaining_med'], 2)}"
    )
print(f"outlier_blocks={[block_id for block_id, _ in outliers]}")
print("scope=client-visible output only; no internal admission/prefill claim")

raise SystemExit(0 if scientific_ready else 2)
PY_ANALYSIS
RC=${PIPESTATUS[0]}
set -e

echo
echo "REPORT_SAVED=$REPORT"
echo "COMMAND_RC=$RC"
echo "TERMINAL_STILL_ALIVE"
exit "$RC"
