#!/usr/bin/env python3
"""
run_burst_calibration.py
========================
Episodischer Runner für die Burst-Kalibrierung des Projekts
"State-Aware Availability Degradation in LLM Serving".

Führt KEINE Policy-Evaluation durch. Vergleicht ausschließlich:
  - no_burst   : Victim ohne Burst
  - fixed_burst: Victim mit genau einem festen Burst

Der vLLM-Server muss bereits mit dem gewünschten --cpu-offload-gb laufen.
OFFLOAD_GB ist reines Ground-Truth-Metadatum und wird nicht geprüft.

Aufruf (Beispiel):
  python run_burst_calibration.py \\
      --offload-gb 12 \\
      --victim-concurrency 4 \\
      --burst-parallelism 4 \\
      --burst-output-len 256 \\
      --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

EXPERIMENT_ID: str = "availability_burst_calibration"
BURST_INPUT_LEN: int = 256
BURST_TEMPERATURE: float = 0.0
VICTIM_INPUT_LEN: int = 256
VICTIM_OUTPUT_LEN: int = 64
VICTIM_TEMPERATURE: float = 0.0

VALID_VICTIM_CONCURRENCIES: set[int] = {4, 8}
VALID_OFFLOAD_GB: set[int] = {0, 12}
VALID_BURST_PARALLELISMS: set[int] = {2, 4, 8}
VALID_BURST_OUTPUT_LENS: set[int] = {128, 256}
VALID_CONDITIONS: set[str] = {"no_burst", "fixed_burst"}

WORKER_SCRIPT_NAME: str = "request_worker.py"

DEFAULT_OUTPUT_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "data" / "burst_calibration"
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimingParams:
    victim_warmup_s: float
    measurement_window_s: float
    post_burst_s: float
    request_timeout_s: float
    drain_timeout_s: float


@dataclass
class EpisodeRecord:
    episode_index: int
    block_index: int
    episode_id: str
    condition: str
    offload_gb: int
    victim_concurrency: int
    burst_parallelism: int
    burst_output_len: int
    victim_seed: int
    burst_seed: int
    start_utc: str = ""
    end_utc: str = ""
    duration_s: float = 0.0
    victim_json: str = ""
    burst_json: str = ""
    victim_exit_code: Optional[int] = None
    burst_exit_code: Optional[int] = None
    status: str = "PENDING"
    error_text: str = ""
    victim_process_start_unix_ns: Optional[int] = None
    burst_process_start_unix_ns: Optional[int] = None
    intervention_time_unix_ns: Optional[int] = None
    intervention_offset_s: Optional[float] = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Burst-Kalibrierungs-Runner für Availability-Projekt."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--api-key", default="pilotkey")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")

    p.add_argument("--offload-gb", type=int, default=0)
    p.add_argument("--victim-concurrency", type=int, default=4)
    p.add_argument("--burst-parallelism", type=int, default=4)
    p.add_argument("--burst-output-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--conditions",
        default="no_burst,fixed_burst",
        help="Kommagetrennte Teilmenge von: no_burst,fixed_burst",
    )

    p.add_argument("--victim-warmup-s", type=float, default=20.0)
    p.add_argument("--measurement-window-s", type=float, default=60.0)
    p.add_argument("--post-burst-s", type=float, default=20.0)
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--drain-timeout-s", type=float, default=600.0)

    p.add_argument("--random-seed", type=int, default=42000)
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []

    if args.victim_concurrency not in VALID_VICTIM_CONCURRENCIES:
        errors.append(
            f"--victim-concurrency muss in {sorted(VALID_VICTIM_CONCURRENCIES)} sein, "
            f"got {args.victim_concurrency}"
        )
    if args.offload_gb not in VALID_OFFLOAD_GB:
        errors.append(
            f"--offload-gb muss in {sorted(VALID_OFFLOAD_GB)} sein, got {args.offload_gb}"
        )
    if args.burst_parallelism not in VALID_BURST_PARALLELISMS:
        errors.append(
            f"--burst-parallelism muss in {sorted(VALID_BURST_PARALLELISMS)} sein, "
            f"got {args.burst_parallelism}"
        )
    if args.burst_output_len not in VALID_BURST_OUTPUT_LENS:
        errors.append(
            f"--burst-output-len muss in {sorted(VALID_BURST_OUTPUT_LENS)} sein, "
            f"got {args.burst_output_len}"
        )
    if args.repeats < 1:
        errors.append(f"--repeats muss >= 1 sein, got {args.repeats}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    invalid_cond = set(conditions) - VALID_CONDITIONS
    if invalid_cond:
        errors.append(f"Ungültige conditions: {invalid_cond}")
    if not conditions:
        errors.append("--conditions darf nicht leer sein.")
    if len(conditions) != len(set(conditions)):
        errors.append("Doppelte Conditions sind nicht erlaubt (z.B. no_burst,no_burst).")

    for name in (
        "victim_warmup_s", "measurement_window_s", "post_burst_s",
    ):
        if getattr(args, name) < 0:
            errors.append(f"--{name.replace('_', '-')} muss >= 0 sein.")

    if args.drain_timeout_s <= 0:
        errors.append("--drain-timeout-s muss > 0 sein.")

    if args.request_timeout_s <= 0:
        errors.append("--request-timeout-s muss > 0 sein.")

    if errors:
        for e in errors:
            print(f"[FEHLER] {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest_atomic(manifest: dict, manifest_path: Path) -> None:
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def write_episodes_csv(records: list[EpisodeRecord], csv_path: Path) -> None:
    import csv as csv_mod
    fields = [
        "episode_index", "block_index", "episode_id", "condition",
        "offload_gb", "victim_concurrency", "burst_parallelism", "burst_output_len",
        "victim_seed", "burst_seed", "start_utc", "end_utc", "duration_s",
        "victim_json", "burst_json", "victim_exit_code", "burst_exit_code",
        "victim_process_start_unix_ns", "burst_process_start_unix_ns",
        "intervention_time_unix_ns", "intervention_offset_s",
        "status", "error_text",
    ]
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k, "") for k in fields})
    os.replace(tmp, csv_path)


# ---------------------------------------------------------------------------
# Serverprüfung
# ---------------------------------------------------------------------------

def check_server(base_url: str, api_key: str) -> bool:
    for path in ("/health", "/version", "/"):
        try:
            req = urllib.request.Request(
                f"{base_url}{path}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status < 400:
                    print(f"  [OK] Server antwortet auf {path} (HTTP {r.status})")
                    return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Episodenreihenfolge
# ---------------------------------------------------------------------------

def build_episode_order(
    conditions: list[str],
    repeats: int,
    base_seed: int,
    offload_gb: int,
    victim_concurrency: int,
    burst_parallelism: int,
    burst_output_len: int,
) -> list[EpisodeRecord]:
    rng = random.Random(base_seed)
    records: list[EpisodeRecord] = []
    episode_index = 0

    for block_idx in range(repeats):
        block_conditions = list(conditions)
        rng.shuffle(block_conditions)
        for condition in block_conditions:
            victim_seed = base_seed + episode_index * 1000 + 1
            burst_seed  = base_seed + episode_index * 1000 + 2
            episode_id = (
                f"calib_o{offload_gb}_vc{victim_concurrency}"
                f"_bp{burst_parallelism}_bo{burst_output_len}"
                f"_block{block_idx + 1:02d}_{condition}"
            )
            records.append(EpisodeRecord(
                episode_index=episode_index,
                block_index=block_idx,
                episode_id=episode_id,
                condition=condition,
                offload_gb=offload_gb,
                victim_concurrency=victim_concurrency,
                burst_parallelism=burst_parallelism,
                burst_output_len=burst_output_len,
                victim_seed=victim_seed,
                burst_seed=burst_seed,
            ))
            episode_index += 1

    return records


# ---------------------------------------------------------------------------
# Worker-Steuerung
# ---------------------------------------------------------------------------

def make_victim_env(
    args: argparse.Namespace,
    rec: EpisodeRecord,
    outfile: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "BASE_URL": args.base_url,
        "ENDPOINT": args.endpoint,
        "API_KEY": args.api_key,
        "MODEL": args.model,
        "MODE": "continuous_victim",
        "ROLE": "victim",
        "CONDITION": rec.condition,
        "EPISODE_ID": rec.episode_id,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "OFFLOAD_GB": str(rec.offload_gb),
        "RUN_CONCURRENCY": str(rec.victim_concurrency),
        "INPUT_LEN": str(VICTIM_INPUT_LEN),
        "OUTPUT_LEN": str(VICTIM_OUTPUT_LEN),
        "TEMPERATURE": str(VICTIM_TEMPERATURE),
        "CONCURRENCY": str(rec.victim_concurrency),
        "REQUEST_COUNT": "0",
        "RANDOM_SEED": str(rec.victim_seed),
        "REQUEST_TIMEOUT_S": str(args.request_timeout_s),
        "DRAIN_TIMEOUT_S": str(args.drain_timeout_s),
        "OUTFILE": str(outfile),
    }


def make_burst_env(
    args: argparse.Namespace,
    rec: EpisodeRecord,
    outfile: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "BASE_URL": args.base_url,
        "ENDPOINT": args.endpoint,
        "API_KEY": args.api_key,
        "MODEL": args.model,
        "MODE": "fixed_count_burst",
        "ROLE": "burst",
        "CONDITION": "fixed_burst",
        "EPISODE_ID": rec.episode_id,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "OFFLOAD_GB": str(rec.offload_gb),
        "RUN_CONCURRENCY": str(rec.victim_concurrency),
        "INPUT_LEN": str(BURST_INPUT_LEN),
        "OUTPUT_LEN": str(rec.burst_output_len),
        "TEMPERATURE": str(BURST_TEMPERATURE),
        "CONCURRENCY": str(rec.burst_parallelism),
        "REQUEST_COUNT": str(rec.burst_parallelism),
        "RANDOM_SEED": str(rec.burst_seed),
        "REQUEST_TIMEOUT_S": str(args.request_timeout_s),
        "DRAIN_TIMEOUT_S": str(args.drain_timeout_s),
        "OUTFILE": str(outfile),
    }


def start_worker(
    worker_path: Path,
    env: dict[str, str],
    stdout_log: Path,
    stderr_log: Path,
) -> subprocess.Popen:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_f = open(stdout_log, "w", encoding="utf-8")
    stderr_f = open(stderr_log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    finally:
        stdout_f.close()
        stderr_f.close()
    return proc


PARENT_DRAIN_GRACE_S: float = 5.0  # Reserve für Worker-Finalisierung und JSON-Schreiben


def stop_victim(proc: subprocess.Popen, drain_timeout_s: float) -> int:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=drain_timeout_s + PARENT_DRAIN_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return proc.returncode


def wait_for_json(json_path: Path, timeout_s: float, poll_interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if json_path.exists() and json_path.stat().st_size > 0:
            return True
        time.sleep(poll_interval)
    return False


GRACEFUL_DRAIN_S: float = 10.0  # Timeout für terminate() vor kill()


def kill_all(procs: list[subprocess.Popen], drain_timeout: float = GRACEFUL_DRAIN_S) -> None:
    """terminate() alle → gemeinsam warten → kill() verbleibende → wait() alle."""
    living = [p for p in procs if p.poll() is None]
    for p in living:
        try:
            p.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + drain_timeout
    for p in living:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for p in living:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass
    for p in procs:
        try:
            p.wait()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Episoden-Runner
# ---------------------------------------------------------------------------

def run_no_burst(
    rec: EpisodeRecord,
    episode_dir: Path,
    worker_path: Path,
    args: argparse.Namespace,
    timing: TimingParams,
    episode_start_ns: int,
) -> None:
    victim_json = episode_dir / "victim.json"
    rec.victim_json = str(victim_json)  # vorab setzen (7)

    victim_env = make_victim_env(args, rec, victim_json)
    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"    Starte Victim (warmup {timing.victim_warmup_s}s + "
          f"window {timing.measurement_window_s}s) …")

    ep_start_ns = episode_start_ns
    rec.victim_process_start_unix_ns = time.time_ns()
    v_proc = start_worker(
        worker_path, victim_env,
        episode_dir / "victim.stdout.log",
        episode_dir / "victim.stderr.log",
    )
    active_procs = [v_proc]

    try:
        # Frühes Scheitern prüfen (8)
        early_check = min(3.0, timing.victim_warmup_s)
        time.sleep(early_check)
        if v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
            raise RuntimeError(
                f"Victim-Worker beendete sich unerwartet nach {early_check}s "
                f"mit Exit-Code {v_proc.returncode}. "
                f"Siehe: {episode_dir / 'victim.stderr.log'}"
            )

        remaining_warmup = timing.victim_warmup_s - early_check
        if remaining_warmup > 0:
            time.sleep(remaining_warmup)

        # Nach vollem Warm-up prüfen (8)
        if v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
            raise RuntimeError(
                f"Victim-Worker beendete sich nach Warm-up mit Exit-Code "
                f"{v_proc.returncode}. Siehe: {episode_dir / 'victim.stderr.log'}"
            )

        # Interventionszeitpunkt = Ende Warm-up (6)
        rec.intervention_time_unix_ns = time.time_ns()
        rec.intervention_offset_s = (time.monotonic_ns() - ep_start_ns) / 1e9

        time.sleep(timing.measurement_window_s)
        print(f"    Stoppe Victim …")
        rc = stop_victim(v_proc, timing.drain_timeout_s)
        active_procs.clear()
    except BaseException as exc:
        kill_all(active_procs)
        if rec.victim_exit_code is None and v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
        if isinstance(exc, Exception):
            raise RuntimeError(f"no_burst episode fehlgeschlagen: {exc}") from exc
        raise

    rec.victim_exit_code = rc

    if rc not in (0, None):
        raise RuntimeError(
            f"Victim-Worker Exit-Code {rc} (erwartet 0). "
            f"Siehe: {episode_dir / 'victim.stderr.log'}"
        )
    if not wait_for_json(victim_json, timeout_s=timing.drain_timeout_s):
        raise RuntimeError(
            f"Victim-JSON nicht gefunden nach drain_timeout: {victim_json}"
        )


def run_fixed_burst(
    rec: EpisodeRecord,
    episode_dir: Path,
    worker_path: Path,
    args: argparse.Namespace,
    timing: TimingParams,
    episode_start_ns: int,
) -> None:
    victim_json = episode_dir / "victim.json"
    burst_json  = episode_dir / "burst.json"

    # Pfade vorab setzen (7)
    rec.victim_json = str(victim_json)
    rec.burst_json  = str(burst_json)

    victim_env = make_victim_env(args, rec, victim_json)
    burst_env  = make_burst_env(args, rec, burst_json)

    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"    Starte Victim (warmup {timing.victim_warmup_s}s) …")

    ep_start_ns = episode_start_ns
    rec.victim_process_start_unix_ns = time.time_ns()
    v_proc = start_worker(
        worker_path, victim_env,
        episode_dir / "victim.stdout.log",
        episode_dir / "victim.stderr.log",
    )
    active_procs: list[subprocess.Popen] = [v_proc]

    try:
        # Frühes Scheitern prüfen (8)
        early_check = min(3.0, timing.victim_warmup_s)
        time.sleep(early_check)
        if v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
            raise RuntimeError(
                f"Victim-Worker beendete sich unerwartet nach {early_check}s "
                f"mit Exit-Code {v_proc.returncode}. "
                f"Siehe: {episode_dir / 'victim.stderr.log'}"
            )

        remaining_warmup = timing.victim_warmup_s - early_check
        if remaining_warmup > 0:
            time.sleep(remaining_warmup)

        # Nach vollem Warm-up prüfen (8)
        if v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
            raise RuntimeError(
                f"Victim-Worker beendete sich nach Warm-up mit Exit-Code "
                f"{v_proc.returncode}. Siehe: {episode_dir / 'victim.stderr.log'}"
            )

        # Deadline erst nach Warm-up setzen (1)
        episode_deadline = time.monotonic() + timing.measurement_window_s

        # Interventionszeitpunkt = unmittelbar vor Burst-Start (6)
        rec.intervention_time_unix_ns = time.time_ns()
        rec.intervention_offset_s = (time.monotonic_ns() - ep_start_ns) / 1e9

        print(f"    Starte Burst (parallelism={rec.burst_parallelism}, "
              f"output_len={rec.burst_output_len}) …")
        rec.burst_process_start_unix_ns = time.time_ns()
        b_proc = start_worker(
            worker_path, burst_env,
            episode_dir / "burst.stdout.log",
            episode_dir / "burst.stderr.log",
        )
        active_procs.append(b_proc)

        # Warte auf Burst-Abschluss
        burst_timeout = timing.request_timeout_s + timing.drain_timeout_s
        try:
            b_rc = b_proc.wait(timeout=burst_timeout)
        except subprocess.TimeoutExpired:
            kill_all(active_procs)
            raise RuntimeError(
                f"Burst-Worker überschritt Timeout ({burst_timeout}s)."
            )

        rec.burst_exit_code = b_rc
        active_procs.remove(b_proc)

        if b_rc != 0:
            raise RuntimeError(
                f"Burst-Worker Exit-Code {b_rc} (erwartet 0). "
                f"Siehe: {episode_dir / 'burst.stderr.log'}"
            )
        if not wait_for_json(burst_json, timeout_s=timing.drain_timeout_s):
            raise RuntimeError(
                f"Burst-JSON nicht gefunden nach drain_timeout: {burst_json}"
            )

        # Deadline-Prüfung vor Post-Burst-Phase
        remaining = episode_deadline - time.monotonic()
        if remaining < timing.post_burst_s:
            kill_all(active_procs)
            raise RuntimeError(
                f"Deadline überschritten vor Post-Burst-Phase: "
                f"verbleibend={remaining:.1f}s, "
                f"benötigt={timing.post_burst_s}s, "
                f"measurement_window_s={timing.measurement_window_s}s"
            )

        print(f"    Burst abgeschlossen. Post-Burst {timing.post_burst_s}s …")
        time.sleep(timing.post_burst_s)

        print(f"    Stoppe Victim …")
        v_rc = stop_victim(v_proc, timing.drain_timeout_s)
        active_procs.clear()

    except BaseException as exc:
        kill_all(active_procs)
        if rec.victim_exit_code is None and v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
        if isinstance(exc, Exception):
            raise RuntimeError(f"fixed_burst episode fehlgeschlagen: {exc}") from exc
        raise

    rec.victim_exit_code = v_rc

    if v_rc not in (0, None):
        raise RuntimeError(
            f"Victim-Worker Exit-Code {v_rc} (erwartet 0). "
            f"Siehe: {episode_dir / 'victim.stderr.log'}"
        )
    if not wait_for_json(victim_json, timeout_s=timing.drain_timeout_s):
        raise RuntimeError(
            f"Victim-JSON nicht gefunden nach drain_timeout: {victim_json}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    timing = TimingParams(
        victim_warmup_s=args.victim_warmup_s,
        measurement_window_s=args.measurement_window_s,
        post_burst_s=args.post_burst_s,
        request_timeout_s=args.request_timeout_s,
        drain_timeout_s=args.drain_timeout_s,
    )

    # Worker-Pfad
    worker_path = Path(__file__).parent / WORKER_SCRIPT_NAME
    if not worker_path.exists():
        print(
            f"[FEHLER] Worker nicht gefunden: {worker_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    worker_sha256 = sha256_of_file(worker_path)
    runner_sha256 = sha256_of_file(Path(__file__))

    # Output-Ordner
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_label = (
        f"calib_offload{args.offload_gb}"
        f"_vc{args.victim_concurrency}"
        f"_bp{args.burst_parallelism}"
        f"_bo{args.burst_output_len}"
        f"_{ts}"
    )
    run_dir = args.output_root / run_label
    if run_dir.exists():
        print(f"[FEHLER] Run-Ordner existiert bereits: {run_dir}", file=sys.stderr)
        sys.exit(1)
    run_dir.mkdir(parents=True, exist_ok=False)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir()

    manifest_path = run_dir / "calibration_manifest.json"
    csv_path      = run_dir / "episodes.csv"

    # Episodenreihenfolge
    records = build_episode_order(
        conditions=conditions,
        repeats=args.repeats,
        base_seed=args.random_seed,
        offload_gb=args.offload_gb,
        victim_concurrency=args.victim_concurrency,
        burst_parallelism=args.burst_parallelism,
        burst_output_len=args.burst_output_len,
    )

    episode_order_summary = [
        {"episode_index": r.episode_index, "episode_id": r.episode_id,
         "condition": r.condition, "block_index": r.block_index,
         "victim_seed": r.victim_seed, "burst_seed": r.burst_seed}
        for r in records
    ]

    manifest: dict = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": now_utc(),
        "model": args.model,
        "base_url": args.base_url,
        "offload_gb": args.offload_gb,
        "victim_profile": {
            "input_len": VICTIM_INPUT_LEN,
            "output_len": VICTIM_OUTPUT_LEN,
            "temperature": VICTIM_TEMPERATURE,
        },
        "victim_concurrency": args.victim_concurrency,
        "burst_profile": {
            "input_len": BURST_INPUT_LEN,
            "output_len": args.burst_output_len,
            "temperature": BURST_TEMPERATURE,
            "parallelism": args.burst_parallelism,
            "request_count": args.burst_parallelism,
        },
        "repeats": args.repeats,
        "conditions": conditions,
        "episode_order": episode_order_summary,
        "base_random_seed": args.random_seed,
        "timing_parameters": {
            "victim_warmup_s": timing.victim_warmup_s,
            "measurement_window_s": timing.measurement_window_s,
            "post_burst_s": timing.post_burst_s,
            "request_timeout_s": timing.request_timeout_s,
            "drain_timeout_s": timing.drain_timeout_s,
        },
        "request_worker_path": str(worker_path),
        "request_worker_sha256": worker_sha256,
        "runner_sha256": runner_sha256,
        "status": "RUNNING",
    }
    write_manifest_atomic(manifest, manifest_path)
    write_episodes_csv(records, csv_path)

    # Banner
    print("=" * 66)
    print(" Burst-Kalibrierung — episodischer Runner")
    print(f" offload_gb         : {args.offload_gb}")
    print(f" victim_concurrency : {args.victim_concurrency}")
    print(f" burst_parallelism  : {args.burst_parallelism}")
    print(f" burst_output_len   : {args.burst_output_len}")
    print(f" repeats            : {args.repeats}")
    print(f" conditions         : {conditions}")
    print(f" total episodes     : {len(records)}")
    print(f" output             : {run_dir}")
    print("=" * 66)

    # Serverprüfung
    print("Prüfe Server …")
    if not check_server(args.base_url, args.api_key):
        manifest["status"] = "FAILED"
        manifest["error_text"] = f"Server unter {args.base_url} nicht erreichbar."
        write_manifest_atomic(manifest, manifest_path)
        print(f"[FEHLER] {manifest['error_text']}", file=sys.stderr)
        sys.exit(1)
    print()

    # Episoden
    failed = False
    try:
        for rec in records:
            episode_dir = episodes_dir / rec.episode_id
            print(f"[{rec.episode_index + 1}/{len(records)}] {rec.episode_id}")
            rec.start_utc = now_utc()
            t0 = time.monotonic()
            episode_start_ns = time.monotonic_ns()

            try:
                if rec.condition == "no_burst":
                    run_no_burst(rec, episode_dir, worker_path, args, timing, episode_start_ns)
                elif rec.condition == "fixed_burst":
                    run_fixed_burst(rec, episode_dir, worker_path, args, timing, episode_start_ns)
                else:
                    raise RuntimeError(f"Unbekannte condition: {rec.condition}")

                rec.status = "OK"

            except Exception as exc:
                rec.status = "FAILED"
                rec.error_text = str(exc)
                print(f"  [FEHLER] {exc}", file=sys.stderr)
                failed = True

            rec.end_utc = now_utc()
            rec.duration_s = round(time.monotonic() - t0, 2)
            write_episodes_csv(records, csv_path)  # atomar (9)
            print(f"  → {rec.status}  ({rec.duration_s}s)\n")

            if failed:
                break

    except KeyboardInterrupt:
        aborted_utc = now_utc()
        # Laufenden Record abschließen
        if records:
            pending = next((r for r in records if r.status in ("PENDING", "RUNNING")), None)
            if pending is not None:
                pending.status     = "FAILED"
                pending.error_text = "KeyboardInterrupt"
                pending.end_utc    = aborted_utc
                if pending.start_utc:
                    try:
                        from datetime import datetime as _dt
                        t0_ = _dt.fromisoformat(pending.start_utc)
                        t1_ = _dt.fromisoformat(aborted_utc)
                        pending.duration_s = round((t1_ - t0_).total_seconds(), 2)
                    except Exception:
                        pass
        manifest["status"]      = "FAILED"
        manifest["error_text"]  = "KeyboardInterrupt"
        manifest["aborted_utc"] = aborted_utc
        write_manifest_atomic(manifest, manifest_path)
        write_episodes_csv(records, csv_path)
        print("\n[ABGEBROCHEN] KeyboardInterrupt.", file=sys.stderr)
        sys.exit(130)

    except BaseException as exc:
        aborted_utc = now_utc()
        manifest["status"]      = "FAILED"
        manifest["error_text"]  = repr(exc)
        manifest["aborted_utc"] = aborted_utc
        write_manifest_atomic(manifest, manifest_path)
        write_episodes_csv(records, csv_path)
        print(f"\n[FEHLER] {exc}", file=sys.stderr)
        sys.exit(1)

    # Manifest abschließen
    manifest["status"] = "FAILED" if failed else "COMPLETE"
    if failed:
        failed_rec = next((r for r in records if r.status == "FAILED"), None)
        manifest["error_text"] = failed_rec.error_text if failed_rec else "Unbekannter Fehler"
    write_manifest_atomic(manifest, manifest_path)
    write_episodes_csv(records, csv_path)

    print("=" * 66)
    if failed:
        print(" Kalibrierung FEHLGESCHLAGEN.")
        sys.exit(1)
    else:
        ok_count = sum(1 for r in records if r.status == "OK")
        print(f" Kalibrierung ABGESCHLOSSEN — {ok_count}/{len(records)} Episoden OK.")
        print(f" Output: {run_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
