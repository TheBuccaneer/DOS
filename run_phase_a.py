#!/usr/bin/env python3
"""
run_phase_a.py
==============
Episodischer Runner für **Phase A: State-dependent Availability in LLM Serving**.

Phase A misst NUR die State x Burst-Interaktion. Es werden ausdrücklich KEINE
Policies gebaut (kein Random / Threshold / State-Aware / Oracle), keine
Mitigation und keine Hauptmessung.

Verglichen werden ausschliesslich zwei Conditions:
  - no_burst    : Victim laeuft ohne Burst
  - fixed_burst : Victim laeuft, Burst startet bei festem burst_offset_s

Validitaetskern (Unterschied zum alten burst_calibration-Runner):
  Das Victim-Messfenster ist in BEIDEN Conditions exakt gleich lang
  (= measurement_window_s). Die Burst-Dauer bestimmt NICHT mehr, wie lange
  das Victim gemessen wird. Dadurch kann die zustandsabhaengige Burst-Dauer
  die spaetere normalized_log_DiD nicht mehr verfaelschen.

Der vLLM-Server muss bereits mit dem gewuenschten --cpu-offload-gb laufen.
OFFLOAD_GB ist reines Ground-Truth-Metadatum und wird nicht geprueft.

Aufruf (Beispiel, eine Phase-A-Zelle = ein (offload, conc), beide Conditions):
  python run_phase_a.py \
      --offload-gb 0 \
      --victim-concurrency 4 \
      --repeats 5

Rollen-/Filterkonvention fuer die spaetere Analyse (siehe Kopf von Abschnitt
"Logging" unten):
  victim-Messrequests = (role == "victim") AND
                        (measurement_window_start <= start_time_unix_ns
                                                  <= measurement_window_end)
  load-Requests       = (role == "burst")   # Worker-Term fuer die Last-Action

Warmup laeuft als SEPARATER, vollstaendig abgeschlossener Victim-Lauf VOR dem
Messfenster und landet in warmup.json. In victim.json gibt es daher keinen
ungezaehlten role==victim-Vorlauf vor measurement_window_start mehr; der
Victim-Worker startet exakt mit measurement_window_start.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request

# ---------------------------------------------------------------------------
# Konstanten (Phase-A-Scope-Freeze)
# ---------------------------------------------------------------------------

EXPERIMENT_ID: str = "phase_a_state_dependent_availability"

VICTIM_INPUT_LEN: int = 256
VICTIM_OUTPUT_LEN: int = 64
VICTIM_TEMPERATURE: float = 0.0

BURST_INPUT_LEN: int = 256
BURST_TEMPERATURE: float = 0.0

# Phase-A-Guards
VALID_VICTIM_CONCURRENCIES: set[int] = {4, 8}
VALID_OFFLOAD_GB: set[int] = {0, 12}
VALID_BURST_PARALLELISMS: set[int] = {2, 4, 8}
VALID_BURST_OUTPUT_LENS: set[int] = {128, 256}
VALID_CONDITIONS: set[str] = {"no_burst", "fixed_burst"}

WORKER_SCRIPT_NAME: str = "request_worker.py"

DEFAULT_OUTPUT_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "data" / "phase_a"
)

# Grace-Reserven fuer Prozess-Finalisierung
PARENT_DRAIN_GRACE_S: float = 5.0
GRACEFUL_DRAIN_S: float = 10.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimingParams:
    victim_warmup_s: float
    measurement_window_s: float
    burst_offset_s: float
    burst_completion_timeout_s: float
    request_timeout_s: float
    drain_timeout_s: float


@dataclass
class EpisodeRecord:
    # Identitaet / Matrix
    episode_index: int
    block_index: int
    repeat_id: int
    episode_id: str
    condition: str
    offload_gb: int
    victim_concurrency: int
    burst_parallelism: int
    burst_output_len: int
    victim_seed: int
    burst_seed: int

    # Fensterdefinition (Wall-Clock, time_ns) - identisch ueber beide Conditions
    measurement_window_s: float = 0.0
    measurement_window_start_unix_ns: Optional[int] = None
    measurement_window_end_unix_ns: Optional[int] = None
    burst_offset_s: float = 0.0
    intervention_time_unix_ns: Optional[int] = None  # window_start + burst_offset_s
    warmup_done: bool = False

    # Burst-Telemetrie (beeinflusst NICHT das Victim-Fenster)
    burst_started: bool = False
    burst_start_time_unix_ns: Optional[int] = None
    burst_end_time_unix_ns: Optional[int] = None
    burst_completed: bool = False
    burst_cancelled_or_timeout: bool = False
    burst_exit_code: Optional[int] = None

    # Lauf-Metadaten
    start_utc: str = ""
    end_utc: str = ""
    duration_s: float = 0.0
    victim_json: str = ""
    burst_json: str = ""
    victim_exit_code: Optional[int] = None
    victim_process_start_unix_ns: Optional[int] = None
    burst_process_start_unix_ns: Optional[int] = None
    status: str = "PENDING"
    error_text: str = ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-A-Runner: State-dependent Availability (no_burst vs fixed_burst)."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--api-key", default="pilotkey")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")

    p.add_argument("--offload-gb", type=int, default=0)
    p.add_argument("--victim-concurrency", type=int, default=4)
    p.add_argument("--burst-parallelism", type=int, default=4)
    p.add_argument("--burst-output-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--conditions",
        default="no_burst,fixed_burst",
        help="Kommagetrennte Teilmenge von: no_burst,fixed_burst "
             "(Default: beide, block-randomisiert interleaved).",
    )

    # Fenster-/Timing-Parameter
    p.add_argument("--victim-warmup-s", type=float, default=20.0,
                   help="Warmup-Dauer VOR dem Messfenster.")
    p.add_argument("--measurement-window-s", type=float, default=90.0,
                   help="Laenge des Victim-Messfensters. IDENTISCH ueber beide Conditions.")
    p.add_argument("--burst-offset-s", type=float, default=30.0,
                   help="Offset des Burst-Starts ab Messfenster-Beginn. Muss < measurement-window-s sein.")
    p.add_argument("--burst-completion-timeout-s", type=float, default=300.0,
                   help="Max. Wartezeit NACH Fensterende, bis ein noch laufender Burst "
                        "kontrolliert beendet wird. Beeinflusst nur Burst-Telemetrie, nicht das Victim-Fenster.")
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--drain-timeout-s", type=float, default=600.0)

    p.add_argument("--random-seed", type=int, default=42000)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
        errors.append(f"Ungueltige conditions: {invalid_cond}")
    if not conditions:
        errors.append("--conditions darf nicht leer sein.")
    if len(conditions) != len(set(conditions)):
        errors.append("Doppelte Conditions sind nicht erlaubt.")

    # Fenster-Konsistenz
    if args.victim_warmup_s < 0:
        errors.append("--victim-warmup-s muss >= 0 sein.")
    if args.measurement_window_s <= 0:
        errors.append("--measurement-window-s muss > 0 sein.")
    if args.burst_offset_s < 0:
        errors.append("--burst-offset-s muss >= 0 sein.")
    if args.burst_offset_s >= args.measurement_window_s:
        errors.append(
            f"--burst-offset-s ({args.burst_offset_s}) muss < "
            f"--measurement-window-s ({args.measurement_window_s}) sein."
        )
    if args.burst_completion_timeout_s < 0:
        errors.append("--burst-completion-timeout-s muss >= 0 sein.")
    if args.request_timeout_s <= 0:
        errors.append("--request-timeout-s muss > 0 sein.")
    if args.drain_timeout_s <= 0:
        errors.append("--drain-timeout-s muss > 0 sein.")

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


CSV_FIELDS = [
    "episode_index", "block_index", "repeat_id", "episode_id", "condition",
    "offload_gb", "victim_concurrency", "burst_parallelism", "burst_output_len",
    "victim_seed", "burst_seed",
    "measurement_window_s",
    "measurement_window_start_unix_ns", "measurement_window_end_unix_ns",
    "burst_offset_s", "intervention_time_unix_ns", "warmup_done",
    "burst_started", "burst_start_time_unix_ns", "burst_end_time_unix_ns",
    "burst_completed", "burst_cancelled_or_timeout", "burst_exit_code",
    "start_utc", "end_utc", "duration_s",
    "victim_json", "burst_json", "victim_exit_code",
    "victim_process_start_unix_ns", "burst_process_start_unix_ns",
    "status", "error_text",
]


def write_episodes_csv(records: list[EpisodeRecord], csv_path: Path) -> None:
    import csv as csv_mod
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k, "") for k in CSV_FIELDS})
    os.replace(tmp, csv_path)


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
# Episodenreihenfolge (block-randomisiert ueber Conditions je Repeat)
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
            burst_seed = base_seed + episode_index * 1000 + 2
            episode_id = (
                f"phaseA_o{offload_gb}_vc{victim_concurrency}"
                f"_bp{burst_parallelism}_bo{burst_output_len}"
                f"_rep{block_idx + 1:02d}_{condition}"
            )
            records.append(EpisodeRecord(
                episode_index=episode_index,
                block_index=block_idx,
                repeat_id=block_idx + 1,
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

def make_victim_env(args: argparse.Namespace, rec: EpisodeRecord, outfile: Path) -> dict[str, str]:
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
        "REQUEST_COUNT": "0",  # vom Worker als "unbounded/continuous" behandelt
        "RANDOM_SEED": str(rec.victim_seed),
        "REQUEST_TIMEOUT_S": str(args.request_timeout_s),
        "DRAIN_TIMEOUT_S": str(args.drain_timeout_s),
        "OUTFILE": str(outfile),
    }


def make_burst_env(args: argparse.Namespace, rec: EpisodeRecord, outfile: Path) -> dict[str, str]:
    return {
        **os.environ,
        "BASE_URL": args.base_url,
        "ENDPOINT": args.endpoint,
        "API_KEY": args.api_key,
        "MODEL": args.model,
        "MODE": "fixed_count_burst",
        "ROLE": "burst",  # Worker akzeptiert nur victim|burst; "burst" == Last-Action ("load")
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


def start_worker(worker_path: Path, env: dict[str, str],
                 stdout_log: Path, stderr_log: Path) -> subprocess.Popen:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_f = open(stdout_log, "w", encoding="utf-8")
    stderr_f = open(stderr_log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            env=env, stdout=stdout_f, stderr=stderr_f,
        )
    finally:
        stdout_f.close()
        stderr_f.close()
    return proc


def stop_victim(proc: subprocess.Popen, drain_timeout_s: float) -> int:
    """Beendet den Victim-Worker (SIGTERM -> drain -> SIGKILL)."""
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


def kill_all(procs: list[subprocess.Popen], drain_timeout: float = GRACEFUL_DRAIN_S) -> None:
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


def _sleep_until(target_monotonic: float) -> None:
    """Schlaeft bis zu einem absoluten monotonic-Zeitpunkt (driftfrei)."""
    remaining = target_monotonic - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _check_victim_alive(v_proc: subprocess.Popen, rec: EpisodeRecord,
                        episode_dir: Path, phase: str) -> None:
    if v_proc.poll() is not None:
        rec.victim_exit_code = v_proc.returncode
        raise RuntimeError(
            f"Victim-Worker beendete sich unerwartet ({phase}) mit Exit-Code "
            f"{v_proc.returncode}. Siehe: {episode_dir / 'victim.stderr.log'}"
        )


# ---------------------------------------------------------------------------
# Einheitlicher Episoden-Runner (identisches Victim-Fenster fuer beide Conditions)
# ---------------------------------------------------------------------------

def run_episode(
    rec: EpisodeRecord,
    episode_dir: Path,
    worker_path: Path,
    args: argparse.Namespace,
    timing: TimingParams,
) -> None:
    """
    Ablauf (identisch fuer no_burst und fixed_burst):

      [Server laeuft bereits im Offload-State]
      start victim (continuous)
      warmup  : victim_warmup_s
      ---- MESSFENSTER START (window_start) ----
      sleep   : burst_offset_s
      intervention_time = window_start + burst_offset_s
        - fixed_burst : start burst (NICHT-blockierend)
        - no_burst    : nichts (nur pseudo_intervention geloggt)
      sleep   : measurement_window_s - burst_offset_s
      ---- MESSFENSTER ENDE (window_end) ----
      stop victim   -> Victim wurde EXAKT measurement_window_s lang beprobt
      danach: noch laufenden Burst bis burst_completion_timeout_s auslaufen
              lassen oder kontrolliert beenden (beeinflusst Victim NICHT mehr).
    """
    victim_json = episode_dir / "victim.json"
    rec.victim_json = str(victim_json)
    episode_dir.mkdir(parents=True, exist_ok=True)

    rec.measurement_window_s = timing.measurement_window_s
    rec.burst_offset_s = timing.burst_offset_s

    victim_env = make_victim_env(args, rec, victim_json)

    b_proc: Optional[subprocess.Popen] = None
    burst_json = episode_dir / "burst.json"

    # ---- WARMUP: separater, vollstaendig abgeschlossener continuous-victim-Lauf
    #      VOR dem Messfenster. Wird gestartet UND gestoppt, bevor das Messfenster
    #      oeffnet. So koennen Warmup-Requests nicht ins Messfenster hineinlaufen
    #      und dort Concurrency blockieren.
    if timing.victim_warmup_s > 0:
        print(f"    Warmup (separat, {timing.victim_warmup_s}s) ...")
        warmup_env = make_victim_env(args, rec, episode_dir / "warmup.json")
        warmup_env["ROLE"] = "victim"
        wproc = start_worker(
            worker_path, warmup_env,
            episode_dir / "warmup.stdout.log",
            episode_dir / "warmup.stderr.log",
        )
        try:
            early = min(3.0, timing.victim_warmup_s)
            time.sleep(early)
            if wproc.poll() is not None:
                raise RuntimeError(
                    f"Warmup-Worker beendete sich unerwartet mit Exit-Code {wproc.returncode}. "
                    f"Siehe: {episode_dir / 'warmup.stderr.log'}"
                )
            _sleep_until(time.monotonic() + max(0.0, timing.victim_warmup_s - early))
        finally:
            # Warmup VOLLSTAENDIG beenden (drain), bevor das Messfenster startet.
            stop_victim(wproc, timing.drain_timeout_s)
        wait_for_json(episode_dir / "warmup.json", timeout_s=timing.drain_timeout_s)
    rec.warmup_done = True

    # ---- frischer continuous victim worker FUER das Messfenster ----
    # Messfenster startet GLEICHZEITIG mit dem Victim-Worker-Start: kein
    # ungezaehlter Victim-Vorlauf vor measurement_window_start.
    print(f"    Starte Victim-Messfenster ({timing.measurement_window_s}s, "
          f"burst_offset {timing.burst_offset_s}s) ...")

    # ---- MESSFENSTER START (== Victim-Worker-Start) ----
    window_start_mono = time.monotonic()
    rec.measurement_window_start_unix_ns = time.time_ns()
    rec.victim_process_start_unix_ns = rec.measurement_window_start_unix_ns
    v_proc = start_worker(
        worker_path, victim_env,
        episode_dir / "victim.stdout.log",
        episode_dir / "victim.stderr.log",
    )
    active_procs: list[subprocess.Popen] = [v_proc]

    try:
        # Alive-Pruefung OHNE eigenen Sleep: erst beim Burst-Offset-Sleep unten
        # vergeht Zeit. Hier nur ein nicht-blockierender Frueh-Check.
        _check_victim_alive(v_proc, rec, episode_dir, "bei Messfenster-Start")

        # ---- bis zum Burst-Offset ----
        _sleep_until(window_start_mono + timing.burst_offset_s)
        rec.intervention_time_unix_ns = time.time_ns()

        if rec.condition == "fixed_burst":
            _check_victim_alive(v_proc, rec, episode_dir, "vor Burst-Start")
            print(f"    Starte Burst (parallelism={rec.burst_parallelism}, "
                  f"output_len={rec.burst_output_len}) bei offset {timing.burst_offset_s}s ...")
            burst_env = make_burst_env(args, rec, burst_json)
            rec.burst_json = str(burst_json)
            rec.burst_process_start_unix_ns = time.time_ns()
            b_proc = start_worker(
                worker_path, burst_env,
                episode_dir / "burst.stdout.log",
                episode_dir / "burst.stderr.log",
            )
            active_procs.append(b_proc)
            rec.burst_started = True
            rec.burst_start_time_unix_ns = rec.burst_process_start_unix_ns
        else:
            # no_burst: nur pseudo_intervention; kein Burst.
            print(f"    [no_burst] pseudo_intervention bei offset {timing.burst_offset_s}s "
                  f"(kein Burst).")

        # ---- restliches Messfenster (Victim-Fenster bleibt fix) ----
        _sleep_until(window_start_mono + timing.measurement_window_s)
        rec.measurement_window_end_unix_ns = time.time_ns()

        # ---- Victim stoppen: exakt measurement_window_s beprobt ----
        print("    Messfenster beendet. Stoppe Victim ...")
        v_rc = stop_victim(v_proc, timing.drain_timeout_s)
        if v_proc in active_procs:
            active_procs.remove(v_proc)
        rec.victim_exit_code = v_rc

        # ---- Burst NACH Fensterende abwickeln (kein Einfluss aufs Victim) ----
        if b_proc is not None:
            if b_proc.poll() is None:
                print(f"    Burst laeuft nach Fensterende noch. Warte bis "
                      f"{timing.burst_completion_timeout_s}s ...")
                try:
                    b_proc.wait(timeout=timing.burst_completion_timeout_s)
                    rec.burst_completed = (b_proc.returncode == 0)
                except subprocess.TimeoutExpired:
                    print("    Burst-Timeout nach Fensterende -> kontrolliert beenden.")
                    kill_all([b_proc])
                    rec.burst_cancelled_or_timeout = True
            else:
                rec.burst_completed = (b_proc.returncode == 0)
            if b_proc in active_procs:
                active_procs.remove(b_proc)
            rec.burst_exit_code = b_proc.returncode
            rec.burst_end_time_unix_ns = time.time_ns()
            # Burst-JSON best effort (nicht erfolgskritisch fuer Phase A)
            wait_for_json(burst_json, timeout_s=timing.drain_timeout_s)

    except BaseException as exc:
        kill_all(active_procs)
        if rec.victim_exit_code is None and v_proc.poll() is not None:
            rec.victim_exit_code = v_proc.returncode
        if isinstance(exc, Exception):
            raise RuntimeError(f"{rec.condition} episode fehlgeschlagen: {exc}") from exc
        raise

    # ---- Victim-Validitaet ----
    if rec.victim_exit_code not in (0, None):
        raise RuntimeError(
            f"Victim-Worker Exit-Code {rec.victim_exit_code} (erwartet 0). "
            f"Siehe: {episode_dir / 'victim.stderr.log'}"
        )
    if not wait_for_json(victim_json, timeout_s=timing.drain_timeout_s):
        raise RuntimeError(f"Victim-JSON nicht gefunden nach drain_timeout: {victim_json}")


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
        burst_offset_s=args.burst_offset_s,
        burst_completion_timeout_s=args.burst_completion_timeout_s,
        request_timeout_s=args.request_timeout_s,
        drain_timeout_s=args.drain_timeout_s,
    )

    worker_path = Path(__file__).parent / WORKER_SCRIPT_NAME
    if not worker_path.exists():
        print(f"[FEHLER] Worker nicht gefunden: {worker_path}", file=sys.stderr)
        sys.exit(1)
    worker_sha256 = sha256_of_file(worker_path)
    runner_sha256 = sha256_of_file(Path(__file__))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_label = (
        f"phaseA_offload{args.offload_gb}"
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

    manifest_path = run_dir / "phase_a_manifest.json"
    csv_path = run_dir / "episodes.csv"

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
         "condition": r.condition, "repeat_id": r.repeat_id,
         "victim_seed": r.victim_seed, "burst_seed": r.burst_seed}
        for r in records
    ]

    manifest: dict = {
        "schema_version": "2.0",
        "experiment_id": EXPERIMENT_ID,
        "phase": "A",
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
            "burst_offset_s": timing.burst_offset_s,
            "burst_completion_timeout_s": timing.burst_completion_timeout_s,
            "request_timeout_s": timing.request_timeout_s,
            "drain_timeout_s": timing.drain_timeout_s,
        },
        "primary_metric": "episode-level victim median TPOT",
        "victim_request_filter": {
            "role": "victim",
            "fully_in_window": (
                "measurement_window_start_unix_ns <= start_time_unix_ns "
                "AND end_time_unix_ns <= measurement_window_end_unix_ns"
            ),
            "keep": (
                "request_success == true AND cancelled == false AND timed_out == false "
                "AND error_text empty AND tpot_ms is not null"
            ),
            "note": "role 'burst' == load action; warmup wird separat vor dem Fenster gemessen und ist NICHT in victim.json.",
        },
        "request_worker_path": str(worker_path),
        "request_worker_sha256": worker_sha256,
        "runner_sha256": runner_sha256,
        "status": "RUNNING",
    }
    write_manifest_atomic(manifest, manifest_path)
    write_episodes_csv(records, csv_path)

    print("=" * 66)
    print(" Phase A — State-dependent Availability (Interaktionspilot)")
    print(f" offload_gb         : {args.offload_gb}")
    print(f" victim_concurrency : {args.victim_concurrency}")
    print(f" burst_parallelism  : {args.burst_parallelism}")
    print(f" burst_output_len   : {args.burst_output_len}")
    print(f" measurement_window : {timing.measurement_window_s}s (identisch ueber Conditions)")
    print(f" burst_offset       : {timing.burst_offset_s}s")
    print(f" repeats            : {args.repeats}")
    print(f" conditions         : {conditions}")
    print(f" total episodes     : {len(records)}")
    print(f" output             : {run_dir}")
    print("=" * 66)

    print("Pruefe Server ...")
    if not check_server(args.base_url, args.api_key):
        manifest["status"] = "FAILED"
        manifest["error_text"] = f"Server unter {args.base_url} nicht erreichbar."
        write_manifest_atomic(manifest, manifest_path)
        print(f"[FEHLER] {manifest['error_text']}", file=sys.stderr)
        sys.exit(1)
    print()

    failed = False
    try:
        for rec in records:
            episode_dir = episodes_dir / rec.episode_id
            print(f"[{rec.episode_index + 1}/{len(records)}] {rec.episode_id}")
            rec.start_utc = now_utc()
            t0 = time.monotonic()
            try:
                run_episode(rec, episode_dir, worker_path, args, timing)
                rec.status = "OK"
            except Exception as exc:
                rec.status = "FAILED"
                rec.error_text = str(exc)
                print(f"  [FEHLER] {exc}", file=sys.stderr)
                failed = True
            rec.end_utc = now_utc()
            rec.duration_s = round(time.monotonic() - t0, 2)
            write_episodes_csv(records, csv_path)
            print(f"  -> {rec.status}  ({rec.duration_s}s)\n")
            if failed:
                break

    except KeyboardInterrupt:
        aborted_utc = now_utc()
        pending = next((r for r in records if r.status in ("PENDING", "RUNNING")), None)
        if pending is not None:
            pending.status = "FAILED"
            pending.error_text = "KeyboardInterrupt"
            pending.end_utc = aborted_utc
        manifest["status"] = "FAILED"
        manifest["error_text"] = "KeyboardInterrupt"
        manifest["aborted_utc"] = aborted_utc
        write_manifest_atomic(manifest, manifest_path)
        write_episodes_csv(records, csv_path)
        print("\n[ABGEBROCHEN] KeyboardInterrupt.", file=sys.stderr)
        sys.exit(130)

    except BaseException as exc:
        aborted_utc = now_utc()
        manifest["status"] = "FAILED"
        manifest["error_text"] = repr(exc)
        manifest["aborted_utc"] = aborted_utc
        write_manifest_atomic(manifest, manifest_path)
        write_episodes_csv(records, csv_path)
        print(f"\n[FEHLER] {exc}", file=sys.stderr)
        sys.exit(1)

    manifest["status"] = "FAILED" if failed else "COMPLETE"
    if failed:
        failed_rec = next((r for r in records if r.status == "FAILED"), None)
        manifest["error_text"] = failed_rec.error_text if failed_rec else "Unbekannter Fehler"
    write_manifest_atomic(manifest, manifest_path)
    write_episodes_csv(records, csv_path)

    print("=" * 66)
    if failed:
        print(" Phase-A-Lauf FEHLGESCHLAGEN.")
        sys.exit(1)
    else:
        ok_count = sum(1 for r in records if r.status == "OK")
        print(f" Phase-A-Lauf ABGESCHLOSSEN — {ok_count}/{len(records)} Episoden OK.")
        print(f" Output: {run_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
