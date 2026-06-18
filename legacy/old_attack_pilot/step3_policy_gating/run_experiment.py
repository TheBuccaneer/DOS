#!/usr/bin/env python3
"""
run_experiment.py
=================
Orchestriert das 300s-Fenster-Experiment.

Bedingung A: Nur Victim läuft 300s, schreibt JSON.
Bedingung B: Victim + Attacker laufen parallel 300s,
             jeder schreibt sein eigenes JSON (wie in v2).

Aufruf:
  OFFLOAD_GB=12 python run_experiment.py
  OFFLOAD_GB=0  WINDOW_SECS=60 RUNS_PER_CONDITION=1 python run_experiment.py

Pakete: httpx numpy  (pip install httpx numpy)
"""
import asyncio, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

OFFLOAD_GB    = int(os.environ.get("OFFLOAD_GB", "0"))
BASE_URL      = os.environ.get("BASE_URL",   "http://127.0.0.1:8000")
ENDPOINT      = os.environ.get("ENDPOINT",   "/v1/chat/completions")
API_KEY       = os.environ.get("API_KEY",    "pilotkey")
MODEL         = os.environ.get("MODEL",      "meta-llama/Llama-3.1-8B-Instruct")
WINDOW_SECS   = int(os.environ.get("WINDOW_SECS",        "300"))
RUNS_PER_COND = int(os.environ.get("RUNS_PER_CONDITION", "5"))

EXPERIMENT_ID = "offload_paper1_utility_calib"
SERVER_LABEL  = f"llama31_8b_offload{OFFLOAD_GB}"

VICTIM_INPUT  = 256;  VICTIM_OUTPUT  = 64;  VICTIM_CONC  = 2;  VICTIM_TEMP  = 0
ATTACKER_INPUT= 2048; ATTACKER_OUTPUT= 16;  ATTACKER_CONC= 8;  ATTACKER_TEMP= 0

HERE = Path(__file__).parent

def _env(role, condition, input_len, output_len, conc, temp, outfile, run_no):
    return {
        **os.environ,
        "BASE_URL":BASE_URL, "ENDPOINT":ENDPOINT, "API_KEY":API_KEY, "MODEL":MODEL,
        "ROLE":role, "CONDITION":condition,
        "INPUT_LEN":str(input_len), "OUTPUT_LEN":str(output_len),
        "CONCURRENCY":str(conc), "TEMPERATURE":str(temp),
        "OUTFILE":str(outfile),
        "EXPERIMENT_ID":EXPERIMENT_ID, "SERVER_LABEL":SERVER_LABEL,
        "OFFLOAD_GB":str(OFFLOAD_GB), "WINDOW_SECS":str(WINDOW_SECS),
        "RUN_NO":str(run_no),
    }

def start_worker(role, condition, input_len, output_len, conc, temp, outfile, run_no):
    env = _env(role, condition, input_len, output_len, conc, temp, outfile, run_no)
    return subprocess.Popen([sys.executable, str(HERE/"victim_worker.py")], env=env)

def stop_worker(proc):
    """Schickt SIGTERM → Worker beendet sich sauber nach aktuellem Request."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    proc.wait()

def check_server():
    import urllib.request, urllib.error
    for path in ("/health", "/version", "/"):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}{path}",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status < 400:
                    print(f"  [OK] Server antwortet auf {path} (HTTP {r.status})")
                    return True
        except Exception:
            pass
    return False

# ── Bedingung A ────────────────────────────────────────────────────────────
def run_condition_a(outdir: Path, run_no: int):
    subdir = outdir / "cond_a_victim_only"
    fname  = f"offload{OFFLOAD_GB}_conc{VICTIM_CONC}_run{run_no}.json"
    print(f"  [A] run {run_no}: starte Victim ({WINDOW_SECS}s) ...")
    proc = start_worker("victim","cond_a",
                        VICTIM_INPUT,VICTIM_OUTPUT,VICTIM_CONC,VICTIM_TEMP,
                        subdir/fname, run_no)
    time.sleep(WINDOW_SECS)
    print(f"  [A] run {run_no}: Fenster vorbei — stoppe Victim sauber ...")
    stop_worker(proc)
    print(f"  [A] run {run_no}: fertig → {subdir/fname}")

# ── Bedingung B ────────────────────────────────────────────────────────────
def run_condition_b(outdir: Path, run_no: int):
    subdir  = outdir / "cond_b_victim_plus_burst"
    v_fname = f"offload{OFFLOAD_GB}_conc{VICTIM_CONC}_run{run_no}.json"
    a_fname = f"offload{OFFLOAD_GB}_attacker_conc{ATTACKER_CONC}_run{run_no}.json"
    print(f"  [B] run {run_no}: starte Victim + Attacker gleichzeitig ({WINDOW_SECS}s) ...")
    v_proc = start_worker("victim",  "cond_b",
                          VICTIM_INPUT,  VICTIM_OUTPUT,  VICTIM_CONC,  VICTIM_TEMP,
                          subdir/v_fname, run_no)
    a_proc = start_worker("attacker","cond_b",
                          ATTACKER_INPUT,ATTACKER_OUTPUT,ATTACKER_CONC,ATTACKER_TEMP,
                          subdir/a_fname, run_no)
    time.sleep(WINDOW_SECS)
    print(f"  [B] run {run_no}: Fenster vorbei — stoppe beide sauber ...")
    stop_worker(v_proc)
    stop_worker(a_proc)
    print(f"  [B] run {run_no}: fertig")
    print(f"      victim   → {subdir/v_fname}")
    print(f"      attacker → {subdir/a_fname}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"bench_runs_{EXPERIMENT_ID}_offload{OFFLOAD_GB}_{ts}")

    print("=" * 62)
    print(" Experiment — Python-Runner (2-Prozess, 2 JSON)")
    print(f" offload_gb    : {OFFLOAD_GB}")
    print(f" victim  conc  : {VICTIM_CONC}  input={VICTIM_INPUT}  output={VICTIM_OUTPUT}")
    print(f" attacker conc : {ATTACKER_CONC} input={ATTACKER_INPUT} output={ATTACKER_OUTPUT}")
    print(f" window_secs   : {WINDOW_SECS}")
    print(f" runs/condition: {RUNS_PER_COND}")
    print(f" outdir        : {outdir}")
    print("=" * 62)

    print("--- Prüfe Server ---")
    if not check_server():
        print(f"[FEHLER] Server unter {BASE_URL} nicht erreichbar.")
        sys.exit(1)
    print()

    print("--- BEDINGUNG A: victim-only baseline ---")
    for r in range(1, RUNS_PER_COND+1):
        run_condition_a(outdir, r); print()

    print("--- BEDINGUNG B: victim + attacker burst ---")
    for r in range(1, RUNS_PER_COND+1):
        run_condition_b(outdir, r); print()

    print("=" * 62)
    print(" Alle Runs abgeschlossen.")
    print(f"  A → {outdir}/cond_a_victim_only/")
    print(f"  B → {outdir}/cond_b_victim_plus_burst/")
    print(f"python extract_vllm_rerun.py {outdir} --outdir extracted_calib")
    print("=" * 62)

if __name__ == "__main__":
    main()
