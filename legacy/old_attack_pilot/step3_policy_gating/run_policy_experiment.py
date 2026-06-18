#!/usr/bin/env python3
"""
run_policy_experiment.py
========================
Misst vier Attack-Policies im gleichen Regime (offload=4, victim conc=2):

  - random_gate     : Burst mit Wahrscheinlichkeit p (--attack-prob, default 0.5)
  - naive_threshold : Burst wenn probe_tpot > --tpot-threshold-ms (ein Merkmal)
  - state_aware     : Burst wenn probe_tpot > --sa-tpot-threshold-ms
                      UND probe_itl > --sa-itl-threshold-ms (zwei Merkmale)
                      → klar von naive_threshold abgesetzt; kein Fallback auf TPOT allein

Unterschied naive vs state_aware:
  naive     = reagiert auf einzelnes Latenz-Signal (könnte auch Jitter sein)
  state_aware = rekonstruiert Serving-Zustand aus zwei korrelierten Metriken
                (robuster, weniger False Positives bei kurzzeitigen Spitzen)

Baselines (no_attack, always_attack) kommen aus run_experiment.py und
müssen nicht neu gemessen werden.

Aufruf:
  OFFLOAD_GB=4 python run_policy_experiment.py
  OFFLOAD_GB=4 python run_policy_experiment.py --tpot-threshold-ms 150
  OFFLOAD_GB=4 python run_policy_experiment.py --runs 5 --window 300

Voraussetzung:
  Server läuft bereits: bash run_server.sh 4
"""
import argparse, asyncio, json, os, random, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import httpx, numpy as np

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tpot-threshold-ms", type=float, default=200.0,
                    help="TPOT-Schwelle für naive_threshold (default: 200ms)")
parser.add_argument("--sa-tpot-threshold-ms", type=float, default=200.0,
                    help="TPOT-Schwelle für state_aware (default: 200ms)")
parser.add_argument("--sa-itl-threshold-ms", type=float, default=300.0,
                    help="ITL-Schwelle für state_aware (default: 300ms). "
                         "state_aware greift an wenn TPOT>sa-tpot UND ITL>sa-itl — "
                         "beide Merkmale müssen überschritten sein.")
parser.add_argument("--attack-prob", type=float, default=0.5,
                    help="Angriffswahrscheinlichkeit für random_gate (default: 0.5). "
                         "Nach dem Lauf budget-matched gegen state_aware anpassen.")
parser.add_argument("--runs", type=int, default=5,
                    help="Runs pro Policy (default: 5)")
parser.add_argument("--window", type=int, default=300,
                    help="Victim-Messfenster in Sekunden (default: 300)")
parser.add_argument("--probe-requests", type=int, default=3,
                    help="Anzahl Probe-Requests pro Fenster (default: 3)")
args = parser.parse_args()

# ── Konfiguration ────────────────────────────────────────────────────────────
OFFLOAD_GB    = int(os.environ.get("OFFLOAD_GB", "4"))
BASE_URL      = os.environ.get("BASE_URL",  "http://127.0.0.1:8000")
ENDPOINT      = os.environ.get("ENDPOINT",  "/v1/chat/completions")
API_KEY       = os.environ.get("API_KEY",   "pilotkey")
MODEL         = os.environ.get("MODEL",     "meta-llama/Llama-3.1-8B-Instruct")

WINDOW_SECS        = args.window
RUNS_PER_POLICY    = args.runs
TPOT_THRESHOLD     = args.tpot_threshold_ms       # naive_threshold
SA_TPOT_THRESHOLD  = args.sa_tpot_threshold_ms    # state_aware TPOT
SA_ITL_THRESHOLD   = args.sa_itl_threshold_ms     # state_aware ITL (zweites Merkmal)
RANDOM_ATTACK_PROB = args.attack_prob
N_PROBE            = args.probe_requests

EXPERIMENT_ID = "offload_paper1_policy_eval"
SERVER_LABEL  = f"llama31_8b_offload{OFFLOAD_GB}"

# Victim — gleich wie in run_experiment.py
VICTIM_INPUT  = 256;  VICTIM_OUTPUT = 64;  VICTIM_CONC = 2;  VICTIM_TEMP = 0

# Attacker — gleich wie in run_experiment.py
ATK_INPUT = 2048; ATK_OUTPUT = 16; ATK_CONC = 8; ATK_TEMP = 0

# Probe — billig: kurze Requests, gleiche Familie wie Victim
PROBE_INPUT   = 256;  PROBE_OUTPUT = 16;  PROBE_CONC = 1

HERE = Path(__file__).parent

POLICIES = ["random_gate", "naive_threshold", "state_aware"]

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────
def _rand_prompt(n):
    chars, words = n * 4, []
    while sum(len(w)+1 for w in words) < chars:
        import string as _s
        words.append("".join(random.choices(_s.ascii_lowercase, k=random.randint(3,10))))
    return " ".join(words)

async def _single_request(client, input_len, output_len):
    """Schickt einen einzelnen Request und gibt (tpot_ms, itl_ms, ttft_ms) zurück."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": _rand_prompt(input_len)}],
        "max_tokens": output_len, "temperature": 0.0, "stream": True
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    ttft_ms = None
    itl = []
    tokens = 0
    prev_ts = None
    try:
        async with client.stream("POST", f"{BASE_URL}{ENDPOINT}",
                                  json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try: chunk = json.loads(d)
                except: continue
                now = time.perf_counter()
                tok = (chunk.get("choices",[{}])[0].get("delta",{}) or {}).get("content") or ""
                if tok:
                    tokens += 1
                    if ttft_ms is None:
                        ttft_ms = (now - t0) * 1000
                        prev_ts = now
                    else:
                        if prev_ts: itl.append((now - prev_ts) * 1000)
                        prev_ts = now
        e2el_ms = (time.perf_counter() - t0) * 1000
        decode_ms = e2el_ms - (ttft_ms or 0)
        tpot_ms = decode_ms / max(1, tokens - 1) if tokens > 1 and decode_ms > 0 else None
        itl_med = float(np.median(itl)) if itl else None
        return {"tpot_ms": tpot_ms, "itl_ms": itl_med, "ttft_ms": ttft_ms,
                "e2el_ms": e2el_ms, "tokens": tokens, "success": True}
    except Exception as e:
        return {"tpot_ms": None, "itl_ms": None, "ttft_ms": None,
                "e2el_ms": None, "tokens": 0, "success": False, "error": str(e)}

async def run_probe(n=N_PROBE):
    """
    Schickt n sequenzielle Probe-Requests und gibt Zusammenfassung zurück.
    Sequenziell (nicht parallel) damit jeder Request einen sauberen TPOT liefert.
    """
    results = []
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(n):
            r = await _single_request(client, PROBE_INPUT, PROBE_OUTPUT)
            if r["success"] and r["tpot_ms"] is not None:
                results.append(r)
    if not results:
        return {"probe_tpot_ms": None, "probe_itl_ms": None,
                "probe_ttft_ms": None, "n_ok": 0}
    tpots = [r["tpot_ms"] for r in results]
    itls  = [r["itl_ms"]  for r in results if r["itl_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    return {
        "probe_tpot_ms": float(np.median(tpots)),
        "probe_itl_ms":  float(np.median(itls))  if itls  else None,
        "probe_ttft_ms": float(np.median(ttfts)) if ttfts else None,
        "n_ok": len(results),
    }

# ── Gate-Entscheidungen ──────────────────────────────────────────────────────
def decide_random_gate(probe_result, attack_prob):
    return random.random() < attack_prob

def decide_naive_threshold(probe_result, threshold_ms):
    tpot = probe_result.get("probe_tpot_ms")
    if tpot is None: return False
    return tpot > threshold_ms

def decide_state_aware(probe_result, sa_tpot_threshold, sa_itl_threshold):
    """
    State-aware gate — klar von naive_threshold abgesetzt.

    naive_threshold : ein Merkmal  (TPOT > T)
    state_aware     : zwei Merkmale (TPOT > T1 UND ITL > T2)

    Begründung: TPOT und ITL sind im offload-degradierten Regime beide erhöht
    und stark korreliert. Ein Angreifer der den Serving-Zustand wirklich rekonstruiert
    (nicht nur auf Latenz reagiert) kann beide Signale gemeinsam auswerten.
    Das macht die Entscheidung robuster gegen kurzzeitige TPOT-Spitzen die kein
    echtes high-offload-Regime anzeigen (z.B. einzelne große Requests).

    Für Paper 2: hier kann ein trainierter Klassifikator (logistic regression,
    decision tree) auf den Probe-Merkmalen eingesetzt werden.
    """
    tpot = probe_result.get("probe_tpot_ms")
    itl  = probe_result.get("probe_itl_ms")
    if tpot is None: return False
    tpot_ok = tpot > sa_tpot_threshold
    # ITL muss explizit über eigener Schwelle liegen — kein Fallback auf TPOT
    if itl is None: return False
    itl_ok = itl > sa_itl_threshold
    decision = tpot_ok and itl_ok
    return decision

# ── Worker-Subprocess ────────────────────────────────────────────────────────
def _env(role, condition, policy, input_len, output_len, conc, temp, outfile, run_no):
    return {
        **os.environ,
        "BASE_URL": BASE_URL, "ENDPOINT": ENDPOINT,
        "API_KEY": API_KEY, "MODEL": MODEL,
        "ROLE": role, "CONDITION": condition,
        "INPUT_LEN": str(input_len), "OUTPUT_LEN": str(output_len),
        "CONCURRENCY": str(conc), "TEMPERATURE": str(temp),
        "OUTFILE": str(outfile),
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SERVER_LABEL": SERVER_LABEL,
        "OFFLOAD_GB": str(OFFLOAD_GB),
        "WINDOW_SECS": str(WINDOW_SECS),
        "RUN_NO": str(run_no),
        "POLICY": policy,
    }

def start_worker(role, condition, policy, input_len, output_len,
                 conc, temp, outfile, run_no):
    env = _env(role, condition, policy, input_len, output_len,
               conc, temp, outfile, run_no)
    return subprocess.Popen([sys.executable, str(HERE / "victim_worker.py")], env=env)

def stop_worker(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    proc.wait()

# ── Server-Check ─────────────────────────────────────────────────────────────
def check_server():
    import urllib.request
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

# ── Ein Policy-Run ────────────────────────────────────────────────────────────
def run_policy_window(policy, run_no, outdir, attack_prob=0.5):
    """
    Ein vollständiger Probe→Decide→(Burst+Victim | Victim-only) Zyklus.

    Gibt zurück:
      dict mit probe_result, decision, victim_file, attacker_file (oder None)
    """
    subdir = outdir / policy
    subdir.mkdir(parents=True, exist_ok=True)

    # 1. Probe
    print(f"    probe ({N_PROBE} requests) ...", end=" ", flush=True)
    probe_result = asyncio.run(run_probe())
    probe_tpot = probe_result.get("probe_tpot_ms")
    print(f"probe_tpot={probe_tpot:.1f}ms" if probe_tpot else "probe_tpot=None")

    # 2. Entscheidung
    decision_reason = {}
    if policy == "random_gate":
        decision = decide_random_gate(probe_result, attack_prob)
        decision_reason = {"rule": "random", "attack_prob": attack_prob}
    elif policy == "naive_threshold":
        decision = decide_naive_threshold(probe_result, TPOT_THRESHOLD)
        decision_reason = {
            "rule": "tpot_only",
            "tpot_threshold_ms": TPOT_THRESHOLD,
            "probe_tpot_ms": probe_result.get("probe_tpot_ms"),
        }
    elif policy == "state_aware":
        decision = decide_state_aware(probe_result, SA_TPOT_THRESHOLD, SA_ITL_THRESHOLD)
        decision_reason = {
            "rule": "tpot_and_itl",
            "sa_tpot_threshold_ms": SA_TPOT_THRESHOLD,
            "sa_itl_threshold_ms":  SA_ITL_THRESHOLD,
            "probe_tpot_ms": probe_result.get("probe_tpot_ms"),
            "probe_itl_ms":  probe_result.get("probe_itl_ms"),
        }
    else:
        raise ValueError(f"Unbekannte Policy: {policy}")

    print(f"    decision: {'ATTACK' if decision else 'no attack'}  reason={decision_reason}")

    # 3. Victim (immer) + optionaler Attacker-Burst
    # condition = "attacked" | "not_attacked" — getrennt von policy-Name
    condition = "attacked" if decision else "not_attacked"
    v_fname = f"offload{OFFLOAD_GB}_{policy}_run{run_no}_victim.json"
    a_fname = f"offload{OFFLOAD_GB}_{policy}_run{run_no}_attacker.json"

    # probe_secs: Zeit die die Probe-Phase dauerte (außerhalb des action window)
    probe_end_ts = time.perf_counter()

    v_proc = start_worker("victim", condition, policy,
                          VICTIM_INPUT, VICTIM_OUTPUT, VICTIM_CONC, VICTIM_TEMP,
                          subdir / v_fname, run_no)
    a_proc = None
    if decision:
        a_proc = start_worker("attacker", condition, policy,
                              ATK_INPUT, ATK_OUTPUT, ATK_CONC, ATK_TEMP,
                              subdir / a_fname, run_no)

    time.sleep(WINDOW_SECS)

    stop_worker(v_proc)
    if a_proc:
        stop_worker(a_proc)

    # Attacker-Budget aus JSON lesen (best-effort)
    atk_requests = 0
    atk_tokens   = 0
    if decision:
        try:
            atk_json = json.loads(Path(subdir / a_fname).read_text())
            atk_requests = atk_json.get("submitted", atk_json.get("completed", 0))
            atk_tokens   = atk_json.get("total_output_tokens", 0)
        except Exception:
            pass  # attacker JSON ggf. noch nicht vollständig — ignorieren

    # 4. Ergebnis-Metadaten speichern
    # Fix: condition und policy getrennt; budget-Felder explizit
    meta = {
        "policy":              policy,
        "condition":           condition,          # "attacked" | "not_attacked"
        "run_no":              run_no,
        "offload_gb":          OFFLOAD_GB,
        "window_secs":         WINDOW_SECS,
        "tpot_threshold_ms":   TPOT_THRESHOLD,
        "sa_tpot_threshold_ms": SA_TPOT_THRESHOLD,
        "sa_itl_threshold_ms":  SA_ITL_THRESHOLD,
        "probe_result":        probe_result,
        "decision":            decision,
        "decision_reason":     decision_reason,
        # Budget-Felder — für Kosten/Wirkung-Berechnung in summarize_policy_results.py
        "attack_budget_requests": atk_requests,
        "attack_budget_tokens":   atk_tokens,
        "victim_file":   str(subdir / v_fname),
        "attacker_file": str(subdir / a_fname) if decision else None,
    }
    meta_path = subdir / f"offload{OFFLOAD_GB}_{policy}_run{run_no}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return meta

# ── Hauptprogramm ─────────────────────────────────────────────────────────────
def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"policy_runs_offload{OFFLOAD_GB}_{ts}")

    print("=" * 62)
    print(" Policy Evaluation — Paper 1 / Standbein 2")
    print(f" offload_gb        : {OFFLOAD_GB}")
    print(f" tpot_threshold_ms     : {TPOT_THRESHOLD}  (naive_threshold)")
    print(f" sa_tpot_threshold_ms  : {SA_TPOT_THRESHOLD}  (state_aware)")
    print(f" sa_itl_threshold_ms   : {SA_ITL_THRESHOLD}  (state_aware)")
    print(f" random_attack_prob    : {RANDOM_ATTACK_PROB}  (random_gate)")
    print(f" runs/policy           : {RUNS_PER_POLICY}")
    print(f" window_secs           : {WINDOW_SECS}")
    print(f" probe_requests        : {N_PROBE}")
    print(f" policies              : {POLICIES}")
    print(f" outdir            : {outdir}")
    print("=" * 62)

    print("--- Prüfe Server ---")
    if not check_server():
        print(f"[FEHLER] Server nicht erreichbar.")
        sys.exit(1)
    print()

    all_results = []

    for policy in POLICIES:
        print(f"--- Policy: {policy} ---")
        for run_no in range(1, RUNS_PER_POLICY + 1):
            print(f"  run {run_no}/{RUNS_PER_POLICY}:")
            result = run_policy_window(policy, run_no, outdir,
                                       attack_prob=RANDOM_ATTACK_PROB)
            all_results.append(result)
            print(f"  run {run_no} fertig: decision={result['decision']}")
            print()

    # Zusammenfassung
    summary_path = outdir / "policy_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))

    print("=" * 62)
    print(" Alle Policy-Runs abgeschlossen.")
    print(f" Ergebnisse: {outdir}/")
    print(f" Zusammenfassung: {summary_path}")
    print()
    print(" Nächster Schritt:")
    print(f"   python summarize_policy_results.py {outdir}")
    print("=" * 62)

if __name__ == "__main__":
    main()
