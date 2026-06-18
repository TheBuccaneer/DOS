#!/usr/bin/env python3
"""
summarize_policy_results.py
===========================
Wertet die Ergebnisse von run_policy_experiment.py aus.

Aufruf:
  python summarize_policy_results.py policy_runs_offload4_<timestamp>/
  python summarize_policy_results.py policy_runs_offload4_<timestamp>/ --baseline-dir bench_runs_.../

Gibt aus:
  - Victim-Schaden pro Policy
  - Angreiferkosten pro Policy
  - Effizienzmetriken: slowdown per attacker request / token
  - Vergleichstabelle aller Policies
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("policy_dir", help="Ausgabeverzeichnis von run_policy_experiment.py")
parser.add_argument("--baseline-dir", default=None,
                    help="Optional: bench_runs_-Verzeichnis mit no_attack/always_attack Baselines")
parser.add_argument("--no-attack-tpot-ms", type=float, default=None,
                    help="Optionaler manueller no-attack median_tpot_ms Wert")
args = parser.parse_args()

policy_dir = Path(args.policy_dir)
if not policy_dir.exists():
    print(f"[FEHLER] Verzeichnis nicht gefunden: {policy_dir}")
    sys.exit(1)

# ── JSON-Hilfsfunktionen ─────────────────────────────────────────────────────
def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as e:
        print(f"  [WARN] Konnte {path} nicht laden: {e}")
        return None

def safe_median(data):
    d = [x for x in data if x is not None]
    return float(np.median(d)) if d else None

# ── Victim-JSON auslesen ─────────────────────────────────────────────────────
def extract_victim_metrics(victim_file):
    if victim_file is None: return {}
    d = load_json(victim_file)
    if d is None: return {}
    return {
        "completed":       d.get("completed", 0),
        "failed":          d.get("failed", 0),
        "median_tpot_ms":  (d.get("tpot_ms") or {}).get("median"),
        "median_itl_ms":   (d.get("itl_ms")  or {}).get("median"),
        "median_ttft_ms":  (d.get("ttft_ms") or {}).get("median"),
        "median_e2el_ms":  (d.get("e2el_ms") or {}).get("median"),
        "request_throughput": d.get("request_throughput"),
    }

def extract_attacker_metrics(attacker_file):
    if attacker_file is None:
        return {"atk_requests": 0, "atk_output_tokens": 0, "atk_completed": 0}
    d = load_json(attacker_file)
    if d is None:
        return {"atk_requests": 0, "atk_output_tokens": 0, "atk_completed": 0}
    return {
        "atk_requests":      d.get("submitted", d.get("completed", 0)),
        "atk_output_tokens": d.get("total_output_tokens", 0),
        "atk_completed":     d.get("completed", 0),
    }

# ── Lade alle Meta-JSONs ─────────────────────────────────────────────────────
meta_files = sorted(policy_dir.rglob("*_meta.json"))
if not meta_files:
    print(f"[FEHLER] Keine *_meta.json Dateien in {policy_dir}")
    sys.exit(1)

rows = []
for mf in meta_files:
    meta = load_json(mf)
    if meta is None: continue
    vm = extract_victim_metrics(meta.get("victim_file"))
    am = extract_attacker_metrics(meta.get("attacker_file") if meta.get("decision") else None)
    row = {
        "policy":          meta.get("policy"),
        "run_no":          meta.get("run_no"),
        "offload_gb":      meta.get("offload_gb"),
        "tpot_threshold":  meta.get("tpot_threshold_ms"),
        "decision":        meta.get("decision"),
        "probe_tpot_ms":   (meta.get("probe_result") or {}).get("probe_tpot_ms"),
        "probe_itl_ms":    (meta.get("probe_result") or {}).get("probe_itl_ms"),
        **vm,
        **am,
    }
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    print("[FEHLER] Keine auswertbaren Daten gefunden.")
    sys.exit(1)

print(f"\nGeladene Runs: {len(df)}")
print(f"Policies: {df['policy'].unique().tolist()}")
print()

# ── No-attack Baseline ───────────────────────────────────────────────────────
# Entweder aus --no-attack-tpot-ms oder aus --baseline-dir oder aus vorhandenen Daten
no_attack_tpot = args.no_attack_tpot_ms

if no_attack_tpot is None and args.baseline_dir:
    bd = Path(args.baseline_dir)
    cond_a_files = list(bd.glob("**/cond_a*/*.json")) + list(bd.glob("**/cond_a_victim_only/*.json"))
    tpots = []
    for f in cond_a_files:
        d = load_json(f)
        if d and d.get("role") == "victim":
            v = (d.get("tpot_ms") or {}).get("median")
            if v: tpots.append(v)
    if tpots:
        no_attack_tpot = float(np.mean(tpots))
        print(f"No-attack baseline aus {args.baseline_dir}: median_tpot={no_attack_tpot:.1f}ms")

if no_attack_tpot is None:
    # Schätze aus no-decision runs (falls vorhanden)
    no_atk_rows = df[df["decision"] == False]
    if not no_atk_rows.empty and no_atk_rows["median_tpot_ms"].notna().any():
        no_attack_tpot = no_atk_rows["median_tpot_ms"].median()
        print(f"No-attack baseline aus nicht-angegriffenen Fenstern: {no_attack_tpot:.1f}ms")
    else:
        print("[WARN] Kein no-attack Baseline verfügbar — delta-Metriken nicht berechnet")

# ── Pro-Policy Aggregation ───────────────────────────────────────────────────
print("=" * 70)
print("POLICY EVALUATION — Victim-Schaden & Angreiferkosten")
print("=" * 70)

summary_rows = []

for policy in df["policy"].unique():
    pdf = df[df["policy"] == policy]
    n_total    = len(pdf)
    n_attacked = pdf["decision"].sum()
    n_no_atk   = n_total - n_attacked

    # Victim-Metriken — nur aus angegriffenen Fenstern (für Schadensberechnung)
    atk_df = pdf[pdf["decision"] == True]
    no_atk_df = pdf[pdf["decision"] == False]

    med_tpot_attacked  = safe_median(atk_df["median_tpot_ms"].tolist()) if not atk_df.empty else None
    med_tpot_no_attack = safe_median(no_atk_df["median_tpot_ms"].tolist()) if not no_atk_df.empty else None
    med_tpot_all       = safe_median(pdf["median_tpot_ms"].tolist())

    med_itl_attacked   = safe_median(atk_df["median_itl_ms"].tolist()) if not atk_df.empty else None
    med_e2el_attacked  = safe_median(atk_df["median_e2el_ms"].tolist()) if not atk_df.empty else None

    # Angreiferkosten
    total_atk_requests = pdf["atk_requests"].sum()
    total_atk_tokens   = pdf["atk_output_tokens"].sum()

    # Delta gegenüber no-attack baseline
    delta_tpot = None
    if med_tpot_attacked and no_attack_tpot:
        delta_tpot = med_tpot_attacked - no_attack_tpot

    # Effizienzmetriken
    slowdown_per_request = None
    slowdown_per_token   = None
    if delta_tpot and total_atk_requests > 0:
        slowdown_per_request = delta_tpot / total_atk_requests
    if delta_tpot and total_atk_tokens > 0:
        slowdown_per_token = delta_tpot / total_atk_tokens

    summary_rows.append({
        "policy":               policy,
        "runs_total":           n_total,
        "runs_attacked":        n_attacked,
        "attack_rate":          f"{n_attacked/n_total*100:.0f}%",
        "med_tpot_attacked_ms": f"{med_tpot_attacked:.1f}" if med_tpot_attacked else "-",
        "med_itl_attacked_ms":  f"{med_itl_attacked:.1f}"  if med_itl_attacked  else "-",
        "med_e2el_attacked_ms": f"{med_e2el_attacked:.1f}" if med_e2el_attacked else "-",
        "delta_tpot_ms":        f"{delta_tpot:+.1f}"       if delta_tpot        else "-",
        "total_atk_requests":   int(total_atk_requests),
        "total_atk_tokens":     int(total_atk_tokens),
        "slowdown_per_req":     f"{slowdown_per_request:.4f}" if slowdown_per_request else "-",
        "slowdown_per_token":   f"{slowdown_per_token:.6f}"   if slowdown_per_token   else "-",
    })

    print(f"\n  Policy: {policy}")
    print(f"    Runs total/attacked : {n_total} / {n_attacked} ({n_attacked/n_total*100:.0f}%)")
    print(f"    Victim median_tpot  : {med_tpot_attacked:.1f}ms (angegriffen)"
          if med_tpot_attacked else f"    Victim median_tpot  : - (keine angegriffenen Fenster)")
    print(f"    Victim median_itl   : {med_itl_attacked:.1f}ms"
          if med_itl_attacked else f"    Victim median_itl   : -")
    print(f"    Delta vs no-attack  : {delta_tpot:+.1f}ms"
          if delta_tpot else f"    Delta vs no-attack  : -")
    print(f"    Attacker requests   : {int(total_atk_requests)}")
    print(f"    Attacker tokens     : {int(total_atk_tokens)}")
    print(f"    Slowdown/req        : {slowdown_per_request:.4f}ms/req"
          if slowdown_per_request else f"    Slowdown/req        : -")
    print(f"    Slowdown/token      : {slowdown_per_token:.6f}ms/tok"
          if slowdown_per_token else f"    Slowdown/token      : -")

# ── Vergleichstabelle ────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("VERGLEICHSTABELLE")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ── Probe-Signal Analyse ─────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("PROBE-SIGNAL ANALYSE")
print("=" * 70)
probe_df = df[df["probe_tpot_ms"].notna()].copy()
if not probe_df.empty:
    print(f"\n  TPOT-Schwelle: {TPOT_THRESHOLD if hasattr(args,'tpot_threshold_ms') else '?'}ms")
    print(f"  Probe TPOT — Median: {probe_df['probe_tpot_ms'].median():.1f}ms")
    print(f"  Probe TPOT — Min:    {probe_df['probe_tpot_ms'].min():.1f}ms")
    print(f"  Probe TPOT — Max:    {probe_df['probe_tpot_ms'].max():.1f}ms")
    threshold_display = getattr(args, "tpot_threshold_ms", None) or TPOT_THRESHOLD if "TPOT_THRESHOLD" in dir() else "?"
    print(f"  Fenster über Schwelle: {(probe_df['probe_tpot_ms'] > 200).sum()} / {len(probe_df)}  (Schwelle=200ms)")

# ── CSV speichern ────────────────────────────────────────────────────────────
out_csv = policy_dir / "policy_comparison.csv"
df.to_csv(out_csv, index=False)
print(f"\n\nRohdaten gespeichert: {out_csv}")

summary_csv = policy_dir / "policy_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Zusammenfassung gespeichert: {summary_csv}")
