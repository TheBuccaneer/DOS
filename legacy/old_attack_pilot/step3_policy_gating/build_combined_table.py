#!/usr/bin/env python3
"""
build_combined_table.py
=======================
Baut die kombinierte Kosten/Wirkung-Tabelle über beide Offload-Regimes.

Benötigt zwei policy_comparison.csv Dateien:
  - eine für offload=0
  - eine für offload=4

Aufruf:
  python build_combined_table.py \
    --offload0 policy_runs_offload0_<timestamp>/policy_comparison.csv \
    --offload4 policy_runs_offload4_<timestamp>/policy_comparison.csv

Output:
  - Tabelle im Terminal
  - combined_table.csv (für Paper/LaTeX)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--offload0", required=True,
                    help="policy_comparison.csv für offload=0")
parser.add_argument("--offload4", required=True,
                    help="policy_comparison.csv für offload=4")
parser.add_argument("--out", default="combined_table.csv",
                    help="Ausgabe-CSV (default: combined_table.csv)")
args = parser.parse_args()

# ── Laden ────────────────────────────────────────────────────────────────────
df0 = pd.read_csv(args.offload0)
df4 = pd.read_csv(args.offload4)
df0["offload_gb"] = 0
df4["offload_gb"] = 4
df = pd.concat([df0, df4], ignore_index=True)

# ── No-attack Baseline (nicht-angegriffene Fenster) ───────────────────────────
no_atk = df[df["decision"] == False]
baseline = {}
for offload in [0, 4]:
    rows = no_atk[no_atk["offload_gb"] == offload]["median_tpot_ms"].dropna()
    baseline[offload] = float(rows.mean()) if len(rows) > 0 else None
    print(f"  No-attack baseline offload={offload}: "
          f"{baseline[offload]:.1f}ms" if baseline[offload] else f"  No-attack baseline offload={offload}: n/a")

print()

# ── Pro Policy + Offload-Level aggregieren ────────────────────────────────────
policies = df["policy"].unique()
rows_out = []

for policy in policies:
    for offload in [0, 4]:
        sub = df[(df["policy"] == policy) & (df["offload_gb"] == offload)]
        if sub.empty:
            continue
        attacked = sub[sub["decision"] == True]
        n_total    = len(sub)
        n_attacked = len(attacked)

        # Victim-Schaden (nur angegriffene Fenster)
        med_tpot = attacked["median_tpot_ms"].median() if n_attacked > 0 else None
        base     = baseline.get(offload)
        delta    = (med_tpot - base) if (med_tpot and base) else None

        # Angreiferkosten
        atk_req = attacked["atk_requests"].sum() if "atk_requests" in attacked.columns else 0
        atk_tok = attacked["atk_output_tokens"].sum() if "atk_output_tokens" in attacked.columns else 0

        rows_out.append({
            "policy":        policy,
            "offload_gb":    offload,
            "n_total":       n_total,
            "n_attacked":    n_attacked,
            "attack_rate":   f"{n_attacked/n_total*100:.0f}%",
            "med_tpot_ms":   round(med_tpot, 1) if med_tpot else None,
            "baseline_ms":   round(base, 1) if base else None,
            "delta_tpot_ms": round(delta, 1) if delta else None,
            "atk_requests":  int(atk_req),
            "atk_tokens":    int(atk_tok),
        })

detail_df = pd.DataFrame(rows_out)

# ── Kombinierte Tabelle über beide Regimes ────────────────────────────────────
combined_rows = []

for policy in policies:
    sub = detail_df[detail_df["policy"] == policy]

    total_req   = sub["atk_requests"].sum()
    total_tok   = sub["atk_tokens"].sum()

    # Schaden: nur offload=4 zählt als "echter" Schaden für die Story
    # offload=0 Angriffe sind verschwendetes Budget
    dmg_4 = sub[sub["offload_gb"] == 4]["delta_tpot_ms"].values
    delta_4 = float(dmg_4[0]) if len(dmg_4) > 0 and dmg_4[0] is not None else None

    attacks_0 = sub[sub["offload_gb"] == 0]["n_attacked"].values
    attacks_4 = sub[sub["offload_gb"] == 4]["n_attacked"].values
    n_att_0 = int(attacks_0[0]) if len(attacks_0) > 0 else 0
    n_att_4 = int(attacks_4[0]) if len(attacks_4) > 0 else 0

    slowdown_req   = round(delta_4 / total_req,   4) if (delta_4 and total_req > 0) else None
    slowdown_token = round(delta_4 / total_tok,   6) if (delta_4 and total_tok > 0) else None

    combined_rows.append({
        "policy":             policy,
        "attacks_offload0":   n_att_0,
        "attacks_offload4":   n_att_4,
        "wasted_budget_off0": int(sub[sub["offload_gb"] == 0]["atk_requests"].sum()),
        "total_atk_requests": int(total_req),
        "total_atk_tokens":   int(total_tok),
        "delta_tpot_off4_ms": delta_4,
        "slowdown_per_req":   slowdown_req,
        "slowdown_per_token": slowdown_token,
    })

combined_df = pd.DataFrame(combined_rows)

# ── Ausgabe ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("COMBINED-REGIME EFFICIENCY TABLE")
print("(Kosten/Wirkung über offload=0 und offload=4 zusammen)")
print("=" * 72)
print()
print(f"{'Policy':20s} {'Att.off0':>8s} {'Att.off4':>8s} "
      f"{'Wasted.req':>10s} {'Total.req':>10s} "
      f"{'ΔTPOT.off4':>12s} {'Slow/req':>12s} {'Slow/tok':>12s}")
print("-" * 100)

for _, row in combined_df.iterrows():
    print(f"{row['policy']:20s} "
          f"{row['attacks_offload0']:>8d} "
          f"{row['attacks_offload4']:>8d} "
          f"{row['wasted_budget_off0']:>10d} "
          f"{row['total_atk_requests']:>10d} "
          f"{row['delta_tpot_off4_ms']:>11.1f}ms "
          f"{row['slowdown_per_req']:>11.4f} "
          f"{row['slowdown_per_token']:>12.6f}" if row['slowdown_per_req'] else
          f"{row['policy']:20s} "
          f"{row['attacks_offload0']:>8d} "
          f"{row['attacks_offload4']:>8d} "
          f"{row['wasted_budget_off0']:>10d} "
          f"{row['total_atk_requests']:>10d} "
          f"{'n/a':>12s} {'n/a':>12s} {'n/a':>12s}")

print()
print("Legende:")
print("  Att.off0      = Angriffe im offload=0 Regime (verschwendetes Budget)")
print("  Att.off4      = Angriffe im offload=4 Regime (echter Schaden)")
print("  Wasted.req    = Attacker-Requests bei offload=0 (kein Nutzen)")
print("  ΔTPOT.off4    = Victim TPOT-Degradation bei offload=4 vs Baseline")
print("  Slow/req      = ΔTPOT / total Attacker-Requests (über beide Regimes)")
print("  Slow/tok      = ΔTPOT / total Attacker-Output-Tokens")

# ── Verbesserungsfaktoren ─────────────────────────────────────────────────────
print()
print("=" * 72)
print("VERBESSERUNGSFAKTOREN (state_aware vs Baselines)")
print("=" * 72)
sa  = combined_df[combined_df["policy"] == "state_aware"]
rg  = combined_df[combined_df["policy"] == "random_gate"]
nt  = combined_df[combined_df["policy"] == "naive_threshold"]

if not sa.empty and not rg.empty:
    sa_eff = sa["slowdown_per_req"].values[0]
    rg_eff = rg["slowdown_per_req"].values[0]
    if sa_eff and rg_eff:
        print(f"  state_aware vs random_gate : {sa_eff/rg_eff:.1f}x effizienter (slowdown/req)")

if not sa.empty and not nt.empty:
    sa_eff = sa["slowdown_per_req"].values[0]
    nt_eff = nt["slowdown_per_req"].values[0]
    if sa_eff and nt_eff:
        print(f"  state_aware vs naive       : {sa_eff/nt_eff:.2f}x effizienter (slowdown/req)")

# ── CSV speichern ─────────────────────────────────────────────────────────────
combined_df.to_csv(args.out, index=False)
print()
print(f"Tabelle gespeichert: {args.out}")
