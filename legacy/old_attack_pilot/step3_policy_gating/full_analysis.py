#!/usr/bin/env python3
"""
full_analysis.py  —  Paper 1 / Standbein 2
==========================================
Baut alle Paper-Claims aus den Policy-Experiment-Daten.

Benötigt:
  --offload0  : policy_comparison CSV für offload=0
  --offload4  : policy_comparison CSV für offload=4
  --aa4       : runs_summary_rerun.csv mit cond_b/cond_a für offload=4

Aufruf:
  python full_analysis.py \
    --offload0 policy_runs_offload0_.../policy_comparison0.csv \
    --offload4 policy_runs_offload4_.../policy_comparison.csv \
    --aa4      ../04/runs_summary_rerun.csv \
    --out-dir  analysis_output/

Output:
  combined_table.csv
  cost_impact_plot.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--offload0",    required=True)
parser.add_argument("--offload4",    required=True)
parser.add_argument("--aa4",         default=None)
parser.add_argument("--bootstrap-n", type=int, default=2000)
parser.add_argument("--out-dir",     default=".")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = ["random_gate", "naive_threshold", "state_aware"]

# ── Laden ─────────────────────────────────────────────────────────────────────
df0 = pd.read_csv(args.offload0); df0["offload_gb"] = 0
df4 = pd.read_csv(args.offload4); df4["offload_gb"] = 4
df  = pd.concat([df0, df4], ignore_index=True)

# ── No-attack baseline ────────────────────────────────────────────────────────
no_atk   = df[df["decision"] == False]
baseline = {}
for off in [0, 4]:
    vals = no_atk[no_atk["offload_gb"] == off]["median_tpot_ms"].dropna()
    baseline[off] = float(vals.mean()) if len(vals) > 0 else None

print("=" * 60)
print("BASELINES")
print("=" * 60)
for off in [0, 4]:
    v = baseline[off]
    print(f"  no_attack  offload={off}: {v:.1f}ms" if v else f"  no_attack  offload={off}: n/a")

# ── Always-attack baseline ────────────────────────────────────────────────────
aa = None
if args.aa4:
    try:
        r  = pd.read_csv(args.aa4)
        va = r[(r["condition"]=="cond_a") & (r["role"]=="victim") & (r["offload_gb"]==4)]
        vb = r[(r["condition"]=="cond_b") & (r["role"]=="victim") & (r["offload_gb"]==4)]
        ab = r[(r["condition"]=="cond_b") & (r["role"]=="attacker") & (r["offload_gb"]==4)]
        if not va.empty and not vb.empty:
            aa = {
                "delta_tpot":   float(vb["median_tpot_ms"].mean() - va["median_tpot_ms"].mean()),
                "atk_requests": int(ab["submitted"].sum()) if "submitted" in ab.columns else int(ab["completed"].sum()),
                "atk_tokens":   int(ab["total_output_tokens"].sum()) if "total_output_tokens" in ab.columns else 0,
            }
            aa["slowdown_per_req"] = aa["delta_tpot"] / aa["atk_requests"] if aa["atk_requests"] > 0 else None
            print(f"  always_attack offload=4: delta={aa['delta_tpot']:.1f}ms  "
                  f"req={aa['atk_requests']}  slow/req={aa['slowdown_per_req']:.2f}ms/req")
    except Exception as e:
        print(f"  [WARN] always-attack load failed: {e}")
print()

# ── Pro-Policy Aggregation ────────────────────────────────────────────────────
detail = []
for policy in POLICIES:
    for off in [0, 4]:
        sub = df[(df["policy"]==policy) & (df["offload_gb"]==off)]
        if sub.empty: continue
        atk     = sub[sub["decision"]==True]
        n_total = len(sub); n_atk = len(atk)
        med     = float(atk["median_tpot_ms"].median()) if n_atk > 0 else None
        base    = baseline.get(off)
        delta   = (med - base) if (med and base) else None
        req     = int(atk["atk_requests"].sum())       if "atk_requests"      in atk.columns else 0
        tok     = int(atk["atk_output_tokens"].sum())  if "atk_output_tokens" in atk.columns else 0
        detail.append(dict(policy=policy, off=off, n_total=n_total, n_atk=n_atk,
                           rate=n_atk/n_total, med=med, delta=delta, req=req, tok=tok))
det = pd.DataFrame(detail)

# ── Kombinierte Tabelle ───────────────────────────────────────────────────────
rows = []
for policy in POLICIES:
    d0 = det[(det["policy"]==policy) & (det["off"]==0)]
    d4 = det[(det["policy"]==policy) & (det["off"]==4)]
    d0 = d0.iloc[0] if not d0.empty else None
    d4 = d4.iloc[0] if not d4.empty else None

    att0 = int(d0["n_atk"]) if d0 is not None else 0
    att4 = int(d4["n_atk"]) if d4 is not None else 0
    w_req= int(d0["req"])   if d0 is not None else 0
    w_tok= int(d0["tok"])   if d0 is not None else 0
    req4 = int(d4["req"])   if d4 is not None else 0
    tok4 = int(d4["tok"])   if d4 is not None else 0
    total_req = w_req + req4
    total_tok = w_tok + tok4
    delta4    = float(d4["delta"]) if (d4 is not None and d4["delta"] is not None) else None
    sel_gap   = (att4/5) - (att0/5)
    waste_frac= w_req / total_req if total_req > 0 else 0.0
    slow_req  = delta4 / total_req if (delta4 and total_req > 0) else None
    slow_tok  = delta4 / total_tok if (delta4 and total_tok > 0) else None

    # always-attack normalization — pct_budget_eff = nur offload=4 Budget
    pct_dmg     = round(delta4 / aa["delta_tpot"] * 100, 1)  if (aa and delta4)   else None
    pct_bud_eff = round(req4   / aa["atk_requests"] * 100, 1) if (aa and req4 > 0) else None

    rows.append({"policy": policy,
                 "att_off0": att0, "att_off4": att4,
                 "rate_off0": f"{att0/5*100:.0f}%", "rate_off4": f"{att4/5*100:.0f}%",
                 "selectivity_gap": round(sel_gap, 2),
                 "wasted_req": w_req, "wasted_frac": round(waste_frac, 3),
                 "total_req": total_req, "total_tok": total_tok,
                 "delta_tpot_ms": round(delta4, 1) if delta4 else None,
                 "slowdown_per_req": round(slow_req, 4) if slow_req else None,
                 "slowdown_per_tok": round(slow_tok, 6) if slow_tok else None,
                 "pct_aa_damage": pct_dmg,
                 "pct_aa_budget_eff": pct_bud_eff})

combined = pd.DataFrame(rows)

# ── Bootstrap CIs (slowdown_per_req) ─────────────────────────────────────────
def bootstrap_ci(policy):
    s0   = df0[df0["policy"]==policy]
    s4   = df4[df4["policy"]==policy]
    base4= baseline.get(4)
    if base4 is None: return None, None
    atk4 = s4[s4["decision"]==True]["median_tpot_ms"].dropna().values
    r4   = s4[s4["decision"]==True]["atk_requests"].fillna(0).values if "atk_requests" in s4.columns else np.zeros(1)
    r0   = s0[s0["decision"]==True]["atk_requests"].fillna(0).values if "atk_requests" in s0.columns else np.zeros(1)
    if len(atk4) == 0: return None, None
    samples = []
    for _ in range(args.bootstrap_n):
        d = atk4[np.random.choice(len(atk4), len(atk4), replace=True)].mean() - base4
        rr4 = r4[np.random.choice(len(r4), len(r4), replace=True)].sum()
        rr0 = r0[np.random.choice(len(r0), len(r0), replace=True)].sum()
        t = rr4 + rr0
        if t > 0: samples.append(d / t)
    if not samples: return None, None
    return round(np.percentile(samples, 2.5), 4), round(np.percentile(samples, 97.5), 4)

combined["ci_lo"], combined["ci_hi"] = zip(*[bootstrap_ci(p) for p in combined["policy"]])

# ── Terminal Output ───────────────────────────────────────────────────────────
print("=" * 72)
print("1. COMBINED-REGIME EFFICIENCY  (slowdown/req mit 95% Bootstrap CI)")
print("=" * 72)
for _, r in combined.iterrows():
    ci = f"  [95% CI: {r['ci_lo']:.2f} – {r['ci_hi']:.2f}]" if r["ci_lo"] else ""
    print(f"  {r['policy']:20s}  {r['slowdown_per_req']:.2f}ms/req{ci}")

print()
print("=" * 72)
print("2. FULL TABLE")
print("=" * 72)
print(combined[["policy","rate_off0","rate_off4","selectivity_gap",
                "wasted_frac","total_req","delta_tpot_ms",
                "slowdown_per_req","slowdown_per_tok"]].to_string(index=False))

print()
print("=" * 72)
print("3. ALWAYS-ATTACK NORMALIZATION")
print("=" * 72)
if aa:
    print(f"  always_attack: {aa['delta_tpot']:.1f}ms delta  "
          f"{aa['atk_requests']} req  {aa['slowdown_per_req']:.2f}ms/req")
    print()
    for _, r in combined.iterrows():
        if r["pct_aa_damage"] and r["pct_aa_budget_eff"]:
            print(f"  {r['policy']:20s}: {r['pct_aa_damage']:.1f}% damage  "
                  f"{r['pct_aa_budget_eff']:.1f}% budget  (offload=4 budget only)")
else:
    print("  (kein --aa4 angegeben)")

print()
print("=" * 72)
print("4. EFFICIENCY LIFT vs RANDOM GATE")
print("=" * 72)
rg_val = combined[combined["policy"]=="random_gate"]["slowdown_per_req"].values[0]
for _, r in combined.iterrows():
    v = r["slowdown_per_req"]
    if v and rg_val:
        print(f"  {r['policy']:20s}: {v/rg_val:.2f}x")

print()
print("=" * 72)
print("5. REGIME SELECTIVITY")
print("=" * 72)
for _, r in combined.iterrows():
    print(f"  {r['policy']:20s}: off0={r['rate_off0']:>4s}  off4={r['rate_off4']:>4s}  "
          f"gap={r['selectivity_gap']:+.2f}  wasted={r['wasted_frac']:.0%}")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors  = {"random_gate":"#e74c3c", "naive_threshold":"#f39c12", "state_aware":"#27ae60"}
markers = {"random_gate":"o",       "naive_threshold":"s",       "state_aware":"D"}
labels  = {"random_gate":"Random Gate", "naive_threshold":"Naive Threshold",
           "state_aware":"State-Aware Gate"}

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for _, r in combined.iterrows():
    p = r["policy"]; x = r["total_req"]; y = r["delta_tpot_ms"]
    if not (x and y): continue
    lo = r["ci_lo"]; hi = r["ci_hi"]
    ax.scatter(x, y, color=colors[p], marker=markers[p], s=160, zorder=4, label=labels[p])
    if lo and hi:
        ax.errorbar(x, y, yerr=[[y - lo*x], [hi*x - y]],
                    fmt="none", color=colors[p], capsize=5, linewidth=1.5, zorder=3)
    ax.annotate(f"{r['slowdown_per_req']:.2f}ms/req",
                xy=(x, y), xytext=(9, 5), textcoords="offset points",
                fontsize=9, color=colors[p], fontweight="bold")

if aa:
    ax.scatter(aa["atk_requests"], aa["delta_tpot"],
               color="#2c3e50", marker="*", s=230, zorder=4, label="Always Attack")
    ax.annotate(f"Always Attack\n{aa['slowdown_per_req']:.2f}ms/req",
                xy=(aa["atk_requests"], aa["delta_tpot"]),
                xytext=(9, -30), textcoords="offset points",
                fontsize=8.5, color="#2c3e50")

ax.set_xlabel("Total Attacker Requests (offload=0 + offload=4)", fontsize=11)
ax.set_ylabel("Victim TPOT Degradation at offload=4 (ms)", fontsize=11)
ax.set_title("Cost–Impact Trade-off: Attack Policies\n"
             "(upper-left = higher efficiency)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.annotate("← fewer requests\n↑ more damage",
            xy=(0.62, 0.06), xycoords="axes fraction",
            fontsize=9, va="bottom", color="gray", style="italic")

plt.tight_layout()
plot_path = OUT / "cost_impact_plot.png"
plt.savefig(plot_path, dpi=150)
plt.close()

# ── Speichern ─────────────────────────────────────────────────────────────────
csv_path = OUT / "combined_table.csv"
combined.to_csv(csv_path, index=False)
print()
print(f"Plot    : {plot_path}")
print(f"Tabelle : {csv_path}")
