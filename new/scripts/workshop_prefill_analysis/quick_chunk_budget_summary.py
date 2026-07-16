#!/usr/bin/env python3
"""
Kurzauswertung des Chunk-Budget-Screens.

Aufruf:
  python3 quick_chunk_budget_summary.py
oder:
  python3 quick_chunk_budget_summary.py /pfad/zum/official_ORDNER
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def newest_result_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    root = Path("new/runs/chunk_budget_screen/results")
    dirs = [p for p in root.glob("official_*") if p.is_dir()]
    if not dirs:
        raise SystemExit("Kein official_*-Ergebnisordner gefunden.")
    return max(dirs, key=lambda p: p.stat().st_mtime).resolve()


def req_median(obj, indices, field):
    vals = []
    for r in obj["victim_requests"]:
        if (
            r.get("status") == "complete"
            and r.get("request_index") in indices
            and r.get(field) is not None
        ):
            vals.append(float(r[field]))
    if len(vals) != len(indices):
        raise ValueError(
            f"{obj['episode_id']}: erwartet {len(indices)} Werte für {field}, "
            f"gefunden {len(vals)}"
        )
    return med(vals)


def pct_loss_from_factor(control_over_burst):
    return (1.0 - 1.0 / control_over_burst) * 100.0


def main() -> int:
    result_dir = newest_result_dir()
    files = sorted((result_dir / "episodes").glob("*.json"))

    episodes = {}
    invalid = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "complete":
            invalid.append((path.name, obj.get("status")))
            continue
        s = obj["schedule_row"]
        key = (
            s["state_label"],
            int(s["max_num_batched_tokens"]),
            s["condition"],
            int(s["repeat"]),
        )
        episodes[key] = obj

    print(f"RESULT_DIR: {result_dir}")
    print(f"COMPLETE EPISODES: {len(episodes)}/36")
    if invalid:
        print(f"INVALID/INCOMPLETE: {invalid}")
    if len(episodes) != 36:
        raise SystemExit("Nicht 36 vollständige reguläre Episoden; Auswertung abgebrochen.")
    print()

    print("SCIENTIFIC RESULTS")
    print("Einheit der Inferenz: gematchtes Episodenpaar, n=3 pro State×Budget.")
    print()

    summary = defaultdict(dict)

    for state in ("low", "high"):
        print("=" * 92)
        print(f"STATE: {state.upper()}")
        print("=" * 92)
        print(
            "budget  active_ΔE2EL_ms  active_ratio  later_ratio  "
            "victim_thr_loss  burst_makespan_ms"
        )

        for budget in (512, 1024, 2048):
            active_deltas = []
            active_ratios = []
            later_ratios = []
            throughput_factors = []
            burst_makespans = []

            for rep in (1, 2, 3):
                c = episodes[(state, budget, "no_burst", rep)]
                b = episodes[(state, budget, "prefill_burst", rep)]

                active_idx = set(range(4))
                later_idx = set(range(4, 20))

                c_active = req_median(c, active_idx, "e2el_ms")
                b_active = req_median(b, active_idx, "e2el_ms")
                c_later = req_median(c, later_idx, "e2el_ms")
                b_later = req_median(b, later_idx, "e2el_ms")

                active_deltas.append(b_active - c_active)
                active_ratios.append(b_active / c_active)
                later_ratios.append(b_later / c_later)

                c_thr = c["aggregate_metrics"]["victim_throughput_tokens_per_s"]
                b_thr = b["aggregate_metrics"]["victim_throughput_tokens_per_s"]
                throughput_factors.append(c_thr / b_thr)

                bm = b["aggregate_metrics"].get("burst_makespan_ms")
                if bm is not None:
                    burst_makespans.append(float(bm))

            row = {
                "active_delta_ms": med(active_deltas),
                "active_ratio": med(active_ratios),
                "later_ratio": med(later_ratios),
                "throughput_factor": med(throughput_factors),
                "throughput_loss_pct": pct_loss_from_factor(med(throughput_factors)),
                "burst_makespan_ms": med(burst_makespans),
                "active_delta_repeats": active_deltas,
                "active_ratio_repeats": active_ratios,
                "later_ratio_repeats": later_ratios,
            }
            summary[state][budget] = row

            print(
                f"{budget:6d}  "
                f"{row['active_delta_ms']:16.3f}  "
                f"{row['active_ratio']:12.4f}x  "
                f"{row['later_ratio']:10.4f}x  "
                f"{row['throughput_loss_pct']:14.1f}%  "
                f"{row['burst_makespan_ms']:17.3f}"
            )

        print()
        print("Repeat-level active-wave ratios:")
        for budget in (512, 1024, 2048):
            vals = summary[state][budget]["active_ratio_repeats"]
            print(f"  budget {budget}: {[round(x, 4) for x in vals]}")
        print()

    print("=" * 92)
    print("BUDGET EFFECT RELATIVE TO DEFAULT 2048")
    print("=" * 92)
    for state in ("low", "high"):
        base = summary[state][2048]
        print(f"{state.upper()}:")
        for budget in (512, 1024):
            row = summary[state][budget]
            stall_reduction = (
                1.0 - row["active_delta_ms"] / base["active_delta_ms"]
            ) * 100.0
            burst_cost = (
                row["burst_makespan_ms"] / base["burst_makespan_ms"] - 1.0
            ) * 100.0
            print(
                f"  {budget} vs 2048: "
                f"active-stall change={stall_reduction:+.1f}% reduction, "
                f"burst-makespan change={burst_cost:+.1f}%"
            )
        print()

    print("READING:")
    print("- Kleinere Budgets sind schützend, wenn active_ΔE2EL und active_ratio")
    print("  geordnet sinken, während later_ratio nahe 1.00 bleibt.")
    print("- Der Preis ist sichtbar, wenn burst_makespan bei kleineren Budgets steigt.")
    print("- Eine State-Invarianz-Lücke liegt vor, wenn dasselbe Budget Low ausreichend")
    print("  schützt, High aber nicht oder deutlich andere Schutz/Kosten-Verhältnisse zeigt.")
    print("- n=3 bleibt ein Screening; keine Signifikanz- oder CI-Claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
