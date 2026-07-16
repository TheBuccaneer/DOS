#!/usr/bin/env python3
"""
Quick scientific summary for Prefill-Screen official results.

Usage:
  python3 quick_prefill_summary.py /path/to/official_RESULT_DIR

If no directory is given, the newest official_* directory below
new/runs/prefill_screen/results is selected automatically.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def med(values):
    return statistics.median(values) if values else float("nan")


def find_result_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    root = Path("new/runs/prefill_screen/results")
    candidates = [p for p in root.glob("official_*") if p.is_dir()]
    if not candidates:
        raise SystemExit("Kein official_*-Ergebnisordner gefunden.")
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def main() -> int:
    result_dir = find_result_dir()
    files = sorted((result_dir / "episodes").glob("*.json"))
    if not files:
        raise SystemExit(f"Keine Episode-Dateien gefunden: {result_dir / 'episodes'}")

    rows = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "complete":
            print(f"WARNUNG: {path.name}: status={obj.get('status')!r}")
            continue

        s = obj["schedule_row"]
        a = obj["aggregate_metrics"]
        rows.append({
            "state": s["state_label"],
            "concurrency": s["concurrency"],
            "condition": s["condition"],
            "repeat": s["repeat"],
            "e2el_ms": a["victim_e2el_ms"]["median"],
            "ttft_ms": a["victim_ttft_ms"]["median"],
            "tpot_ms": a["victim_client_observed_tpot_ms"]["median"],
            "itl_ms": a["victim_itl_ms"]["median"],
            "throughput_tok_s": a["victim_throughput_tokens_per_s"],
        })

    print(f"RESULT_DIR: {result_dir}")
    print(f"COMPLETE EPISODES: {len(rows)}/24")
    print()

    by_cell = defaultdict(list)
    by_key = {}
    for row in rows:
        by_cell[(row["state"], row["concurrency"], row["condition"])].append(row)
        by_key[
            (row["state"], row["concurrency"], row["condition"], row["repeat"])
        ] = row

    print("CELL MEDIANS ACROSS REPEATS")
    print("state conc condition          n   E2EL_ms   TTFT_ms   TPOT_ms   ITL_ms   throughput_tok_s")
    for state in ("low", "high"):
        for conc in (4, 8):
            for cond in ("no_burst", "prefill_burst"):
                rs = by_cell[(state, conc, cond)]
                print(
                    f"{state:5s} {conc:4d} {cond:18s} {len(rs):2d} "
                    f"{med([r['e2el_ms'] for r in rs]):10.3f} "
                    f"{med([r['ttft_ms'] for r in rs]):10.3f} "
                    f"{med([r['tpot_ms'] for r in rs]):10.3f} "
                    f"{med([r['itl_ms'] for r in rs]):9.3f} "
                    f"{med([r['throughput_tok_s'] for r in rs]):17.3f}"
                )

    print()
    print("PRIMARY: PAIRED E2EL DEGRADATION (prefill_burst / no_burst)")
    ratios_by_state_conc = {}

    for conc in (4, 8):
        for state in ("low", "high"):
            ratios = []
            for repeat in (1, 2, 3):
                burst = by_key[(state, conc, "prefill_burst", repeat)]["e2el_ms"]
                control = by_key[(state, conc, "no_burst", repeat)]["e2el_ms"]
                ratios.append(burst / control)

            ratios_by_state_conc[(state, conc)] = ratios
            print(
                f"{state:5s} conc={conc}: "
                f"per-repeat={[round(x, 4) for x in ratios]}  "
                f"median={med(ratios):.4f}x"
            )

        low = ratios_by_state_conc[("low", conc)]
        high = ratios_by_state_conc[("high", conc)]
        interaction = [h / l for h, l in zip(high, low)]

        print(
            f"INTERACTION conc={conc}: "
            f"per-repeat High/Low={[round(x, 4) for x in interaction]}  "
            f"median={med(interaction):.4f}x  "
            f"ratio-of-medians={med(high) / med(low):.4f}x  "
            f"exp(mean log-DiD)="
            f"{math.exp(statistics.mean(math.log(h) - math.log(l) for h, l in zip(high, low))):.4f}x"
        )
        print()

    print("Interpretation:")
    print("  > 1: derselbe Burst schadet High relativ stärker")
    print("  < 1: derselbe Burst schadet Low relativ stärker")
    print("  ca. 0.80–1.25: wahrscheinlich keine starke State×Load-Interaktion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
