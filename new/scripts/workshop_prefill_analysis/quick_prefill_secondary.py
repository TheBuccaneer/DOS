#!/usr/bin/env python3
"""
Secondary paired analysis for Prefill-Screen.

Usage:
  python3 quick_prefill_secondary.py
  python3 quick_prefill_secondary.py /path/to/official_RESULT_DIR
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def med(xs):
    return statistics.median(xs)


def newest_result_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    root = Path("new/runs/prefill_screen/results")
    dirs = [p for p in root.glob("official_*") if p.is_dir()]
    if not dirs:
        raise SystemExit("Kein official_*-Ordner gefunden.")
    return max(dirs, key=lambda p: p.stat().st_mtime).resolve()


def pct(x):
    return f"{(x - 1.0) * 100:+.1f}%"


def main() -> int:
    result_dir = newest_result_dir()
    files = sorted((result_dir / "episodes").glob("*.json"))
    if len(files) != 24:
        raise SystemExit(f"Erwartet 24 Episoden, gefunden: {len(files)}")

    by_key = {}

    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "complete":
            raise SystemExit(f"{path.name}: status={obj.get('status')!r}")

        s = obj["schedule_row"]
        a = obj["aggregate_metrics"]
        victims = [
            r for r in obj["victim_requests"]
            if r.get("status") == "complete"
        ]
        e2el_values = [r["e2el_ms"] for r in victims if r.get("e2el_ms") is not None]

        row = {
            "e2el_median": a["victim_e2el_ms"]["median"],
            "e2el_p95": a["victim_e2el_ms"]["p95"],
            "e2el_max": max(e2el_values),
            "ttft_median": a["victim_ttft_ms"]["median"],
            "ttft_p95": a["victim_ttft_ms"]["p95"],
            "tpot_median": a["victim_client_observed_tpot_ms"]["median"],
            "tpot_p95": a["victim_client_observed_tpot_ms"]["p95"],
            "itl_median": a["victim_itl_ms"]["median"],
            "itl_p95": a["victim_itl_ms"]["p95"],
            "throughput": a["victim_throughput_tokens_per_s"],
        }

        key = (
            s["state_label"],
            s["concurrency"],
            s["condition"],
            s["repeat"],
        )
        by_key[key] = row

    metrics = [
        ("e2el_median", "higher"),
        ("e2el_p95", "higher"),
        ("e2el_max", "higher"),
        ("ttft_median", "higher"),
        ("ttft_p95", "higher"),
        ("tpot_median", "higher"),
        ("tpot_p95", "higher"),
        ("itl_median", "higher"),
        ("itl_p95", "higher"),
        ("throughput", "lower"),
    ]

    print(f"RESULT_DIR: {result_dir}")
    print("All effects are paired within state × concurrency × repeat.")
    print("For every printed degradation factor, >1 means worse under burst.")
    print()

    for metric, bad_direction in metrics:
        print("=" * 78)
        print(metric.upper())
        print("=" * 78)

        for conc in (4, 8):
            ratios_by_state = {}

            for state in ("low", "high"):
                ratios = []
                absolute = []

                for rep in (1, 2, 3):
                    control = by_key[(state, conc, "no_burst", rep)][metric]
                    burst = by_key[(state, conc, "prefill_burst", rep)][metric]

                    if bad_direction == "higher":
                        degradation = burst / control
                        delta = burst - control
                    else:
                        degradation = control / burst
                        delta = burst - control

                    ratios.append(degradation)
                    absolute.append(delta)

                ratios_by_state[state] = ratios
                worse_count = sum(x > 1.0 for x in ratios)

                print(
                    f"{state:5s} conc={conc}: "
                    f"degradation={[round(x, 4) for x in ratios]}  "
                    f"median={med(ratios):.4f}x ({pct(med(ratios))})  "
                    f"worse={worse_count}/3  "
                    f"absolute_delta_median={med(absolute):.3f}"
                )

            low = ratios_by_state["low"]
            high = ratios_by_state["high"]
            interaction = [h / l for h, l in zip(high, low)]

            print(
                f"INTERACTION conc={conc}: "
                f"High/Low={[round(x, 4) for x in interaction]}  "
                f"median={med(interaction):.4f}x ({pct(med(interaction))})"
            )
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
