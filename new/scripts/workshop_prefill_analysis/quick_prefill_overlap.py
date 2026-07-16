#!/usr/bin/env python3
"""
Request-level mechanism screen for Prefill-Screen.

The analysis avoids the naive comparison "actual overlap vs no overlap"
inside burst episodes, because long-running requests are mechanically more
likely to overlap. Instead, for every matched burst/control episode pair it:

1. takes the real burst duration from the burst episode,
2. places a same-duration pseudo-burst window at the control episode's trigger,
3. classifies request exposure using the CONTROL request timeline,
4. compares the same request_index between burst and control.

This is a screening/mechanism analysis, not a causal proof.

Usage:
  python3 quick_prefill_overlap.py
  python3 quick_prefill_overlap.py /path/to/official_RESULT_DIR
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else float("nan")


def newest_result_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    root = Path("new/runs/prefill_screen/results")
    candidates = [p for p in root.glob("official_*") if p.is_dir()]
    if not candidates:
        raise SystemExit("Kein official_*-Ergebnisordner gefunden.")
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def overlap_ms(start_ns: int, end_ns: int, win_start_ns: int, win_end_ns: int) -> float:
    return max(0, min(end_ns, win_end_ns) - max(start_ns, win_start_ns)) / 1e6


def rankdata(values):
    """Average ranks for ties, 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[order[k]] = avg_rank
        pos = end
    return ranks


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if den == 0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / den


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def request_map(obj):
    out = {}
    for record in obj["victim_requests"]:
        if record.get("status") != "complete":
            continue
        out[record["request_index"]] = record
    return out


def main() -> int:
    result_dir = newest_result_dir()
    files = sorted((result_dir / "episodes").glob("*.json"))
    if len(files) != 24:
        raise SystemExit(f"Erwartet 24 Episode-Dateien, gefunden: {len(files)}")

    episodes = {}
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "complete":
            raise SystemExit(f"{path.name}: status={obj.get('status')!r}")
        s = obj["schedule_row"]
        key = (s["state_label"], s["concurrency"], s["condition"], s["repeat"])
        episodes[key] = obj

    rows = []

    for state in ("low", "high"):
        for conc in (4, 8):
            for repeat in (1, 2, 3):
                control = episodes[(state, conc, "no_burst", repeat)]
                burst = episodes[(state, conc, "prefill_burst", repeat)]

                interval = burst.get("burst_interval")
                if not isinstance(interval, dict):
                    raise SystemExit(
                        f"Fehlendes burst_interval: state={state}, conc={conc}, repeat={repeat}"
                    )

                burst_start = interval["start_ns"]
                burst_end = interval["end_ns"]
                burst_duration_ns = burst_end - burst_start

                control_trigger = control["trigger"]["trigger_perf_ns"]
                pseudo_start = control_trigger
                pseudo_end = control_trigger + burst_duration_ns

                control_requests = request_map(control)
                burst_requests = request_map(burst)

                if set(control_requests) != set(range(20)):
                    raise SystemExit(
                        f"Control hat nicht 20 vollständige Request-Indizes: "
                        f"{state}, c{conc}, r{repeat}"
                    )
                if set(burst_requests) != set(range(20)):
                    raise SystemExit(
                        f"Burst hat nicht 20 vollständige Request-Indizes: "
                        f"{state}, c{conc}, r{repeat}"
                    )

                for index in range(20):
                    c = control_requests[index]
                    b = burst_requests[index]

                    pseudo_ov = overlap_ms(
                        c["request_start_ns"],
                        c["stream_end_ns"],
                        pseudo_start,
                        pseudo_end,
                    )
                    actual_ov = overlap_ms(
                        b["request_start_ns"],
                        b["stream_end_ns"],
                        burst_start,
                        burst_end,
                    )

                    rows.append(
                        {
                            "state": state,
                            "conc": conc,
                            "repeat": repeat,
                            "request_index": index,
                            "wave": index // conc,
                            "pseudo_overlap_ms": pseudo_ov,
                            "pseudo_exposed": pseudo_ov > 0,
                            "actual_overlap_ms": actual_ov,
                            "actual_exposed": actual_ov > 0,
                            "delta_e2el_ms": b["e2el_ms"] - c["e2el_ms"],
                            "ratio_e2el": b["e2el_ms"] / c["e2el_ms"],
                            "delta_ttft_ms": b["ttft_ms"] - c["ttft_ms"],
                            "delta_tpot_ms": (
                                b["client_observed_tpot_ms"]
                                - c["client_observed_tpot_ms"]
                            ),
                        }
                    )

    print(f"RESULT_DIR: {result_dir}")
    print("Matched request-level mechanism screen")
    print("Exposure is defined from a control-based pseudo-burst window.")
    print()

    for state in ("low", "high"):
        for conc in (4, 8):
            subset = [r for r in rows if r["state"] == state and r["conc"] == conc]
            exposed = [r for r in subset if r["pseudo_exposed"]]
            unexposed = [r for r in subset if not r["pseudo_exposed"]]

            print("=" * 88)
            print(f"{state.upper()}  concurrency={conc}")
            print("=" * 88)
            print(
                f"requests: total={len(subset)}, "
                f"pseudo_exposed={len(exposed)}, pseudo_unexposed={len(unexposed)}"
            )

            for label, group in (("EXPOSED", exposed), ("UNEXPOSED", unexposed)):
                if not group:
                    print(f"{label}: n=0")
                    continue
                print(
                    f"{label}: n={len(group)}  "
                    f"median ΔE2EL={median([r['delta_e2el_ms'] for r in group]):.3f} ms  "
                    f"median E2EL ratio={median([r['ratio_e2el'] for r in group]):.4f}x  "
                    f"median ΔTPOT={median([r['delta_tpot_ms'] for r in group]):.3f} ms  "
                    f"positive ΔE2EL="
                    f"{sum(r['delta_e2el_ms'] > 0 for r in group)}/{len(group)}"
                )

            rho = spearman(
                [r["pseudo_overlap_ms"] for r in subset],
                [r["delta_e2el_ms"] for r in subset],
            )
            print(f"Spearman(pseudo_overlap_ms, ΔE2EL_ms): {rho:.4f}")

            print("By fixed request wave (request_index // concurrency):")
            for wave in sorted({r["wave"] for r in subset}):
                group = [r for r in subset if r["wave"] == wave]
                print(
                    f"  wave={wave}: n={len(group):2d}  "
                    f"median ΔE2EL={median([r['delta_e2el_ms'] for r in group]):10.3f} ms  "
                    f"median ratio={median([r['ratio_e2el'] for r in group]):.4f}x  "
                    f"positive={sum(r['delta_e2el_ms'] > 0 for r in group)}/{len(group)}"
                )

            by_index = defaultdict(list)
            for r in subset:
                by_index[r["request_index"]].append(r["delta_e2el_ms"])

            ranked = sorted(
                (
                    (index, median(deltas), sum(x > 0 for x in deltas))
                    for index, deltas in by_index.items()
                ),
                key=lambda x: x[1],
                reverse=True,
            )

            print("Top request indices by median paired ΔE2EL across repeats:")
            for index, delta, positive_count in ranked[:6]:
                print(
                    f"  index={index:2d}, wave={index // conc}, "
                    f"median ΔE2EL={delta:10.3f} ms, positive={positive_count}/3"
                )

            print("Maximum-E2EL request index by repeat:")
            for repeat in (1, 2, 3):
                control = episodes[(state, conc, "no_burst", repeat)]
                burst = episodes[(state, conc, "prefill_burst", repeat)]
                cmax = max(
                    control["victim_requests"],
                    key=lambda r: r.get("e2el_ms") if r.get("e2el_ms") is not None else -1,
                )
                bmax = max(
                    burst["victim_requests"],
                    key=lambda r: r.get("e2el_ms") if r.get("e2el_ms") is not None else -1,
                )
                print(
                    f"  repeat={repeat}: control index={cmax['request_index']:2d} "
                    f"({cmax['e2el_ms']:.3f} ms), "
                    f"burst index={bmax['request_index']:2d} "
                    f"({bmax['e2el_ms']:.3f} ms)"
                )
            print()

    print("Interpretation guardrails:")
    print("- A larger ΔE2EL in pseudo-exposed requests supports localization.")
    print("- A monotonic wave pattern supports a queue/straggler mechanism.")
    print("- This remains observational screening, not causal proof.")
    print("- p95, max, throughput and makespan must not be counted as independent effects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
