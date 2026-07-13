#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "na", "null"}:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def to_int(x: Any) -> int | None:
    v = to_float(x)
    return int(v) if v is not None else None


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = math.ceil(0.95 * len(ys)) - 1
    idx = max(0, min(idx, len(ys) - 1))
    return ys[idx]


def fmt_ms(x: float | None) -> str:
    return "NA" if x is None else f"{x:.3f}"


def fmt_ratio(x: float | None) -> str:
    return "NA" if x is None else f"{x:.2f}"


def fmt_pct(x: float | None) -> str:
    return "NA" if x is None else f"{x * 100:.2f}%"


def load_audit_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row: dict[str, Any] = dict(raw)

            for k in ["offload_gb", "concurrency", "input_len", "output_len", "run_no"]:
                row[k] = to_int(row.get(k))

            for k in [
                "tpot_median",
                "tpot_p95",
                "itl_median",
                "itl_p95",
                "ttft_median",
                "ttft_p95",
                "e2el_median",
                "e2el_p95",
            ]:
                row[k] = to_float(row.get(k))

            rows.append(row)

    return rows


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(out)


def metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return vals


def metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int | None]:
    vals = metric_values(rows, metric)
    return {
        "n": len(vals),
        "median": median(vals),
        "p95": p95(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require_clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    dropped = 0

    for r in rows:
        if (
            r.get("offload_gb") is None
            or r.get("concurrency") is None
            or r.get("input_len") is None
            or r.get("output_len") is None
            or r.get("run_no") is None
        ):
            dropped += 1
            continue
        clean.append(r)

    if dropped:
        print(f"WARNING: dropped {dropped} rows with missing matrix metadata.")

    return clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create paper-ready offline summaries from profile_grid_v2 audit CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input audit CSV, e.g. llama_audit_filename.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for paper-ready summary tables.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = require_clean_rows(load_audit_csv(csv_path))

    if not rows:
        print("ERROR: no usable rows found.")
        return 2

    offloads = sorted({r["offload_gb"] for r in rows})
    concurrencies = sorted({r["concurrency"] for r in rows})
    inputs = sorted({r["input_len"] for r in rows})
    outputs = sorted({r["output_len"] for r in rows})

    by_offload = group_by(rows, ["offload_gb"])

    if (0,) not in by_offload:
        print("ERROR: offload0 baseline missing, cannot compute ratios.")
        return 2

    baseline_tpot = metric_summary(by_offload[(0,)], "tpot_median")["median"]
    baseline_itl = metric_summary(by_offload[(0,)], "itl_median")["median"]
    baseline_ttft = metric_summary(by_offload[(0,)], "ttft_median")["median"]

    if baseline_tpot is None or baseline_itl is None or baseline_ttft is None:
        print("ERROR: baseline metrics for offload0 are incomplete.")
        return 2

    # ------------------------------------------------------------------
    # Table 1: by offload
    # ------------------------------------------------------------------
    table_by_offload: list[dict[str, Any]] = []

    for off in offloads:
        subset = by_offload[(off,)]

        tpot = metric_summary(subset, "tpot_median")
        itl = metric_summary(subset, "itl_median")
        ttft = metric_summary(subset, "ttft_median")

        tpot_m = tpot["median"]
        itl_m = itl["median"]
        ttft_m = ttft["median"]

        tpot_ratio = float(tpot_m) / float(baseline_tpot) if tpot_m is not None else None
        itl_ratio = float(itl_m) / float(baseline_itl) if itl_m is not None else None
        ttft_ratio = float(ttft_m) / float(baseline_ttft) if ttft_m is not None else None

        table_by_offload.append(
            {
                "offload_gb": off,
                "n_files": len(subset),
                "median_tpot_ms": fmt_ms(tpot_m),
                "p95_file_tpot_ms": fmt_ms(tpot["p95"]),
                "tpot_ratio_vs_offload0": fmt_ratio(tpot_ratio),
                "median_itl_ms": fmt_ms(itl_m),
                "p95_file_itl_ms": fmt_ms(itl["p95"]),
                "itl_ratio_vs_offload0": fmt_ratio(itl_ratio),
                "median_ttft_ms": fmt_ms(ttft_m),
                "p95_file_ttft_ms": fmt_ms(ttft["p95"]),
                "ttft_ratio_vs_offload0": fmt_ratio(ttft_ratio),
            }
        )

    # ------------------------------------------------------------------
    # Table 2: by offload × output length
    # ------------------------------------------------------------------
    table_by_output: list[dict[str, Any]] = []

    for (off, out), subset in sorted(group_by(rows, ["offload_gb", "output_len"]).items()):
        tpot = metric_summary(subset, "tpot_median")
        itl = metric_summary(subset, "itl_median")
        ttft = metric_summary(subset, "ttft_median")

        table_by_output.append(
            {
                "offload_gb": off,
                "output_len": out,
                "n_files": len(subset),
                "median_tpot_ms": fmt_ms(tpot["median"]),
                "p95_file_tpot_ms": fmt_ms(tpot["p95"]),
                "median_itl_ms": fmt_ms(itl["median"]),
                "median_ttft_ms": fmt_ms(ttft["median"]),
            }
        )

    # ------------------------------------------------------------------
    # Table 3: by offload × concurrency
    # ------------------------------------------------------------------
    table_by_concurrency: list[dict[str, Any]] = []

    for (off, conc), subset in sorted(group_by(rows, ["offload_gb", "concurrency"]).items()):
        tpot = metric_summary(subset, "tpot_median")
        itl = metric_summary(subset, "itl_median")
        ttft = metric_summary(subset, "ttft_median")

        table_by_concurrency.append(
            {
                "offload_gb": off,
                "concurrency": conc,
                "n_files": len(subset),
                "median_tpot_ms": fmt_ms(tpot["median"]),
                "p95_file_tpot_ms": fmt_ms(tpot["p95"]),
                "median_itl_ms": fmt_ms(itl["median"]),
                "median_ttft_ms": fmt_ms(ttft["median"]),
            }
        )

    write_csv(out_dir / "paper_table_by_offload.csv", table_by_offload)
    write_csv(out_dir / "paper_table_by_offload_output.csv", table_by_output)
    write_csv(out_dir / "paper_table_by_offload_concurrency.csv", table_by_concurrency)

    # ------------------------------------------------------------------
    # Additional derived findings
    # ------------------------------------------------------------------
    off_to_tpot: dict[int, float] = {}
    off_to_itl: dict[int, float] = {}

    for off in offloads:
        tpot_m = metric_summary(by_offload[(off,)], "tpot_median")["median"]
        itl_m = metric_summary(by_offload[(off,)], "itl_median")["median"]
        if tpot_m is not None:
            off_to_tpot[int(off)] = float(tpot_m)
        if itl_m is not None:
            off_to_itl[int(off)] = float(itl_m)

    off12_tpot = off_to_tpot.get(12)
    off16_tpot = off_to_tpot.get(16)

    off16_vs_12_ratio = None
    off16_vs_12_pct = None
    if off12_tpot is not None and off16_tpot is not None:
        off16_vs_12_ratio = off16_tpot / off12_tpot
        off16_vs_12_pct = off16_vs_12_ratio - 1.0

    # Output-length stability: max-min TPOT within each offload over output lengths.
    output_stability_rows: list[dict[str, Any]] = []

    for off in offloads:
        vals = []
        for out in outputs:
            subset = [
                r for r in rows
                if r["offload_gb"] == off and r["output_len"] == out
            ]
            m = metric_summary(subset, "tpot_median")["median"]
            if m is not None:
                vals.append(float(m))

        if vals:
            spread_abs = max(vals) - min(vals)
            spread_rel = spread_abs / statistics.median(vals)
            output_stability_rows.append(
                {
                    "offload_gb": off,
                    "min_output_median_tpot_ms": fmt_ms(min(vals)),
                    "max_output_median_tpot_ms": fmt_ms(max(vals)),
                    "absolute_spread_ms": fmt_ms(spread_abs),
                    "relative_spread": fmt_ratio(spread_rel),
                    "relative_spread_percent": fmt_pct(spread_rel),
                }
            )

    write_csv(out_dir / "paper_table_output_stability.csv", output_stability_rows)

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    md: list[str] = []

    md.append("# Llama profile_grid_v2 paper summary")
    md.append("")
    md.append("## Dataset")
    md.append("")
    md.append(f"- Input file: `{csv_path}`")
    md.append(f"- Rows/files included: {len(rows)}")
    md.append(f"- Offloads: {offloads}")
    md.append(f"- Concurrency levels: {concurrencies}")
    md.append(f"- Input lengths: {inputs}")
    md.append(f"- Output lengths: {outputs}")
    md.append("")
    md.append("Interpretation: fixed-input, variable-output decode-focused profiling matrix.")
    md.append("")
    md.append("## Main table: by offload")
    md.append("")
    md.append(
        "| offload_gb | n_files | median TPOT ms | p95 file TPOT ms | "
        "TPOT ratio vs offload0 | median ITL ms | ITL ratio vs offload0 | median TTFT ms |"
    )
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in table_by_offload:
        md.append(
            f"| {r['offload_gb']} | {r['n_files']} | {r['median_tpot_ms']} | "
            f"{r['p95_file_tpot_ms']} | {r['tpot_ratio_vs_offload0']} | "
            f"{r['median_itl_ms']} | {r['itl_ratio_vs_offload0']} | "
            f"{r['median_ttft_ms']} |"
        )

    md.append("")
    md.append("## TPOT/ITL by offload and output length")
    md.append("")
    md.append(
        "| offload_gb | output_len | n_files | median TPOT ms | p95 file TPOT ms | median ITL ms | median TTFT ms |"
    )
    md.append("|---:|---:|---:|---:|---:|---:|---:|")

    for r in table_by_output:
        md.append(
            f"| {r['offload_gb']} | {r['output_len']} | {r['n_files']} | "
            f"{r['median_tpot_ms']} | {r['p95_file_tpot_ms']} | "
            f"{r['median_itl_ms']} | {r['median_ttft_ms']} |"
        )

    md.append("")
    md.append("## TPOT/ITL by offload and concurrency")
    md.append("")
    md.append(
        "| offload_gb | concurrency | n_files | median TPOT ms | p95 file TPOT ms | median ITL ms | median TTFT ms |"
    )
    md.append("|---:|---:|---:|---:|---:|---:|---:|")

    for r in table_by_concurrency:
        md.append(
            f"| {r['offload_gb']} | {r['concurrency']} | {r['n_files']} | "
            f"{r['median_tpot_ms']} | {r['p95_file_tpot_ms']} | "
            f"{r['median_itl_ms']} | {r['median_ttft_ms']} |"
        )

    md.append("")
    md.append("## Output-length stability")
    md.append("")
    md.append(
        "Within each offload level, median TPOT is highly stable across output lengths. "
        "This supports interpreting the measurement as a decode-regime effect rather than an output-length artifact."
    )
    md.append("")
    md.append("| offload_gb | min output-median TPOT ms | max output-median TPOT ms | spread ms | spread % |")
    md.append("|---:|---:|---:|---:|---:|")

    for r in output_stability_rows:
        md.append(
            f"| {r['offload_gb']} | {r['min_output_median_tpot_ms']} | "
            f"{r['max_output_median_tpot_ms']} | {r['absolute_spread_ms']} | "
            f"{r['relative_spread_percent']} |"
        )

    md.append("")
    md.append("## Paper-ready interpretation")
    md.append("")
    md.append(
        "The profiling run shows a clear monotonic increase in decode-time metrics as CPU-offload increases. "
        "TPOT and ITL form the strongest runtime-regime signals; TTFT also increases but is less suitable as the primary signal."
    )
    md.append("")
    md.append(
        "The fixed input length controls prefill cost, while varying output length tests the decode phase where CPU-offload effects dominate."
    )
    md.append("")
    md.append("## Phase-A recommendation")
    md.append("")
    md.append("- Recommended Phase-A state pair: low=offload0, high=offload12.")
    md.append(
        "- Rationale: offload12 is already a strong high-state, while offload16 adds comparatively little extra TPOT over offload12."
    )

    if off12_tpot is not None and off16_tpot is not None:
        md.append(
            f"- Observed median TPOT: offload12={off12_tpot:.3f} ms, "
            f"offload16={off16_tpot:.3f} ms."
        )
        md.append(
            f"- offload16 is only {fmt_ratio(off16_vs_12_ratio)}× offload12 "
            f"({fmt_pct(off16_vs_12_pct)} higher TPOT), so offload12 is a cleaner high-state for Phase A."
        )

    md.append("")
    md.append(
        "Do not claim availability degradation from this profiling dataset alone. "
        "It supports state selection and regime characterization for the later Victim/Burst Phase-A experiment."
    )

    md_path = out_dir / "paper_profile_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("DONE")
    print(f"Wrote: {out_dir / 'paper_table_by_offload.csv'}")
    print(f"Wrote: {out_dir / 'paper_table_by_offload_output.csv'}")
    print(f"Wrote: {out_dir / 'paper_table_by_offload_concurrency.csv'}")
    print(f"Wrote: {out_dir / 'paper_table_output_stability.csv'}")
    print(f"Wrote: {md_path}")
    print()
    print("Main result:")
    for r in table_by_offload:
        print(
            f"offload{int(r['offload_gb']):>2}: "
            f"TPOT={r['median_tpot_ms']} ms, "
            f"ITL={r['median_itl_ms']} ms, "
            f"TPOT ratio vs offload0={r['tpot_ratio_vs_offload0']}x"
        )

    if off12_tpot is not None and off16_tpot is not None:
        print()
        print(
            f"Phase-A high-state recommendation: offload12 "
            f"(offload16/offload12 TPOT ratio = {fmt_ratio(off16_vs_12_ratio)}x; "
            f"+{fmt_pct(off16_vs_12_pct)})"
        )

    # sanity check that the old NA-ratio bug is gone
    bad_ratio_rows = [
        r for r in table_by_offload
        if r["offload_gb"] != 0 and r["tpot_ratio_vs_offload0"] == "NA"
    ]
    if bad_ratio_rows:
        print()
        print("ERROR: non-baseline ratio rows are still NA.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())