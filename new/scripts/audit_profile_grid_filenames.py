#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return None, "top-level JSON is not an object"
        return d, None
    except Exception as e:
        return None, repr(e)


def metadata(d: dict[str, Any]) -> dict[str, Any]:
    md = d.get("metadata", {})
    return md if isinstance(md, dict) else {}


def as_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def as_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def re_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def parse_from_path(path: Path) -> dict[str, int | None]:
    """
    Expected examples:
      llama_offload16_conc1_in256_out32_run1.json
      llama_offload16_conc12_input256_output128_run5.json
    Offload is also read from parent folder offload16.
    """
    stem = path.stem
    parent = path.parent.name

    offload = re_int(r"offload[_-]?(\d+)", parent)
    if offload is None:
        offload = re_int(r"offload[_-]?(\d+)", stem)

    conc = re_int(r"(?:^|[_-])conc(?:urrency)?[_-]?(\d+)", stem)
    input_len = re_int(r"(?:^|[_-])(?:in|input|inputlen|input_len)[_-]?(\d+)", stem)
    output_len = re_int(r"(?:^|[_-])(?:out|output|outputlen|output_len)[_-]?(\d+)", stem)
    run_no = re_int(r"(?:^|[_-])run[_-]?(\d+)", stem)

    return {
        "offload_gb": offload,
        "concurrency": conc,
        "input_len": input_len,
        "output_len": output_len,
        "run_no": run_no,
    }


def first_present(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def get_completed_failed(d: dict[str, Any]) -> tuple[Any, Any]:
    completed = None
    failed = None

    candidates = [
        d,
        d.get("summary", {}),
        d.get("stats", {}),
        d.get("result", {}),
        d.get("metadata", {}),
    ]

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        if completed is None:
            completed = first_present(
                obj,
                [
                    "completed",
                    "num_completed",
                    "successful",
                    "success",
                    "success_count",
                    "completed_requests",
                ],
            )

        if failed is None:
            failed = first_present(
                obj,
                [
                    "failed",
                    "num_failed",
                    "failures",
                    "failure_count",
                    "failed_requests",
                    "errors",
                ],
            )

    return completed, failed


def walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)


def collect_numeric(d: dict[str, Any], names: list[str]) -> list[float]:
    wanted = set(names)
    vals: list[float] = []

    for obj in walk(d):
        for k, v in obj.items():
            if k in wanted:
                x = as_float_or_none(v)
                if x is not None:
                    vals.append(x)

    return vals


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--offloads", default="0,2,4,8,12,16")
    ap.add_argument("--concs", default="1,2,4,8,12,16")
    ap.add_argument("--inputs", default="256")
    ap.add_argument("--outputs", default="32,64,128")
    ap.add_argument("--runs", default="1,2,3,4,5")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    offloads = parse_int_list(args.offloads)
    concs = parse_int_list(args.concs)
    inputs = parse_int_list(args.inputs)
    outputs = parse_int_list(args.outputs)
    runs = parse_int_list(args.runs)

    expected_per_offload = len(concs) * len(inputs) * len(outputs) * len(runs)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    print("=== PROFILE GRID AUDIT FROM FILENAMES ===")
    print(f"root: {root}")
    print(f"expected files per offload: {expected_per_offload}")
    print()

    for off in offloads:
        folder = root / f"offload{off}"
        files = sorted(folder.glob("*.json"))

        print(f"--- offload{off} ---")
        print(f"files: {len(files)} / expected {expected_per_offload}")

        if len(files) != expected_per_offload:
            warnings.append(f"offload{off}: file count {len(files)} != {expected_per_offload}")

        combos = collections.Counter()
        completed_counter = collections.Counter()
        failed_counter = collections.Counter()
        conc_counter = collections.Counter()
        input_counter = collections.Counter()
        output_counter = collections.Counter()
        run_counter = collections.Counter()

        for path in files:
            d, err = safe_json(path)
            if err:
                errors.append(f"BAD JSON: {path}: {err}")
                continue
            assert d is not None

            md = metadata(d)
            parsed = parse_from_path(path)

            offload_gb = as_int_or_none(md.get("offload_gb")) or parsed["offload_gb"]
            concurrency = as_int_or_none(md.get("concurrency")) or parsed["concurrency"]
            input_len = as_int_or_none(md.get("input_len")) or parsed["input_len"]
            output_len = as_int_or_none(md.get("output_len")) or parsed["output_len"]
            run_no = as_int_or_none(md.get("run_no")) or parsed["run_no"]

            completed, failed = get_completed_failed(d)

            if offload_gb != off:
                errors.append(f"OFFLOAD MISMATCH: {path}: got {offload_gb}, expected folder offload{off}")

            if failed not in (0, "0", None):
                errors.append(f"FAILED REQUESTS: {path}: failed={failed}")

            combo = (concurrency, input_len, output_len, run_no)
            combos[combo] += 1

            completed_counter[str(completed)] += 1
            failed_counter[str(failed)] += 1
            conc_counter[str(concurrency)] += 1
            input_counter[str(input_len)] += 1
            output_counter[str(output_len)] += 1
            run_counter[str(run_no)] += 1

            tpot_vals = collect_numeric(d, ["tpot_ms", "tpot", "median_tpot_ms", "mean_tpot_ms"])
            itl_vals = collect_numeric(d, ["itl_ms", "itl_median_ms", "median_itl_ms", "mean_itl_ms", "itl_mean_ms"])
            ttft_vals = collect_numeric(d, ["ttft_ms", "ttft", "median_ttft_ms", "mean_ttft_ms"])
            e2el_vals = collect_numeric(d, ["e2el_ms", "e2e_ms", "latency_ms", "end_to_end_ms", "total_latency_ms"])

            rows.append(
                {
                    "file": str(path),
                    "offload_gb": offload_gb,
                    "concurrency": concurrency,
                    "input_len": input_len,
                    "output_len": output_len,
                    "run_no": run_no,
                    "completed": completed,
                    "failed": failed,
                    "tpot_median": median(tpot_vals),
                    "tpot_p95": p95(tpot_vals),
                    "itl_median": median(itl_vals),
                    "itl_p95": p95(itl_vals),
                    "ttft_median": median(ttft_vals),
                    "ttft_p95": p95(ttft_vals),
                    "e2el_median": median(e2el_vals),
                    "e2el_p95": p95(e2el_vals),
                    "n_tpot_values": len(tpot_vals),
                    "n_itl_values": len(itl_vals),
                    "n_ttft_values": len(ttft_vals),
                    "n_e2el_values": len(e2el_vals),
                }
            )

        print("concurrency:", dict(sorted(conc_counter.items())))
        print("input_len:", dict(sorted(input_counter.items())))
        print("output_len:", dict(sorted(output_counter.items())))
        print("run_no:", dict(sorted(run_counter.items())))
        print("completed:", dict(completed_counter))
        print("failed:", dict(failed_counter))

        missing = []
        for c in concs:
            for i in inputs:
                for o in outputs:
                    for r in runs:
                        if combos[(c, i, o, r)] == 0:
                            missing.append((c, i, o, r))

        duplicates = [(k, v) for k, v in combos.items() if v > 1]

        print(f"missing combos: {len(missing)}")
        if missing:
            print("first missing:", missing[:10])
            warnings.append(f"offload{off}: missing combos={len(missing)}")

        print(f"duplicate combos: {len(duplicates)}")
        if duplicates:
            print("first duplicates:", duplicates[:10])
            warnings.append(f"offload{off}: duplicate combos={len(duplicates)}")

        print()

    print("=== SUMMARY BY OFFLOAD ===")
    for off in offloads:
        subset = [r for r in rows if r["offload_gb"] == off]
        print(f"offload{off}:")
        for m in ["tpot_median", "itl_median", "ttft_median", "e2el_median"]:
            vals = [r[m] for r in subset if isinstance(r[m], (int, float))]
            print(f"  {m:>12}: median={fmt_ms(median(vals))} ms, p95={fmt_ms(p95(vals))} ms, n={len(vals)}")
        print()

    print("=== TPOT/ITL BY OFFLOAD × OUTPUT_LEN ===")
    for off in offloads:
        for out in outputs:
            subset = [r for r in rows if r["offload_gb"] == off and r["output_len"] == out]
            tpot = [r["tpot_median"] for r in subset if isinstance(r["tpot_median"], (int, float))]
            itl = [r["itl_median"] for r in subset if isinstance(r["itl_median"], (int, float))]
            print(
                f"offload{off:>2} out{out:>3}: "
                f"TPOT median={fmt_ms(median(tpot)):>10} ms, "
                f"ITL median={fmt_ms(median(itl)):>10} ms, "
                f"n_files={len(subset)}"
            )
    print()

    print("=== TPOT BY OFFLOAD × CONCURRENCY ===")
    for off in offloads:
        for c in concs:
            subset = [r for r in rows if r["offload_gb"] == off and r["concurrency"] == c]
            tpot = [r["tpot_median"] for r in subset if isinstance(r["tpot_median"], (int, float))]
            print(
                f"offload{off:>2} conc{c:>2}: "
                f"TPOT median={fmt_ms(median(tpot)):>10} ms, "
                f"n_files={len(subset)}"
            )
    print()

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {csv_path}")
        print()

    print("=== FINAL VERDICT ===")
    if errors:
        print("FAIL")
        for e in errors[:50]:
            print("ERROR:", e)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        return 1

    if warnings:
        print("PASS WITH WARNINGS")
        for w in warnings:
            print("WARNING:", w)
        return 0

    print("PASS: complete matrix, no JSON errors, no failed requests, no missing/duplicate cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())