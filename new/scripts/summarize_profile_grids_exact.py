#!/usr/bin/env python3
"""
Exact summary for the profile_grid_v2 JSON format.

The JSON files already contain run-level metrics such as:

    median_ttft_ms
    median_tpot_ms
    median_itl_ms
    median_e2el_ms

Therefore this script does not attempt to reconstruct request-level metrics.

Expected grid per model:

    offload_gb:  0, 2, 4, 8, 12, 16
    concurrency: 1, 2, 4, 8, 12, 16
    output_len:  32, 64, 128
    run_no:      1, 2, 3, 4, 5

Expected total per model:

    540 JSON files
    10,800 completed requests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable


EXPECTED_OFFLOADS = (0, 2, 4, 8, 12, 16)
EXPECTED_CONCURRENCIES = (1, 2, 4, 8, 12, 16)
EXPECTED_OUTPUT_LENGTHS = (32, 64, 128)
EXPECTED_RUNS = (1, 2, 3, 4, 5)

EXPECTED_INPUT_LEN = 256
EXPECTED_WARMUPS = 1
EXPECTED_COMPLETED = 20
EXPECTED_FAILED = 0


REQUIRED_METRICS = (
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
)


def parse_dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Use MODEL=PATH, for example llama=/data/profile_grid_v2/llama"
        )

    model, path = value.split("=", 1)
    model = model.strip()
    path = path.strip()

    if not model or not path:
        raise argparse.ArgumentTypeError(
            "Both MODEL and PATH must be non-empty."
        )

    return model, Path(path).expanduser()


def to_int(value: Any, field: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: field {field!r} is not an integer: {value!r}"
        ) from exc


def to_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: field {field!r} is not numeric: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{path}: field {field!r} is not finite: {value!r}"
        )

    return result


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)

    if not ordered:
        raise ValueError("Cannot calculate percentile of empty data.")

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def read_run(model: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required_metadata = (
        "model_key",
        "model_name",
        "offload_gb",
        "concurrency",
        "input_len",
        "output_len",
        "num_warmups",
        "run_no",
        "completed",
        "failed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    )

    missing = [
        field
        for field in (*required_metadata, *REQUIRED_METRICS)
        if field not in data
    ]

    if missing:
        raise ValueError(
            f"{path}: missing fields: {', '.join(missing)}"
        )

    row: dict[str, Any] = {
        "dataset_model": model,
        "path": str(path),
        "date": data.get("date"),
        "experiment_id": data.get("experiment_id"),
        "server_config_label": data.get("server_config_label"),
        "model_key": str(data["model_key"]),
        "model_name": str(data["model_name"]),
        "offload_gb": to_int(data["offload_gb"], "offload_gb", path),
        "concurrency": to_int(
            data["concurrency"],
            "concurrency",
            path,
        ),
        "input_len": to_int(data["input_len"], "input_len", path),
        "output_len": to_int(data["output_len"], "output_len", path),
        "num_warmups": to_int(
            data["num_warmups"],
            "num_warmups",
            path,
        ),
        "run_no": to_int(data["run_no"], "run_no", path),
        "completed": to_int(data["completed"], "completed", path),
        "failed": to_int(data["failed"], "failed", path),
        "duration_s": to_float(data["duration"], "duration", path),
        "total_input_tokens": to_int(
            data["total_input_tokens"],
            "total_input_tokens",
            path,
        ),
        "total_output_tokens": to_int(
            data["total_output_tokens"],
            "total_output_tokens",
            path,
        ),
        "request_throughput": to_float(
            data["request_throughput"],
            "request_throughput",
            path,
        ),
        "output_throughput": to_float(
            data["output_throughput"],
            "output_throughput",
            path,
        ),
        "total_token_throughput": to_float(
            data["total_token_throughput"],
            "total_token_throughput",
            path,
        ),
    }

    for field in REQUIRED_METRICS:
        row[field] = to_float(data[field], field, path)

    validation_errors: list[str] = []

    if row["offload_gb"] not in EXPECTED_OFFLOADS:
        validation_errors.append(
            f"unexpected offload={row['offload_gb']}"
        )

    if row["concurrency"] not in EXPECTED_CONCURRENCIES:
        validation_errors.append(
            f"unexpected concurrency={row['concurrency']}"
        )

    if row["output_len"] not in EXPECTED_OUTPUT_LENGTHS:
        validation_errors.append(
            f"unexpected output_len={row['output_len']}"
        )

    if row["run_no"] not in EXPECTED_RUNS:
        validation_errors.append(
            f"unexpected run_no={row['run_no']}"
        )

    if row["input_len"] != EXPECTED_INPUT_LEN:
        validation_errors.append(
            f"input_len={row['input_len']}"
        )

    if row["num_warmups"] != EXPECTED_WARMUPS:
        validation_errors.append(
            f"num_warmups={row['num_warmups']}"
        )

    if row["completed"] != EXPECTED_COMPLETED:
        validation_errors.append(
            f"completed={row['completed']}"
        )

    if row["failed"] != EXPECTED_FAILED:
        validation_errors.append(
            f"failed={row['failed']}"
        )

    if row["model_key"].lower() != model.lower():
        validation_errors.append(
            f"model_key={row['model_key']!r}, expected {model!r}"
        )

    row["validation_status"] = (
        "OK"
        if not validation_errors
        else "; ".join(validation_errors)
    )

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []

    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]

    return {
        f"{field}_n": len(values),
        f"{field}_median": median(values),
        f"{field}_p05": percentile(values, 0.05),
        f"{field}_p95": percentile(values, 0.95),
        f"{field}_min": min(values),
        f"{field}_max": max(values),
    }


def aggregate_rows(
    run_rows: list[dict[str, Any]],
    grouping_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in run_rows:
        key = tuple(row[field] for field in grouping_fields)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []

    summary_fields = (
        "median_ttft_ms",
        "median_tpot_ms",
        "median_itl_ms",
        "median_e2el_ms",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "p95_itl_ms",
        "p95_e2el_ms",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "duration_s",
    )

    for key in sorted(grouped):
        rows = grouped[key]

        summary: dict[str, Any] = {
            field: value
            for field, value in zip(grouping_fields, key)
        }

        summary["files"] = len(rows)
        summary["completed_requests"] = sum(
            int(row["completed"]) for row in rows
        )
        summary["failed_requests"] = sum(
            int(row["failed"]) for row in rows
        )

        for field in summary_fields:
            summary.update(summarize_values(rows, field))

        summary_rows.append(summary)

    return summary_rows


def add_ratios_vs_offload_zero(
    rows: list[dict[str, Any]],
    context_fields: tuple[str, ...],
) -> None:
    """
    Add ratios relative to offload 0 within the same model/context.

    For the by-offload table, context_fields is only dataset_model.

    For output/concurrency tables, the comparison remains within the
    same output length or concurrency.
    """
    metrics = (
        "median_tpot_ms_median",
        "median_itl_ms_median",
        "median_ttft_ms_median",
        "median_e2el_ms_median",
        "output_throughput_median",
    )

    baseline_lookup: dict[
        tuple[Any, ...],
        dict[str, float],
    ] = {}

    for row in rows:
        if int(row["offload_gb"]) != 0:
            continue

        key = tuple(row[field] for field in context_fields)

        baseline_lookup[key] = {
            metric: float(row[metric])
            for metric in metrics
        }

    for row in rows:
        key = tuple(row[field] for field in context_fields)
        baseline = baseline_lookup.get(key)

        for metric in metrics:
            ratio_name = f"{metric}_ratio_vs_offload0"

            if baseline is None or baseline[metric] == 0:
                row[ratio_name] = None
            else:
                row[ratio_name] = (
                    float(row[metric])
                    / baseline[metric]
                )


def format_value(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "NA"

    return f"{float(value):.{decimals}f}"


def print_offload_table(
    model: str,
    rows: list[dict[str, Any]],
) -> None:
    selected = [
        row
        for row in rows
        if row["dataset_model"] == model
    ]

    print()
    print("=" * 120)
    print(f"MODEL: {model}")
    print("=" * 120)
    print(
        f"{'Offload':>7} "
        f"{'Files':>6} "
        f"{'TPOT ms':>12} "
        f"{'TPOT ×':>9} "
        f"{'ITL ms':>12} "
        f"{'TTFT ms':>12} "
        f"{'E2EL ms':>12} "
        f"{'Out tok/s':>12}"
    )
    print("-" * 120)

    for row in selected:
        print(
            f"{int(row['offload_gb']):>7} "
            f"{int(row['files']):>6} "
            f"{format_value(row['median_tpot_ms_median']):>12} "
            f"{format_value(
                row['median_tpot_ms_median_ratio_vs_offload0'],
                2,
            ):>9} "
            f"{format_value(row['median_itl_ms_median']):>12} "
            f"{format_value(row['median_ttft_ms_median']):>12} "
            f"{format_value(row['median_e2el_ms_median']):>12} "
            f"{format_value(row['output_throughput_median']):>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize profile_grid_v2 using exact JSON fields."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_argument,
        metavar="MODEL=PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_run_rows: list[dict[str, Any]] = []
    extraction_errors: list[str] = []

    for model, root in args.dataset:
        if not root.exists():
            print(
                f"ERROR: directory does not exist: {root}",
                file=sys.stderr,
            )
            return 1

        json_files = sorted(root.rglob("*.json"))

        print()
        print(f"Reading {model}: {len(json_files)} JSON files")

        for index, path in enumerate(json_files, start=1):
            try:
                row = read_run(model, path)
                all_run_rows.append(row)
            except Exception as exc:
                extraction_errors.append(
                    f"[{model}] {path}: {type(exc).__name__}: {exc}"
                )

            if index % 100 == 0:
                print(f"  processed {index}/{len(json_files)}")

    if extraction_errors:
        print()
        print("EXTRACTION ERRORS:")

        for error in extraction_errors[:30]:
            print(f"  {error}")

        if len(extraction_errors) > 30:
            print(
                f"  ... plus {len(extraction_errors) - 30} more"
            )

        return 1

    validation_failures = [
        row
        for row in all_run_rows
        if row["validation_status"] != "OK"
    ]

    run_csv = args.output_dir / "profile_run_metrics.csv"
    write_csv(run_csv, all_run_rows)

    by_offload = aggregate_rows(
        all_run_rows,
        ("dataset_model", "offload_gb"),
    )
    add_ratios_vs_offload_zero(
        by_offload,
        ("dataset_model",),
    )

    by_output = aggregate_rows(
        all_run_rows,
        (
            "dataset_model",
            "offload_gb",
            "output_len",
        ),
    )
    add_ratios_vs_offload_zero(
        by_output,
        (
            "dataset_model",
            "output_len",
        ),
    )

    by_concurrency = aggregate_rows(
        all_run_rows,
        (
            "dataset_model",
            "offload_gb",
            "concurrency",
        ),
    )
    add_ratios_vs_offload_zero(
        by_concurrency,
        (
            "dataset_model",
            "concurrency",
        ),
    )

    offload_csv = args.output_dir / "profile_by_offload.csv"
    output_csv = args.output_dir / "profile_by_offload_output.csv"
    concurrency_csv = (
        args.output_dir
        / "profile_by_offload_concurrency.csv"
    )

    write_csv(offload_csv, by_offload)
    write_csv(output_csv, by_output)
    write_csv(concurrency_csv, by_concurrency)

    models = sorted(
        {str(row["dataset_model"]) for row in all_run_rows}
    )

    for model in models:
        print_offload_table(model, by_offload)

    print()
    print("=" * 120)
    print("VALIDATION")
    print("=" * 120)
    print(f"Total files:               {len(all_run_rows)}")
    print(
        "Completed requests:        "
        f"{sum(int(row['completed']) for row in all_run_rows)}"
    )
    print(
        "Failed requests:           "
        f"{sum(int(row['failed']) for row in all_run_rows)}"
    )
    print(
        "Validation failures:       "
        f"{len(validation_failures)}"
    )

    for model in models:
        model_rows = [
            row
            for row in all_run_rows
            if row["dataset_model"] == model
        ]

        print()
        print(f"{model}:")
        print(f"  files:                   {len(model_rows)}")
        print(
            "  completed requests:      "
            f"{sum(int(row['completed']) for row in model_rows)}"
        )
        print(
            "  failed requests:         "
            f"{sum(int(row['failed']) for row in model_rows)}"
        )
        print(
            "  warmup values:           "
            f"{sorted({int(row['num_warmups']) for row in model_rows})}"
        )
        print(
            "  validation failures:     "
            f"{sum(
                row['validation_status'] != 'OK'
                for row in model_rows
            )}"
        )

    if validation_failures:
        print()
        print("First validation failures:")

        for row in validation_failures[:20]:
            print(
                f"  [{row['validation_status']}] "
                f"{row['path']}"
            )

    print()
    print("Generated files:")
    print(f"  {run_csv}")
    print(f"  {offload_csv}")
    print(f"  {output_csv}")
    print(f"  {concurrency_csv}")

    if validation_failures:
        print()
        print("FAIL: extraction succeeded, but validation failed.")
        return 1

    print()
    print(
        "PASS: exact metric extraction and aggregation completed."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
