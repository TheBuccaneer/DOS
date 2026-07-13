#!/usr/bin/env python3
"""
Extract and summarize profiling metrics for multiple models.

Per JSON file:
- parses offload, concurrency, output length and repeat from the path
- locates the request-result list
- ignores explicitly failed requests
- extracts TTFT, TPOT, ITL and E2EL
- calculates one median per run

Per model and offload level:
- median across run medians
- p05 and p95 across run medians
- ratio relative to offload 0

Expected invocation:

python3 summarize_profile_grids.py \
  --dataset llama=/path/to/llama \
  --dataset qwen=/path/to/qwen \
  --output-dir=/path/to/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Filename/path parsing
# ---------------------------------------------------------------------------

PATTERNS = {
    "offload_gb": (
        r"(?:cpu[_-]?)?offload(?:[_-]?gb)?[_=-]?(\d+)",
        r"(?:^|[/\\])(?:o|off)[_-]?(\d+)(?:[/\\]|$)",
    ),
    "concurrency": (
        r"(?:concurrency|conc|c)[_=-]?(\d+)",
    ),
    "output_len": (
        r"(?:output[_-]?(?:len|length)?|out(?:put)?|ol)[_=-]?(\d+)",
    ),
    "repeat": (
        r"(?:repeat|rep|run|r)[_=-]?(\d+)",
    ),
}


# ---------------------------------------------------------------------------
# Metric aliases
# ---------------------------------------------------------------------------

SCALAR_ALIASES = {
    "ttft_ms": (
        "ttft_ms",
        "time_to_first_token_ms",
        "first_token_latency_ms",
        "ttft",
        "time_to_first_token",
    ),
    "tpot_ms": (
        "tpot_ms",
        "time_per_output_token_ms",
        "time_per_token_ms",
        "mean_tpot_ms",
        "tpot",
        "time_per_output_token",
    ),
    "itl_ms": (
        "itl_mean_ms",
        "mean_itl_ms",
        "avg_itl_ms",
        "itl_median_ms",
        "median_itl_ms",
        "inter_token_latency_ms",
        "itl_ms",
        "itl",
    ),
    "e2el_ms": (
        "e2el_ms",
        "end_to_end_latency_ms",
        "end_to_end_ms",
        "request_latency_ms",
        "latency_ms",
        "total_latency_ms",
        "e2el",
        "end_to_end_latency",
    ),
}

ITL_SEQUENCE_ALIASES = (
    "itl_sequence_ms",
    "itls_ms",
    "inter_token_latencies_ms",
    "itl_values_ms",
    "itl_sequence",
    "itls",
    "inter_token_latencies",
)

OUTPUT_TOKEN_ALIASES = (
    "output_tokens",
    "num_output_tokens",
    "generated_tokens",
    "completion_tokens",
    "output_token_count",
)


@dataclass
class RunSummary:
    model: str
    path: str
    offload_gb: int
    concurrency: int
    output_len: int
    repeat: int

    request_count: int
    successful_requests: int
    failed_requests: int

    ttft_count: int
    tpot_count: int
    itl_count: int
    e2el_count: int

    median_ttft_ms: float | None
    median_tpot_ms: float | None
    median_itl_ms: float | None
    median_e2el_ms: float | None

    derived_tpot_count: int
    request_list_path: str


def parse_dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Use MODEL=PATH, for example llama=/data/llama"
        )

    model, path = value.split("=", 1)
    model = model.strip()
    path = path.strip()

    if not model or not path:
        raise argparse.ArgumentTypeError(
            "Both MODEL and PATH must be non-empty."
        )

    return model, Path(path).expanduser()


def extract_dimension(text: str, name: str) -> int | None:
    for pattern in PATTERNS[name]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])

    return None


def parse_dimensions(path: Path) -> tuple[int, int, int, int]:
    text = path.as_posix()

    values = {
        name: extract_dimension(text, name)
        for name in (
            "offload_gb",
            "concurrency",
            "output_len",
            "repeat",
        )
    }

    missing = [name for name, value in values.items() if value is None]

    if missing:
        raise ValueError(
            f"Could not parse {', '.join(missing)} from {path}"
        )

    return (
        int(values["offload_gb"]),
        int(values["concurrency"]),
        int(values["output_len"]),
        int(values["repeat"]),
    )


# ---------------------------------------------------------------------------
# Recursive JSON helpers
# ---------------------------------------------------------------------------

def iter_dict_lists(
    obj: Any,
    path: str = "$",
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """
    Yield JSON lists containing dictionaries.

    The runner's request list may be stored below keys such as:
    requests, results, measurements or request_results.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"

            if isinstance(value, list):
                dict_items = [
                    item for item in value if isinstance(item, dict)
                ]

                if dict_items:
                    yield child_path, dict_items

            yield from iter_dict_lists(value, child_path)

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from iter_dict_lists(value, f"{path}[{index}]")


def recursive_find_key(
    obj: Any,
    aliases: Iterable[str],
) -> tuple[str, Any] | None:
    wanted = {alias.lower() for alias in aliases}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                return str(key), value

        for value in obj.values():
            result = recursive_find_key(value, aliases)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = recursive_find_key(value, aliases)
            if result is not None:
                return result

    return None


def request_metric_score(request: dict[str, Any]) -> int:
    score = 0

    for aliases in SCALAR_ALIASES.values():
        if recursive_find_key(request, aliases) is not None:
            score += 1

    if recursive_find_key(request, ITL_SEQUENCE_ALIASES) is not None:
        score += 1

    if recursive_find_key(request, OUTPUT_TOKEN_ALIASES) is not None:
        score += 1

    return score


def find_request_list(
    data: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Pick the dictionary-list that most closely resembles request results.
    """
    candidates: list[
        tuple[int, int, str, list[dict[str, Any]]]
    ] = []

    for path, items in iter_dict_lists(data):
        metric_score = sum(
            request_metric_score(item)
            for item in items
        )

        candidates.append(
            (
                metric_score,
                len(items),
                path,
                items,
            )
        )

    if not candidates:
        raise ValueError("No list containing request dictionaries found.")

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    metric_score, _, path, items = candidates[0]

    if metric_score == 0:
        raise ValueError(
            "Dictionary lists were found, but none contained known metrics."
        )

    return path, items


# ---------------------------------------------------------------------------
# Metric conversion
# ---------------------------------------------------------------------------

def numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        value = float(value)

        if math.isfinite(value):
            return value

        return None

    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None

        if math.isfinite(parsed):
            return parsed

    return None


def convert_to_ms(
    key: str,
    value: Any,
    bare_unit: str,
) -> float | None:
    number = numeric_value(value)

    if number is None:
        return None

    lowered = key.lower()

    if (
        lowered.endswith("_ms")
        or "milliseconds" in lowered
        or "millisecond" in lowered
    ):
        return number

    if (
        lowered.endswith("_seconds")
        or lowered.endswith("_second")
        or lowered.endswith("_secs")
        or lowered.endswith("_sec")
        or lowered.endswith("_s")
    ):
        return number * 1000.0

    if bare_unit == "ms":
        return number

    if bare_unit == "s":
        return number * 1000.0

    # Automatic treatment of bare names such as "ttft" or "tpot".
    # LLM benchmark tools often store these values in seconds.
    if abs(number) < 10.0:
        return number * 1000.0

    return number


def extract_scalar_metric(
    request: dict[str, Any],
    metric: str,
    bare_unit: str,
) -> float | None:
    result = recursive_find_key(
        request,
        SCALAR_ALIASES[metric],
    )

    if result is None:
        return None

    key, value = result
    return convert_to_ms(key, value, bare_unit)


def extract_itl_from_sequence(
    request: dict[str, Any],
    bare_unit: str,
) -> float | None:
    result = recursive_find_key(
        request,
        ITL_SEQUENCE_ALIASES,
    )

    if result is None:
        return None

    key, values = result

    if not isinstance(values, list):
        return None

    converted = [
        converted_value
        for value in values
        if (
            converted_value := convert_to_ms(
                key,
                value,
                bare_unit,
            )
        )
        is not None
    ]

    if not converted:
        return None

    # One value per request:
    # average ITL within that request.
    return mean(converted)


def extract_output_tokens(
    request: dict[str, Any],
) -> int | None:
    result = recursive_find_key(
        request,
        OUTPUT_TOKEN_ALIASES,
    )

    if result is None:
        return None

    _, value = result
    number = numeric_value(value)

    if number is None or number < 0:
        return None

    return int(round(number))


def request_succeeded(request: dict[str, Any]) -> bool:
    success_result = recursive_find_key(
        request,
        ("success", "successful", "completed"),
    )

    if success_result is not None:
        _, value = success_result

        if isinstance(value, bool):
            return value

    status_result = recursive_find_key(
        request,
        ("status", "request_status"),
    )

    if status_result is not None:
        _, value = status_result
        status = str(value).strip().lower()

        if status in {
            "failed",
            "failure",
            "error",
            "timeout",
            "cancelled",
            "canceled",
        }:
            return False

    timeout_result = recursive_find_key(
        request,
        ("timeout", "timed_out"),
    )

    if timeout_result is not None:
        _, value = timeout_result

        if value is True:
            return False

    error_result = recursive_find_key(
        request,
        ("error", "exception", "error_message"),
    )

    if error_result is not None:
        _, value = error_result

        if value not in (None, "", False):
            return False

    return True


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    return float(median(values))


def summarize_json_file(
    model: str,
    path: Path,
    bare_unit: str,
) -> RunSummary:
    offload, concurrency, output_len, repeat = parse_dimensions(path)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    request_list_path, requests = find_request_list(data)

    ttft_values: list[float] = []
    tpot_values: list[float] = []
    itl_values: list[float] = []
    e2el_values: list[float] = []

    successful = 0
    failed = 0
    derived_tpot_count = 0

    for request in requests:
        if not request_succeeded(request):
            failed += 1
            continue

        successful += 1

        ttft = extract_scalar_metric(
            request,
            "ttft_ms",
            bare_unit,
        )
        tpot = extract_scalar_metric(
            request,
            "tpot_ms",
            bare_unit,
        )
        itl = extract_scalar_metric(
            request,
            "itl_ms",
            bare_unit,
        )
        e2el = extract_scalar_metric(
            request,
            "e2el_ms",
            bare_unit,
        )

        if itl is None:
            itl = extract_itl_from_sequence(
                request,
                bare_unit,
            )

        # Derive TPOT only if it was not explicitly stored.
        # Standard decode definition:
        # (E2EL - TTFT) / (output_tokens - 1)
        if tpot is None:
            output_tokens = extract_output_tokens(request)

            if (
                e2el is not None
                and ttft is not None
                and output_tokens is not None
                and output_tokens > 1
                and e2el >= ttft
            ):
                tpot = (
                    (e2el - ttft)
                    / (output_tokens - 1)
                )
                derived_tpot_count += 1

        if ttft is not None:
            ttft_values.append(ttft)

        if tpot is not None:
            tpot_values.append(tpot)

        if itl is not None:
            itl_values.append(itl)

        if e2el is not None:
            e2el_values.append(e2el)

    return RunSummary(
        model=model,
        path=str(path),
        offload_gb=offload,
        concurrency=concurrency,
        output_len=output_len,
        repeat=repeat,
        request_count=len(requests),
        successful_requests=successful,
        failed_requests=failed,
        ttft_count=len(ttft_values),
        tpot_count=len(tpot_values),
        itl_count=len(itl_values),
        e2el_count=len(e2el_values),
        median_ttft_ms=safe_median(ttft_values),
        median_tpot_ms=safe_median(tpot_values),
        median_itl_ms=safe_median(itl_values),
        median_e2el_ms=safe_median(e2el_values),
        derived_tpot_count=derived_tpot_count,
        request_list_path=request_list_path,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def percentile(
    values: list[float],
    probability: float,
) -> float | None:
    """
    Linear percentile, equivalent to NumPy's usual linear interpolation.
    """
    if not values:
        return None

    ordered = sorted(values)

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


def aggregate_metric(
    rows: list[RunSummary],
    attribute: str,
) -> dict[str, float | int | None]:
    values = [
        float(value)
        for row in rows
        if (value := getattr(row, attribute)) is not None
    ]

    return {
        "n_runs": len(values),
        "median": safe_median(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def build_offload_rows(
    run_rows: list[RunSummary],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int],
        list[RunSummary],
    ] = defaultdict(list)

    for row in run_rows:
        grouped[(row.model, row.offload_gb)].append(row)

    baseline: dict[
        tuple[str, str],
        float | None,
    ] = {}

    for model in sorted({row.model for row in run_rows}):
        zero_rows = grouped.get((model, 0), [])

        for metric_name, attribute in (
            ("ttft", "median_ttft_ms"),
            ("tpot", "median_tpot_ms"),
            ("itl", "median_itl_ms"),
            ("e2el", "median_e2el_ms"),
        ):
            baseline[(model, metric_name)] = aggregate_metric(
                zero_rows,
                attribute,
            )["median"]

    output_rows: list[dict[str, Any]] = []

    for model, offload in sorted(grouped):
        rows = grouped[(model, offload)]

        result: dict[str, Any] = {
            "model": model,
            "offload_gb": offload,
            "files": len(rows),
            "successful_requests": sum(
                row.successful_requests for row in rows
            ),
            "failed_requests": sum(
                row.failed_requests for row in rows
            ),
        }

        for metric_name, attribute in (
            ("ttft", "median_ttft_ms"),
            ("tpot", "median_tpot_ms"),
            ("itl", "median_itl_ms"),
            ("e2el", "median_e2el_ms"),
        ):
            summary = aggregate_metric(rows, attribute)

            result[f"{metric_name}_n_runs"] = summary["n_runs"]
            result[f"{metric_name}_median_ms"] = summary["median"]
            result[f"{metric_name}_p05_ms"] = summary["p05"]
            result[f"{metric_name}_p95_ms"] = summary["p95"]
            result[f"{metric_name}_min_ms"] = summary["min"]
            result[f"{metric_name}_max_ms"] = summary["max"]

            base = baseline[(model, metric_name)]
            current = summary["median"]

            if (
                base is not None
                and current is not None
                and base != 0
            ):
                result[f"{metric_name}_ratio_vs_offload0"] = (
                    current / base
                )
            else:
                result[f"{metric_name}_ratio_vs_offload0"] = None

        output_rows.append(result)

    return output_rows


# ---------------------------------------------------------------------------
# CSV and console output
# ---------------------------------------------------------------------------

def write_dict_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows available for {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_dataclass_csv(
    path: Path,
    rows: list[RunSummary],
) -> None:
    dictionaries = [asdict(row) for row in rows]
    write_dict_csv(path, dictionaries)


def format_number(
    value: Any,
    decimals: int = 3,
) -> str:
    if value is None:
        return "NA"

    if isinstance(value, int):
        return str(value)

    return f"{float(value):.{decimals}f}"


def print_offload_table(
    model: str,
    rows: list[dict[str, Any]],
) -> None:
    model_rows = [
        row for row in rows if row["model"] == model
    ]

    print()
    print("=" * 105)
    print(f"MODEL: {model}")
    print("=" * 105)
    print(
        f"{'Offload':>7}  "
        f"{'Runs':>5}  "
        f"{'TPOT ms':>12}  "
        f"{'TPOT ratio':>11}  "
        f"{'ITL ms':>12}  "
        f"{'TTFT ms':>12}  "
        f"{'E2EL ms':>12}"
    )
    print("-" * 105)

    for row in model_rows:
        print(
            f"{row['offload_gb']:>7}  "
            f"{row['files']:>5}  "
            f"{format_number(row['tpot_median_ms']):>12}  "
            f"{format_number(row['tpot_ratio_vs_offload0'], 2):>11}  "
            f"{format_number(row['itl_median_ms']):>12}  "
            f"{format_number(row['ttft_median_ms']):>12}  "
            f"{format_number(row['e2el_median_ms']):>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract TPOT, ITL, TTFT and E2EL from profiling grids."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_argument,
        metavar="MODEL=PATH",
        help="Repeat once per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--bare-unit",
        choices=("auto", "ms", "s"),
        default="auto",
        help=(
            "Unit for metric keys without an explicit suffix. "
            "Default: auto."
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[RunSummary] = []
    errors: list[tuple[str, Path, str]] = []

    for model, root in args.dataset:
        if not root.exists():
            print(
                f"ERROR: Dataset directory does not exist: {root}",
                file=sys.stderr,
            )
            return 1

        json_files = sorted(root.rglob("*.json"))

        print()
        print(f"Reading {model}: {len(json_files)} JSON files")

        for index, path in enumerate(json_files, start=1):
            try:
                row = summarize_json_file(
                    model=model,
                    path=path,
                    bare_unit=args.bare_unit,
                )
                run_rows.append(row)
            except Exception as exc:
                errors.append(
                    (
                        model,
                        path,
                        f"{type(exc).__name__}: {exc}",
                    )
                )

            if index % 100 == 0:
                print(f"  processed {index}/{len(json_files)}")

    if errors:
        print()
        print("Extraction errors:")
        for model, path, error in errors[:30]:
            print(f"  [{model}] {path}")
            print(f"    {error}")

        if len(errors) > 30:
            print(f"  ... plus {len(errors) - 30} more errors")

        return 1

    run_csv = args.output_dir / "profile_run_metrics.csv"
    write_dataclass_csv(run_csv, run_rows)

    offload_rows = build_offload_rows(run_rows)
    offload_csv = args.output_dir / "profile_by_offload.csv"
    write_dict_csv(offload_csv, offload_rows)

    for model in sorted({row.model for row in run_rows}):
        print_offload_table(model, offload_rows)

    print()
    print("=" * 105)
    print(f"Run-level CSV: {run_csv}")
    print(f"Offload CSV:   {offload_csv}")
    print(f"Total runs:    {len(run_rows)}")
    print(
        "Total successful requests:",
        sum(row.successful_requests for row in run_rows),
    )
    print(
        "Total failed requests:    ",
        sum(row.failed_requests for row in run_rows),
    )

    missing_tpot = sum(
        row.median_tpot_ms is None for row in run_rows
    )
    missing_itl = sum(
        row.median_itl_ms is None for row in run_rows
    )
    missing_ttft = sum(
        row.median_ttft_ms is None for row in run_rows
    )
    missing_e2el = sum(
        row.median_e2el_ms is None for row in run_rows
    )

    print()
    print("Runs without extracted metric:")
    print(f"  TPOT: {missing_tpot}")
    print(f"  ITL:  {missing_itl}")
    print(f"  TTFT: {missing_ttft}")
    print(f"  E2EL: {missing_e2el}")

    print()
    print("PASS: metric extraction and offload summary completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
