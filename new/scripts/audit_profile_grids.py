#!/usr/bin/env python3
"""
Audit two profiling grids, e.g. Llama and Qwen.

Expected grid per model:
    offload_gb:  0, 2, 4, 8, 12, 16
    concurrency: 1, 2, 4, 8, 12, 16
    output_len:  32, 64, 128
    repeat/run:   1, 2, 3, 4, 5
    requests:     20 per JSON file

Expected totals per model:
    6 * 6 * 3 * 5 = 540 JSON files
    540 * 20 = 10,800 completed requests

The script parses dimensions from filenames and parent directories.
It supports names such as:

    offload_12_conc_4_out_64_run_3.json
    offload12_concurrency4_output64_repeat3.json

and directories such as:

    offload_12/
    offload12/
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_OFFLOADS = (0, 2, 4, 8, 12, 16)
EXPECTED_CONCURRENCIES = (1, 2, 4, 8, 12, 16)
EXPECTED_OUTPUT_LENGTHS = (32, 64, 128)
EXPECTED_REPEATS = (1, 2, 3, 4, 5)
EXPECTED_REQUESTS_PER_FILE = 20


@dataclass(frozen=True, order=True)
class Cell:
    offload_gb: int
    concurrency: int
    output_len: int
    repeat: int


@dataclass
class FileAudit:
    model: str
    path: str
    offload_gb: int | None
    concurrency: int | None
    output_len: int | None
    repeat: int | None
    json_valid: bool
    completed: int | None
    failed: int | None
    status: str
    error: str = ""


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


def parse_int(value: Any) -> int | None:
    """Convert ordinary integer-like values safely."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*", value)
        if match:
            return int(match.group(1))

    return None


def recursive_find(data: Any, keys: Iterable[str]) -> Any:
    """
    Find the first matching key recursively.

    Filename-derived values remain authoritative for matrix dimensions.
    This function is mainly used for request counts.
    """
    wanted = {key.lower() for key in keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in wanted:
                return value

        for value in data.values():
            found = recursive_find(value, keys)
            if found is not None:
                return found

    elif isinstance(data, list):
        for value in data:
            found = recursive_find(value, keys)
            if found is not None:
                return found

    return None


def extract_dimension(text: str, dimension: str) -> int | None:
    """Extract a matrix dimension from a path or filename."""
    for pattern in PATTERNS[dimension]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])

    return None


def extract_cell(path: Path) -> Cell | None:
    """
    Parse the experimental cell from the full path.

    The full path is used because offload is often encoded in a parent
    directory while concurrency/output/repeat are encoded in the filename.
    """
    text = path.as_posix()

    offload = extract_dimension(text, "offload_gb")
    concurrency = extract_dimension(text, "concurrency")
    output_len = extract_dimension(text, "output_len")
    repeat = extract_dimension(text, "repeat")

    if None in (offload, concurrency, output_len, repeat):
        return None

    return Cell(
        offload_gb=offload,
        concurrency=concurrency,
        output_len=output_len,
        repeat=repeat,
    )


def count_request_results(data: Any) -> tuple[int | None, int | None]:
    """
    Determine completed and failed request counts.

    It first checks explicit summary fields. If these are unavailable,
    it falls back to common request-list structures.
    """
    completed_value = recursive_find(
        data,
        (
            "completed",
            "completed_requests",
            "num_completed",
            "successful_requests",
            "success_count",
            "num_successful",
        ),
    )
    failed_value = recursive_find(
        data,
        (
            "failed",
            "failed_requests",
            "num_failed",
            "failure_count",
            "error_count",
            "num_errors",
        ),
    )

    completed = parse_int(completed_value)
    failed = parse_int(failed_value)

    if completed is not None or failed is not None:
        return completed, failed

    request_list = None

    if isinstance(data, list):
        request_list = data
    elif isinstance(data, dict):
        for key in (
            "requests",
            "results",
            "request_results",
            "measurements",
            "samples",
        ):
            candidate = data.get(key)
            if isinstance(candidate, list):
                request_list = candidate
                break

    if request_list is None:
        return None, None

    successful = 0
    unsuccessful = 0
    classified = 0

    for request in request_list:
        if not isinstance(request, dict):
            continue

        success = request.get("success")

        if isinstance(success, bool):
            classified += 1
            if success:
                successful += 1
            else:
                unsuccessful += 1
            continue

        status = str(request.get("status", "")).lower()
        error = request.get("error")
        timeout = request.get("timeout")

        if status in {"success", "completed", "ok"}:
            classified += 1
            successful += 1
        elif (
            status in {"failed", "error", "timeout"}
            or error not in (None, "", False)
            or timeout is True
        ):
            classified += 1
            unsuccessful += 1

    if classified:
        return successful, unsuccessful

    # The file contains a request list, but no explicit success flag.
    return len(request_list), 0


def expected_cells() -> set[Cell]:
    return {
        Cell(offload, concurrency, output_len, repeat)
        for offload, concurrency, output_len, repeat in itertools.product(
            EXPECTED_OFFLOADS,
            EXPECTED_CONCURRENCIES,
            EXPECTED_OUTPUT_LENGTHS,
            EXPECTED_REPEATS,
        )
    }


def audit_file(model: str, path: Path) -> tuple[FileAudit, Cell | None]:
    cell = extract_cell(path)

    dimensions = {
        "offload_gb": cell.offload_gb if cell else None,
        "concurrency": cell.concurrency if cell else None,
        "output_len": cell.output_len if cell else None,
        "repeat": cell.repeat if cell else None,
    }

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return (
            FileAudit(
                model=model,
                path=str(path),
                **dimensions,
                json_valid=False,
                completed=None,
                failed=None,
                status="INVALID_JSON",
                error=f"{type(exc).__name__}: {exc}",
            ),
            cell,
        )

    completed, failed = count_request_results(data)

    problems: list[str] = []

    if cell is None:
        problems.append("UNPARSEABLE_FILENAME")

    if completed is None:
        problems.append("COMPLETED_UNKNOWN")
    elif completed != EXPECTED_REQUESTS_PER_FILE:
        problems.append(f"COMPLETED_{completed}")

    if failed is None:
        problems.append("FAILED_UNKNOWN")
    elif failed != 0:
        problems.append(f"FAILED_{failed}")

    status = "OK" if not problems else ";".join(problems)

    return (
        FileAudit(
            model=model,
            path=str(path),
            **dimensions,
            json_valid=True,
            completed=completed,
            failed=failed,
            status=status,
        ),
        cell,
    )


def write_csv(path: Path, rows: list[FileAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "model",
        "path",
        "offload_gb",
        "concurrency",
        "output_len",
        "repeat",
        "json_valid",
        "completed",
        "failed",
        "status",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def audit_dataset(
    model: str,
    root: Path,
    output_dir: Path,
) -> bool:
    print()
    print("=" * 78)
    print(f"MODEL: {model}")
    print(f"ROOT:  {root}")
    print("=" * 78)

    if not root.exists():
        print("FAIL: Dataset directory does not exist.")
        return False

    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print("FAIL: No JSON files found.")
        return False

    audits: list[FileAudit] = []
    cell_to_paths: dict[Cell, list[Path]] = {}

    for path in json_files:
        audit, cell = audit_file(model, path)
        audits.append(audit)

        if cell is not None:
            cell_to_paths.setdefault(cell, []).append(path)

    expected = expected_cells()
    observed = set(cell_to_paths)

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = {
        cell: paths
        for cell, paths in cell_to_paths.items()
        if len(paths) > 1
    }

    invalid_json = [row for row in audits if not row.json_valid]
    unparseable = [
        row for row in audits
        if "UNPARSEABLE_FILENAME" in row.status
    ]
    incomplete = [
        row for row in audits
        if row.completed is not None
        and row.completed != EXPECTED_REQUESTS_PER_FILE
    ]
    failed_requests = [
        row for row in audits
        if row.failed is not None and row.failed != 0
    ]
    unknown_counts = [
        row for row in audits
        if row.completed is None or row.failed is None
    ]

    total_completed = sum(
        row.completed for row in audits if row.completed is not None
    )
    total_failed = sum(
        row.failed for row in audits if row.failed is not None
    )

    expected_file_count = len(expected)
    expected_request_count = (
        expected_file_count * EXPECTED_REQUESTS_PER_FILE
    )

    print(f"JSON files found:          {len(json_files)}")
    print(f"Expected JSON files:       {expected_file_count}")
    print(f"Parsed unique cells:       {len(observed)}")
    print(f"Missing cells:             {len(missing)}")
    print(f"Unexpected cells:          {len(unexpected)}")
    print(f"Duplicate cells:           {len(duplicates)}")
    print(f"Invalid JSON files:        {len(invalid_json)}")
    print(f"Unparseable filenames:     {len(unparseable)}")
    print(f"Incomplete files:          {len(incomplete)}")
    print(f"Files with failures:       {len(failed_requests)}")
    print(f"Unknown request counts:    {len(unknown_counts)}")
    print(f"Completed requests found:  {total_completed}")
    print(f"Expected completed:        {expected_request_count}")
    print(f"Failed requests found:     {total_failed}")

    csv_path = output_dir / f"{model}_profile_grid_audit.csv"
    write_csv(csv_path, audits)
    print(f"Audit CSV:                 {csv_path}")

    if missing:
        print("\nFirst missing cells:")
        for cell in missing[:20]:
            print(f"  {cell}")
        if len(missing) > 20:
            print(f"  ... plus {len(missing) - 20} more")

    if unexpected:
        print("\nUnexpected cells:")
        for cell in unexpected[:20]:
            print(f"  {cell}")

    if duplicates:
        print("\nDuplicate cells:")
        for cell, paths in list(duplicates.items())[:20]:
            print(f"  {cell}")
            for duplicate_path in paths:
                print(f"    {duplicate_path}")

    problematic_rows = [
        row for row in audits if row.status != "OK"
    ]

    if problematic_rows:
        print("\nFirst problematic files:")
        for row in problematic_rows[:20]:
            print(f"  [{row.status}] {row.path}")
            if row.error:
                print(f"    {row.error}")

    passed = all(
        (
            len(json_files) == expected_file_count,
            not missing,
            not unexpected,
            not duplicates,
            not invalid_json,
            not unparseable,
            not incomplete,
            not failed_requests,
            not unknown_counts,
            total_completed == expected_request_count,
            total_failed == 0,
        )
    )

    print()
    if passed:
        print(
            "PASS: complete matrix, valid JSON, no missing/duplicate cells, "
            "and no failed requests."
        )
    else:
        print("FAIL: one or more audit checks did not pass.")

    return passed


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Llama and Qwen profiling grids."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_argument,
        metavar="MODEL=PATH",
        help=(
            "Dataset name and root path. Repeat for each model, for example "
            "--dataset llama=/data/llama --dataset qwen=/data/qwen"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile_audit"),
        help="Directory for audit CSV files.",
    )

    args = parser.parse_args()

    all_passed = True

    for model, root in args.dataset:
        passed = audit_dataset(
            model=model,
            root=root,
            output_dir=args.output_dir,
        )
        all_passed = all_passed and passed

    print()
    print("=" * 78)
    print("OVERALL RESULT:", "PASS" if all_passed else "FAIL")
    print("=" * 78)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
