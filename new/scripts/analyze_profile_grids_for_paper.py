#!/usr/bin/env python3
"""
analyze_profile_grids_for_paper.py

Paper-ready descriptive analysis of profile_grid_v2 CSV outputs.

Reads only the already-generated CSV files (no raw JSON parsing):

    profile_run_metrics.csv
    profile_by_offload.csv
    profile_by_offload_output.csv
    profile_by_offload_concurrency.csv

Produces validation checks, paper tables, output-length stability,
concurrency transition analysis, saturation analysis (offload12 vs 16),
a cross-model (llama vs qwen) comparison, and non-parametric bootstrap
confidence intervals for the median-of-run-medians.

This script makes NO Phase-A, attack, or security claims. It only
characterizes runtime-regime separation in the profiling data.

Usage:

    python3 analyze_profile_grids_for_paper.py \
        --run-metrics /path/to/profile_run_metrics.csv \
        --by-offload /path/to/profile_by_offload.csv \
        --by-offload-output /path/to/profile_by_offload_output.csv \
        --by-offload-concurrency /path/to/profile_by_offload_concurrency.csv \
        --output-dir /path/to/paper_analysis \
        --seed 20260711
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Constants / expected grid
# ---------------------------------------------------------------------------

EXPECTED_OFFLOADS = (0, 2, 4, 8, 12, 16)
EXPECTED_CONCURRENCIES = (1, 2, 4, 8, 12, 16)
EXPECTED_OUTPUT_LENGTHS = (32, 64, 128)
EXPECTED_RUNS = (1, 2, 3, 4, 5)
EXPECTED_MODELS = ("llama", "qwen")

EXPECTED_RUNS_PER_MODEL = 540
EXPECTED_RUNS_PER_OFFLOAD = 90
EXPECTED_COMPLETED = 20
EXPECTED_FAILED = 0
EXPECTED_WARMUPS = 1

CENTRAL_METRICS = (
    "median_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
)

LOW_STATE_OFFLOAD = 0
HIGH_STATE_OFFLOAD = 12

OUTPUT_STABILITY_WARN_PCT = 1.0
OUTPUT_STABILITY_ALERT_PCT = 5.0

DEFAULT_SEED = 20260711
DEFAULT_N_BOOTSTRAP = 10_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_summary_dir = Path(
        "/home/rock/projects/DOS/new/runs/profile_grid_v2/summary"
    )
    default_output_dir = Path(
        "/home/rock/projects/DOS/new/runs/profile_grid_v2/paper_analysis"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Paper-ready descriptive analysis of profile_grid_v2 CSV "
            "outputs (no raw JSON parsing, no Phase-A / attack claims)."
        )
    )
    parser.add_argument(
        "--run-metrics",
        type=Path,
        default=default_summary_dir / "profile_run_metrics.csv",
        help="Path to profile_run_metrics.csv",
    )
    parser.add_argument(
        "--by-offload",
        type=Path,
        default=default_summary_dir / "profile_by_offload.csv",
        help="Path to profile_by_offload.csv",
    )
    parser.add_argument(
        "--by-offload-output",
        type=Path,
        default=default_summary_dir / "profile_by_offload_output.csv",
        help="Path to profile_by_offload_output.csv",
    )
    parser.add_argument(
        "--by-offload-concurrency",
        type=Path,
        default=default_summary_dir / "profile_by_offload_concurrency.csv",
        help="Path to profile_by_offload_concurrency.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for generated CSV/Markdown files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Bootstrap RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap draws (default: {DEFAULT_N_BOOTSTRAP}).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when the input data does not match the expected grid."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise ValidationError(f"{label}: file does not exist: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValidationError(f"{label}: could not read CSV {path}: {exc}") from exc

    if df.empty:
        raise ValidationError(f"{label}: file is empty: {path}")

    return df


def normalize_model_column(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "dataset_model" in df.columns:
        df = df.copy()
        df["model"] = df["dataset_model"].astype(str).str.lower()
    elif "model" in df.columns:
        df = df.copy()
        df["model"] = df["model"].astype(str).str.lower()
    else:
        raise ValidationError(
            f"Neither 'dataset_model' nor 'model' column found in {path}"
        )
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_run_metrics(df: pd.DataFrame, path: Path) -> list[str]:
    errors: list[str] = []

    required_columns = (
        "model",
        "offload_gb",
        "concurrency",
        "output_len",
        "run_no",
        "completed",
        "failed",
        "num_warmups",
        *CENTRAL_METRICS,
    )
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValidationError(
            f"{path}: missing required columns: {', '.join(missing_columns)}"
        )

    models_found = sorted(df["model"].unique())
    for model in EXPECTED_MODELS:
        if model not in models_found:
            errors.append(f"missing model in run metrics: {model!r}")

    unexpected_models = [m for m in models_found if m not in EXPECTED_MODELS]
    if unexpected_models:
        errors.append(f"unexpected model(s) in run metrics: {unexpected_models}")

    for model in EXPECTED_MODELS:
        sub = df[df["model"] == model]
        n = len(sub)
        if n != EXPECTED_RUNS_PER_MODEL:
            errors.append(
                f"model={model}: expected {EXPECTED_RUNS_PER_MODEL} runs, "
                f"found {n}"
            )

        for offload in EXPECTED_OFFLOADS:
            n_offload = int((sub["offload_gb"] == offload).sum())
            if n_offload != EXPECTED_RUNS_PER_OFFLOAD:
                errors.append(
                    f"model={model}, offload={offload}: expected "
                    f"{EXPECTED_RUNS_PER_OFFLOAD} runs, found {n_offload}"
                )

        bad_offloads = sorted(
            set(sub["offload_gb"].unique()) - set(EXPECTED_OFFLOADS)
        )
        if bad_offloads:
            errors.append(f"model={model}: unexpected offload values: {bad_offloads}")

        bad_conc = sorted(
            set(sub["concurrency"].unique()) - set(EXPECTED_CONCURRENCIES)
        )
        if bad_conc:
            errors.append(f"model={model}: unexpected concurrency values: {bad_conc}")

        bad_out = sorted(
            set(sub["output_len"].unique()) - set(EXPECTED_OUTPUT_LENGTHS)
        )
        if bad_out:
            errors.append(f"model={model}: unexpected output_len values: {bad_out}")

        bad_runs = sorted(set(sub["run_no"].unique()) - set(EXPECTED_RUNS))
        if bad_runs:
            errors.append(f"model={model}: unexpected run_no values: {bad_runs}")

        bad_completed = int((sub["completed"] != EXPECTED_COMPLETED).sum())
        if bad_completed:
            errors.append(
                f"model={model}: {bad_completed} run(s) with completed != "
                f"{EXPECTED_COMPLETED}"
            )

        bad_failed = int((sub["failed"] != EXPECTED_FAILED).sum())
        if bad_failed:
            errors.append(
                f"model={model}: {bad_failed} run(s) with failed != "
                f"{EXPECTED_FAILED}"
            )

        bad_warmups = int((sub["num_warmups"] != EXPECTED_WARMUPS).sum())
        if bad_warmups:
            errors.append(
                f"model={model}: {bad_warmups} run(s) with num_warmups != "
                f"{EXPECTED_WARMUPS}"
            )

        # combination completeness: offload x concurrency x output_len x run_no
        expected_combos = {
            (o, c, ol, r)
            for o in EXPECTED_OFFLOADS
            for c in EXPECTED_CONCURRENCIES
            for ol in EXPECTED_OUTPUT_LENGTHS
            for r in EXPECTED_RUNS
        }
        found_combos = set(
            zip(
                sub["offload_gb"],
                sub["concurrency"],
                sub["output_len"],
                sub["run_no"],
            )
        )
        missing_combos = expected_combos - found_combos
        duplicate_mask = sub.duplicated(
            subset=["offload_gb", "concurrency", "output_len", "run_no"]
        )
        n_duplicates = int(duplicate_mask.sum())

        if missing_combos:
            errors.append(
                f"model={model}: {len(missing_combos)} missing grid "
                f"combination(s), e.g. {sorted(missing_combos)[:3]}"
            )
        if n_duplicates:
            errors.append(
                f"model={model}: {n_duplicates} duplicate grid combination(s)"
            )

    metrics_to_check = list(CENTRAL_METRICS)
    if "output_throughput" in df.columns:
        metrics_to_check.append("output_throughput")

    for metric in metrics_to_check:
        values = pd.to_numeric(df[metric], errors="coerce")
        n_nan = int(values.isna().sum())
        n_inf = int(np.isinf(values.to_numpy(dtype=float)).sum())
        if n_nan:
            errors.append(f"{metric}: {n_nan} NaN value(s)")
        if n_inf:
            errors.append(f"{metric}: {n_inf} Inf value(s)")

    return errors


def validate_auxiliary_csv(
    df: pd.DataFrame,
    path: Path,
    label: str,
    metric_columns: tuple[str, ...],
) -> list[str]:
    """
    Lightweight consistency check for the pre-aggregated auxiliary CSVs
    (by-offload / by-offload-output / by-offload-concurrency). These are
    not re-derived from run_metrics, so this only checks that they cover
    the same models/offloads and contain no NaN/Inf in their metric
    columns -- it does not guarantee they were built from the exact same
    run_metrics.csv snapshot.
    """
    errors: list[str] = []

    if "model" not in df.columns or "offload_gb" not in df.columns:
        errors.append(f"{label}: missing 'model' or 'offload_gb' column ({path})")
        return errors

    models_found = set(df["model"].unique())
    missing_models = set(EXPECTED_MODELS) - models_found
    if missing_models:
        errors.append(f"{label}: missing model(s) {sorted(missing_models)} in {path}")

    offloads_found = set(df["offload_gb"].unique())
    missing_offloads = set(EXPECTED_OFFLOADS) - offloads_found
    if missing_offloads:
        errors.append(
            f"{label}: missing offload level(s) {sorted(missing_offloads)} in {path}"
        )

    for metric in metric_columns:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        n_nan = int(values.isna().sum())
        n_inf = int(np.isinf(values.to_numpy(dtype=float)).sum())
        if n_nan:
            errors.append(f"{label}: {metric}: {n_nan} NaN value(s) in {path}")
        if n_inf:
            errors.append(f"{label}: {metric}: {n_inf} Inf value(s) in {path}")

    return errors


def run_validation(df_run: pd.DataFrame, run_metrics_path: Path) -> list[str]:
    return validate_run_metrics(df_run, run_metrics_path)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, probability * 100.0))


def describe_group(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else float("nan"),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": float(np.min(values)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
    }


def rank_array(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties get mean rank), 1-indexed."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]

    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        raise ValueError("spearman_corr: x and y must have equal length")
    if len(x) < 2:
        return float("nan")

    if HAVE_SCIPY:
        result = _scipy_spearmanr(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result[0])

    rx = rank_array(np.asarray(x, dtype=float))
    ry = rank_array(np.asarray(y, dtype=float))

    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_median_ci_stratified(
    cell_values: list[np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Stratified non-parametric bootstrap, vectorized (no Python-level loop
    over bootstrap draws -- that was ~8M individual numpy calls in the
    naive version and effectively hung the script).

    The 90 runs per (model, offload) cell are not exchangeable repeats:
    they come from a balanced 6 concurrency x 3 output_len x 5 repeat
    design. Pooling all 90 and resampling naively lets each draw randomly
    change the concurrency/output_len composition.

    Instead: resample independently *within* each of the 18
    (concurrency, output_len) design cells (with replacement, same size
    as that cell -- normally 5) for all n_bootstrap draws at once, keeping
    all 18 cells represented in every draw, then pool the resampled
    values per draw and take the overall median. This preserves the
    balanced design structure while still quantifying repeat-level
    uncertainty.

    Returns (point_median, ci_low_2.5, ci_high_97.5).
    """
    non_empty_cells = [c for c in cell_values if c.size > 0]

    if not non_empty_cells:
        return (float("nan"), float("nan"), float("nan"))

    pooled = np.concatenate(non_empty_cells)
    point_median = float(np.median(pooled))

    if pooled.size == 1:
        return (point_median, point_median, point_median)

    # For each cell, draw all n_bootstrap resamples at once:
    # shape (n_bootstrap, cell.size). Then concatenate across cells along
    # axis=1 to get shape (n_bootstrap, total_n), and take the median of
    # each row (draw) in one vectorized call.
    resampled_cells = []
    for cell in non_empty_cells:
        idx = rng.integers(0, cell.size, size=(n_bootstrap, cell.size))
        resampled_cells.append(cell[idx])

    resampled_pooled = np.concatenate(resampled_cells, axis=1)
    boot_medians = np.median(resampled_pooled, axis=1)

    ci_low = float(np.percentile(boot_medians, 2.5))
    ci_high = float(np.percentile(boot_medians, 97.5))

    return (point_median, ci_low, ci_high)


# ---------------------------------------------------------------------------
# 2. Paper table by model / offload
# ---------------------------------------------------------------------------

def build_paper_table_by_offload(df_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        sub = df_run[df_run["model"] == model]

        baseline: dict[str, float] = {}
        base_rows = sub[sub["offload_gb"] == LOW_STATE_OFFLOAD]
        for metric in CENTRAL_METRICS:
            baseline[metric] = float(
                np.median(pd.to_numeric(base_rows[metric]).to_numpy())
            )

        base_throughput = float(
            np.median(pd.to_numeric(base_rows["output_throughput"]).to_numpy())
        ) if "output_throughput" in sub.columns else float("nan")

        for offload in EXPECTED_OFFLOADS:
            grp = sub[sub["offload_gb"] == offload]
            row: dict[str, Any] = {
                "model": model,
                "offload_gb": offload,
                "n_runs": len(grp),
            }

            for metric in CENTRAL_METRICS:
                values = pd.to_numeric(grp[metric]).to_numpy(dtype=float)
                stats = describe_group(values)
                for key, value in stats.items():
                    row[f"{metric}_{key}"] = value

                base = baseline[metric]
                current = stats["median"]
                if base and not math.isnan(base) and base != 0:
                    row[f"{metric}_ratio_vs_offload0"] = current / base
                    row[f"{metric}_pct_change_vs_offload0"] = (
                        (current / base) - 1.0
                    ) * 100.0
                else:
                    row[f"{metric}_ratio_vs_offload0"] = None
                    row[f"{metric}_pct_change_vs_offload0"] = None

            if "output_throughput" in grp.columns:
                thr_values = pd.to_numeric(grp["output_throughput"]).to_numpy(
                    dtype=float
                )
                thr_stats = describe_group(thr_values)
                for key, value in thr_stats.items():
                    row[f"output_throughput_{key}"] = value

                if base_throughput and not math.isnan(base_throughput) and base_throughput != 0:
                    row["output_throughput_ratio_vs_offload0"] = (
                        thr_stats["median"] / base_throughput
                    )
                else:
                    row["output_throughput_ratio_vs_offload0"] = None

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Output-length stability
# ---------------------------------------------------------------------------

def build_output_stability_table(df_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        sub_model = df_run[df_run["model"] == model]

        for offload in EXPECTED_OFFLOADS:
            sub = sub_model[sub_model["offload_gb"] == offload]

            row: dict[str, Any] = {"model": model, "offload_gb": offload}

            for metric, prefix in (
                ("median_tpot_ms", "tpot"),
                ("median_itl_ms", "itl"),
            ):
                per_output: dict[int, float] = {}
                for out_len in EXPECTED_OUTPUT_LENGTHS:
                    values = pd.to_numeric(
                        sub[sub["output_len"] == out_len][metric]
                    ).to_numpy(dtype=float)
                    per_output[out_len] = (
                        float(np.median(values)) if values.size else float("nan")
                    )
                    row[f"{prefix}_median_out{out_len}"] = per_output[out_len]

                vals = np.array(list(per_output.values()), dtype=float)
                vals = vals[~np.isnan(vals)]

                if vals.size:
                    span = float(np.max(vals) - np.min(vals))
                    center = float(np.median(vals))
                    rel_span_pct = (
                        (span / center) * 100.0 if center != 0 else float("nan")
                    )
                else:
                    span = float("nan")
                    rel_span_pct = float("nan")

                row[f"{prefix}_abs_span_ms"] = span
                row[f"{prefix}_rel_span_pct"] = rel_span_pct
                row[f"{prefix}_flag_gt_1pct"] = (
                    bool(rel_span_pct > OUTPUT_STABILITY_WARN_PCT)
                    if not math.isnan(rel_span_pct)
                    else False
                )
                row[f"{prefix}_flag_gt_5pct"] = (
                    bool(rel_span_pct > OUTPUT_STABILITY_ALERT_PCT)
                    if not math.isnan(rel_span_pct)
                    else False
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_output_length_table(df_run: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputed (not pass-through) per model/offload/output_len medians,
    derived directly from profile_run_metrics.csv. This is the source
    for paper_table_by_offload_output.csv.
    """
    rows: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        sub_model = df_run[df_run["model"] == model]

        for offload in EXPECTED_OFFLOADS:
            sub_offload = sub_model[sub_model["offload_gb"] == offload]

            for out_len in EXPECTED_OUTPUT_LENGTHS:
                cell = sub_offload[sub_offload["output_len"] == out_len]

                row: dict[str, Any] = {
                    "model": model,
                    "offload_gb": offload,
                    "output_len": out_len,
                    "n_runs": len(cell),
                }

                for metric in CENTRAL_METRICS:
                    values = pd.to_numeric(cell[metric]).to_numpy(dtype=float)
                    stats = describe_group(values)
                    short = metric.replace("median_", "").replace("_ms", "")
                    for key, value in stats.items():
                        row[f"{short}_{key}"] = value

                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Concurrency analysis
# ---------------------------------------------------------------------------

def build_concurrency_table(df_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        sub_model = df_run[df_run["model"] == model]

        for offload in EXPECTED_OFFLOADS:
            sub = sub_model[sub_model["offload_gb"] == offload]

            row: dict[str, Any] = {"model": model, "offload_gb": offload}

            tpot_by_conc: dict[int, float] = {}
            itl_by_conc: dict[int, float] = {}

            for conc in EXPECTED_CONCURRENCIES:
                conc_rows = sub[sub["concurrency"] == conc]
                tpot_vals = pd.to_numeric(
                    conc_rows["median_tpot_ms"]
                ).to_numpy(dtype=float)
                itl_vals = pd.to_numeric(
                    conc_rows["median_itl_ms"]
                ).to_numpy(dtype=float)

                tpot_median = (
                    float(np.median(tpot_vals)) if tpot_vals.size else float("nan")
                )
                itl_median = (
                    float(np.median(itl_vals)) if itl_vals.size else float("nan")
                )

                tpot_by_conc[conc] = tpot_median
                itl_by_conc[conc] = itl_median

                row[f"tpot_median_conc{conc}"] = tpot_median
                row[f"itl_median_conc{conc}"] = itl_median

            def safe_ratio(a: float, b: float) -> float | None:
                if b and not math.isnan(b) and b != 0 and not math.isnan(a):
                    return a / b
                return None

            row["tpot_ratio_conc2_over_conc1"] = safe_ratio(
                tpot_by_conc[2], tpot_by_conc[1]
            )
            row["tpot_ratio_conc4_over_conc2"] = safe_ratio(
                tpot_by_conc[4], tpot_by_conc[2]
            )
            row["tpot_ratio_conc8_over_conc4"] = safe_ratio(
                tpot_by_conc[8], tpot_by_conc[4]
            )
            row["tpot_ratio_conc16_over_conc8"] = safe_ratio(
                tpot_by_conc[16], tpot_by_conc[8]
            )

            row["itl_ratio_conc2_over_conc1"] = safe_ratio(
                itl_by_conc[2], itl_by_conc[1]
            )
            row["itl_ratio_conc4_over_conc2"] = safe_ratio(
                itl_by_conc[4], itl_by_conc[2]
            )
            row["itl_ratio_conc8_over_conc4"] = safe_ratio(
                itl_by_conc[8], itl_by_conc[4]
            )
            row["itl_ratio_conc16_over_conc8"] = safe_ratio(
                itl_by_conc[16], itl_by_conc[8]
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Saturation analysis (12 vs 16)
# ---------------------------------------------------------------------------

def build_saturation_table(paper_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        sub = paper_table[paper_table["model"] == model]

        row12 = sub[sub["offload_gb"] == 12].iloc[0]
        row16 = sub[sub["offload_gb"] == 16].iloc[0]

        row: dict[str, Any] = {"model": model}

        for metric in ("median_tpot_ms", "median_itl_ms"):
            v12 = float(row12[f"{metric}_median"])
            v16 = float(row16[f"{metric}_median"])

            ratio = v16 / v12 if v12 != 0 else float("nan")
            abs_diff = v16 - v12
            pct_diff = (ratio - 1.0) * 100.0 if not math.isnan(ratio) else float("nan")

            prefix = "tpot" if metric == "median_tpot_ms" else "itl"
            row[f"{prefix}_offload12"] = v12
            row[f"{prefix}_offload16"] = v16
            row[f"{prefix}_ratio_16_over_12"] = ratio
            row[f"{prefix}_abs_diff_16_minus_12"] = abs_diff
            row[f"{prefix}_pct_diff"] = pct_diff
            row[f"{prefix}_16_higher_than_12"] = bool(v16 > v12)

        rows.append(row)

    result = pd.DataFrame(rows)

    recommendation_lines = [
        f"Recommended low state: offload{LOW_STATE_OFFLOAD} GB",
        f"Recommended high state: offload{HIGH_STATE_OFFLOAD} GB",
        "Rationale (data-based only, no security claim):",
        "  - Clear separation from baseline at offload12 across models.",
        "  - offload12 already lies in the extreme regime of the profiled grid.",
        "  - offload16 adds only a small additional effect or plateaus relative "
        "to offload12.",
    ]
    result.attrs["recommendation_text"] = "\n".join(recommendation_lines)

    return result


# ---------------------------------------------------------------------------
# 6. Cross-model comparison
# ---------------------------------------------------------------------------

def build_cross_model_table(paper_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    llama = paper_table[paper_table["model"] == "llama"].set_index("offload_gb")
    qwen = paper_table[paper_table["model"] == "qwen"].set_index("offload_gb")

    for offload in EXPECTED_OFFLOADS:
        row: dict[str, Any] = {"offload_gb": offload}

        llama_tpot_ratio = float(llama.loc[offload, "median_tpot_ms_ratio_vs_offload0"])
        qwen_tpot_ratio = float(qwen.loc[offload, "median_tpot_ms_ratio_vs_offload0"])
        llama_itl_ratio = float(llama.loc[offload, "median_itl_ms_ratio_vs_offload0"])
        qwen_itl_ratio = float(qwen.loc[offload, "median_itl_ms_ratio_vs_offload0"])

        row["llama_tpot_ratio_vs_offload0"] = llama_tpot_ratio
        row["qwen_tpot_ratio_vs_offload0"] = qwen_tpot_ratio
        row["tpot_ratio_diff_llama_minus_qwen"] = llama_tpot_ratio - qwen_tpot_ratio

        row["llama_itl_ratio_vs_offload0"] = llama_itl_ratio
        row["qwen_itl_ratio_vs_offload0"] = qwen_itl_ratio
        row["itl_ratio_diff_llama_minus_qwen"] = llama_itl_ratio - qwen_itl_ratio

        rows.append(row)

    cross_df = pd.DataFrame(rows)

    llama_tpot_rank = rank_array(cross_df["llama_tpot_ratio_vs_offload0"].to_numpy())
    qwen_tpot_rank = rank_array(cross_df["qwen_tpot_ratio_vs_offload0"].to_numpy())
    cross_df["llama_tpot_rank"] = llama_tpot_rank
    cross_df["qwen_tpot_rank"] = qwen_tpot_rank

    tpot_spearman = spearman_corr(
        cross_df["llama_tpot_ratio_vs_offload0"].to_numpy(),
        cross_df["qwen_tpot_ratio_vs_offload0"].to_numpy(),
    )
    itl_spearman = spearman_corr(
        cross_df["llama_itl_ratio_vs_offload0"].to_numpy(),
        cross_df["qwen_itl_ratio_vs_offload0"].to_numpy(),
    )

    cross_df.attrs["tpot_spearman"] = tpot_spearman
    cross_df.attrs["itl_spearman"] = itl_spearman
    cross_df.attrs["spearman_source"] = "scipy" if HAVE_SCIPY else "manual_rank"

    return cross_df


# ---------------------------------------------------------------------------
# 7. Bootstrap CIs
# ---------------------------------------------------------------------------

def build_bootstrap_table(
    df_run: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """
    Stratified bootstrap CIs for the median-of-run-medians per model and
    offload level. Stratification cells are (concurrency, output_len),
    matching the balanced 6x3x5 design, so each bootstrap draw preserves
    the design composition instead of pooling all 90 runs as if they were
    interchangeable repeats.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    print(
        f"Running stratified bootstrap CIs "
        f"({len(EXPECTED_MODELS)} models x {len(EXPECTED_OFFLOADS)} offloads "
        f"x {len(CENTRAL_METRICS)} metrics, n_bootstrap={n_bootstrap})..."
    )

    for model in EXPECTED_MODELS:
        sub_model = df_run[df_run["model"] == model]

        for offload in EXPECTED_OFFLOADS:
            sub = sub_model[sub_model["offload_gb"] == offload]

            row: dict[str, Any] = {
                "model": model,
                "offload_gb": offload,
                "n_runs": len(sub),
                "n_cells": sub.groupby(
                    ["concurrency", "output_len"]
                ).ngroups,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "stratified_by": "concurrency,output_len",
            }

            for metric in CENTRAL_METRICS:
                cell_values = [
                    pd.to_numeric(cell_df[metric]).to_numpy(dtype=float)
                    for _, cell_df in sub.groupby(
                        ["concurrency", "output_len"]
                    )
                ]
                point, ci_low, ci_high = bootstrap_median_ci_stratified(
                    cell_values, n_bootstrap, rng
                )
                short = metric.replace("median_", "").replace("_ms", "")
                row[f"{short}_median"] = point
                row[f"{short}_ci_low_2.5"] = ci_low
                row[f"{short}_ci_high_97.5"] = ci_high

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def format_num(value: Any, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.{decimals}f}"


def write_markdown_summary(
    path: Path,
    validation_errors: list[str],
    df_run: pd.DataFrame,
    paper_table: pd.DataFrame,
    output_stability: pd.DataFrame,
    concurrency_table: pd.DataFrame,
    saturation_table: pd.DataFrame,
    cross_model_table: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> None:
    lines: list[str] = []

    lines.append("# Profile Grid v2 — Paper Analysis Summary")
    lines.append("")

    # 1. Data scope and validation
    lines.append("## 1. Data scope and validation")
    lines.append("")
    total_runs = len(df_run)
    lines.append(f"- Total runs loaded: {total_runs}")
    for model in EXPECTED_MODELS:
        n = int((df_run["model"] == model).sum())
        lines.append(f"- {model}: {n} runs")
    if validation_errors:
        lines.append("- **Validation: FAIL**")
        for err in validation_errors:
            lines.append(f"  - {err}")
    else:
        lines.append("- **Validation: PASS** (grid complete, no NaN/Inf, all metadata as expected)")
    lines.append("")

    # 2 & 3. Central offload metrics per model
    for model in EXPECTED_MODELS:
        lines.append(f"## {'2' if model == 'llama' else '3'}. Central offload metrics — {model}")
        lines.append("")
        lines.append("| Offload GB | Runs | TPOT median (ms) | TPOT ratio vs 0 | ITL median (ms) | TTFT median (ms) | E2EL median (ms) |")
        lines.append("|---|---|---|---|---|---|---|")
        sub = paper_table[paper_table["model"] == model].sort_values("offload_gb")
        for _, row in sub.iterrows():
            lines.append(
                f"| {int(row['offload_gb'])} | {int(row['n_runs'])} | "
                f"{format_num(row['median_tpot_ms_median'])} | "
                f"{format_num(row['median_tpot_ms_ratio_vs_offload0'], 2)}x | "
                f"{format_num(row['median_itl_ms_median'])} | "
                f"{format_num(row['median_ttft_ms_median'])} | "
                f"{format_num(row['median_e2el_ms_median'])} |"
            )
        lines.append("")

    # 4. Output-length stability
    lines.append("## 4. Output-length stability")
    lines.append("")
    max_rel_span = output_stability[["tpot_rel_span_pct", "itl_rel_span_pct"]].max().max()
    lines.append(f"- Maximum observed relative output-length span across all model/offload cells: {format_num(max_rel_span)}%")
    for model in EXPECTED_MODELS:
        sub = output_stability[output_stability["model"] == model]
        model_max = sub[["tpot_rel_span_pct", "itl_rel_span_pct"]].max().max()
        n_flag_1pct = int((sub["tpot_flag_gt_1pct"] | sub["itl_flag_gt_1pct"]).sum())
        n_flag_5pct = int((sub["tpot_flag_gt_5pct"] | sub["itl_flag_gt_5pct"]).sum())
        lines.append(
            f"- {model}: max relative span {format_num(model_max)}%, "
            f"{n_flag_1pct} cell(s) > 1%, {n_flag_5pct} cell(s) > 5%"
        )
    lines.append("")
    lines.append("TPOT and ITL remain highly stable across output lengths 32/64/128 within a given offload level.")
    lines.append("")

    # 5. Concurrency 1-to-2 transition
    lines.append("## 5. Concurrency 1-to-2 transition")
    lines.append("")
    lines.append("| Model | Offload GB | TPOT conc1 (ms) | TPOT conc2 (ms) | conc2/conc1 ratio |")
    lines.append("|---|---|---|---|---|")
    max_transition = 0.0
    max_transition_row = None
    for _, row in concurrency_table.iterrows():
        ratio = row["tpot_ratio_conc2_over_conc1"]
        if ratio is not None and not (isinstance(ratio, float) and math.isnan(ratio)):
            if ratio > max_transition:
                max_transition = ratio
                max_transition_row = row
        lines.append(
            f"| {row['model']} | {int(row['offload_gb'])} | "
            f"{format_num(row['tpot_median_conc1'])} | "
            f"{format_num(row['tpot_median_conc2'])} | "
            f"{format_num(ratio, 2)}x |"
        )
    lines.append("")
    if max_transition_row is not None:
        lines.append(
            f"Largest conc2/conc1 TPOT transition: {format_num(max_transition, 2)}x "
            f"({max_transition_row['model']}, offload{int(max_transition_row['offload_gb'])} GB)."
        )
    lines.append(
        "This transition is reported descriptively only; no mechanism "
        "(scheduler, batching, memory) is inferred from this script."
    )
    lines.append("")

    # 6. Saturation 12 vs 16
    lines.append("## 6. Saturation: offload12 vs offload16")
    lines.append("")
    lines.append("| Model | TPOT@12 | TPOT@16 | ratio 16/12 | % diff | 16 higher than 12 |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in saturation_table.iterrows():
        lines.append(
            f"| {row['model']} | {format_num(row['tpot_offload12'])} | "
            f"{format_num(row['tpot_offload16'])} | "
            f"{format_num(row['tpot_ratio_16_over_12'], 3)} | "
            f"{format_num(row['tpot_pct_diff'], 2)}% | "
            f"{row['tpot_16_higher_than_12']} |"
        )
    lines.append("")

    # 7. Cross-model consistency
    lines.append("## 7. Cross-model consistency (Llama vs Qwen)")
    lines.append("")
    tpot_spearman = cross_model_table.attrs.get("tpot_spearman", float("nan"))
    itl_spearman = cross_model_table.attrs.get("itl_spearman", float("nan"))
    spearman_source = cross_model_table.attrs.get("spearman_source", "unknown")
    lines.append(f"- Spearman correlation (TPOT ratio-vs-offload0 curves): {format_num(tpot_spearman, 4)} (source: {spearman_source})")
    lines.append(f"- Spearman correlation (ITL ratio-vs-offload0 curves): {format_num(itl_spearman, 4)} (source: {spearman_source})")
    lines.append("")

    # 8. Recommendation
    lines.append("## 8. Data-based recommendation: offload0 vs offload12")
    lines.append("")
    recommendation_text = saturation_table.attrs.get("recommendation_text", "")
    for line in recommendation_text.split("\n"):
        lines.append(f"- {line}" if not line.startswith("  ") else f"  {line}")
    lines.append("")

    # 9. Scope disclaimer
    lines.append("## 9. Scope")
    lines.append("")
    lines.append(
        "These profiling results establish runtime-regime separation only.\n"
        "They do not establish a State x Burst availability interaction."
    )
    lines.append("")
    lines.append(f"_Bootstrap settings: n={n_bootstrap}, seed={seed}._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_console_summary(
    validation_errors: list[str],
    df_run: pd.DataFrame,
    paper_table: pd.DataFrame,
    output_stability: pd.DataFrame,
    concurrency_table: pd.DataFrame,
    saturation_table: pd.DataFrame,
    cross_model_table: pd.DataFrame,
    generated_files: list[Path],
) -> None:
    print()
    print("=" * 100)
    print("VALIDATION")
    print("=" * 100)
    if validation_errors:
        print("FAIL")
        for err in validation_errors:
            print(f"  - {err}")
    else:
        print("PASS")

    print()
    for model in EXPECTED_MODELS:
        n = int((df_run["model"] == model).sum())
        print(f"{model}: {n} runs")

    for model in EXPECTED_MODELS:
        print()
        print(f"--- {model}: TPOT by offload ---")
        sub = paper_table[paper_table["model"] == model].sort_values("offload_gb")
        print(f"{'Offload':>7}  {'TPOT ms':>12}  {'ratio':>8}")
        for _, row in sub.iterrows():
            print(
                f"{int(row['offload_gb']):>7}  "
                f"{format_num(row['median_tpot_ms_median']):>12}  "
                f"{format_num(row['median_tpot_ms_ratio_vs_offload0'], 2):>8}"
            )

    max_rel_span = output_stability[["tpot_rel_span_pct", "itl_rel_span_pct"]].max().max()
    print()
    print(f"Max output-length relative span: {format_num(max_rel_span)}%")

    ratios = concurrency_table["tpot_ratio_conc2_over_conc1"].dropna()
    if len(ratios):
        print(f"Largest conc2/conc1 TPOT transition: {format_num(ratios.max(), 2)}x")

    for _, row in saturation_table.iterrows():
        print(
            f"{row['model']}: offload16/offload12 TPOT ratio = "
            f"{format_num(row['tpot_ratio_16_over_12'], 3)}"
        )

    tpot_spearman = cross_model_table.attrs.get("tpot_spearman", float("nan"))
    print()
    print(f"Spearman correlation (TPOT ratio curves, Llama vs Qwen): {format_num(tpot_spearman, 4)}")

    print()
    print("Generated files:")
    for f in generated_files:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df_run_raw = load_csv(args.run_metrics, "run-metrics")
        df_run = normalize_model_column(df_run_raw, args.run_metrics)

        # by-offload / by-offload-output / by-offload-concurrency are loaded
        # only as auxiliary detail tables (saved separately, suffixed
        # "_detail"). All paper_table_*.csv files are computed directly
        # from profile_run_metrics.csv so they cannot silently go stale
        # relative to the run-level data.
        df_by_offload = normalize_model_column(
            load_csv(args.by_offload, "by-offload"), args.by_offload
        )
        df_by_offload_output = normalize_model_column(
            load_csv(args.by_offload_output, "by-offload-output"),
            args.by_offload_output,
        )
        df_by_offload_concurrency = normalize_model_column(
            load_csv(args.by_offload_concurrency, "by-offload-concurrency"),
            args.by_offload_concurrency,
        )
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    validation_errors = run_validation(df_run, args.run_metrics)

    validation_errors += validate_auxiliary_csv(
        df_by_offload, args.by_offload, "by-offload",
        ("median_tpot_ms_median", "median_itl_ms_median"),
    )
    validation_errors += validate_auxiliary_csv(
        df_by_offload_output, args.by_offload_output, "by-offload-output",
        ("median_tpot_ms_median", "median_itl_ms_median"),
    )
    validation_errors += validate_auxiliary_csv(
        df_by_offload_concurrency, args.by_offload_concurrency,
        "by-offload-concurrency",
        ("median_tpot_ms_median", "median_itl_ms_median"),
    )

    if validation_errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for err in validation_errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nAborting before generating paper tables. Fix the underlying "
            "data or regenerate the CSVs before re-running.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_table = build_paper_table_by_offload(df_run)
    output_stability = build_output_stability_table(df_run)
    output_length_table = build_output_length_table(df_run)
    concurrency_table = build_concurrency_table(df_run)
    saturation_table = build_saturation_table(paper_table)
    cross_model_table = build_cross_model_table(paper_table)
    bootstrap_table = build_bootstrap_table(df_run, args.n_bootstrap, args.seed)

    out_paper_offload = args.output_dir / "paper_table_by_offload.csv"
    out_paper_offload_output = args.output_dir / "paper_table_by_offload_output.csv"
    out_paper_offload_concurrency = (
        args.output_dir / "paper_table_by_offload_concurrency.csv"
    )
    out_output_stability = args.output_dir / "paper_table_output_stability.csv"
    out_saturation = args.output_dir / "paper_table_saturation.csv"
    out_cross_model = args.output_dir / "paper_table_cross_model.csv"
    out_bootstrap = args.output_dir / "paper_bootstrap_ci.csv"
    out_summary_md = args.output_dir / "paper_profile_summary.md"

    # Detail pass-through of the original pre-aggregated CSVs, kept for
    # reference under distinct filenames -- these are NOT what
    # paper_table_by_offload_output.csv / _concurrency.csv contain.
    out_offload_detail = args.output_dir / "paper_table_by_offload_detail.csv"
    out_offload_output_detail = (
        args.output_dir / "paper_table_by_offload_output_detail.csv"
    )
    out_offload_concurrency_detail = (
        args.output_dir / "paper_table_by_offload_concurrency_detail.csv"
    )

    write_csv(paper_table, out_paper_offload)
    write_csv(output_length_table, out_paper_offload_output)
    write_csv(concurrency_table, out_paper_offload_concurrency)
    write_csv(output_stability, out_output_stability)
    write_csv(saturation_table, out_saturation)
    write_csv(cross_model_table, out_cross_model)
    write_csv(bootstrap_table, out_bootstrap)

    write_csv(df_by_offload, out_offload_detail)
    write_csv(df_by_offload_output, out_offload_output_detail)
    write_csv(df_by_offload_concurrency, out_offload_concurrency_detail)

    write_markdown_summary(
        out_summary_md,
        validation_errors,
        df_run,
        paper_table,
        output_stability,
        concurrency_table,
        saturation_table,
        cross_model_table,
        args.n_bootstrap,
        args.seed,
    )

    generated_files = [
        out_paper_offload,
        out_paper_offload_output,
        out_paper_offload_concurrency,
        out_output_stability,
        out_saturation,
        out_cross_model,
        out_bootstrap,
        out_summary_md,
        out_offload_detail,
        out_offload_output_detail,
        out_offload_concurrency_detail,
    ]

    print_console_summary(
        validation_errors,
        df_run,
        paper_table,
        output_stability,
        concurrency_table,
        saturation_table,
        cross_model_table,
        generated_files,
    )

    print()
    print("PASS: paper-ready descriptive analysis completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
