"""
raw_combined_defense_sweep_parallel.py  –  V3.2 (parallel)
============================================================
Offline defense simulation against the CPU-offload timing channel.
Parallel variant of raw_combined_defense_sweep.py V3.1 — same defenses,
same feature sets, same CV logic, same output schema. Only the execution
strategy changed: each (defense, param, jitter_rep) config is now an
independent task with its own deterministically-seeded RNG (see
_seed_for_task), dispatched across worker processes via joblib.Parallel
instead of executed sequentially with one shared, order-dependent RNG.

Usage
-----
python raw_combined_defense_sweep_parallel.py \\
    --raw-json-dir-llama ../porto/runs/2026-04-26_lama/base_runs \\
    --raw-json-dir-qwen  ../porto/runs/2026-06-01_Qwen \\
    --cv groupkfold \\
    --n-jobs 32

# --n-jobs 1 reproduces sequential (original-equivalent) execution.
# Default --n-jobs is cpu_count()-2.

Changes vs V3.1
----------------
1.  PRIMARY_FS = "all_primary" (ttft_ms, itl_mean/median/p95/p99, decode_time_ms).
    e2el_ms removed from primary because it is ttft+decode (redundant as a feature).
    e2el_only retained as an ablation feature set.

2.  Hard sanity checks in finalise_df(): raise ValueError on matrix mismatches
    (total rows, per-dataset, per-offload, per-concurrency cell counts, NaN/Inf).
    Strict=True for raw-JSON mode; strict=False (soft warnings) for CSV mode.

3.  Deterministic stochastic defenses: df_defended is built exactly ONCE per
    (defense, param, jitter_rep) and the SAME dataframe is used for both
    overhead computation and adaptive/frozen evaluation. No re-draws inside
    run_one_config() for non-fold-calibrated defenses.

4.  run_one_config() receives a pre-built df_defended for all non-fold-calibrated
    defenses. For fold-calibrated (calibrated_floor), df_defended=None and
    calibration happens per-fold without test leakage.

5.  calibrated_floor ITL target: quantile computed on FLATTENED per-token
    itl_seq_ms of high-state TRAIN rows (not per-request itl_median_ms quantile).

6.  oracle_low_equalize: only modifies Low-state (label==0) requests.
    High-state requests are left unchanged. Implemented in apply_oracle_low_df().

7.  Manifest extended: primary_featureset, rows_total,
    counts_per_dataset_offload_concurrency, script version V3.1.

Usage
-----
# raw JSONs (preferred – true per-token ITLs)
python raw_combined_defense_sweep.py \\
    --raw-json-dir-llama ../porto/runs/2026-04-26_lama/base_runs \\
    --raw-json-dir-qwen  ../porto/runs/2026-06-01_Qwen

# pre-extracted CSV (synthetic Gamma-reconstructed ITL sequences)
python raw_combined_defense_sweep.py \\
    --csv results/defense_raw/raw_requests_0_vs_12.csv

# smoke test
python raw_combined_defense_sweep.py --raw-json-dir-llama ... --raw-json-dir-qwen ... --smoke

# LOCO CV
python raw_combined_defense_sweep.py --raw-json-dir-llama ... --raw-json-dir-qwen ... --cv loco
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
RNG_GLOBAL = np.random.default_rng(SEED)

ZERO_OVERHEAD = {k: 0.0 for k in [
    "overhead_total_median_ms_all", "overhead_total_median_ms_low",
    "overhead_total_median_ms_high", "overhead_total_p95_ms_low",
    "overhead_total_p95_ms_high", "relative_overhead_total_low",
    "relative_overhead_total_high", "overhead_ttft_median_ms_low",
    "overhead_decode_median_ms_low",
]}


def _seed_for_task(base_seed: int, dataset_name: str, defense: str,
                   param, jrep: int) -> int:
    """Deterministic per-task seed, independent of execution order/worker count.

    The original script threaded ONE shared RNG sequentially through the
    entire sweep. That is unsafe to parallelise: forked/copied RNG state
    across processes causes correlated or duplicated jitter draws, silently
    reducing the effective number of independent repeats. Each task instead
    derives its own RNG purely from its own identity (dataset, defense,
    param, jitter_rep) via a stable hash, so results are bit-identical
    regardless of how many workers run or in what order tasks complete.
    """
    key = f"{dataset_name}|{defense}|{param}|{jrep}".encode("utf-8")
    h = zlib.crc32(key)
    return (base_seed * 1_000_003 + h) % (2**31 - 1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DOS_ROOT = HERE.parent.parent          # adjust if script moves
RESULTS_DIR = DOS_ROOT / "results" / "defense_raw"

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
FEATURE_SETS: dict[str, list[str]] = {
    "ttft_only":       ["ttft_ms"],
    "itl_median_only": ["itl_median_ms"],
    "itl_core":        ["itl_mean_ms", "itl_median_ms", "itl_p95_ms", "itl_p99_ms"],
    "decode_only":     ["decode_time_ms"],
    "e2el_only":       ["e2el_ms"],
    # Primary: does NOT include e2el_ms because e2el = ttft + decode (redundant).
    # e2el_only remains as an ablation.
    "all_primary":     ["ttft_ms", "itl_mean_ms", "itl_median_ms",
                        "itl_p95_ms", "itl_p99_ms", "decode_time_ms"],
}
PRIMARY_FS = "all_primary"

# NOTE: itl_min_ms is deliberately excluded (artefact values at sequence start).

# ---------------------------------------------------------------------------
# Defense parameter profiles
# ---------------------------------------------------------------------------
COMBINED_JITTER_PROFILES: list[tuple[float, float]] = [
    # (J_ttft_ms, J_itl_ms)
    (0,    0),
    (100,  5),
    (200,  10),
    (400,  25),
    (800,  50),
    (1600, 100),
    (3200, 200),
    (5000, 400),
    (8000, 800),
]

COMBINED_FLOOR_PROFILES: list[tuple[float, float]] = [
    # (target_ttft_ms, target_itl_ms)
    (100,  25),
    (200,  50),
    (400,  100),
    (800,  200),
    (1600, 400),
    (3200, 800),
    (5000, 1200),
]

# Single-component controls
TTFT_PAD_VALUES   = [25, 50, 100, 200, 400, 800, 1200, 1600]
TTFT_JITTER_VALUES = [25, 50, 100, 200, 400, 800, 1200, 1600]
TTFT_FLOOR_VALUES  = [50, 100, 200, 400, 800, 1200, 1600]
ITL_JITTER_VALUES  = [5, 10, 25, 50, 100, 200, 400, 800]
ITL_FLOOR_VALUES   = [5, 10, 25, 50, 100, 200, 400, 800]

# Calibrated floor quantiles
CALIB_QUANTILES = [0.25, 0.50, 0.75, 0.90]

# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifiers(rf_n_jobs: int = -1) -> dict:
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=SEED, n_jobs=rf_n_jobs, min_samples_leaf=2)),
        ]),
    }

# ---------------------------------------------------------------------------
# ITL sequence reconstruction from aggregates
# ---------------------------------------------------------------------------
def reconstruct_itl_sequence(itl_mean_ms: float, itl_std_ms: float,
                              itl_count: int, rng: np.random.Generator) -> np.ndarray:
    """
    Synthesise a plausible ITL sequence from aggregate statistics.
    Uses a Gamma distribution parameterised by mean and std.
    Falls back to constant sequence if std == 0 or mean <= 0.
    """
    if itl_count <= 0:
        return np.array([])
    if itl_mean_ms <= 0 or itl_std_ms <= 0:
        return np.full(itl_count, max(itl_mean_ms, 0.0))
    cv2  = (itl_std_ms / itl_mean_ms) ** 2
    if cv2 <= 0:
        return np.full(itl_count, itl_mean_ms)
    shape = 1.0 / cv2
    scale = itl_mean_ms * cv2
    seq   = rng.gamma(shape, scale, size=itl_count)
    return np.maximum(seq, 0.0)


# ---------------------------------------------------------------------------
# Feature derivation from raw timing
# ---------------------------------------------------------------------------
def derive_features(ttft_ms: float, itl_seq_ms: np.ndarray,
                    drop_first_itl: bool = False) -> dict:
    """Compute all aggregate features from defended timing values."""
    seq = itl_seq_ms[1:] if drop_first_itl and len(itl_seq_ms) > 1 else itl_seq_ms
    if len(seq) == 0:
        seq = np.array([0.0])
    decode_ms = float(np.sum(seq))
    return {
        "ttft_ms":      ttft_ms,
        "itl_mean_ms":  float(np.mean(seq)),
        "itl_median_ms":float(np.median(seq)),
        "itl_std_ms":   float(np.std(seq)),
        "itl_p95_ms":   float(np.percentile(seq, 95)),
        "itl_p99_ms":   float(np.percentile(seq, 99)),
        "decode_time_ms": decode_ms,
        "e2el_ms":      ttft_ms + decode_ms,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
PAT_JSON = re.compile(
    r"(?:qwen_)?offload(?P<offload>\d+)_conc(?P<conc>\d+)_run(?P<run>\d+)\.json$"
)

def load_raw_jsons(llama_root: Path, qwen_root: Path) -> pd.DataFrame:
    """Load from original benchmark JSON files (true per-token ITLs)."""
    roots = {"llama": llama_root, "qwen": qwen_root}
    rows = []
    for dataset, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(f"JSON root not found: {root}")
        for path in sorted(root.rglob("*.json")):
            m = PAT_JSON.search(path.name)
            if not m:
                continue
            offload = int(m.group("offload"))
            if offload not in (0, 12):
                continue
            conc   = int(m.group("conc"))
            run_id = int(m.group("run"))
            with path.open() as f:
                data = json.load(f)
            ttfts      = data.get("ttfts")
            itls_all   = data.get("itls")
            if ttfts is None or itls_all is None:
                print(f"  SKIP (missing ttfts/itls): {path}")
                continue
            for req_idx, (ttft_s, itl_seq_s) in enumerate(zip(ttfts, itls_all)):
                itl_ms  = np.asarray(itl_seq_s, dtype=float) * 1000.0
                ttft_ms = float(ttft_s) * 1000.0
                if len(itl_ms) == 0:
                    continue
                rows.append({
                    "dataset":        dataset,
                    "offload_gb":     offload,
                    "label":          int(offload == 12),
                    "run_concurrency":conc,
                    "run_id":         run_id,
                    "request_idx":    req_idx,
                    "ttft_ms":        ttft_ms,
                    "itl_seq_ms":     itl_ms,      # stored as object
                    "itl_count":      len(itl_ms),
                    "itl_mean_ms":    float(np.mean(itl_ms)),
                    "itl_median_ms":  float(np.median(itl_ms)),
                    "itl_std_ms":     float(np.std(itl_ms)),
                    "itl_p95_ms":     float(np.percentile(itl_ms, 95)),
                    "itl_p99_ms":     float(np.percentile(itl_ms, 99)),
                    "decode_time_ms": float(np.sum(itl_ms)),
                    "e2el_ms":        ttft_ms + float(np.sum(itl_ms)),
                    "has_true_itl_seq": True,
                })
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} requests from raw JSONs.")
    return df  # finalise_df(strict=True) called by caller


def load_csv(csv_path: Path, rng: np.random.Generator) -> pd.DataFrame:
    """Load from pre-extracted CSV; reconstruct synthetic ITL sequences."""
    df = pd.read_csv(csv_path)
    df = df[df["offload_gb"].isin([0, 12])].copy()
    df["label"]          = (df["offload_gb"] == 12).astype(int)
    df["has_true_itl_seq"] = False

    # Reconstruct ITL sequences
    seqs = []
    for _, row in df.iterrows():
        seq = reconstruct_itl_sequence(
            itl_mean_ms=row["itl_mean_ms"],
            itl_std_ms=row.get("itl_std_ms", row["itl_mean_ms"] * 0.1),
            itl_count=int(row["itl_count"]),
            rng=rng,
        )
        seqs.append(seq)
    df["itl_seq_ms"] = seqs
    print(f"  Loaded {len(df)} requests from CSV (synthetic ITL sequences).")
    return df  # finalise_df(strict=False) called by caller


EXPECTED_DATASETS    = {"llama", "qwen"}
EXPECTED_OFFLOADS    = {0, 12}
EXPECTED_CONCURRENCY = {1, 2, 4, 8, 12, 16}
EXPECTED_TOTAL       = 2400
EXPECTED_PER_DS      = 1200
EXPECTED_PER_DS_OFF  = 600
EXPECTED_PER_DS_OFF_CONC = 100


def _check_hard(condition: bool, msg: str) -> None:
    """Raise ValueError if condition is False."""
    if not condition:
        raise ValueError(f"[SANITY FAIL] {msg}")


def finalise_df(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Add group key, validate matrix, drop bad rows.

    Parameters
    ----------
    strict : if True (default), raise ValueError on any matrix mismatch.
             Set to False only for CSV mode where row count may differ slightly.
    """
    df = df.copy()
    df["group"] = (
        df["dataset"].astype(str) + "_o" +
        df["offload_gb"].astype(str) + "_c" +
        df["run_concurrency"].astype(str) + "_r" +
        df["run_id"].astype(str)
    )

    # ------------------------------------------------------------------
    # Hard sanity checks (raise on violation)
    # ------------------------------------------------------------------
    actual_ds  = set(df["dataset"].unique())
    actual_off = set(df["offload_gb"].unique())

    _check_hard(
        actual_ds <= EXPECTED_DATASETS,
        f"Unexpected datasets: {actual_ds - EXPECTED_DATASETS}",
    )
    _check_hard(
        actual_off <= EXPECTED_OFFLOADS,
        f"Unexpected offload_gb values: {actual_off - EXPECTED_OFFLOADS}",
    )

    if "run_concurrency" in df.columns:
        actual_conc = set(df["run_concurrency"].unique())
        _check_hard(
            actual_conc <= EXPECTED_CONCURRENCY,
            f"Unexpected concurrency values: {actual_conc - EXPECTED_CONCURRENCY}",
        )

    if strict:
        n = len(df)
        _check_hard(n == EXPECTED_TOTAL,
                    f"Expected {EXPECTED_TOTAL} total rows, got {n}.")

        for ds in EXPECTED_DATASETS:
            cnt_ds = len(df[df["dataset"] == ds])
            _check_hard(
                cnt_ds == EXPECTED_PER_DS,
                f"Expected {EXPECTED_PER_DS} rows for dataset={ds}, got {cnt_ds}.",
            )
            for off in EXPECTED_OFFLOADS:
                cnt_do = len(df[(df["dataset"] == ds) & (df["offload_gb"] == off)])
                _check_hard(
                    cnt_do == EXPECTED_PER_DS_OFF,
                    f"Expected {EXPECTED_PER_DS_OFF} rows for {ds}/offload={off}, got {cnt_do}.",
                )
                if "run_concurrency" in df.columns:
                    for conc in EXPECTED_CONCURRENCY:
                        cnt_doc = len(df[
                            (df["dataset"] == ds) &
                            (df["offload_gb"] == off) &
                            (df["run_concurrency"] == conc)
                        ])
                        _check_hard(
                            cnt_doc == EXPECTED_PER_DS_OFF_CONC,
                            f"Expected {EXPECTED_PER_DS_OFF_CONC} rows for "
                            f"{ds}/offload={off}/conc={conc}, got {cnt_doc}.",
                        )
    else:
        # CSV mode: softer checks
        n = len(df)
        if abs(n - EXPECTED_TOTAL) > 10:
            warnings.warn(f"Expected ~{EXPECTED_TOTAL} rows, got {n}.", stacklevel=2)
        for ds in EXPECTED_DATASETS:
            for off in EXPECTED_OFFLOADS:
                cnt = len(df[(df["dataset"] == ds) & (df["offload_gb"] == off)])
                if cnt == 0:
                    warnings.warn(
                        f"No rows for dataset={ds}, offload_gb={off}.", stacklevel=2)

    # ------------------------------------------------------------------
    # NaN/Inf check on all_primary features
    # ------------------------------------------------------------------
    primary_cols = FEATURE_SETS["all_primary"]
    feat_cols    = primary_cols + ["e2el_ms"]  # also clean e2el_ms
    for col in feat_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    nan_counts = {c: int(df[c].isna().sum())
                  for c in primary_cols if c in df.columns and df[c].isna().any()}
    if nan_counts:
        _check_hard(
            False,
            f"NaN/Inf in all_primary features after cleaning: {nan_counts}",
        )

    print(f"  [sanity] All checks passed. {len(df)} requests loaded.")
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Defense functions – operate on (ttft_ms, itl_seq_ms) → (ttft_def, itl_def)
# ---------------------------------------------------------------------------
Defense = Callable[[float, np.ndarray, np.random.Generator], tuple[float, np.ndarray]]


def def_none(ttft: float, itl: np.ndarray, rng: np.random.Generator
             ) -> tuple[float, np.ndarray]:
    return ttft, itl.copy()


# --- single-component controls ---

def make_ttft_const_pad(pad: float) -> Defense:
    def fn(ttft, itl, rng):
        return ttft + pad, itl.copy()
    return fn

def make_ttft_floor(target: float) -> Defense:
    def fn(ttft, itl, rng):
        return max(ttft, target), itl.copy()
    return fn

def make_ttft_jitter_nonneg(J: float) -> Defense:
    def fn(ttft, itl, rng):
        return ttft + float(rng.uniform(0, J)), itl.copy()
    return fn

def make_itl_jitter_nonneg(J: float) -> Defense:
    def fn(ttft, itl, rng):
        noise = rng.uniform(0, J, size=len(itl))
        return ttft, itl + noise
    return fn

def make_itl_floor(target: float) -> Defense:
    def fn(ttft, itl, rng):
        return ttft, np.maximum(itl, target)
    return fn


# --- combined defenses ---

def make_combined_jitter(J_ttft: float, J_itl: float) -> Defense:
    def fn(ttft, itl, rng):
        ttft_def = ttft + float(rng.uniform(0, J_ttft)) if J_ttft > 0 else ttft
        if J_itl > 0:
            noise    = rng.uniform(0, J_itl, size=len(itl))
            itl_def  = itl + noise
        else:
            itl_def  = itl.copy()
        return ttft_def, itl_def
    return fn

def make_combined_floor(target_ttft: float, target_itl: float) -> Defense:
    def fn(ttft, itl, rng):
        return max(ttft, target_ttft), np.maximum(itl, target_itl)
    return fn


def make_oracle_low_equalize(target_ttft: float, target_itl: float) -> Defense:
    """Oracle: only modifies Low-state (label=0) requests.
    High-state requests are left unchanged.
    The label must be threaded in via a wrapper; see apply_oracle_low_df().
    This function is a NOOP placeholder; the real logic is in apply_oracle_low_df().
    """
    def fn(ttft, itl, rng):
        # Placeholder — apply_oracle_low_df handles label-aware application.
        return ttft, itl.copy()
    return fn


# ---------------------------------------------------------------------------
# Apply defense to a full dataframe
# ---------------------------------------------------------------------------
def apply_defense_df(df: pd.DataFrame, defense_fn: Defense, rng: np.random.Generator,
                     drop_first_itl: bool = False) -> pd.DataFrame:
    """Apply defense_fn to every request, re-derive all features."""
    records = []
    for _, row in df.iterrows():
        ttft_def, itl_def = defense_fn(
            float(row["ttft_ms"]), np.asarray(row["itl_seq_ms"]), rng)
        feats = derive_features(ttft_def, itl_def, drop_first_itl=drop_first_itl)
        feats["itl_seq_ms"] = itl_def  # carry defended sequence forward
        records.append(feats)
    feat_df = pd.DataFrame(records)
    # Carry over non-feature columns
    carry = ["dataset", "offload_gb", "label", "run_concurrency", "run_id",
             "request_idx", "group", "itl_count", "has_true_itl_seq"]
    for col in carry:
        if col in df.columns:
            feat_df[col] = df[col].values
    return feat_df


def apply_oracle_low_df(df: pd.DataFrame, target_ttft: float, target_itl: float,
                        rng: np.random.Generator,
                        drop_first_itl: bool = False) -> pd.DataFrame:
    """Oracle defense: floor Low-state requests only; High-state unchanged.
    Low  (label==0): ttft_def = max(ttft, target_ttft),
                     itl_def  = max(itl_i, target_itl) per token
    High (label==1): unchanged
    This is a theoretical lower bound on overhead, NOT a deployment defense.
    """
    floor_fn = make_combined_floor(target_ttft, target_itl)
    records  = []
    for _, row in df.iterrows():
        if int(row["label"]) == 0:
            ttft_def, itl_def = floor_fn(
                float(row["ttft_ms"]), np.asarray(row["itl_seq_ms"]), rng)
        else:
            ttft_def = float(row["ttft_ms"])
            itl_def  = np.asarray(row["itl_seq_ms"]).copy()
        feats = derive_features(ttft_def, itl_def, drop_first_itl=drop_first_itl)
        feats["itl_seq_ms"] = itl_def
        records.append(feats)
    feat_df = pd.DataFrame(records)
    carry   = ["dataset", "offload_gb", "label", "run_concurrency", "run_id",
               "request_idx", "group", "itl_count", "has_true_itl_seq"]
    for col in carry:
        if col in df.columns:
            feat_df[col] = df[col].values
    return feat_df


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------
def compute_overhead(df_orig: pd.DataFrame, df_def: pd.DataFrame) -> dict:
    label = df_orig["label"]
    mask_low  = label == 0
    mask_high = label == 1

    def med_diff(col):
        d = df_def[col] - df_orig[col]
        return {
            "all":  float(d.median()),
            "low":  float(d[mask_low].median())  if mask_low.any()  else np.nan,
            "high": float(d[mask_high].median()) if mask_high.any() else np.nan,
            "p95_low":  float(d[mask_low].quantile(0.95))  if mask_low.any()  else np.nan,
            "p95_high": float(d[mask_high].quantile(0.95)) if mask_high.any() else np.nan,
        }

    e2el  = med_diff("e2el_ms")
    ttft  = med_diff("ttft_ms")
    dec   = med_diff("decode_time_ms")

    def rel(col, mask):
        d = (df_def[col] - df_orig[col]) / df_orig.loc[mask, col].replace(0, np.nan)
        return float(d[mask].median()) if mask.any() else np.nan

    return {
        "overhead_total_median_ms_all":   e2el["all"],
        "overhead_total_median_ms_low":   e2el["low"],
        "overhead_total_median_ms_high":  e2el["high"],
        "overhead_total_p95_ms_low":      e2el["p95_low"],
        "overhead_total_p95_ms_high":     e2el["p95_high"],
        "relative_overhead_total_low":    rel("e2el_ms", mask_low),
        "relative_overhead_total_high":   rel("e2el_ms", mask_high),
        "overhead_ttft_median_ms_low":    ttft["low"],
        "overhead_decode_median_ms_low":  dec["low"],
    }


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------
def make_cv_splitter(cv_mode: str, n_splits: int):
    if cv_mode == "loco":
        return LeaveOneGroupOut()
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=SEED)
    except TypeError:
        return GroupKFold(n_splits=n_splits)


def get_cv_groups(df: pd.DataFrame, cv_mode: str) -> np.ndarray:
    if cv_mode == "loco":
        return df["run_concurrency"].values
    return df["group"].values


# ---------------------------------------------------------------------------
# Evaluation of one fold
# ---------------------------------------------------------------------------
def eval_fold(clf, X_tr, y_tr, X_te, y_te, split_id: int,
              meta: dict, overhead: dict,
              effective_target_ttft: float | None = None,
              effective_target_itl:  float | None = None) -> dict:
    clf_fit = clone(clf)
    clf_fit.fit(X_tr, y_tr)
    y_pred = clf_fit.predict(X_te)
    try:
        y_prob = clf_fit.predict_proba(X_te)[:, 1]
        auroc  = float(roc_auc_score(y_te, y_prob))
    except Exception:
        auroc = np.nan

    y_te_arr  = np.asarray(y_te)
    mask_low  = y_te_arr == 0
    mask_high = y_te_arr == 1

    row = dict(meta)
    row.update({
        "split_id":          split_id,
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "auroc":             auroc,
        "accuracy":          float(accuracy_score(y_te, y_pred)),
        "n_train":           int(len(y_tr)),
        "n_test":            int(len(y_te)),
        "n_low_test":        int(mask_low.sum()),
        "n_high_test":       int(mask_high.sum()),
        "effective_target_ttft_ms": effective_target_ttft if effective_target_ttft is not None else np.nan,
        "effective_target_itl_ms":  effective_target_itl  if effective_target_itl  is not None else np.nan,
    })
    row.update(overhead)
    return row


# ---------------------------------------------------------------------------
# Core: run one defense configuration through CV
# ---------------------------------------------------------------------------
def run_one_config(
    df_orig: pd.DataFrame,
    df_defended: pd.DataFrame | None,   # pre-built; None only for fold_calibrated
    dataset_name: str,
    defense: str,
    defense_family: str,
    defense_param: str | float,
    jitter_rep: int,
    featureset_name: str,
    feat_cols: list[str],
    clfs: dict,
    cv_splitter,
    cv_groups: np.ndarray,
    cv_mode: str,
    overhead_dict: dict,                # pre-computed for non-fold-calibrated
    rng: np.random.Generator,
    drop_first_itl: bool = False,
    oracle: bool = False,
    fold_calibrated: bool = False,
    calib_q: float | None = None,
) -> list[dict]:
    """
    Run adaptive + frozen evaluation for one (defense, param, jitter_rep) config.

    Non-fold-calibrated defenses:
      df_defended is pre-built once outside this function.
      overhead_dict is pre-computed from that same df_defended.
      Adaptive: train on df_defended[train], test on df_defended[test].
      Frozen:   train on df_orig[train],    test on df_defended[test].

    fold_calibrated=True (calibrated_floor):
      df_defended=None; calibration is computed fresh per fold from train split.
      target_ttft = quantile(high_train.ttft_ms, q)
      target_itl  = quantile(flattened high_train.itl_seq_ms, q)
      Applied to both train and test splits with a combined_floor defense.
      No test-data leakage.
    """
    y      = df_orig["label"].values

    def safe_X(d: pd.DataFrame) -> np.ndarray:
        return d[[c for c in feat_cols if c in d.columns]].values

    X_orig = safe_X(df_orig)

    rows     = []
    split_id = 0

    for train_idx, test_idx in cv_splitter.split(df_orig, y, cv_groups):
        df_tr_orig = df_orig.iloc[train_idx].copy()
        df_te_orig = df_orig.iloc[test_idx].copy()
        y_tr       = y[train_idx]
        y_te       = y[test_idx]

        eff_ttft     = None
        eff_itl      = None
        fold_overhead = overhead_dict  # default: pre-computed global overhead

        if fold_calibrated:
            # ----------------------------------------------------------------
            # Calibrate exclusively from high-state TRAIN rows
            # ----------------------------------------------------------------
            high_tr = df_tr_orig[df_tr_orig["label"] == 1]
            if high_tr.empty:
                # Edge case: skip fold if no high-state train rows
                split_id += 1
                continue

            eff_ttft = float(high_tr["ttft_ms"].quantile(calib_q))

            # Use flattened per-token ITL sequences from train high-state rows
            # for the most accurate quantile target.
            itl_seqs_flat = np.concatenate(
                [np.asarray(s) for s in high_tr["itl_seq_ms"]]
            )
            eff_itl = float(np.quantile(itl_seqs_flat, calib_q))

            calib_fn  = make_combined_floor(eff_ttft, eff_itl)
            df_tr_def = apply_defense_df(df_tr_orig, calib_fn, rng, drop_first_itl)
            df_te_def = apply_defense_df(df_te_orig, calib_fn, rng, drop_first_itl)

            # Fold-specific overhead: computed from train+test in this fold
            fold_overhead = compute_overhead(
                pd.concat([df_tr_orig, df_te_orig], ignore_index=True),
                pd.concat([df_tr_def,  df_te_def],  ignore_index=True),
            )
        else:
            # ----------------------------------------------------------------
            # Non-fold-calibrated: slice pre-built defended df
            # ----------------------------------------------------------------
            df_tr_def = df_defended.iloc[train_idx].copy()
            df_te_def = df_defended.iloc[test_idx].copy()

        X_tr_def  = safe_X(df_tr_def)
        X_te_def  = safe_X(df_te_def)
        X_tr_orig = X_orig[train_idx]

        meta = {
            "dataset":        dataset_name,
            "cv":             cv_mode,
            "defense":        defense,
            "defense_family": defense_family,
            "defense_param":  defense_param,
            "jitter_rep":     jitter_rep,
            "featureset":     featureset_name,
            "oracle":         oracle,
        }

        for clf_name, clf in clfs.items():
            # Adaptive: train AND test on defended data
            meta_a = dict(meta, attacker_type="adaptive", classifier=clf_name)
            rows.append(eval_fold(
                clf, X_tr_def, y_tr, X_te_def, y_te, split_id,
                meta_a, fold_overhead, eff_ttft, eff_itl,
            ))
            # Frozen: train on original, test on defended
            meta_f = dict(meta, attacker_type="frozen", classifier=clf_name)
            rows.append(eval_fold(
                clf, X_tr_orig, y_tr, X_te_def, y_te, split_id,
                meta_f, fold_overhead, eff_ttft, eff_itl,
            ))

        split_id += 1

    return rows


# ---------------------------------------------------------------------------
# Reconstruct a defense function from (name, raw_param) inside a worker.
# Needed because nested closures from make_* are rebuilt locally per task
# rather than pickled across process boundaries.
# ---------------------------------------------------------------------------
def build_defense_fn(defense: str, raw_param) -> Defense:
    if defense == "ttft_const_pad":
        return make_ttft_const_pad(float(raw_param))
    if defense == "ttft_floor":
        return make_ttft_floor(float(raw_param))
    if defense == "ttft_jitter_nonneg":
        return make_ttft_jitter_nonneg(float(raw_param))
    if defense == "itl_jitter_nonneg":
        return make_itl_jitter_nonneg(float(raw_param))
    if defense == "itl_floor":
        return make_itl_floor(float(raw_param))
    if defense == "combined_jitter":
        j_ttft, j_itl = raw_param
        return make_combined_jitter(j_ttft, j_itl)
    if defense == "combined_floor":
        t_ttft, t_itl = raw_param
        return make_combined_floor(t_ttft, t_itl)
    raise ValueError(f"build_defense_fn: unknown defense '{defense}'")


# ---------------------------------------------------------------------------
# ONE independent unit of work: build the defended dataframe once (with a
# task-private RNG), then run all feature sets / folds / classifiers /
# attacker types for this single (defense, param, jitter_rep). This mirrors
# exactly what the original sequential loop body did per iteration — only
# now it is a standalone, picklable function dispatched to a worker process
# by joblib.Parallel.
# ---------------------------------------------------------------------------
def _execute_task(task: dict, df: pd.DataFrame, cv_mode: str,
                  n_splits: int, drop_first_itl: bool, rf_n_jobs: int) -> list[dict]:
    rng = np.random.default_rng(task["seed"])
    defense  = task["defense"]
    family   = task["family"]
    param    = task["param"]
    jrep     = task["jrep"]
    fold_cal = task["fold_cal"]
    calib_q  = task["calib_q"]
    oracle   = task["oracle"]
    dataset_name = task["dataset"]

    cv        = make_cv_splitter(cv_mode, n_splits)
    cv_groups = get_cv_groups(df, cv_mode)
    clfs      = make_classifiers(rf_n_jobs=rf_n_jobs)

    if defense == "none":
        d_def, oh = df, ZERO_OVERHEAD
    elif fold_cal:
        d_def, oh = None, ZERO_OVERHEAD  # calibrated per-fold inside run_one_config
    elif family == "oracle":
        high_rows   = df[df["label"] == 1]
        target_ttft = float(high_rows["ttft_ms"].median())
        target_itl  = float(high_rows["itl_median_ms"].median())
        d_def = apply_oracle_low_df(df, target_ttft, target_itl, rng, drop_first_itl)
        oh    = compute_overhead(df, d_def)
    else:
        fn    = build_defense_fn(defense, task["raw_param"])
        d_def = apply_defense_df(df, fn, rng, drop_first_itl)
        oh    = compute_overhead(df, d_def)

    rows: list[dict] = []
    for fs_name, fs_cols in FEATURE_SETS.items():
        avail = [c for c in fs_cols if c in df.columns]
        if not avail:
            continue
        rows.extend(run_one_config(
            df_orig=df, df_defended=d_def, dataset_name=dataset_name,
            defense=defense, defense_family=family, defense_param=param,
            jitter_rep=jrep, featureset_name=fs_name, feat_cols=avail,
            clfs=clfs, cv_splitter=cv, cv_groups=cv_groups, cv_mode=cv_mode,
            overhead_dict=oh, rng=rng, drop_first_itl=drop_first_itl,
            oracle=oracle, fold_calibrated=fold_cal, calib_q=calib_q,
        ))
    return rows


# ---------------------------------------------------------------------------
# Full sweep for one dataset
# ---------------------------------------------------------------------------
def sweep_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    base_seed: int,
    cv_mode: str,
    n_splits: int,
    jitter_repeats: int,
    drop_first_itl: bool,
    smoke: bool,
    n_jobs: int = 1,
) -> list[dict]:
    """
    PARALLEL VERSION: build the full list of (defense, param, jitter_rep)
    tasks for this dataset (identical loop structure/values/smoke behaviour
    to the original), then execute them via joblib.Parallel. Each task is
    fully self-contained (own seeded RNG, own defended dataframe, own CV
    loop over folds/classifiers/attacker types), so tasks are embarrassingly
    parallel and their result is independent of execution order.
    """
    j_reps   = 2 if smoke else jitter_repeats
    comb_j   = COMBINED_JITTER_PROFILES[:3] if smoke else COMBINED_JITTER_PROFILES
    comb_f   = COMBINED_FLOOR_PROFILES[:3]  if smoke else COMBINED_FLOOR_PROFILES
    calib_qs = [0.50]                       if smoke else CALIB_QUANTILES

    tasks: list[dict] = []

    def _add(defense: str, family: str, param, jrep: int, *,
             raw_param=None, oracle: bool = False,
             fold_cal: bool = False, calib_q: float | None = None) -> None:
        """Register one (defense, param, jrep) config as a pending task."""
        seed = _seed_for_task(base_seed, dataset_name, defense, param, jrep)
        tasks.append(dict(
            dataset=dataset_name, defense=defense, family=family, param=param,
            raw_param=raw_param if raw_param is not None else param,
            jrep=jrep, oracle=oracle, fold_cal=fold_cal, calib_q=calib_q,
            seed=seed,
        ))

    # -----------------------------------------------------------------------
    # 0. Baseline
    # -----------------------------------------------------------------------
    _add("none", "none", 0.0, 0)

    # -----------------------------------------------------------------------
    # Single-component controls
    # -----------------------------------------------------------------------
    for pad in (TTFT_PAD_VALUES[:2] if smoke else TTFT_PAD_VALUES):
        _add("ttft_const_pad", "single_component_control", float(pad), 0)

    for J in (TTFT_JITTER_VALUES[:2] if smoke else TTFT_JITTER_VALUES):
        for rep in range(j_reps):
            _add("ttft_jitter_nonneg", "single_component_control", float(J), rep)

    for t in (TTFT_FLOOR_VALUES[:2] if smoke else TTFT_FLOOR_VALUES):
        _add("ttft_floor", "single_component_control", float(t), 0)

    for J in (ITL_JITTER_VALUES[:2] if smoke else ITL_JITTER_VALUES):
        for rep in range(j_reps):
            _add("itl_jitter_nonneg", "single_component_control", float(J), rep)

    for t in (ITL_FLOOR_VALUES[:2] if smoke else ITL_FLOOR_VALUES):
        _add("itl_floor", "single_component_control", float(t), 0)

    # -----------------------------------------------------------------------
    # Combined jitter  (main defense 1)
    # -----------------------------------------------------------------------
    for J_ttft, J_itl in comb_j:
        if J_ttft == 0 and J_itl == 0:
            continue
        for rep in range(j_reps):
            param = f"ttft{J_ttft}_itl{J_itl}"
            _add("combined_jitter", "combined_main", param, rep,
                 raw_param=(J_ttft, J_itl))

    # -----------------------------------------------------------------------
    # Combined floor  (main defense 2)
    # -----------------------------------------------------------------------
    for t_ttft, t_itl in comb_f:
        param = f"ttft{t_ttft}_itl{t_itl}"
        _add("combined_floor", "combined_main", param, 0,
             raw_param=(t_ttft, t_itl))

    # -----------------------------------------------------------------------
    # Train-calibrated high-quantile floor  (fold-clean, no pre-built df_defended)
    # -----------------------------------------------------------------------
    for q in calib_qs:
        param = f"q{int(q*100)}"
        _add("calibrated_floor", "combined_main", param, 0,
             fold_cal=True, calib_q=q)

    # -----------------------------------------------------------------------
    # Oracle low-only equalization  (theoretical; NOT a deployment defense)
    # -----------------------------------------------------------------------
    _add("oracle_low_equalize", "oracle", "median_high", 0, oracle=True)

    # RandomForest already parallelises internally (n_jobs=-1) per fit.
    # If we ALSO parallelise the outer task loop across processes, stacking
    # both would oversubscribe the machine (n_jobs outer procs x all-cores
    # inner). So: outer parallel -> RF gets 1 core per fit; outer sequential
    # (n_jobs=1) -> RF may use all cores per fit, as before.
    rf_n_jobs = -1 if n_jobs == 1 else 1
    print(f"  [{dataset_name}] {len(tasks)} defense configs queued "
          f"(x{len(FEATURE_SETS)} featuresets each) -> n_jobs={n_jobs}, "
          f"rf_n_jobs={rf_n_jobs}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_execute_task)(t, df, cv_mode, n_splits, drop_first_itl, rf_n_jobs)
        for t in tasks
    )

    all_rows: list[dict] = []
    for rows in results:
        all_rows.extend(rows)
    return all_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
BY_SPLIT_COLS = [
    "dataset", "cv", "defense", "defense_family", "defense_param", "jitter_rep",
    "attacker_type", "classifier", "featureset", "split_id",
    "balanced_accuracy", "auroc", "accuracy",
    "n_train", "n_test", "n_low_test", "n_high_test",
    "overhead_total_median_ms_all", "overhead_total_median_ms_low",
    "overhead_total_median_ms_high", "overhead_total_p95_ms_low",
    "overhead_total_p95_ms_high", "relative_overhead_total_low",
    "relative_overhead_total_high", "overhead_ttft_median_ms_low",
    "overhead_decode_median_ms_low",
    "effective_target_ttft_ms", "effective_target_itl_ms", "oracle",
]

def build_summary(by_split: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        "dataset", "cv", "defense", "defense_family", "defense_param",
        "attacker_type", "classifier", "featureset", "oracle",
    ]
    agg = (
        by_split.groupby(group_keys, dropna=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std= ("balanced_accuracy", "std"),
            auroc_mean=            ("auroc",             "mean"),
            auroc_std=             ("auroc",             "std"),
            accuracy_mean=         ("accuracy",          "mean"),
            n_splits=              ("split_id",          "nunique"),
            n_jitter_reps=         ("jitter_rep",        "nunique"),
            overhead_total_median_ms_low= ("overhead_total_median_ms_low", "mean"),
            overhead_total_median_ms_high=("overhead_total_median_ms_high","mean"),
            overhead_total_p95_ms_low=    ("overhead_total_p95_ms_low",    "mean"),
            overhead_ttft_median_ms_low=  ("overhead_ttft_median_ms_low",  "mean"),
            overhead_decode_median_ms_low=("overhead_decode_median_ms_low","mean"),
            relative_overhead_total_low=  ("relative_overhead_total_low",  "mean"),
            effective_target_ttft_mean=   ("effective_target_ttft_ms",     "mean"),
            effective_target_ttft_std=    ("effective_target_ttft_ms",     "std"),
            effective_target_itl_mean=    ("effective_target_itl_ms",      "mean"),
            effective_target_itl_std=     ("effective_target_itl_ms",      "std"),
        )
        .reset_index()
    )
    return agg


def build_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Feature-set ablation: baseline adaptive BA across feature sets."""
    return summary[
        (summary["defense"] == "none") &
        (summary["attacker_type"] == "adaptive")
    ][["dataset", "cv", "featureset", "classifier",
       "balanced_accuracy_mean", "balanced_accuracy_std",
       "auroc_mean", "auroc_std"]].copy()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(summary: pd.DataFrame, cv_mode: str):
    print("\n" + "=" * 72)
    print("DIAGNOSTICS")
    print("=" * 72)

    # 1. Baseline
    print("\n[1] Baseline adaptive BA / AUROC")
    base = summary[
        (summary["defense"]       == "none") &
        (summary["attacker_type"] == "adaptive") &
        (summary["featureset"]    == PRIMARY_FS)
    ]
    if not base.empty:
        print(base[["dataset", "cv", "classifier",
                    "balanced_accuracy_mean", "auroc_mean"]].to_string(index=False))

    # 2. Feature ablation (baseline, LogReg)
    print("\n[2] Feature ablation – baseline, adaptive, LogReg")
    abl = summary[
        (summary["defense"]       == "none") &
        (summary["attacker_type"] == "adaptive") &
        (summary["classifier"]    == "LogReg")
    ]
    if not abl.empty:
        print(abl[["dataset", "featureset",
                   "balanced_accuracy_mean", "auroc_mean"]].to_string(index=False))

    # 3. Combined defenses
    for ds in summary["dataset"].unique():
        print(f"\n[3] {ds} – Combined defenses (adaptive LogReg, {PRIMARY_FS})")
        filt = summary[
            (summary["dataset"]        == ds) &
            (summary["attacker_type"]  == "adaptive") &
            (summary["classifier"]     == "LogReg") &
            (summary["featureset"]     == PRIMARY_FS) &
            (summary["defense_family"] == "combined_main") &
            (~summary["oracle"].astype(bool))
        ]
        if filt.empty:
            print("  (no data)")
            continue

        for dname in filt["defense"].unique():
            sub = filt[filt["defense"] == dname].sort_values("overhead_total_median_ms_low")
            best_idx = sub["balanced_accuracy_mean"].idxmin()
            best = sub.loc[best_idx]
            print(f"\n  {dname}")
            print(f"    Best (lowest) adaptive BA: {best.balanced_accuracy_mean:.4f}  "
                  f"auroc={best.auroc_mean:.4f}  "
                  f"overhead_low={best.overhead_total_median_ms_low:.1f} ms  "
                  f"param={best.defense_param}")
            dropped = sub[sub["balanced_accuracy_mean"] < 0.75]
            if dropped.empty:
                print("    → Adaptive BA never drops below 0.75 in scanned range.")
            else:
                r = dropped.iloc[0]
                print(f"    → First BA<0.75: param={r.defense_param}  "
                      f"BA={r.balanced_accuracy_mean:.4f}  "
                      f"overhead_low={r.overhead_total_median_ms_low:.1f} ms")

    # 4. Single-component vs combined comparison
    print(f"\n[4] Single-component vs combined at similar overhead (adaptive LogReg, {PRIMARY_FS})")
    for ds in summary["dataset"].unique():
        s_filt = summary[
            (summary["dataset"]        == ds) &
            (summary["attacker_type"]  == "adaptive") &
            (summary["classifier"]     == "LogReg") &
            (summary["featureset"]     == PRIMARY_FS) &
            (summary["defense"]        != "none") &
            (~summary["oracle"].astype(bool))
        ].copy()
        if s_filt.empty:
            continue
        rows = []
        for family, grp in s_filt.groupby("defense_family"):
            idx = grp["balanced_accuracy_mean"].idxmin()
            rows.append(grp.loc[idx].copy())
        best_per_family = pd.DataFrame(rows)
        show_cols = [c for c in [
            "defense_family", "defense", "defense_param",
            "balanced_accuracy_mean", "auroc_mean", "overhead_total_median_ms_low",
        ] if c in best_per_family.columns]
        print(f"\n  {ds}:")
        print(best_per_family[show_cols].to_string(index=False))

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(summary: pd.DataFrame, by_split: pd.DataFrame, plots_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available – skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    clf_show = "LogReg"
    markers  = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]

    for ds in summary["dataset"].unique():
        base = summary[
            (summary["dataset"]       == ds) &
            (summary["defense"]       == "none") &
            (summary["attacker_type"] == "adaptive") &
            (summary["classifier"]    == clf_show) &
            (summary["featureset"]    == PRIMARY_FS)
        ]
        base_ba = base["balanced_accuracy_mean"].values[0] if not base.empty else None

        # Plot 1: overhead_low vs adaptive BA, main defenses
        fig, ax = plt.subplots(figsize=(12, 6))
        sub_main = summary[
            (summary["dataset"]        == ds) &
            (summary["attacker_type"]  == "adaptive") &
            (summary["classifier"]     == clf_show) &
            (summary["featureset"]     == PRIMARY_FS) &
            (summary["defense"]        != "none") &
            (~summary["oracle"].astype(bool))
        ]
        for i, (family, grp) in enumerate(sub_main.groupby("defense")):
            grp = grp.sort_values("overhead_total_median_ms_low")
            ax.plot(grp["overhead_total_median_ms_low"], grp["balanced_accuracy_mean"],
                    marker=markers[i % len(markers)], label=family, linewidth=1.5)

        if base_ba is not None:
            ax.axhline(base_ba, color="blue", linestyle=":", linewidth=1, label="baseline")
        ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, label="BA=0.75")
        ax.axhline(0.50, color="gray",   linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Median Overhead Low-State E2E (ms)")
        ax.set_ylabel("Adaptive Balanced Accuracy")
        ax.set_title(f"{ds} – Defense Tradeoff ({clf_show}, {PRIMARY_FS})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plots_dir / f"{ds}_plot1_tradeoff.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # Plot 2: AUROC vs overhead
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (family, grp) in enumerate(sub_main.groupby("defense")):
            grp = grp.sort_values("overhead_total_median_ms_low")
            ax.plot(grp["overhead_total_median_ms_low"], grp["auroc_mean"],
                    marker=markers[i % len(markers)], label=family, linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Median Overhead Low-State E2E (ms)")
        ax.set_ylabel("Adaptive AUROC")
        ax.set_title(f"{ds} – AUROC vs Overhead ({clf_show})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plots_dir / f"{ds}_plot2_auroc.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # Plot 3: frozen vs adaptive for combined defenses
        for defense_name in ["combined_jitter", "combined_floor", "calibrated_floor"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for att in ["frozen", "adaptive"]:
                sub = summary[
                    (summary["dataset"]       == ds) &
                    (summary["defense"]       == defense_name) &
                    (summary["attacker_type"] == att) &
                    (summary["classifier"]    == clf_show) &
                    (summary["featureset"]    == PRIMARY_FS)
                ].sort_values("overhead_total_median_ms_low")
                if sub.empty:
                    continue
                ax.plot(sub["overhead_total_median_ms_low"],
                        sub["balanced_accuracy_mean"],
                        marker="o", label=att, linewidth=1.5)
                ax.fill_between(
                    sub["overhead_total_median_ms_low"],
                    sub["balanced_accuracy_mean"] - sub["balanced_accuracy_std"],
                    sub["balanced_accuracy_mean"] + sub["balanced_accuracy_std"],
                    alpha=0.15,
                )
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Median Overhead Low-State E2E (ms)")
            ax.set_ylabel("Balanced Accuracy")
            ax.set_title(f"{ds} | {defense_name} – Frozen vs Adaptive ({clf_show})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = plots_dir / f"{ds}_plot3_{defense_name}_frozen_vs_adaptive.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Saved: {out}")

        # Plot 4: single-component vs combined (violin per defense family)
        fam_data = {}
        for fam, grp in by_split[
            (by_split["dataset"]       == ds) &
            (by_split["attacker_type"] == "adaptive") &
            (by_split["classifier"]    == clf_show) &
            (by_split["featureset"]    == PRIMARY_FS) &
            (by_split["defense"]       != "none") &
            (~by_split["oracle"].astype(bool))
        ].groupby("defense_family"):
            fam_data[fam] = grp["balanced_accuracy"].values
        if len(fam_data) >= 2:
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = list(fam_data.keys())
            data   = [fam_data[k] for k in labels]
            ax.violinplot(data, positions=range(len(labels)), showmedians=True)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_ylabel("Balanced Accuracy (per split)")
            ax.set_title(f"{ds} – Defense Family BA Distribution")
            fig.tight_layout()
            out = plots_dir / f"{ds}_plot4_family_violin.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(results_dir: Path, args, df_all: pd.DataFrame,
                   has_true_itl: bool, by_split_rows: int, summary_rows: int):
    # Build per-cell counts for manifest
    if "run_concurrency" in df_all.columns:
        cell_counts = {
            str(k): int(v)
            for k, v in df_all.groupby(
                ["dataset", "offload_gb", "run_concurrency"]
            ).size().items()
        }
    else:
        cell_counts = {}

    ds_off_counts = {
        str(k): int(v)
        for k, v in df_all.groupby(["dataset", "offload_gb"]).size().items()
    }

    manifest = {
        "script":           "raw_combined_defense_sweep.py  V3.1",
        "primary_featureset": PRIMARY_FS,
        "csv_source":       str(args.csv) if hasattr(args, "csv") and args.csv else None,
        "raw_json_llama":   str(args.raw_json_dir_llama) if args.raw_json_dir_llama else None,
        "raw_json_qwen":    str(args.raw_json_dir_qwen)  if args.raw_json_dir_qwen  else None,
        "itl_mode":         "true_per_token" if has_true_itl else "synthetic_gamma_reconstruction",
        "cv":               args.cv,
        "n_splits":         args.n_splits,
        "jitter_repeats":   args.jitter_repeats,
        "drop_first_itl":   args.drop_first_itl,
        "smoke":            args.smoke,
        "n_jobs":           args.n_jobs,
        "rows_total":       len(df_all),
        "counts_per_dataset_offload": ds_off_counts,
        "counts_per_dataset_offload_concurrency": cell_counts,
        "by_split_rows":    by_split_rows,
        "summary_rows":     summary_rows,
        "note": (
            "ITL sequences are SYNTHETIC (Gamma-reconstructed from aggregates) "
            "when loaded from CSV.  Results are conservative: true per-token "
            "defenses applied to real sequences may differ."
            if not has_true_itl else
            "True per-token ITL sequences loaded from raw JSON files."
        ),
    }
    out = results_dir / "raw_combined_manifest.json"
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Raw combined defense sweep – V3")
    # Data sources (mutually exclusive: CSV or raw JSONs)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--csv", type=Path,
                     help="Pre-extracted raw_requests_0_vs_12.csv")
    grp.add_argument("--raw-json-dir-llama", type=Path,
                     help="Root dir for Llama raw JSON benchmark files")
    parser.add_argument("--raw-json-dir-qwen", type=Path,
                        help="Root dir for Qwen raw JSON benchmark files "
                             "(required when --raw-json-dir-llama is given)")
    # CV
    parser.add_argument("--cv", choices=["groupkfold", "loco"],
                        default="groupkfold")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Folds for GroupKFold (ignored for loco).")
    # Jitter
    parser.add_argument("--jitter-repeats", type=int, default=20)
    # Features
    parser.add_argument("--drop-first-itl", action="store_true",
                        help="Drop first token ITL before feature computation.")
    # Output
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    # Options
    parser.add_argument("--no-combined", action="store_true",
                        help="Skip combined Llama+Qwen sweep.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test mode: cut down repeats and profiles.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2),
                        help="Parallel worker processes for the defense sweep "
                             "(one process per defense/param/jitter_rep task). "
                             "Default: cpu_count()-2. Use 1 for sequential "
                             "(original) behaviour.")
    args = parser.parse_args()

    if args.raw_json_dir_llama and not args.raw_json_dir_qwen:
        parser.error("--raw-json-dir-qwen is required when --raw-json-dir-llama is given.")

    rng = np.random.default_rng(args.seed)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("\n=== Loading data ===")
    has_true_itl = False

    if args.raw_json_dir_llama:
        df_raw       = load_raw_jsons(args.raw_json_dir_llama, args.raw_json_dir_qwen)
        has_true_itl = True
        df_all       = finalise_df(df_raw, strict=True)   # hard checks for JSON mode
    else:
        df_raw = load_csv(args.csv, rng)
        df_all = finalise_df(df_raw, strict=False)        # soft checks for CSV mode

    print(f"\nDataset summary:")
    counts = df_all.groupby(["dataset", "offload_gb", "run_concurrency"]).size()
    print(counts.to_string())

    # Split per model
    df_llama = df_all[df_all["dataset"] == "llama"].copy().reset_index(drop=True)
    df_qwen  = df_all[df_all["dataset"] == "qwen"].copy().reset_index(drop=True)

    # -------------------------------------------------------------------
    # Run sweeps
    # -------------------------------------------------------------------
    all_by_split: list[dict] = []

    for ds_name, df_ds in [("llama", df_llama), ("qwen", df_qwen)]:
        print(f"\n=== Sweep: {ds_name}  n={len(df_ds)} ===")
        rows = sweep_dataset(df_ds, ds_name, args.seed,
                             cv_mode=args.cv, n_splits=args.n_splits,
                             jitter_repeats=args.jitter_repeats,
                             drop_first_itl=args.drop_first_itl,
                             smoke=args.smoke, n_jobs=args.n_jobs)
        all_by_split.extend(rows)

    if not args.no_combined:
        df_combined = df_all.copy()
        print(f"\n=== Sweep: combined  n={len(df_combined)} ===")
        rows = sweep_dataset(df_combined, "combined", args.seed,
                             cv_mode=args.cv, n_splits=args.n_splits,
                             jitter_repeats=args.jitter_repeats,
                             drop_first_itl=args.drop_first_itl,
                             smoke=args.smoke, n_jobs=args.n_jobs)
        all_by_split.extend(rows)

    # -------------------------------------------------------------------
    # Build DataFrames
    # -------------------------------------------------------------------
    by_split_df = pd.DataFrame(all_by_split)
    for col in BY_SPLIT_COLS:
        if col not in by_split_df.columns:
            by_split_df[col] = np.nan
    by_split_df = by_split_df[BY_SPLIT_COLS]
    by_split_df["oracle"] = by_split_df["oracle"].fillna(False).astype(bool)

    summary_df  = build_summary(by_split_df)
    ablation_df = build_ablation_table(summary_df)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    out_by_split = results_dir / "raw_combined_by_split.csv"
    out_summary  = results_dir / "raw_combined_summary.csv"
    out_ablation = results_dir / "raw_feature_ablation.csv"

    by_split_df.to_csv(out_by_split, index=False, float_format="%.5f")
    summary_df.to_csv(out_summary,   index=False, float_format="%.5f")
    ablation_df.to_csv(out_ablation, index=False, float_format="%.5f")

    print(f"\nSaved by_split  : {out_by_split}  ({len(by_split_df)} rows)")
    print(f"Saved summary   : {out_summary}   ({len(summary_df)} rows)")
    print(f"Saved ablation  : {out_ablation}  ({len(ablation_df)} rows)")

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    make_plots(summary_df, by_split_df, results_dir / "plots")

    # -------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------
    write_manifest(results_dir, args, df_all,
                   has_true_itl, len(by_split_df), len(summary_df))

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------
    print_diagnostics(summary_df, args.cv)
    print("\nDone.")


if __name__ == "__main__":
    main()
