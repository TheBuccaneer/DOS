"""
offline_defense_sweep.py  –  V2
================================
Offline simulation of timing-obfuscation defenses against an
offload-regime side-channel in vLLM serving.

Datasets  : llama_requests_summary.csv, qwen_requests_summary.csv
Binary    : offload_gb == 0  (Low)  vs  offload_gb == 12  (High)
Attackers : frozen   – trained on original data, evaluated on defended test split
            adaptive – trained AND tested on defended data (GroupKFold / LOCO)

Changes vs V1
-------------
1.  Per-split output: defense_sweep_by_split.csv now holds one row per fold,
    with columns:
      dataset, defense, defense_param, jitter_rep, attacker_type, classifier,
      featureset, split_id, balanced_accuracy, auroc, accuracy,
      n_train, n_test, n_low_test, n_high_test,
      overhead_median_ms_all, overhead_median_ms_low, overhead_median_ms_high,
      relative_overhead_low, relative_overhead_high
    The summary CSV is derived from aggregating this file.

2.  Jitter repeats (--jitter-repeats N, default 20):
    Stochastic defenses (ttft_jitter_nonneg, ttft_jitter_sym,
    itl_jitter_nonneg, itl_jitter_sym) run N independent noise draws.
    Deterministic defenses use jitter_rep=0.

3.  Consistent ITL jitter:
    One noise value is drawn per request and applied identically to
      itl_mean_ms, itl_median_ms, itl_p95_ms, itl_p99_ms,
      first_4_itl_mean_ms, last_4_itl_mean_ms
    decode_time_ms is adjusted as  decode_time_ms += noise * itl_count
    (Variant B).  The featureset label carries "_itl_jitter" suffix to
    document the decode_time_ms adjustment.

4.  Fold-clean low-to-high equalization:
    The padding value is computed from the TRAIN split only, then applied
    to both train and test.  The effective padding per fold is recorded in
    the by_split output; the summary reports effective_pad_mean / _std.

5.  Floor / Homogenization defense (new):
      ttft_floor_pad:  ttft_ms = max(ttft_ms, target_ms)
      itl_floor_pad:   itl_*  = max(itl_*, target_ms)  for all ITL aggregates
    Overhead is computed as defended - original (separately for Low/High).

6.  Leave-One-Concurrency-Out CV (--cv loco):
    Groups on run_concurrency instead of run_id.

7.  Combined dataset:
    group key = dataset_name + "_" + run_id to avoid cross-dataset leakage.
    dataset_name is NOT used as a feature by default (--combined-dataset-feature
    flag to enable).
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_LLAMA_CSV = HERE.parent.parent / "legacy" / "estimator_inputs" / "llama_requests_summary.csv"
DEFAULT_QWEN_CSV  = HERE.parent.parent / "legacy" / "estimator_inputs" / "qwen_requests_summary.csv"
RESULTS_DIR       = HERE.parent.parent / "results" / "defense_sweep"

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
FEATURES_CORE = [
    "ttft_ms",
    "itl_mean_ms",
    "itl_median_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "decode_time_ms",
]

FEATURES_EXTENDED = FEATURES_CORE + [
    "itl_std_ms",
    "first_4_itl_mean_ms",
    "last_4_itl_mean_ms",
    "tail_ratio_p99_over_median",
    "run_concurrency",
]

# ITL aggregate columns modified by ITL-jitter / ITL-floor defenses
ITL_AGG_COLS = [
    "itl_mean_ms",
    "itl_median_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "first_4_itl_mean_ms",
    "last_4_itl_mean_ms",
]

# ---------------------------------------------------------------------------
# Defense parameter sweeps
# ---------------------------------------------------------------------------
TTFT_PAD_VALUES        = [25, 50, 100, 200, 400, 800, 1200]        # constant pad (ms)
TTFT_JITTER_VALUES     = [25, 50, 100, 200, 400, 800, 1200]        # jitter scale (ms)
ITL_JITTER_VALUES      = [5, 10, 25, 50, 100, 200, 400, 800]       # jitter scale (ms)
TTFT_FLOOR_VALUES      = [50, 100, 200, 400, 800, 1200, 1600]      # floor target (ms)
ITL_FLOOR_VALUES       = [25, 50, 100, 200, 400, 800, 1200, 1600]  # floor target (ms)

N_SPLITS_DEFAULT = 5
SEED             = 42


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifiers() -> dict:
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=SEED, n_jobs=-1, min_samples_leaf=2)),
        ]),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["request_success"] == True].copy()
    if "error_text" in df.columns:
        df = df[df["error_text"].isna() | (df["error_text"].astype(str).str.strip() == "")]
    df = df[df["offload_gb"].isin([0, 12])].copy()
    df["label"]   = (df["offload_gb"] == 12).astype(int)   # 0=Low, 1=High
    df["dataset"] = dataset_name
    # group for GroupKFold: unique per-model run
    df["group"]   = dataset_name + "_run" + df["run_id"].astype(str)

    all_feat_cols = list(set(FEATURES_EXTENDED) & set(df.columns))
    df[all_feat_cols] = df[all_feat_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=all_feat_cols).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  [{dataset_name}] Dropped {before - after} rows with NaN/Inf in features.")
    print(f"  [{dataset_name}] Loaded {len(df)} rows  "
          f"(Low={(df.label==0).sum()}, High={(df.label==1).sum()})")
    return df


# ---------------------------------------------------------------------------
# CV splitter factory
# ---------------------------------------------------------------------------
def make_cv(cv_mode: str, n_splits: int):
    """Return a fitted-ready CV splitter and the group column name."""
    if cv_mode == "loco":
        return LeaveOneGroupOut(), "run_concurrency"
    else:
        return GroupKFold(n_splits=n_splits), "group"


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------
def compute_overhead(original: pd.Series, defended: pd.Series,
                     label: pd.Series) -> dict:
    delta     = defended - original
    mask_low  = label == 0
    mask_high = label == 1
    return {
        "overhead_median_ms_all":  float(delta.median()),
        "overhead_median_ms_low":  float(delta[mask_low].median())  if mask_low.any()  else np.nan,
        "overhead_median_ms_high": float(delta[mask_high].median()) if mask_high.any() else np.nan,
        "relative_overhead_low":   float(
            (delta[mask_low] / original[mask_low].replace(0, np.nan)).median()
        ) if mask_low.any() else np.nan,
        "relative_overhead_high":  float(
            (delta[mask_high] / original[mask_high].replace(0, np.nan)).median()
        ) if mask_high.any() else np.nan,
    }


# ---------------------------------------------------------------------------
# Per-fold evaluation helpers
# ---------------------------------------------------------------------------
def _eval_fold(clf, X_tr, y_tr, X_te, y_te, train_idx, test_idx, df_full, label_arr,
               split_id, dataset, defense, defense_param, jitter_rep,
               attacker_type, clf_name, featureset, overhead_dict,
               effective_pad=None) -> dict:
    """Fit clf on X_tr/y_tr, evaluate on X_te/y_te.  Return one by_split row."""
    clf_fit = clone(clf)
    clf_fit.fit(X_tr, y_tr)
    y_pred = clf_fit.predict(X_te)
    try:
        y_prob = clf_fit.predict_proba(X_te)[:, 1]
        auroc  = float(roc_auc_score(y_te, y_prob))
    except Exception:
        auroc = np.nan

    y_te_arr   = np.asarray(y_te)
    mask_low   = y_te_arr == 0
    mask_high  = y_te_arr == 1

    row = {
        "dataset":        dataset,
        "defense":        defense,
        "defense_param":  defense_param,
        "jitter_rep":     jitter_rep,
        "attacker_type":  attacker_type,
        "classifier":     clf_name,
        "featureset":     featureset,
        "split_id":       split_id,
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "auroc":          auroc,
        "accuracy":       float(accuracy_score(y_te, y_pred)),
        "n_train":        int(len(y_tr)),
        "n_test":         int(len(y_te)),
        "n_low_test":     int(mask_low.sum()),
        "n_high_test":    int(mask_high.sum()),
    }
    row.update(overhead_dict)
    if effective_pad is not None:
        row["effective_pad"] = float(effective_pad)
    return row


# ---------------------------------------------------------------------------
# Apply defenses
# ---------------------------------------------------------------------------
def apply_ttft_constant_pad(df: pd.DataFrame, pad_ms: float) -> pd.DataFrame:
    d = df.copy()
    d["ttft_ms"] = d["ttft_ms"] + pad_ms
    return d


def apply_ttft_floor(df: pd.DataFrame, target_ms: float) -> pd.DataFrame:
    d = df.copy()
    d["ttft_ms"] = np.maximum(d["ttft_ms"], target_ms)
    return d


def apply_ttft_nonneg_jitter(df: pd.DataFrame, J: float,
                              rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    d["ttft_ms"] = d["ttft_ms"] + rng.uniform(0, J, size=len(d))
    return d


def apply_ttft_sym_jitter(df: pd.DataFrame, sigma: float,
                           rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    d["ttft_ms"] = np.maximum(d["ttft_ms"] + rng.normal(0, sigma, size=len(d)), 0.0)
    return d


def apply_itl_nonneg_jitter(df: pd.DataFrame, J: float,
                             rng: np.random.Generator) -> pd.DataFrame:
    """One noise value per request, applied identically to all ITL aggregates
    and propagated to decode_time_ms (Variant B: decode_time_ms += noise * itl_count).
    """
    d     = df.copy()
    noise = rng.uniform(0, J, size=len(d))
    for col in ITL_AGG_COLS:
        if col in d.columns:
            d[col] = d[col] + noise
    if "decode_time_ms" in d.columns and "itl_count" in d.columns:
        d["decode_time_ms"] = d["decode_time_ms"] + noise * d["itl_count"]
    return d


def apply_itl_sym_jitter(df: pd.DataFrame, sigma: float,
                          rng: np.random.Generator) -> pd.DataFrame:
    """One noise value per request (symmetric), clamped >= 0 after adding."""
    d     = df.copy()
    noise = rng.normal(0, sigma, size=len(d))
    for col in ITL_AGG_COLS:
        if col in d.columns:
            d[col] = np.maximum(d[col] + noise, 0.0)
    if "decode_time_ms" in d.columns and "itl_count" in d.columns:
        d["decode_time_ms"] = np.maximum(
            d["decode_time_ms"] + noise * d["itl_count"], 0.0)
    return d


def apply_itl_floor(df: pd.DataFrame, target_ms: float) -> pd.DataFrame:
    """Floor all ITL aggregate columns to target_ms."""
    d = df.copy()
    for col in ITL_AGG_COLS:
        if col in d.columns:
            d[col] = np.maximum(d[col], target_ms)
    # decode_time_ms: do NOT floor (it aggregates token count * itl, not a per-token stat)
    return d


# ---------------------------------------------------------------------------
# Fold-clean low-to-high equalization helpers
# (pad is computed from train split only, then applied to train + test)
# ---------------------------------------------------------------------------
def _compute_l2h_pad(df_train: pd.DataFrame, feature: str, q: float | None) -> float:
    low_mask  = df_train["label"] == 0
    high_mask = df_train["label"] == 1
    low_med   = df_train.loc[low_mask,  feature].median()
    if q is None:
        target = df_train.loc[high_mask, feature].median()
    else:
        target = df_train.loc[high_mask, feature].quantile(q)
    return float(max(target - low_med, 0.0))


def _apply_l2h_pad(df: pd.DataFrame, pad: float) -> pd.DataFrame:
    """Apply pad to Low-state ITL columns."""
    d        = df.copy()
    low_mask = d["label"] == 0
    for col in ITL_AGG_COLS:
        if col in d.columns:
            d.loc[low_mask, col] = d.loc[low_mask, col] + pad
    return d


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------
def run_sweep(df: pd.DataFrame, dataset_name: str, featureset_name: str,
              feature_cols: list, rng: np.random.Generator,
              cv_mode: str, n_splits: int, jitter_repeats: int) -> list[dict]:
    """
    Returns a list of by_split rows (one per fold x attacker x classifier).
    """
    clfs = make_classifiers()
    cv, group_col = make_cv(cv_mode, n_splits)

    y      = df["label"].values
    groups = df[group_col].values

    def _safe_X(d, cols):
        return d[[c for c in cols if c in d.columns]].values

    # -----------------------------------------------------------------------
    # Helper: run one (defense, defended_df, feature_cols_to_use) configuration
    # through all folds and return by_split rows.
    # Handles both adaptive and frozen attackers.
    # For low-to-high equalization, fold_pad_fn(df_train) -> float supplies
    # the padding amount computed per fold from train data only.
    # -----------------------------------------------------------------------
    def _run_config(defense: str, defense_param: float, jitter_rep: int,
                    df_defended: pd.DataFrame | None,  # None = determined per fold
                    feat_cols: list,
                    overhead_dict: dict,
                    *,
                    fold_pad_fn=None,          # callable(df_train) -> float, for l2h
                    df_defended_frozen: pd.DataFrame | None = None,  # override for frozen
                    ) -> list[dict]:
        """
        df_defended : fully pre-applied defense dataframe (for all defenses except l2h).
        fold_pad_fn : if set, l2h padding is computed fresh per fold from train split.
        df_defended_frozen : if set, use this df for frozen attacker test split
                             (same as df_defended in most cases).
        """
        rows = []
        split_id = 0

        if df_defended_frozen is None and df_defended is not None:
            df_defended_frozen = df_defended

        X_orig = _safe_X(df, feat_cols)

        for train_idx, test_idx in cv.split(df, y, groups):
            df_tr_orig = df.iloc[train_idx]
            df_te_orig = df.iloc[test_idx]
            y_tr       = y[train_idx]
            y_te       = y[test_idx]

            if fold_pad_fn is not None:
                # Fold-clean l2h: compute pad from train, apply to both splits
                pad_val    = fold_pad_fn(df_tr_orig)
                df_tr_def  = _apply_l2h_pad(df_tr_orig, pad_val)
                df_te_def  = _apply_l2h_pad(df_te_orig, pad_val)
                X_tr_def   = _safe_X(df_tr_def, feat_cols)
                X_te_def   = _safe_X(df_te_def, feat_cols)
                X_te_frozen = X_te_def   # frozen also sees defended test
                eff_pad    = pad_val
            else:
                X_tr_def   = _safe_X(df_defended.iloc[train_idx], feat_cols)
                X_te_def   = _safe_X(df_defended.iloc[test_idx],  feat_cols)
                X_te_frozen = _safe_X(df_defended_frozen.iloc[test_idx], feat_cols)
                eff_pad    = None

            for clf_name, clf in clfs.items():
                # Adaptive: train AND test on defended data
                rows.append(_eval_fold(
                    clf, X_tr_def, y_tr, X_te_def, y_te,
                    train_idx, test_idx, df, y, split_id,
                    dataset_name, defense, defense_param, jitter_rep,
                    "adaptive", clf_name, featureset_name, overhead_dict,
                    effective_pad=eff_pad,
                ))
                # Frozen: train on original, test on defended
                X_tr_orig = X_orig[train_idx]
                rows.append(_eval_fold(
                    clf, X_tr_orig, y_tr, X_te_frozen, y_te,
                    train_idx, test_idx, df, y, split_id,
                    dataset_name, defense, defense_param, jitter_rep,
                    "frozen", clf_name, featureset_name, overhead_dict,
                    effective_pad=eff_pad,
                ))
            split_id += 1
        return rows

    by_split_rows = []

    # -----------------------------------------------------------------------
    # 0. Baseline
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] Baseline ...")
    zero_oh = {k: 0.0 for k in [
        "overhead_median_ms_all", "overhead_median_ms_low", "overhead_median_ms_high",
        "relative_overhead_low", "relative_overhead_high",
    ]}
    by_split_rows += _run_config("none", 0.0, 0, df, feature_cols, zero_oh)

    # -----------------------------------------------------------------------
    # 1. TTFT constant padding  (deterministic, jitter_rep=0)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] TTFT constant padding ...")
    orig_ttft = df["ttft_ms"]
    for pad in TTFT_PAD_VALUES:
        d_def = apply_ttft_constant_pad(df, pad)
        oh    = compute_overhead(orig_ttft, d_def["ttft_ms"], df["label"])
        by_split_rows += _run_config("ttft_const_pad", float(pad), 0, d_def, feature_cols, oh)

    # -----------------------------------------------------------------------
    # 2. TTFT floor padding  (deterministic, jitter_rep=0)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] TTFT floor padding ...")
    for target in TTFT_FLOOR_VALUES:
        d_def = apply_ttft_floor(df, target)
        oh    = compute_overhead(orig_ttft, d_def["ttft_ms"], df["label"])
        by_split_rows += _run_config("ttft_floor_pad", float(target), 0, d_def, feature_cols, oh)

    # -----------------------------------------------------------------------
    # 3a. TTFT nonneg jitter  (stochastic)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] TTFT nonneg jitter ...")
    for J in TTFT_JITTER_VALUES:
        for rep in range(jitter_repeats):
            d_def = apply_ttft_nonneg_jitter(df, J, rng)
            oh    = compute_overhead(orig_ttft, d_def["ttft_ms"], df["label"])
            by_split_rows += _run_config("ttft_jitter_nonneg", float(J), rep, d_def, feature_cols, oh)

    # -----------------------------------------------------------------------
    # 3b. TTFT symmetric jitter  (stochastic)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] TTFT sym jitter ...")
    for sigma in TTFT_JITTER_VALUES:
        for rep in range(jitter_repeats):
            d_def = apply_ttft_sym_jitter(df, sigma, rng)
            oh    = compute_overhead(orig_ttft, d_def["ttft_ms"], df["label"])
            by_split_rows += _run_config("ttft_jitter_sym", float(sigma), rep, d_def, feature_cols, oh)

    # -----------------------------------------------------------------------
    # ITL defenses: feature cols exclude decode_time_ms because it is
    # adjusted inside the defense functions (Variant B).
    # The featureset label gains "_itl_jitter" suffix to document this.
    # -----------------------------------------------------------------------
    itl_feat_cols = [c for c in feature_cols if c != "decode_time_ms"]
    itl_fs_name   = featureset_name + "_itl_jitter"
    orig_itl_mean = df["itl_mean_ms"]

    # -----------------------------------------------------------------------
    # 4a. ITL nonneg jitter  (stochastic)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] ITL nonneg jitter ...")
    for J in ITL_JITTER_VALUES:
        for rep in range(jitter_repeats):
            d_def = apply_itl_nonneg_jitter(df, J, rng)
            oh    = compute_overhead(orig_itl_mean, d_def["itl_mean_ms"], df["label"])
            by_split_rows += _run_config("itl_jitter_nonneg", float(J), rep, d_def,
                                         itl_feat_cols, oh)

    # -----------------------------------------------------------------------
    # 4b. ITL symmetric jitter  (stochastic)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] ITL sym jitter ...")
    for sigma in ITL_JITTER_VALUES:
        for rep in range(jitter_repeats):
            d_def = apply_itl_sym_jitter(df, sigma, rng)
            oh    = compute_overhead(orig_itl_mean, d_def["itl_mean_ms"], df["label"])
            by_split_rows += _run_config("itl_jitter_sym", float(sigma), rep, d_def,
                                         itl_feat_cols, oh)

    # -----------------------------------------------------------------------
    # 5. ITL floor padding  (deterministic, jitter_rep=0)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] ITL floor padding ...")
    for target in ITL_FLOOR_VALUES:
        d_def = apply_itl_floor(df, target)
        oh    = compute_overhead(orig_itl_mean, d_def["itl_mean_ms"], df["label"])
        by_split_rows += _run_config("itl_floor_pad", float(target), 0, d_def,
                                     itl_feat_cols, oh)

    # -----------------------------------------------------------------------
    # 6. Low-to-high equalization  (fold-clean, deterministic per fold)
    # -----------------------------------------------------------------------
    print(f"  [{dataset_name}] Low-to-high equalization ...")

    def _l2h_oh(df_full: pd.DataFrame, feature: str, q: float | None) -> dict:
        """Global overhead estimate (only for reporting; pad is fold-specific)."""
        low_mask  = df_full["label"] == 0
        high_mask = df_full["label"] == 1
        low_med   = df_full.loc[low_mask, feature].median()
        target    = (df_full.loc[high_mask, feature].median() if q is None
                     else df_full.loc[high_mask, feature].quantile(q))
        pad_approx = max(target - low_med, 0.0)
        d_approx   = df_full.copy()
        d_approx.loc[low_mask, "itl_mean_ms"] = (
            d_approx.loc[low_mask, "itl_mean_ms"] + pad_approx)
        return compute_overhead(orig_itl_mean, d_approx["itl_mean_ms"], df_full["label"])

    # Median equalization
    oh = _l2h_oh(df, "itl_mean_ms", None)
    by_split_rows += _run_config(
        "low_to_high_median", 0.0, 0, None, itl_feat_cols, oh,
        fold_pad_fn=lambda df_tr: _compute_l2h_pad(df_tr, "itl_mean_ms", None),
    )

    # Quantile equalization q = 0.25, 0.50, 0.75, 0.90
    for q in [0.25, 0.50, 0.75, 0.90]:
        oh = _l2h_oh(df, "itl_mean_ms", q)
        by_split_rows += _run_config(
            f"low_to_high_q{int(q*100)}", float(q), 0, None, itl_feat_cols, oh,
            fold_pad_fn=lambda df_tr, _q=q: _compute_l2h_pad(df_tr, "itl_mean_ms", _q),
        )

    return by_split_rows


# ---------------------------------------------------------------------------
# Aggregate by_split rows -> summary
# ---------------------------------------------------------------------------
BY_SPLIT_COLS = [
    "dataset", "defense", "defense_param", "jitter_rep", "attacker_type",
    "classifier", "featureset", "split_id",
    "balanced_accuracy", "auroc", "accuracy",
    "n_train", "n_test", "n_low_test", "n_high_test",
    "overhead_median_ms_all", "overhead_median_ms_low", "overhead_median_ms_high",
    "relative_overhead_low", "relative_overhead_high",
    "effective_pad",
]

def build_summary(by_split: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold rows into summary (mean/std over folds x jitter_reps)."""
    group_keys = [
        "dataset", "defense", "defense_param",
        "attacker_type", "classifier", "featureset",
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
            n_low_test_mean=       ("n_low_test",        "mean"),
            n_high_test_mean=      ("n_high_test",       "mean"),
            overhead_median_ms_all=  ("overhead_median_ms_all",  "mean"),
            overhead_median_ms_low=  ("overhead_median_ms_low",  "mean"),
            overhead_median_ms_high= ("overhead_median_ms_high", "mean"),
            relative_overhead_low=   ("relative_overhead_low",   "mean"),
            relative_overhead_high=  ("relative_overhead_high",  "mean"),
            effective_pad_mean=    ("effective_pad",     "mean"),
            effective_pad_std=     ("effective_pad",     "std"),
        )
        .reset_index()
    )
    return agg


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
    clf_show  = "LogReg"
    adaptive  = summary[
        (summary["attacker_type"] == "adaptive") &
        (summary["classifier"]    == clf_show)
    ].copy()

    defenses = [d for d in adaptive["defense"].unique() if d != "none"]
    markers  = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]

    for ds in adaptive["dataset"].unique():
        sub_ds = adaptive[adaptive["dataset"] == ds]

        # Plot 1: overhead_low vs adaptive BA (per defense type)
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, dname in enumerate(defenses):
            sub = sub_ds[sub_ds["defense"] == dname].sort_values("overhead_median_ms_low")
            if sub.empty:
                continue
            ax.plot(sub["overhead_median_ms_low"], sub["balanced_accuracy_mean"],
                    marker=markers[i % len(markers)], label=dname, linewidth=1.5)
        ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, label="BA=0.75")
        ax.axhline(0.50, color="gray",   linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Median Overhead Low-State (ms)")
        ax.set_ylabel("Adaptive Balanced Accuracy")
        ax.set_title(f"Defense Tradeoff – {ds} ({clf_show})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plots_dir / f"plot1_{ds}_overhead_vs_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # Plot 2: per defense, frozen vs adaptive over defense_param
        for dname in defenses:
            fig, ax = plt.subplots(figsize=(9, 5))
            for att in ["frozen", "adaptive"]:
                sub = summary[
                    (summary["dataset"]       == ds) &
                    (summary["defense"]       == dname) &
                    (summary["attacker_type"] == att) &
                    (summary["classifier"]    == clf_show)
                ].sort_values("defense_param")
                if sub.empty:
                    continue
                ax.plot(sub["defense_param"], sub["balanced_accuracy_mean"],
                        marker="o", label=att, linewidth=1.5)
                ax.fill_between(
                    sub["defense_param"],
                    sub["balanced_accuracy_mean"] - sub["balanced_accuracy_std"],
                    sub["balanced_accuracy_mean"] + sub["balanced_accuracy_std"],
                    alpha=0.15,
                )
            ax.axhline(0.50, color="gray", linestyle="--", linewidth=1, label="chance")
            ax.set_xlabel("Defense Parameter")
            ax.set_ylabel("Balanced Accuracy")
            ax.set_title(f"{ds} | {dname} – Frozen vs Adaptive ({clf_show})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe = dname.replace("/", "_")
            out = plots_dir / f"plot2_{ds}_{safe}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Saved: {out}")

        # Plot 3: violin of per-split BA for adaptive attacker, grouped by defense
        # (only for stochastic defenses with multiple jitter_reps)
        stochastic = [d for d in defenses if "jitter" in d]
        if stochastic:
            sub_stoch = by_split[
                (by_split["dataset"]       == ds) &
                (by_split["attacker_type"] == "adaptive") &
                (by_split["classifier"]    == clf_show) &
                (by_split["defense"].isin(stochastic))
            ]
            if not sub_stoch.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                groups_v = [sub_stoch.loc[sub_stoch["defense"] == d, "balanced_accuracy"].values
                            for d in stochastic]
                ax.violinplot(groups_v, positions=range(len(stochastic)), showmedians=True)
                ax.set_xticks(range(len(stochastic)))
                ax.set_xticklabels(stochastic, rotation=30, ha="right", fontsize=8)
                ax.axhline(0.50, color="gray", linestyle="--", linewidth=1)
                ax.set_ylabel("Balanced Accuracy (per split)")
                ax.set_title(f"{ds} – Stochastic Defense BA Distribution ({clf_show})")
                fig.tight_layout()
                out = plots_dir / f"plot3_{ds}_violin_stochastic.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Terminal diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("QUICK DIAGNOSTICS")
    print("=" * 70)

    # Baseline adaptive per model
    print("\n--- Baseline adaptive BA / AUROC ---")
    base = summary[
        (summary["defense"] == "none") &
        (summary["attacker_type"] == "adaptive")
    ]
    if not base.empty:
        print(base[["dataset", "classifier", "featureset",
                    "balanced_accuracy_mean", "auroc_mean"]].to_string(index=False))

    # Per defense type: best (lowest) adaptive BA for LogReg + core
    for ds in summary["dataset"].unique():
        print(f"\n--- {ds}: Best (lowest) adaptive BA per defense type  (LogReg, core) ---")
        filt = summary[
            (summary["dataset"]       == ds) &
            (summary["attacker_type"] == "adaptive") &
            (summary["classifier"]    == "LogReg") &
            (summary["featureset"].str.startswith("core")) &
            (summary["defense"]       != "none")
        ]
        if filt.empty:
            print("  (no data)")
            continue
        idx  = filt.groupby("defense")["balanced_accuracy_mean"].idxmin()
        best = filt.loc[idx, ["defense", "defense_param", "balanced_accuracy_mean",
                               "auroc_mean", "overhead_median_ms_low",
                               "relative_overhead_low"]]
        print(best.to_string(index=False))

        # First point where adaptive BA < 0.75
        print(f"\n--- {ds}: First point adaptive BA < 0.75  (LogReg, core) ---")
        dropped = filt[filt["balanced_accuracy_mean"] < 0.75].sort_values(
            "overhead_median_ms_low")
        if dropped.empty:
            print("  Adaptive BA never drops below 0.75 in scanned range.")
        else:
            row = dropped.iloc[0]
            print(f"  defense={row.defense}  param={row.defense_param:.1f}  "
                  f"BA={row.balanced_accuracy_mean:.4f}  "
                  f"low_overhead={row.overhead_median_ms_low:.1f} ms  "
                  f"rel_overhead_low={row.relative_overhead_low:.3f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Offline defense sweep for offload timing channel – V2.")
    parser.add_argument("--llama-csv",    type=Path, default=DEFAULT_LLAMA_CSV)
    parser.add_argument("--qwen-csv",     type=Path, default=DEFAULT_QWEN_CSV)
    parser.add_argument("--results-dir",  type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-combined",  action="store_true",
                        help="Skip combined Llama+Qwen sweep.")
    parser.add_argument("--combined-dataset-feature", action="store_true",
                        help="Add dataset name as a feature in the combined sweep.")
    parser.add_argument("--featureset",   choices=["core", "extended", "both"],
                        default="both")
    parser.add_argument("--cv",           choices=["groupkfold", "loco"],
                        default="groupkfold",
                        help="Cross-validation strategy: groupkfold (default) or "
                             "loco (Leave-One-Concurrency-Out).")
    parser.add_argument("--n-splits",     type=int, default=N_SPLITS_DEFAULT,
                        help="Number of folds for GroupKFold (ignored for loco).")
    parser.add_argument("--jitter-repeats", type=int, default=20,
                        help="Number of independent noise draws for stochastic defenses.")
    parser.add_argument("--seed",         type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("\n=== Loading datasets ===")
    df_llama = load_dataset(args.llama_csv, "llama")
    df_qwen  = load_dataset(args.qwen_csv,  "qwen")

    featuresets: dict[str, list[str]] = {}
    if args.featureset in ("core",     "both"):
        featuresets["core"]     = FEATURES_CORE
    if args.featureset in ("extended", "both"):
        featuresets["extended"] = FEATURES_EXTENDED

    all_by_split: list[dict] = []

    # -------------------------------------------------------------------
    # Per-model sweeps
    # -------------------------------------------------------------------
    for ds_name, df in [("llama", df_llama), ("qwen", df_qwen)]:
        for fs_name, feat_cols in featuresets.items():
            avail = [c for c in feat_cols if c in df.columns]
            print(f"\n=== Sweep: dataset={ds_name}  featureset={fs_name}  "
                  f"features={avail} ===")
            rows = run_sweep(df, ds_name, fs_name, avail, rng,
                             cv_mode=args.cv,
                             n_splits=args.n_splits,
                             jitter_repeats=args.jitter_repeats)
            all_by_split.extend(rows)

    # -------------------------------------------------------------------
    # Combined Llama + Qwen (optional)
    # -------------------------------------------------------------------
    if not args.no_combined:
        df_combined = pd.concat([df_llama, df_qwen], ignore_index=True)
        # group key already unique: dataset_name + "_run" + run_id
        for fs_name, feat_cols in featuresets.items():
            avail = [c for c in feat_cols if c in df_combined.columns]
            if args.combined_dataset_feature and "dataset" not in avail:
                # encode dataset as numeric feature
                df_combined["dataset_id"] = (df_combined["dataset"] == "qwen").astype(float)
                avail = avail + ["dataset_id"]
            print(f"\n=== Sweep: dataset=combined  featureset={fs_name}  "
                  f"features={avail} ===")
            rows = run_sweep(df_combined, "combined", fs_name, avail, rng,
                             cv_mode=args.cv,
                             n_splits=args.n_splits,
                             jitter_repeats=args.jitter_repeats)
            all_by_split.extend(rows)

    # -------------------------------------------------------------------
    # Build DataFrames
    # -------------------------------------------------------------------
    by_split_df = pd.DataFrame(all_by_split)
    # Ensure all expected columns exist
    for col in BY_SPLIT_COLS:
        if col not in by_split_df.columns:
            by_split_df[col] = np.nan
    by_split_df = by_split_df[BY_SPLIT_COLS]

    summary_df = build_summary(by_split_df)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    out_by_split = results_dir / "defense_sweep_by_split.csv"
    out_summary  = results_dir / "defense_sweep_summary.csv"

    by_split_df.to_csv(out_by_split, index=False, float_format="%.5f")
    summary_df.to_csv(out_summary,   index=False, float_format="%.5f")

    print(f"\nSaved by_split : {out_by_split}  ({len(by_split_df)} rows)")
    print(f"Saved summary  : {out_summary}   ({len(summary_df)} rows)")

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    make_plots(summary_df, by_split_df, results_dir / "plots")

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------
    print_diagnostics(summary_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
