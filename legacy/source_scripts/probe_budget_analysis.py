#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BUDGETS = [1, 2, 3, 5, 10, 20]
OFFLOAD_CLASSES = [0, 2, 4, 8, 12, 16]

FEATURES = [
    "median_ttft_ms",
    "mean_ttft_ms",
    "median_itl_mean_ms",
    "mean_itl_mean_ms",
    "median_itl_median_ms",
    "p95_itl_mean_ms",
    "median_decode_time_ms",
]


def load_requests(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "extracted_rerun" / "requests_summary_rerun.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)

    needed = [
        "file_path",
        "file_name",
        "offload_gb",
        "run_concurrency",
        "run_id",
        "request_idx",
        "ttft_ms",
        "itl_mean_ms",
        "itl_median_ms",
        "decode_time_ms",
        "request_success",
    ]

    # Older Llama extracts may not have source_subdir yet.
    # Prefer source_subdir when present; otherwise fall back to parent_dir;
    # otherwise use a constant value.
    if "source_subdir" in df.columns:
        source_col = "source_subdir"
    elif "parent_dir" in df.columns:
        source_col = "parent_dir"
    else:
        source_col = None

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    keep_cols = needed + ([source_col] if source_col is not None else [])
    df = df[keep_cols].copy()

    if source_col is None:
        df["source_subdir"] = "unknown"
    elif source_col != "source_subdir":
        df["source_subdir"] = df[source_col].astype(str)
    df = df[df["request_success"] == True].copy()

    for c in ["offload_gb", "run_concurrency", "run_id", "request_idx"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ["ttft_ms", "itl_mean_ms", "itl_median_ms", "decode_time_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "offload_gb",
        "run_concurrency",
        "run_id",
        "request_idx",
        "ttft_ms",
        "itl_mean_ms",
        "itl_median_ms",
        "decode_time_ms",
    ]).copy()

    df["offload_gb"] = df["offload_gb"].astype(int)
    df["run_concurrency"] = df["run_concurrency"].astype(int)
    df["run_id"] = df["run_id"].astype(int)

    df = df[df["offload_gb"].isin(OFFLOAD_CLASSES)].copy()

    # Eindeutiger Run-Key
    df["run_key"] = (
        df["source_subdir"].astype(str)
        + "::"
        + df["file_name"].astype(str)
    )

    return df.reset_index(drop=True)


def sample_budget_features(reqs: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for run_key, g in reqs.groupby("run_key"):
        if len(g) < budget:
            continue

        sampled_idx = rng.choice(g.index.to_numpy(), size=budget, replace=False)
        s = reqs.loc[sampled_idx]

        first = g.iloc[0]

        row = {
            "run_key": run_key,
            "offload_gb": int(first["offload_gb"]),
            "run_concurrency": int(first["run_concurrency"]),
            "run_id": int(first["run_id"]),
            "budget": budget,

            "median_ttft_ms": float(s["ttft_ms"].median()),
            "mean_ttft_ms": float(s["ttft_ms"].mean()),

            "median_itl_mean_ms": float(s["itl_mean_ms"].median()),
            "mean_itl_mean_ms": float(s["itl_mean_ms"].mean()),

            "median_itl_median_ms": float(s["itl_median_ms"].median()),
            "p95_itl_mean_ms": float(s["itl_mean_ms"].quantile(0.95)),

            "median_decode_time_ms": float(s["decode_time_ms"].median()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def eval_logreg(df: pd.DataFrame, mode: str, seed: int) -> dict:
    X = df[FEATURES].copy()

    if mode == "binary":
        y = (df["offload_gb"] > 0).astype(int)
    elif mode == "multiclass":
        y = df["offload_gb"].astype(int)
    else:
        raise ValueError(mode)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train)

    pred = clf.predict(X_test_s)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
    }


def analyze_one_model(label: str, run_dir: Path, repeats: int) -> list[dict]:
    reqs = load_requests(run_dir)
    rows = []

    print(f"\n=== {label} ===")
    print(f"Requests: {len(reqs)}")
    print(f"Runs:     {reqs['run_key'].nunique()}")

    for budget in BUDGETS:
        print(f"  budget={budget}")

        for rep in range(repeats):
            seed = 100000 + rep * 100 + budget

            budget_df = sample_budget_features(reqs, budget=budget, seed=seed)

            for mode in ["binary", "multiclass"]:
                r = eval_logreg(budget_df, mode=mode, seed=seed)

                rows.append({
                    "llm_model": label,
                    "budget_requests": budget,
                    "mode": mode,
                    "repeat": rep,
                    "n_runs": len(budget_df),
                    "accuracy": r["accuracy"],
                    "balanced_accuracy": r["balanced_accuracy"],
                })

    return rows


def summarize(all_trials: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_trials
        .groupby(["llm_model", "mode", "budget_requests"], as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            n_runs=("n_runs", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .sort_values(["llm_model", "mode", "budget_requests"])
    )

    for c in [
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]:
        summary[c] = summary[c].round(4)

    summary["n_runs"] = summary["n_runs"].round(1)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("probe_budget_results"))
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(analyze_one_model("Llama-3.1-8B-Instruct", args.llama_dir, args.repeats))
    rows.extend(analyze_one_model("Qwen2.5-7B-Instruct", args.qwen_dir, args.repeats))

    all_trials = pd.DataFrame(rows)
    summary = summarize(all_trials)

    all_trials_path = args.outdir / "probe_budget_all_trials.csv"
    summary_path = args.outdir / "probe_budget_summary.csv"

    all_trials.to_csv(all_trials_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Probe budget summary ===")
    print(summary.to_string(index=False))

    print("\nWrote:")
    print(f"  {all_trials_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()