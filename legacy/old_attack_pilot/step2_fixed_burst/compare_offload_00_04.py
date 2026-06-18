from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(".")
CSV_00 = BASE / "00" / "runs_summary_rerun.csv"
CSV_04 = BASE / "04" / "runs_summary_rerun.csv"

METRICS = [
    "median_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
]


def load_runs(path: Path, offload_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Nur Victim-Runs
    df = df[df["role"] == "victim"].copy()

    # Für Sicherheit numerisch machen
    for col in ["run_id", "offload_gb"] + METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["offload_label"] = offload_label
    return df


def summarize_block(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")

    grouped = (
        df.groupby(["offload_label", "condition"], dropna=False)[METRICS]
        .mean()
        .reset_index()
        .sort_values(["offload_label", "condition"])
    )

    if grouped.empty:
        print("[WARN] Keine Daten in diesem Block.")
        return

    for offload in grouped["offload_label"].unique():
        print(f"\n--- offload {offload} ---")
        sub = grouped[grouped["offload_label"] == offload].copy()

        # Schöne Ausgabe
        print(sub[["condition"] + METRICS].to_string(index=False))

        # Delta B - A
        a = sub[sub["condition"] == "cond_a"]
        b = sub[sub["condition"] == "cond_b"]

        if len(a) == 1 and len(b) == 1:
            print(f"\nDelta cond_b - cond_a (offload {offload})")
            for metric in METRICS:
                va = float(a.iloc[0][metric])
                vb = float(b.iloc[0][metric])
                delta = vb - va
                rel = (delta / va * 100.0) if va != 0 else float("nan")
                print(
                    f"  {metric:16s}  "
                    f"cond_a={va:12.3f}  "
                    f"cond_b={vb:12.3f}  "
                    f"delta={delta:12.3f}  "
                    f"rel={rel:8.2f}%"
                )
        else:
            print(f"[WARN] cond_a/cond_b für offload {offload} nicht vollständig gefunden.")

    # Zusatz: Offload 4 vs 0 innerhalb gleicher condition
    print("\n--- Zusatz: offload 4 - offload 0 je condition ---")
    for cond in ["cond_a", "cond_b"]:
        s0 = grouped[(grouped["offload_label"] == "0") & (grouped["condition"] == cond)]
        s4 = grouped[(grouped["offload_label"] == "4") & (grouped["condition"] == cond)]

        if len(s0) == 1 and len(s4) == 1:
            print(f"\nCondition {cond}")
            for metric in METRICS:
                v0 = float(s0.iloc[0][metric])
                v4 = float(s4.iloc[0][metric])
                delta = v4 - v0
                rel = (delta / v0 * 100.0) if v0 != 0 else float("nan")
                print(
                    f"  {metric:16s}  "
                    f"off0={v0:12.3f}  "
                    f"off4={v4:12.3f}  "
                    f"delta={delta:12.3f}  "
                    f"rel={rel:8.2f}%"
                )


def main() -> None:
    if not CSV_00.exists():
        raise FileNotFoundError(f"Fehlt: {CSV_00}")
    if not CSV_04.exists():
        raise FileNotFoundError(f"Fehlt: {CSV_04}")

    df00 = load_runs(CSV_00, "0")
    df04 = load_runs(CSV_04, "4")

    all_df = pd.concat([df00, df04], ignore_index=True)

    # Alle Runs
    summarize_block(all_df, "ALLE RUNS")

    # Nur Runs 2..5
    df_25 = all_df[all_df["run_id"].between(2, 5)].copy()
    summarize_block(df_25, "NUR RUNS 2..5")


if __name__ == "__main__":
    main()
