"""
freeze_llama_estimator.py
=========================
Friert vor dem Availability-Pilot zwei Llama-State-Policies ein:
  1. Naive Threshold-Policy  : itl_mean_ms >= T  (Feature vorab festgelegt)
  2. State-Aware-Policy      : Logistic Regression auf ttft_ms,
                               itl_median_ms, itl_mean_ms

Beobachtungseinheit: ein einzelner Probe-Request
  input_len  = 256
  output_len = 64
  temperature = 0

Trainingsdata: llama_requests_summary.csv  (Request-Level)
Unterscheidet ausschließlich:
  Low State  : offload_gb == 0
  High State : offload_gb == 12

Ausführung:
    python freeze_llama_estimator.py [--force] [--input CSV] ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Freeze-Konstanten  (nicht per CLI änderbar)
# ---------------------------------------------------------------------------

LOW_OFFLOAD: int = 0
HIGH_OFFLOAD: int = 12
MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"

PROBE_REQUESTS: int = 1
PROBE_INPUT_LEN: int = 256
PROBE_OUTPUT_LEN: int = 64
PROBE_TEMPERATURE: float = 0.0

# Feature-Set für den State-Aware-Klassifikator
SA_FEATURES: list[str] = [
    "ttft_ms",
    "itl_median_ms",
    "itl_mean_ms",
]

# Feature für die naive Baseline — vorab festgelegt, nicht optimiert
NAIVE_FEATURE: str = "itl_mean_ms"

# Concurrency-Filter — vorab festgelegt
MIN_RUN_CONCURRENCY: int = 2

REQUIRED_COLUMNS: list[str] = list(dict.fromkeys(SA_FEATURES + ["offload_gb", "run_concurrency"]))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path(
    r"C:\projects\DOS\legacy\estimator_inputs\llama_requests_summary.csv"
)
DEFAULT_MODEL_OUT = Path(
    r"C:\projects\DOS\configs\state_estimator\llama_state_aware.joblib"
)
DEFAULT_FREEZE_JSON = Path(
    r"C:\projects\DOS\configs\state_estimator\llama_estimator_freeze.json"
)
DEFAULT_FOLDS_CSV = Path(
    r"C:\projects\DOS\results\estimator_freeze\llama_loco_folds.csv"
)
DEFAULT_OOF_CSV = Path(
    r"C:\projects\DOS\results\estimator_freeze\llama_oof_predictions.csv"
)
DEFAULT_REPORT = Path(
    r"C:\projects\DOS\results\estimator_freeze\llama_estimator_report.txt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze Llama state estimator.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    p.add_argument("--folds-csv", type=Path, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--oof-csv", type=Path, default=DEFAULT_OOF_CSV)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument(
        "--force",
        action="store_true",
        help="Überschreibe vorhandene Freeze-Artefakte.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def check_no_overwrite(paths: list[Path], force: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        names = "\n  ".join(str(p) for p in existing)
        sys.exit(
            f"Fehler: Folgende Freeze-Artefakte existieren bereits:\n  {names}\n"
            "Verwende --force zum Überschreiben."
        )


# ---------------------------------------------------------------------------
# Daten laden & validieren
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"Fehler: Eingabedatei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"Fehler: Fehlende Spalten in der CSV: {missing}")

    for col in REQUIRED_COLUMNS:
        try:
            pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as e:
            sys.exit(f"Fehler: Spalte '{col}' enthält nicht-numerische Werte: {e}")

    nan_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if nan_cols:
        sys.exit(f"Fehler: NaN-Werte in Spalten: {nan_cols}")

    inf_cols = [c for c in SA_FEATURES if np.isinf(df[c].values).any()]
    if inf_cols:
        sys.exit(f"Fehler: Unendliche Werte in Spalten: {inf_cols}")

    # Eindeutige Quellzeilennummer — unabhängig vom CSV-Index
    df = df.copy()
    df["_source_row_number"] = np.arange(len(df), dtype=int)

    return df


def validate_probe_profile(df: pd.DataFrame) -> str:
    """
    Prüft input_len/actual_output_len gegen das eingefrorene Profil,
    falls die Spalten vorhanden sind. Gibt den Validierungsstatus zurück.
    """
    input_col = "input_len" if "input_len" in df.columns else None
    output_col = "actual_output_len" if "actual_output_len" in df.columns else None

    if input_col is None and output_col is None:
        return (
            "documented_from_legacy_configuration_not_fully_validated_from_csv"
        )

    problems: list[str] = []
    if input_col:
        bad = df[input_col].dropna()
        bad = bad[~bad.between(PROBE_INPUT_LEN - 2, PROBE_INPUT_LEN + 2)]
        if not bad.empty:
            problems.append(
                f"{len(bad)} Requests mit input_len außerhalb "
                f"[{PROBE_INPUT_LEN-2}, {PROBE_INPUT_LEN+2}]"
            )
    if output_col:
        bad = df[output_col].dropna()
        bad = bad[bad != PROBE_OUTPUT_LEN]
        if not bad.empty:
            problems.append(
                f"{len(bad)} Requests mit actual_output_len != {PROBE_OUTPUT_LEN}"
            )

    if problems:
        sys.exit(
            "Fehler: Probe-Profil stimmt nicht mit CSV überein:\n  "
            + "\n  ".join(problems)
        )

    validated = []
    if input_col:
        validated.append(f"input_len≈{PROBE_INPUT_LEN}")
    if output_col:
        validated.append(f"actual_output_len={PROBE_OUTPUT_LEN}")
    return "validated_from_csv_columns: " + ", ".join(validated)


def filter_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["run_concurrency"] >= MIN_RUN_CONCURRENCY].copy()
    mask = df["offload_gb"].isin([LOW_OFFLOAD, HIGH_OFFLOAD])
    out = df[mask].copy().reset_index(drop=True)
    out["original_row_index"] = out["_source_row_number"]
    out["label"] = (out["offload_gb"] == HIGH_OFFLOAD).astype(int)
    return out


def validate_binary_df(df: pd.DataFrame) -> None:
    classes = sorted(df["label"].unique())
    if classes != [0, 1]:
        sys.exit(f"Fehler: Erwarte Klassen [0, 1], gefunden: {classes}")

    groups = sorted(df["run_concurrency"].unique())
    if len(groups) < 2:
        sys.exit(f"Fehler: Mindestens zwei Concurrency-Gruppen nötig, gefunden: {groups}")

    for g in groups:
        fold_classes = df.loc[df["run_concurrency"] == g, "label"].unique()
        if len(fold_classes) < 2:
            sys.exit(
                f"Fehler: Concurrency-Gruppe {g} enthält nur Klasse(n) {fold_classes}. "
                "LOCO-CV nicht möglich."
            )
        train_classes = df.loc[df["run_concurrency"] != g, "label"].unique()
        if len(train_classes) < 2:
            sys.exit(
                f"Fehler: Trainingsdaten ohne Gruppe {g} enthalten nur Klasse(n) "
                f"{train_classes}."
            )


# ---------------------------------------------------------------------------
# Naive Threshold
# ---------------------------------------------------------------------------

def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.sort(np.unique(values))
    if len(unique) < 2:
        return unique
    return (unique[:-1] + unique[1:]) / 2.0


def best_threshold_for_feature(
    train_vals: np.ndarray, y_train: np.ndarray
) -> tuple[float, float]:
    """Gibt (threshold, train_balanced_accuracy) zurück."""
    best_thr, best_ba = 0.0, -1.0
    for thr in candidate_thresholds(train_vals):
        preds = (train_vals >= thr).astype(int)
        ba = balanced_accuracy_score(y_train, preds)
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr, best_ba


# ---------------------------------------------------------------------------
# Leave-One-Concurrency-Out CV
# ---------------------------------------------------------------------------

def run_loco_cv(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    concurrencies = sorted(df["run_concurrency"].unique())
    fold_records: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for held_out in concurrencies:
        train_mask = df["run_concurrency"] != held_out
        test_mask = df["run_concurrency"] == held_out

        X_train = df.loc[train_mask, SA_FEATURES]
        y_train = df.loc[train_mask, "label"].values
        X_test = df.loc[test_mask, SA_FEATURES]
        y_test = df.loc[test_mask, "label"].values

        test_indices = df.index[test_mask].tolist()

        # --- Naive Threshold (itl_mean_ms, vorab festgelegt) ---
        thr, _ = best_threshold_for_feature(
            X_train[NAIVE_FEATURE].values, y_train
        )
        naive_preds = (X_test[NAIVE_FEATURE].values >= thr).astype(int)
        naive_ba = balanced_accuracy_score(y_test, naive_preds)
        naive_acc = float((naive_preds == y_test).mean())
        tn, fp, fn, tp = confusion_matrix(y_test, naive_preds, labels=[0, 1]).ravel()

        fold_records.append({
            "held_out_concurrency": int(held_out),
            "policy": "naive_threshold",
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "balanced_accuracy": naive_ba,
            "accuracy": naive_acc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "selected_feature": NAIVE_FEATURE,
            "selected_threshold_ms": thr,
        })

        for i, idx in enumerate(test_indices):
            oof_rows.append({
                "original_row_index": int(df.loc[idx, "original_row_index"]),
                "run_concurrency": int(held_out),
                "offload_gb": int(df.loc[idx, "offload_gb"]),
                "true_label": int(y_test[i]),
                "policy": "naive_threshold",
                "predicted_label": int(naive_preds[i]),
                "high_state_probability": None,
                "held_out_concurrency": int(held_out),
            })

        # --- State-Aware (Logistic Regression) ---
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            )),
        ])
        pipe.fit(X_train, y_train)
        lr_preds = pipe.predict(X_test)
        lr_proba = pipe.predict_proba(X_test)[:, 1]
        lr_ba = balanced_accuracy_score(y_test, lr_preds)
        lr_acc = float((lr_preds == y_test).mean())
        tn2, fp2, fn2, tp2 = confusion_matrix(
            y_test, lr_preds, labels=[0, 1]
        ).ravel()

        fold_records.append({
            "held_out_concurrency": int(held_out),
            "policy": "state_aware",
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "balanced_accuracy": lr_ba,
            "accuracy": lr_acc,
            "tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2),
            "selected_feature": None,
            "selected_threshold_ms": None,
        })

        for i, idx in enumerate(test_indices):
            oof_rows.append({
                "original_row_index": int(df.loc[idx, "original_row_index"]),
                "run_concurrency": int(held_out),
                "offload_gb": int(df.loc[idx, "offload_gb"]),
                "true_label": int(y_test[i]),
                "policy": "state_aware",
                "predicted_label": int(lr_preds[i]),
                "high_state_probability": float(lr_proba[i]),
                "held_out_concurrency": int(held_out),
            })

    return fold_records, pd.DataFrame(oof_rows)


# ---------------------------------------------------------------------------
# Finale Modelle auf allen Daten
# ---------------------------------------------------------------------------

def fit_final_models(df: pd.DataFrame) -> tuple[Pipeline, float]:
    X = df[SA_FEATURES]
    y = df["label"].values

    final_thr, _ = best_threshold_for_feature(
        df[NAIVE_FEATURE].values, y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        )),
    ])
    pipe.fit(X, y)

    return pipe, final_thr


# ---------------------------------------------------------------------------
# Paketversionen
# ---------------------------------------------------------------------------

def get_versions() -> dict[str, str]:
    pkgs = ["scikit-learn", "pandas", "numpy", "joblib"]
    result: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in pkgs:
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            result[pkg] = "unknown"
    return result


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------

def build_report(
    csv_path: Path,
    df_binary: pd.DataFrame,
    excluded: list[int],
    excluded_conc: list[int],
    included_conc: list[int],
    fold_records: list[dict[str, Any]],
    naive_cv_mean: float,
    naive_cv_std: float,
    lr_cv_mean: float,
    lr_cv_std: float,
    final_thr: float,
    freeze_ts: str,
    versions: dict[str, str],
) -> str:
    lines: list[str] = []
    a = lines.append

    a("=" * 72)
    a("LLAMA STATE ESTIMATOR — FREEZE REPORT")
    a("=" * 72)
    a(f"Freeze Timestamp   : {freeze_ts}")
    a(f"Source CSV         : {csv_path}")
    a(f"Low State          : offload_gb == {LOW_OFFLOAD}")
    a(f"High State         : offload_gb == {HIGH_OFFLOAD}")
    a(f"Excluded Offload   : {excluded}")
    a(f"Min Concurrency    : {MIN_RUN_CONCURRENCY}")
    a(f"Excluded Conc.     : {excluded_conc}")
    a(f"Included Conc.     : {included_conc}")
    a(f"Concurrency note   : Concurrency 1 was excluded a priori because it "
      f"represents a distinct low-load regime and is outside the frozen pilot scope.")
    a(f"Probe Requests     : {PROBE_REQUESTS}")
    a(f"Probe Input/Output : {PROBE_INPUT_LEN} / {PROBE_OUTPUT_LEN} tokens")
    a(f"Probe Temperature  : {PROBE_TEMPERATURE}")
    a(f"SA Features        : {SA_FEATURES}")
    a(f"Naive Feature      : {NAIVE_FEATURE}  (vorab festgelegt)")
    a(f"Rows (binary)      : {len(df_binary)}")
    counts = df_binary["label"].value_counts().sort_index()
    a(f"Class counts       : Low={counts.get(0, 0)}  High={counts.get(1, 0)}")
    a("")

    a("LEAVE-ONE-CONCURRENCY-OUT ERGEBNISSE")
    a("-" * 72)
    hdr = (
        f"{'Conc':>5}  {'Policy':<20}  "
        f"{'BA':>6}  {'Acc':>6}  "
        f"{'TN':>4}{'FP':>4}{'FN':>4}{'TP':>4}  "
        f"{'Thr (ms)':>10}"
    )
    a(hdr)
    a("-" * 72)
    for r in fold_records:
        thr_str = (
            f"{r['selected_threshold_ms']:.2f}"
            if r["selected_threshold_ms"] is not None
            else ""
        )
        a(
            f"{r['held_out_concurrency']:>5}  {r['policy']:<20}  "
            f"{r['balanced_accuracy']:>6.4f}  {r['accuracy']:>6.4f}  "
            f"{r['tn']:>4}{r['fp']:>4}{r['fn']:>4}{r['tp']:>4}  "
            f"{thr_str:>10}"
        )

    a("")
    a("AGGREGIERTE CV-ERGEBNISSE  (Std: ddof=0, Populationsformel)")
    a("-" * 72)
    a(f"Naive Threshold ({NAIVE_FEATURE}) : "
      f"BA = {naive_cv_mean:.4f} ± {naive_cv_std:.4f}")
    a(f"State-Aware (LR)               : "
      f"BA = {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")

    a("")
    a("EINGEFRORENE POLICIES")
    a("-" * 72)
    a(f"Naive  : {NAIVE_FEATURE} >= {final_thr:.4f} ms → High State")
    a(f"SA     : Logistic Regression, Entscheidungsschwelle 0.5")
    a(f"Status : FROZEN_BEFORE_AVAILABILITY_PILOT")

    a("")
    a("PAKETVERSIONEN")
    a("-" * 72)
    for k, v in versions.items():
        a(f"  {k:<18}: {v}")

    a("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atomares Schreiben
# ---------------------------------------------------------------------------

def write_all_artifacts(
    tmp_dir: Path,
    final_targets: dict[str, Path],
    completion_path: Path,
    final_pipe: Pipeline,
    freeze_config: dict[str, Any],
    folds_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    report_text: str,
) -> str:
    """
    Schreibt alle Artefakte in tmp_dir, berechnet den Modell-SHA256,
    ersetzt dann die Zieldateien einzeln mit os.replace (Windows-sicher),
    und schreibt als Letztes den Completion-Marker.
    Gibt den Modell-SHA256 zurück.
    """
    tmp_model  = tmp_dir / "model.joblib"
    tmp_json   = tmp_dir / "freeze.json"
    tmp_folds  = tmp_dir / "folds.csv"
    tmp_oof    = tmp_dir / "oof.csv"
    tmp_report = tmp_dir / "report.txt"

    # Modell zuerst — SHA256 wird danach berechnet
    joblib.dump(final_pipe, tmp_model)
    model_sha256 = sha256_of_file(tmp_model)

    # Modell-Hash in freeze_config einbetten
    freeze_config["model_artifact_sha256"] = model_sha256

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(freeze_config, f, indent=2)
    folds_df.to_csv(tmp_folds, index=False)
    oof_df.to_csv(tmp_oof, index=False)
    tmp_report.write_text(report_text, encoding="utf-8")

    # Zielverzeichnisse anlegen
    for target in list(final_targets.values()) + [completion_path]:
        target.parent.mkdir(parents=True, exist_ok=True)

    # Atomar ersetzen (Windows-sicher)
    os.replace(tmp_model,  final_targets["model"])
    os.replace(tmp_json,   final_targets["json"])
    os.replace(tmp_folds,  final_targets["folds"])
    os.replace(tmp_oof,    final_targets["oof"])
    os.replace(tmp_report, final_targets["report"])

    return model_sha256


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    completion_path = args.freeze_json.parent / "llama_estimator_freeze.COMPLETE"

    final_targets = {
        "model":  args.model_out,
        "json":   args.freeze_json,
        "folds":  args.folds_csv,
        "oof":    args.oof_csv,
        "report": args.report,
    }

    check_no_overwrite(list(final_targets.values()) + [completion_path], args.force)

    # --- Daten ---
    print(f"Lade: {args.input}")
    df_raw = load_and_validate(args.input)
    csv_hash = sha256_of_file(args.input)
    script_sha256 = sha256_of_file(Path(__file__))

    probe_profile_validation = validate_probe_profile(df_raw)
    print(f"Probe-Profil-Validierung: {probe_profile_validation}")

    excluded = sorted(
        int(v) for v in df_raw["offload_gb"].unique()
        if int(v) not in (LOW_OFFLOAD, HIGH_OFFLOAD)
    )
    print(
        f"Behalte offload_gb ∈ {{{LOW_OFFLOAD}, {HIGH_OFFLOAD}}} GB, "
        f"verwerfe {excluded}"
    )

    df_binary = filter_binary(df_raw)
    validate_binary_df(df_binary)

    # included/excluded concurrencies aus dem bereits gefilterten DataFrame
    included_conc = sorted(int(v) for v in df_binary["run_concurrency"].unique())
    excluded_conc = sorted(
        int(v) for v in df_raw["run_concurrency"].unique()
        if int(v) < MIN_RUN_CONCURRENCY
    )

    counts = df_binary["label"].value_counts().sort_index()
    print(
        f"Binärer Datensatz: {len(df_binary)} Requests  "
        f"(Low={counts.get(0, 0)}, High={counts.get(1, 0)})"
    )

    # --- LOCO CV ---
    print("\nStarte Leave-One-Concurrency-Out CV …")
    fold_records, oof_df = run_loco_cv(df_binary)
    folds_df = pd.DataFrame(fold_records)

    naive_bas = folds_df.loc[
        folds_df["policy"] == "naive_threshold", "balanced_accuracy"
    ].values
    lr_bas = folds_df.loc[
        folds_df["policy"] == "state_aware", "balanced_accuracy"
    ].values

    naive_cv_mean = float(naive_bas.mean())
    naive_cv_std  = float(naive_bas.std())   # ddof=0
    lr_cv_mean    = float(lr_bas.mean())
    lr_cv_std     = float(lr_bas.std())

    # --- Finale Modelle ---
    print("Trainiere finale Modelle auf allen 0-vs-12-Daten …")
    final_pipe, final_thr = fit_final_models(df_binary)

    # --- Freeze-Config (model_artifact_sha256 wird in write_all_artifacts ergänzt) ---
    freeze_ts = datetime.now(timezone.utc).isoformat()
    versions  = get_versions()
    lr_params = final_pipe.named_steps["clf"].get_params()

    freeze_config: dict[str, Any] = {
        "freeze_timestamp": freeze_ts,
        "freeze_script_sha256": script_sha256,
        "source_csv": str(args.input),
        "source_csv_sha256": csv_hash,
        "model_name": MODEL_NAME,
        "low_offload_gb": LOW_OFFLOAD,
        "high_offload_gb": HIGH_OFFLOAD,
        "features": SA_FEATURES,
        "naive_feature": NAIVE_FEATURE,
        "excluded_offload_values": excluded,
        "minimum_run_concurrency": MIN_RUN_CONCURRENCY,
        "excluded_concurrencies": excluded_conc,
        "included_concurrencies": included_conc,
        "probe_requests": PROBE_REQUESTS,
        "probe_input_len": PROBE_INPUT_LEN,
        "probe_output_len": PROBE_OUTPUT_LEN,
        "probe_temperature": PROBE_TEMPERATURE,
        "probe_profile_validation": probe_profile_validation,
        "number_of_rows": len(df_binary),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "naive_threshold_feature": NAIVE_FEATURE,
        "naive_threshold_ms": float(final_thr),
        "naive_threshold_cv_balanced_accuracy_mean": naive_cv_mean,
        "naive_threshold_cv_balanced_accuracy_std": naive_cv_std,
        "state_aware_cv_balanced_accuracy_mean": lr_cv_mean,
        "state_aware_cv_balanced_accuracy_std": lr_cv_std,
        "state_aware_decision_threshold": 0.5,
        "logistic_regression_parameters": {
            k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
            for k, v in lr_params.items()
        },
        "random_state": 42,
        "cv_std_ddof": 0,
        "package_versions": versions,
        "status": "FROZEN_BEFORE_AVAILABILITY_PILOT",
    }

    # --- Bericht ---
    report_text = build_report(
        csv_path=args.input,
        df_binary=df_binary,
        excluded=excluded,
        excluded_conc=excluded_conc,
        included_conc=included_conc,
        fold_records=fold_records,
        naive_cv_mean=naive_cv_mean,
        naive_cv_std=naive_cv_std,
        lr_cv_mean=lr_cv_mean,
        lr_cv_std=lr_cv_std,
        final_thr=final_thr,
        freeze_ts=freeze_ts,
        versions=versions,
    )

    # --- Atomar schreiben ---
    with tempfile.TemporaryDirectory() as tmp:
        model_sha256 = write_all_artifacts(
            tmp_dir=Path(tmp),
            final_targets=final_targets,
            completion_path=completion_path,
            final_pipe=final_pipe,
            freeze_config=freeze_config,
            folds_df=folds_df,
            oof_df=oof_df,
            report_text=report_text,
        )

    # --- Completion-Marker (als Letztes) ---
    completion_content: dict[str, Any] = {
        "freeze_timestamp": freeze_ts,
        "source_csv_sha256": csv_hash,
        "model_artifact_sha256": model_sha256,
        "completion_marker": completion_path.name,
        "status": "FROZEN_BEFORE_AVAILABILITY_PILOT",
    }
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(completion_content, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FREEZE ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Naive Threshold  : BA = {naive_cv_mean:.4f} ± {naive_cv_std:.4f}")
    print(f"State-Aware (LR) : BA = {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")
    print(f"Naive Policy     : {NAIVE_FEATURE} >= {final_thr:.4f} ms")
    print(f"Modell-SHA256    : {model_sha256[:16]}…")
    print(f"Status           : FROZEN_BEFORE_AVAILABILITY_PILOT")
    print()
    for name, path in final_targets.items():
        print(f"  {name:<8}: {path}")
    print(f"  complete : {completion_path}")


if __name__ == "__main__":
    main()
