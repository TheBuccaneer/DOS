#!/usr/bin/env python3
"""
infer_offload_binary.py

Erste binäre Zustandsinferenz aus vLLM-Benchmark-Metriken:
    Klasse 0: offload_gb == 0  (reines GPU-Serving)
    Klasse 1: offload_gb  > 0  (CPU-Offloading aktiv)

Verwendung:
    python infer_offload_binary.py runs_summary.csv
    python infer_offload_binary.py runs_summary.csv --test-size 0.3 --random-state 42
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

FEATURES = [
    "median_tpot_ms",
    "median_itl_ms",
    "median_ttft_ms",
]


# ---------------------------------------------------------------------------
# Daten laden & vorbereiten
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["offload_gb"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[FEHLER] Fehlende Spalten: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df[required].dropna()
    df["label"] = (df["offload_gb"].astype(float) > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# (A) Einfache Threshold-Regel
# ---------------------------------------------------------------------------

def find_best_threshold(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Sucht für jedes Feature den Schwellwert, der auf dem Trainingssatz
    die beste Balanced Accuracy ergibt.
    Gibt zurück: (bestes Feature, beste Schwelle, erzielte Balanced Accuracy)
    """
    best_feature, best_thresh, best_score = None, None, -1.0

    for feature in FEATURES:
        values = X_train[feature].values
        # Kandidaten: alle Mittelpunkte zwischen sortierten unique-Werten
        candidates = np.unique(values)
        thresholds = (candidates[:-1] + candidates[1:]) / 2

        for thresh in thresholds:
            # Regel: über Schwelle -> offload aktiv (Label 1)
            preds = (values >= thresh).astype(int)
            score = balanced_accuracy_score(y_train, preds)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_thresh = thresh

    return best_feature, best_thresh, best_score


def apply_threshold(X: pd.DataFrame, feature: str, threshold: float) -> np.ndarray:
    return (X[feature].values >= threshold).astype(int)


# ---------------------------------------------------------------------------
# (B) Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


def predict_logistic(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    return model.predict(scaler.transform(X))


# ---------------------------------------------------------------------------
# Ausgabe-Hilfsfunktionen
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_metrics(y_true, y_pred, label: str) -> None:
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"\n  Confusion Matrix ({label}):")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0:    {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  Actual 1:    {cm[1,0]:>6}   {cm[1,1]:>6}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["offload=0", "offload>0"],
                                digits=4))


def print_feature_separation(X: pd.DataFrame, y: pd.Series) -> None:
    """Zeigt Mittelwert pro Klasse für jedes Feature."""
    print()
    header = f"  {'Feature':<20}  {'Mean (offload=0)':>18}  {'Mean (offload>0)':>18}  {'Diff %':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for feat in FEATURES:
        m0 = X.loc[y == 0, feat].mean()
        m1 = X.loc[y == 1, feat].mean()
        diff_pct = (m1 - m0) / (m0 + 1e-9) * 100
        print(f"  {feat:<20}  {m0:>18.3f}  {m1:>18.3f}  {diff_pct:>+8.1f}%")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binäre Offload-Zustandsinferenz aus vLLM-Metriken."
    )
    parser.add_argument("csv_path", help="Pfad zur runs_summary.csv")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Anteil Testdaten (Standard: 0.3)")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Daten laden ---
    df = load_data(args.csv_path)
    X = df[FEATURES]
    y = df["label"]

    section("DATENSATZ")
    print(f"\n  Samples gesamt:  {len(df)}")
    print(f"  Features:        {FEATURES}")
    vc = y.value_counts().sort_index()
    print(f"\n  Klassenverteilung:")
    print(f"    Label 0 (offload=0): {vc.get(0, 0):>4} Samples")
    print(f"    Label 1 (offload>0): {vc.get(1, 0):>4} Samples")

    # --- Feature-Trennung auf Gesamtdaten ---
    section("FEATURE-TRENNUNG (Gesamtdaten)")
    print_feature_separation(X, y)

    # --- Train/Test-Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"\n  Train: {len(X_train)} Samples  |  Test: {len(X_test)} Samples")

    # -----------------------------------------------------------------------
    # (A) Threshold-Regel
    # -----------------------------------------------------------------------
    section("(A) THRESHOLD-REGEL")
    best_feat, best_thresh, train_bacc = find_best_threshold(X_train, y_train)
    print(f"\n  Bestes Feature:  {best_feat}")
    print(f"  Schwelle:        {best_thresh:.4f} ms")
    print(f"  Balanced Acc. (Train): {train_bacc:.4f}")

    y_pred_thresh = apply_threshold(X_test, best_feat, best_thresh)
    print_metrics(y_test, y_pred_thresh, "Threshold")

    # -----------------------------------------------------------------------
    # (B) Logistic Regression
    # -----------------------------------------------------------------------
    section("(B) LOGISTIC REGRESSION")
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    y_pred_lr = predict_logistic(lr_model, scaler, X_test)
    print_metrics(y_test, y_pred_lr, "LogReg")

    # Koeffizienten ausgeben
    print("  Feature-Koeffizienten (standardisiert):")
    for feat, coef in zip(FEATURES, lr_model.coef_[0]):
        bar = "█" * int(abs(coef) * 5)
        sign = "+" if coef >= 0 else "-"
        print(f"    {feat:<20}  {sign}{abs(coef):.4f}  {bar}")

    # -----------------------------------------------------------------------
    # Zusammenfassung
    # -----------------------------------------------------------------------
    section("ZUSAMMENFASSUNG")
    acc_thresh = accuracy_score(y_test, y_pred_thresh)
    acc_lr     = accuracy_score(y_test, y_pred_lr)
    bacc_thresh = balanced_accuracy_score(y_test, y_pred_thresh)
    bacc_lr     = balanced_accuracy_score(y_test, y_pred_lr)

    print(f"\n  {'Modell':<25}  {'Accuracy':>10}  {'Bal. Acc.':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Threshold (' + best_feat[:12] + ')':<25}  {acc_thresh:>10.4f}  {bacc_thresh:>10.4f}")
    print(f"  {'Logistic Regression':<25}  {acc_lr:>10.4f}  {bacc_lr:>10.4f}")
    print(f"\n  Threshold-Feature: {best_feat} >= {best_thresh:.4f} ms  -> offload>0")
    print()


if __name__ == "__main__":
    main()
