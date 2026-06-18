#!/usr/bin/env python3
"""
extract_vllm_rerun.py  —  Python-Runner-Format (überarbeitet)

Liest rekursiv einen Wurzelordner mit Python-Runner-JSONs und erzeugt:

  A. runs_summary_rerun.csv      — eine Zeile pro JSON/Run
  B. requests_summary_rerun.csv  — eine Zeile pro Request innerhalb eines Runs

Neues Dateinamensschema:
  offload<N>_conc<M>_run<R>.json
  offload<N>_attacker_conc<M>_run<R>.json

Wichtigste Unterschiede zum alten vLLM-bench-Format:
  1. Latenz-Metriken sind verschachtelt: ttft_ms.mean, ttft_ms.p95, ...
  2. Request-Rohdaten liegen in `individual_request_results` (nicht mehr in
     separaten Top-Level-Arrays wie ttfts / itls / errors).
  3. `itl_sequence` im Request ist bereits in Millisekunden (NICHT in Sekunden).
  4. Durchsatz-Felder heißen jetzt duration_s, drain_s, submitted, run_no, concurrency.
  5. role (victim/attacker) wird aus Dateinamen ODER parent_dir inferiert.

Verwendung:
  python extract_vllm_rerun.py /pfad/zur/wurzel
  python extract_vllm_rerun.py /pfad/zur/wurzel --outdir extracted_rerun --print-head 5
  python extract_vllm_rerun.py /pfad/zur/wurzel --parquet
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bekannte gültige Werte (für Sanity Checks)
# ---------------------------------------------------------------------------

VALID_OFFLOAD_GB    = {0, 2, 4, 8, 12, 16}
VALID_CONCURRENCIES = {1, 2, 4, 8, 12, 16}

# ---------------------------------------------------------------------------
# Run-Level Spaltenreihenfolge  (angepasst auf neues Format)
# ---------------------------------------------------------------------------

RUN_COLUMNS = [
    "file_path", "file_name", "parent_dir", "source_subdir",
    "experiment_id", "server_config_label", "model_name",
    "offload_gb", "run_concurrency", "run_id",
    "condition", "role",
    "input_len", "output_len", "temperature",
    # NEU: Durchsatz-/Zeitfelder des Python-Runners
    "window_secs", "duration_s", "drain_s",
    "submitted", "completed", "failed",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    # Latenz-Metriken (flat, aus verschachtelten Blöcken)
    "mean_ttft_ms",   "median_ttft_ms",   "std_ttft_ms",
    "p50_ttft_ms",    "p95_ttft_ms",      "p99_ttft_ms",
    "mean_tpot_ms",   "median_tpot_ms",   "std_tpot_ms",
    "p50_tpot_ms",    "p95_tpot_ms",      "p99_tpot_ms",
    "mean_itl_ms",    "median_itl_ms",    "std_itl_ms",
    "p50_itl_ms",     "p95_itl_ms",       "p99_itl_ms",
    "mean_e2el_ms",   "median_e2el_ms",   "std_e2el_ms",
    "p50_e2el_ms",    "p95_e2el_ms",      "p99_e2el_ms",
]

# ---------------------------------------------------------------------------
# Request-Level Spaltenreihenfolge  (angepasst auf individual_request_results)
# ---------------------------------------------------------------------------

REQ_COLUMNS = [
    "file_path", "file_name", "parent_dir", "source_subdir",
    "experiment_id", "server_config_label", "model_name",
    "offload_gb", "run_concurrency", "run_id",
    "condition", "role",
    "request_idx",
    "input_len", "target_output_len", "actual_output_len",
    "request_success", "error_text",
    "start_time",
    "ttft_s", "ttft_ms",
    "decode_time_ms",
    "e2el_ms",
    "tpot_ms",
    "itl_count",
    "itl_mean_ms", "itl_median_ms", "itl_std_ms",
    "itl_min_ms",  "itl_max_ms",    "itl_sum_ms",
    "itl_p95_ms",  "itl_p99_ms",
    "first_4_itl_mean_ms", "last_4_itl_mean_ms",
    "tail_ratio_p99_over_median",
    "generated_text_preview",
]

# ---------------------------------------------------------------------------
# Hilfsfunktionen: Metadaten aus Pfad/Dateinamen extrahieren
# ---------------------------------------------------------------------------

def _int_or_none(s) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def extract_offload_gb(name: str) -> int | None:
    """Extrahiert offload_gb aus Dateiname oder Ordnername."""
    import re
    m = re.search(r"(?:offload|offset)_?(\d+)", name)
    return _int_or_none(m.group(1)) if m else None


def extract_run_concurrency(stem: str) -> int | None:
    import re
    m = re.search(r"_conc(\d+)(?:_|$)", stem)
    return _int_or_none(m.group(1)) if m else None


def extract_run_id(stem: str) -> int | None:
    import re
    m = re.search(r"_run(\d+)$", stem)
    return _int_or_none(m.group(1)) if m else None


def _infer_role_from_stem(stem: str) -> str | None:
    """Inferiert role aus Dateinamen — attacker-Dateien enthalten '_attacker_'."""
    if "attacker" in stem:
        return "attacker"
    return None


# ---------------------------------------------------------------------------
# Scalar-Getter (kein Crash bei fehlenden Feldern)
# ---------------------------------------------------------------------------

def _get(data: dict, key: str, default=None):
    v = data.get(key, default)
    if isinstance(v, (list, dict)):
        return default
    return v


def _coerce_int(v, default=None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_float(v, default=None) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Fallback-Inferenz für condition/role aus Unterordnernamen
# Konvention: cond_a_victim_only, cond_b_victim_plus_burst
# ---------------------------------------------------------------------------

def _infer_condition(parent_dir: str) -> str | None:
    import re
    m = re.search(r"(cond_[a-z])", parent_dir)
    return m.group(1) if m else None


def _infer_role(parent_dir: str) -> str | None:
    if "attacker" in parent_dir:
        return "attacker"
    if "victim" in parent_dir:
        return "victim"
    return None


# ---------------------------------------------------------------------------
# NEU: Verschachtelte Latenz-Metrikblöcke flach extrahieren
#
# Altes Format: flache Top-Level-Felder  mean_ttft_ms, p95_ttft_ms, ...
# Neues Format: verschachtelter Block    ttft_ms: { mean, median, std, p50, p95, p99 }
#
# Diese Funktion liest beide Varianten und bevorzugt das neue Format.
# ---------------------------------------------------------------------------

def _extract_metric_block(data: dict, metric: str) -> dict:
    """
    Extrahiert einen verschachtelten Metrik-Block (neues Format) in flache Spalten.
    Fallback auf alte flache Felder (falls doch altes Format vorliegt).

    metric: 'ttft', 'tpot', 'itl', 'e2el'
    Gibt zurück: { 'mean_ttft_ms': ..., 'p95_ttft_ms': ..., ... }
    """
    result = {}
    block_key = f"{metric}_ms"   # z.B. "ttft_ms"
    block = data.get(block_key)  # erwartet dict mit mean/median/std/p50/p95/p99

    for stat in ("mean", "median", "std"):
        col = f"{stat}_{metric}_ms"
        if isinstance(block, dict):
            result[col] = _coerce_float(block.get(stat))
        else:
            # Fallback: altes flaches Format
            result[col] = _coerce_float(data.get(col))

    for pct in ("p50", "p95", "p99"):
        col = f"{pct}_{metric}_ms"
        if isinstance(block, dict):
            result[col] = _coerce_float(block.get(pct))
        else:
            result[col] = _coerce_float(data.get(col))

    return result


# ---------------------------------------------------------------------------
# Run-Level Verarbeitung
# ---------------------------------------------------------------------------

def process_run(json_path: Path) -> dict | None:
    """Liest eine JSON-Datei und gibt ein Run-Level-Dict zurück."""
    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Ungültiges JSON, überspringe: {json_path}  ({e})", file=sys.stderr)
        return None
    except OSError as e:
        print(f"  [WARN] Nicht lesbar, überspringe: {json_path}  ({e})", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"  [WARN] Kein JSON-Objekt (top-level), überspringe: {json_path}", file=sys.stderr)
        return None

    stem       = json_path.stem
    parent_dir = json_path.parent.name

    # offload_gb: explizite None-Prüfung (0 ist falsy!)
    _og_stem = extract_offload_gb(stem)
    _og_dir  = extract_offload_gb(parent_dir)
    _og_json = _coerce_int(data.get("offload_gb"))
    # Priorität: Dateiname > Ordnername > JSON-Feld
    if _og_stem is not None:
        offload_gb = _og_stem
    elif _og_dir is not None:
        offload_gb = _og_dir
    else:
        offload_gb = _og_json
    # Warnung bei Widerspruch zwischen nicht-None Quellen
    _og_sources = {s: v for s, v in [("stem", _og_stem), ("dir", _og_dir), ("json", _og_json)] if v is not None}
    if len(set(_og_sources.values())) > 1:
        print(f"  [WARN] offload_gb-Mismatch in {json_path.name}: {_og_sources}", file=sys.stderr)

    # run_concurrency: Dateiname hat Priorität, dann JSON-Felder
    run_concurrency = (
        extract_run_concurrency(stem)
        or _coerce_int(data.get("concurrency"))
        or _coerce_int(data.get("max_concurrency"))
    )

    # run_id: aus _runN im Dateinamen, Fallback auf run_no im JSON
    run_id = (
        extract_run_id(stem)
        or _coerce_int(data.get("run_no"))
    )

    # role: Dateiname hat Priorität (attacker-Dateien tragen _attacker_ im Namen),
    # dann JSON-Feld, dann Fallback aus parent_dir-Konvention.
    role = (
        _infer_role_from_stem(stem)
        or _get(data, "role")
        or _infer_role(parent_dir)
    )

    row: dict = {
        "file_path":           str(json_path.resolve()),
        "file_name":           json_path.name,
        "parent_dir":          parent_dir,
        # source_subdir: der direkte Elternordner des JSON — z.B. cond_a_victim_only.
        # Wichtig bei gleichen Dateinamen in verschiedenen Unterordnern (Utility-Test).
        "source_subdir":       parent_dir,
        "experiment_id":       _get(data, "experiment_id"),
        "server_config_label": _get(data, "server_config_label"),
        "model_name":          _get(data, "model_name") or _get(data, "model_id"),
        "offload_gb":          offload_gb,
        "run_concurrency":     run_concurrency,
        "run_id":              run_id,
        # condition/role: primär aus JSON/Dateiname, Fallback aus parent_dir-Name
        "condition":           _get(data, "condition") or _infer_condition(parent_dir),
        "role":                role,
        # Prompt-Parameter
        "input_len":           _coerce_int(data.get("input_len")),
        "output_len":          _coerce_int(data.get("output_len")),
        "temperature":         _coerce_float(data.get("temperature")),
        # NEU: Python-Runner-Zeitfelder (ersetzen/ergänzen das alte "duration")
        "window_secs":         _coerce_float(data.get("window_secs")),
        "duration_s":          _coerce_float(data.get("duration_s")),
        "drain_s":             _coerce_float(data.get("drain_s")),
        # NEU: submitted statt num_prompts als primäre Zählung
        "submitted":           _coerce_int(data.get("submitted")),
        "completed":           _coerce_int(data.get("completed")),
        "failed":              _coerce_int(data.get("failed")),
        "total_input_tokens":  _coerce_int(data.get("total_input_tokens")),
        "total_output_tokens": _coerce_int(data.get("total_output_tokens")),
        "request_throughput":  _coerce_float(data.get("request_throughput")),
        "output_throughput":   _coerce_float(data.get("output_throughput")),
        "total_token_throughput": _coerce_float(data.get("total_token_throughput")),
    }

    # NEU: Latenz-Metriken aus verschachtelten Blöcken extrahieren
    for metric in ("ttft", "tpot", "itl", "e2el"):
        row.update(_extract_metric_block(data, metric))

    return row


# ---------------------------------------------------------------------------
# Request-Level Verarbeitung
# ---------------------------------------------------------------------------

def _itl_stats_ms(itl_list: list) -> dict:
    """
    Berechnet ITL-Statistiken aus einer Liste, die BEREITS in Millisekunden vorliegt.

    WICHTIG: Im neuen Python-Runner-Format ist itl_sequence schon in ms.
             Daher KEINE Multiplikation mit 1000 mehr!
    """
    empty = {k: None for k in [
        "itl_count", "itl_mean_ms", "itl_median_ms", "itl_std_ms",
        "itl_min_ms", "itl_max_ms", "itl_sum_ms", "itl_p95_ms", "itl_p99_ms",
        "decode_time_ms", "first_4_itl_mean_ms", "last_4_itl_mean_ms",
        "tail_ratio_p99_over_median",
    ]}
    if not itl_list or len(itl_list) == 0:
        return empty

    arr = np.array(itl_list, dtype=float)   # bereits in ms — kein * 1000!
    n   = len(arr)
    p95 = float(np.percentile(arr, 95)) if n >= 2 else float(arr[0])
    p99 = float(np.percentile(arr, 99)) if n >= 2 else float(arr[0])
    med = float(np.median(arr))
    return {
        "itl_count":            n,
        "itl_mean_ms":          float(np.mean(arr)),
        "itl_median_ms":        med,
        "itl_std_ms":           float(np.std(arr)) if n >= 2 else None,
        "itl_min_ms":           float(np.min(arr)),
        "itl_max_ms":           float(np.max(arr)),
        "itl_sum_ms":           float(np.sum(arr)),
        "itl_p95_ms":           p95,
        "itl_p99_ms":           p99,
        "decode_time_ms":       float(np.sum(arr)),   # ≈ Summe aller ITLs
        "first_4_itl_mean_ms":  float(np.mean(arr[:4]))  if n >= 4 else float(np.mean(arr)),
        "last_4_itl_mean_ms":   float(np.mean(arr[-4:])) if n >= 4 else float(np.mean(arr)),
        "tail_ratio_p99_over_median": (p99 / med) if med > 0 else None,
    }


def process_requests(json_path: Path, run_meta: dict) -> list[dict]:
    """
    NEU: Liest individual_request_results aus der JSON-Datei.

    Altes Format: separate Top-Level-Arrays  ttfts / itls / errors / ...
    Neues Format: Liste von Dicts unter      individual_request_results
    """
    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    # NEU: Request-Rohdaten aus individual_request_results
    individual = data.get("individual_request_results", []) or []
    if not isinstance(individual, list):
        individual = []

    target_output_len = _coerce_int(data.get("output_len"))

    n = len(individual)
    if n == 0:
        return []

    # Basis-Metadaten aus run_meta übernehmen
    base_keys = [
        "file_path", "file_name", "parent_dir", "source_subdir",
        "experiment_id", "server_config_label", "model_name",
        "offload_gb", "run_concurrency", "run_id",
        "condition", "role",
    ]
    base = {k: run_meta[k] for k in base_keys if k in run_meta}

    rows = []
    for i, req in enumerate(individual):
        if not isinstance(req, dict):
            continue

        # TTFT: neues Format liefert ttft_s (Sekunden)
        ttft_s = _coerce_float(req.get("ttft_s"))
        ttft_ms = (ttft_s * 1000.0) if ttft_s is not None else None

        # ITL-Sequenz: NEU bereits in ms, NICHT nochmal konvertieren
        itl_list = req.get("itl_sequence", []) or []
        if not isinstance(itl_list, list):
            itl_list = []
        itl_stats = _itl_stats_ms(itl_list)

        # Fehlerfeld
        err = req.get("error_text") or req.get("error") or ""
        success_raw = req.get("request_success")
        if success_raw is not None:
            request_success = bool(success_raw)
        else:
            request_success = (err == "" or err is None)

        # generated_text: Vorschau begrenzen
        gen_text = req.get("generated_text") or req.get("output_text") or ""
        preview  = str(gen_text)[:120] if gen_text else ""

        row = {
            **base,
            "request_idx":        i,
            "input_len":          _coerce_int(req.get("input_len")),
            "target_output_len":  target_output_len,
            "actual_output_len":  _coerce_int(req.get("actual_output_len")
                                              or req.get("output_len")),
            "request_success":    request_success,
            "error_text":         err if err else "",
            "start_time":         _coerce_float(req.get("start_time")),
            "ttft_s":             ttft_s,
            "ttft_ms":            ttft_ms,
            # e2el und tpot direkt aus dem Request-Dict (neue Felder)
            "e2el_ms":            _coerce_float(req.get("e2el_ms")),
            "tpot_ms":            _coerce_float(req.get("tpot_ms")),
            "generated_text_preview": preview,
            **itl_stats,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Sanity Checks  (angepasst auf neues Format)
# ---------------------------------------------------------------------------

def sanity_check_runs(df: pd.DataFrame) -> None:
    print("\n[SANITY CHECK] Run-Level Tabelle")

    # Offload-Werte
    unexpected_offload = set(df["offload_gb"].dropna().astype(int)) - VALID_OFFLOAD_GB
    if unexpected_offload:
        print(f"  [WARN] Unerwartete offload_gb-Werte: {sorted(unexpected_offload)}")
    else:
        print(f"  [OK]   offload_gb-Werte: {sorted(df['offload_gb'].dropna().astype(int).unique())}")

    # Concurrency-Werte
    unexpected_conc = set(df["run_concurrency"].dropna().astype(int)) - VALID_CONCURRENCIES
    if unexpected_conc:
        print(f"  [WARN] Unerwartete run_concurrency-Werte: {sorted(unexpected_conc)}")
    else:
        print(f"  [OK]   run_concurrency-Werte: {sorted(df['run_concurrency'].dropna().astype(int).unique())}")

    # NEU: completed + failed == submitted  (statt num_prompts)
    needed = ["completed", "failed", "submitted"]
    mask = df[needed].notna().all(axis=1)
    sub  = df[mask].copy()
    if not sub.empty:
        sub["_total"] = sub["completed"].astype(int) + sub["failed"].astype(int)
        sub["_sub"]   = sub["submitted"].astype(int)
        bad = sub[sub["_total"] != sub["_sub"]]
        if len(bad) > 0:
            print(f"  [WARN] {len(bad)} Runs: completed+failed != submitted")
            for _, r in bad.iterrows():
                label = f"{r['file_name']} [{r.get('source_subdir', r['parent_dir'])}]"
                print(f"         {label}  completed={r['completed']} failed={r['failed']} submitted={r['submitted']}")
        else:
            print(f"  [OK]   completed+failed == submitted für alle {len(sub)} Runs mit vollständigen Daten")
    else:
        print(f"  [INFO] Keine vollständigen completed/failed/submitted-Daten für Check")

    # NEU: Warnung wenn failed > 0
    if "failed" in df.columns:
        failed_runs = df[df["failed"].fillna(0).astype(int) > 0]
        if not failed_runs.empty:
            for _, r in failed_runs.iterrows():
                label = f"{r['file_name']} [{r.get('source_subdir', r['parent_dir'])}]"
                print(f"  [WARN] failed={int(r['failed'])} in {label}")

    # NEU: Warnung wenn completed sehr klein im Verhältnis zu submitted
    if {"completed", "submitted"}.issubset(df.columns):
        csub = df[df["submitted"].notna() & df["completed"].notna()].copy()
        csub["_ratio"] = csub["completed"].astype(float) / csub["submitted"].astype(float).replace(0, np.nan)
        low = csub[csub["_ratio"] < 0.5]
        if not low.empty:
            for _, r in low.iterrows():
                label = f"{r['file_name']} [{r.get('source_subdir', r['parent_dir'])}]"
                print(f"  [WARN] Niedrige Completion-Rate ({r['_ratio']:.0%}): {label}")


def sanity_check_requests(df_runs: pd.DataFrame, df_req: pd.DataFrame) -> None:
    print("\n[SANITY CHECK] Request-Level Tabelle")

    # Compound-Key: file_name + source_subdir — verhindert falsche Aggregation
    # bei gleichen Dateinamen in verschiedenen Bedingungsunterordnern.
    group_keys = ["file_name", "source_subdir"]
    counts = df_req.groupby(group_keys)["request_idx"].count().rename("req_rows")

    # NEU: Referenz ist submitted (statt num_prompts)
    ref_col = "submitted" if "submitted" in df_runs.columns else None
    ref_cols = [c for c in [ref_col, "completed"] if c is not None]
    ref = df_runs.set_index(group_keys)[ref_cols].copy()
    for c in ref_cols:
        ref[c] = ref[c].apply(_coerce_int)

    merged = ref.join(counts, how="left")
    merged["req_rows"] = merged["req_rows"].fillna(0).astype(int)

    issues = 0
    for idx, row in merged.iterrows():
        label  = f"{idx[0]} [{idx[1]}]"
        sub    = row.get("submitted")
        comp   = row.get("completed")
        nreq   = row["req_rows"]

        if sub is not None and nreq != sub:
            direction = "weniger" if nreq < sub else "mehr"
            print(f"  [WARN] {direction} Request-Zeilen als submitted: {label}  req_rows={nreq} submitted={sub}")
            issues += 1
        if comp is not None and nreq > 0 and nreq < comp:
            print(f"  [WARN] req_rows={nreq} < completed={comp} in {label}")
            issues += 1

    if issues == 0:
        print(f"  [OK]   Request-Zeilenzahl plausibel für alle {len(merged)} Runs")

    # NEU: Optional — Warnung wenn actual_output_len != target_output_len
    if {"actual_output_len", "target_output_len"}.issubset(df_req.columns):
        mismatch = df_req[
            df_req["actual_output_len"].notna() &
            df_req["target_output_len"].notna() &
            (df_req["actual_output_len"].astype(int) != df_req["target_output_len"].astype(int))
        ]
        if not mismatch.empty:
            n_mis = len(mismatch)
            pct   = n_mis / len(df_req) * 100 if len(df_req) > 0 else 0
            print(f"  [INFO] {n_mis} Requests ({pct:.1f}%) mit actual_output_len != target_output_len")


# ---------------------------------------------------------------------------
# Ausgabe-Vorschau
# ---------------------------------------------------------------------------

def print_head(df: pd.DataFrame, n: int, title: str, cols: list[str]) -> None:
    print(f"\n--- {title}: erste {min(n, len(df))} Zeile(n) ---")
    preview = [c for c in cols if c in df.columns][:8]
    print(df[preview].head(n).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Sortierung
# ---------------------------------------------------------------------------

def sort_df(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    base = ["offload_gb", "run_concurrency", "run_id"]
    sort_cols = [c for c in (base + (extra_cols or [])) if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrahiert Run- und Request-Level-Tabellen aus Python-Runner-JSONs."
    )
    parser.add_argument("input_dir", type=Path,
                        help="Wurzelverzeichnis mit den Benchmark-Ordnern.")
    parser.add_argument("--outdir", "-o", type=Path, default=Path("."),
                        help="Ausgabeverzeichnis (Standard: aktuelles Verzeichnis).")
    parser.add_argument("--print-head", type=int, default=0, metavar="N",
                        help="Zeigt N Beispielzeilen beider Tabellen nach dem Schreiben.")
    parser.add_argument("--parquet", action="store_true",
                        help="Speichert die Tabellen zusätzlich als Parquet.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"[FEHLER] Verzeichnis nicht gefunden: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- JSON-Dateien sammeln und sortieren ---
    def sort_key(p: Path):
        _og_stem = extract_offload_gb(p.stem)
        _og_dir  = extract_offload_gb(p.parent.name)
        og = _og_stem if _og_stem is not None else (_og_dir if _og_dir is not None else 10**9)
        rc = extract_run_concurrency(p.stem)
        rc = rc if rc is not None else 10**9
        ri = extract_run_id(p.stem)
        ri = ri if ri is not None else 10**9
        return (og, rc, ri)

    json_files = sorted(args.input_dir.rglob("*.json"), key=sort_key)

    if not json_files:
        print(f"[INFO] Keine JSON-Dateien unter: {args.input_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"[INFO] {len(json_files)} JSON-Dateien gefunden.")

    # --- Verarbeiten ---
    run_rows  = []
    req_rows  = []
    n_skipped = 0

    for jp in json_files:
        run = process_run(jp)
        if run is None:
            n_skipped += 1
            continue
        run_rows.append(run)
        req_rows.extend(process_requests(jp, run))

    print(f"[INFO] {len(run_rows)} Runs verarbeitet, {n_skipped} übersprungen.")
    print(f"[INFO] {len(req_rows)} Requests extrahiert.")

    if not run_rows:
        print("[WARN] Keine Daten zum Schreiben.", file=sys.stderr)
        sys.exit(0)

    # --- DataFrames ---
    df_runs = sort_df(pd.DataFrame(run_rows).reindex(columns=RUN_COLUMNS))
    df_reqs = (
        sort_df(pd.DataFrame(req_rows).reindex(columns=REQ_COLUMNS), extra_cols=["request_idx"])
        if req_rows else pd.DataFrame(columns=REQ_COLUMNS)
    )

    # --- Sanity Checks ---
    sanity_check_runs(df_runs)
    if not df_reqs.empty:
        sanity_check_requests(df_runs, df_reqs)

    # --- CSV schreiben ---
    runs_csv = args.outdir / "runs_summary_rerun.csv"
    reqs_csv = args.outdir / "requests_summary_rerun.csv"

    df_runs.to_csv(runs_csv, index=False)
    print(f"\n[INFO] Runs-Tabelle:    {runs_csv}  ({len(df_runs)} Zeilen, {len(df_runs.columns)} Spalten)")

    df_reqs.to_csv(reqs_csv, index=False)
    print(f"[INFO] Request-Tabelle: {reqs_csv}  ({len(df_reqs)} Zeilen, {len(df_reqs.columns)} Spalten)")

    # --- Optional: Parquet ---
    if args.parquet:
        try:
            runs_pq = args.outdir / "runs_summary_rerun.parquet"
            reqs_pq = args.outdir / "requests_summary_rerun.parquet"
            df_runs.to_parquet(runs_pq, index=False)
            df_reqs.to_parquet(reqs_pq, index=False)
            print(f"[INFO] Parquet: {runs_pq}, {reqs_pq}")
        except ImportError:
            print("[WARN] pyarrow/fastparquet nicht installiert — Parquet übersprungen.", file=sys.stderr)

    # --- Vorschau ---
    if args.print_head > 0:
        print_head(df_runs, args.print_head, "runs_summary_rerun",
                   ["file_name", "source_subdir", "condition", "role",
                    "offload_gb", "run_concurrency", "run_id", "mean_ttft_ms"])
        print_head(df_reqs, args.print_head, "requests_summary_rerun",
                   ["file_name", "source_subdir", "condition", "role",
                    "run_id", "request_idx", "ttft_ms", "itl_mean_ms"])


if __name__ == "__main__":
    main()
