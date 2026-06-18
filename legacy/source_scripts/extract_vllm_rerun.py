#!/usr/bin/env python3
"""
extract_vllm_rerun.py

Liest rekursiv einen Wurzelordner mit vLLM-Re-Run-JSONs (neues Format)
und erzeugt zwei CSV/Parquet-Tabellen:

  A. runs_summary_rerun.csv      — eine Zeile pro JSON/Run
  B. requests_summary_rerun.csv  — eine Zeile pro Request innerhalb eines Runs

Neues Dateinamensschema:
  offload<N>_conc<M>_run<R>.json
  z.B. offload0_conc1_run1.json, offload8_conc16_run5.json

Ordnernamen-Beispiel (offload_gb wird aus beiden Quellen abgeleitet):
  bench_runs_offload_paper1_rerun_offload8_20260320_093921/

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
# Run-Level Spaltenreihenfolge
# ---------------------------------------------------------------------------

RUN_COLUMNS = [
    "file_path", "file_name", "parent_dir", "source_subdir",
    "experiment_id", "server_config_label", "model_name", "date",
    "offload_gb", "run_concurrency", "run_id",
    "condition", "role",
    "num_prompts", "num_warmups", "input_len", "output_len", "temperature",
    "request_rate", "burstiness", "max_concurrency",
    "duration", "completed", "failed",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
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
# Request-Level Spaltenreihenfolge
# ---------------------------------------------------------------------------

REQ_COLUMNS = [
    "file_path", "file_name", "parent_dir", "source_subdir",
    "experiment_id", "server_config_label", "model_name", "date",
    "offload_gb", "run_concurrency", "run_id",
    "condition", "role",
    "request_idx",
    "input_len", "target_output_len", "actual_output_len",
    "ttft_s", "ttft_ms",
    "request_success", "error_text",
    "start_time",
    "itl_count",
    "itl_mean_ms", "itl_median_ms", "itl_std_ms",
    "itl_min_ms",  "itl_max_ms",    "itl_sum_ms",
    "itl_p95_ms",  "itl_p99_ms",
    "decode_time_ms",
    "first_4_itl_mean_ms", "last_4_itl_mean_ms",
    "tail_ratio_p99_over_median",
    "generated_text_preview",
]

# ---------------------------------------------------------------------------
# Hilfsfunktionen: Metadaten aus Pfad/Dateinamen extrahieren
# ---------------------------------------------------------------------------

def _int_or_none(s: str | None) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def extract_offload_gb(name: str) -> int | None:
    """
    Extrahiert offload_gb aus Dateiname oder Ordnername.
    Akzeptiert: offload<N>, offload_<N>, offset<N>, offset_<N>
    """
    import re
    m = re.search(r"(?:offload|offset)_?(\d+)", name)
    return _int_or_none(m.group(1)) if m else None


def extract_run_concurrency(stem: str) -> int | None:
    import re
    m = re.search(r"_conc(\d+)_", stem)
    return _int_or_none(m.group(1)) if m else None


def extract_run_id(stem: str) -> int | None:
    import re
    m = re.search(r"_run(\d+)$", stem)
    return _int_or_none(m.group(1)) if m else None


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
# Fallback-Inferenz für condition/role aus Unterordnernamen.
# Greift nur, wenn das JSON keine expliziten Metadaten enthält.
# Konvention des Utility-Skripts: cond_a_victim_only, cond_b_victim_plus_burst
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

    # offload_gb: explizit None-Prüfung statt or-Kette (0 ist falsy!)
    _og_stem   = extract_offload_gb(stem)
    _og_dir    = extract_offload_gb(parent_dir)
    _og_json   = _coerce_int(data.get("offload_gb"))
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

    run_concurrency = (
        extract_run_concurrency(stem)
        or _coerce_int(data.get("concurrency"))
        or _coerce_int(data.get("max_concurrency"))
    )
    run_id = (
        extract_run_id(stem)
        or _coerce_int(data.get("run_no"))
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
        "date":                _get(data, "date"),
        "offload_gb":          offload_gb,
        "run_concurrency":     run_concurrency,
        "run_id":              run_id,
        # condition/role: primär aus JSON-Metadaten, Fallback aus parent_dir-Name
        "condition":           _get(data, "condition") or _infer_condition(parent_dir),
        "role":                _get(data, "role")      or _infer_role(parent_dir),
        "num_prompts":         _coerce_int(data.get("num_prompts")),
        "num_warmups":         _coerce_int(data.get("num_warmups")),
        "input_len":           _coerce_int(data.get("input_len")),
        "output_len":          _coerce_int(data.get("output_len")),
        "temperature":         _coerce_float(data.get("temperature")),
        "request_rate":        _get(data, "request_rate"),
        "burstiness":          _coerce_float(data.get("burstiness")),
        "max_concurrency":     _coerce_int(data.get("max_concurrency")),
        "duration":            _coerce_float(data.get("duration")),
        "completed":           _coerce_int(data.get("completed")),
        "failed":              _coerce_int(data.get("failed")),
        "total_input_tokens":  _coerce_int(data.get("total_input_tokens")),
        "total_output_tokens": _coerce_int(data.get("total_output_tokens")),
        "request_throughput":  _coerce_float(data.get("request_throughput")),
        "output_throughput":   _coerce_float(data.get("output_throughput")),
        "total_token_throughput": _coerce_float(data.get("total_token_throughput")),
    }

    # Latenz-Metriken (alle Suffixe)
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "std"):
            key = f"{stat}_{metric}_ms"
            row[key] = _coerce_float(data.get(key))
        for pct in ("p50", "p95", "p99"):
            key = f"{pct}_{metric}_ms"
            row[key] = _coerce_float(data.get(key))

    return row


# ---------------------------------------------------------------------------
# Request-Level Verarbeitung
# ---------------------------------------------------------------------------

def _itl_stats(itl_list: list) -> dict:
    """Berechnet alle ITL-Statistiken in Millisekunden aus einer Liste (Sekunden)."""
    if not itl_list or len(itl_list) == 0:
        return {k: None for k in [
            "itl_count", "itl_mean_ms", "itl_median_ms", "itl_std_ms",
            "itl_min_ms", "itl_max_ms", "itl_sum_ms", "itl_p95_ms", "itl_p99_ms",
            "decode_time_ms", "first_4_itl_mean_ms", "last_4_itl_mean_ms",
            "tail_ratio_p99_over_median",
        ]}
    arr = np.array(itl_list, dtype=float) * 1000.0  # s -> ms
    # Erster Wert ist oft ein Spike (erstes Token nach TTFT), trotzdem mitrechnen
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
        "decode_time_ms":       float(np.sum(arr)),   # ≈ sum of all ITLs
        "first_4_itl_mean_ms":  float(np.mean(arr[:4]))  if n >= 4 else float(np.mean(arr)),
        "last_4_itl_mean_ms":   float(np.mean(arr[-4:])) if n >= 4 else float(np.mean(arr)),
        "tail_ratio_p99_over_median": (p99 / med) if med > 0 else None,
    }


def process_requests(json_path: Path, run_meta: dict) -> list[dict]:
    """
    Liest request-level Arrays aus der JSON-Datei und gibt eine Liste von Dicts zurück.
    Robuste Behandlung von fehlenden/leeren Feldern.
    """
    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    ttfts         = data.get("ttfts",         []) or []
    input_lens    = data.get("input_lens",    []) or []
    output_lens   = data.get("output_lens",   []) or []
    itls          = data.get("itls",          []) or []
    start_times   = data.get("start_times",   []) or []
    errors        = data.get("errors",        []) or []
    gen_texts     = data.get("generated_texts", []) or []

    target_output_len = _coerce_int(data.get("output_len"))
    # n über alle relevanten Arrays bestimmen, Fallback auf num_prompts
    _array_lens = [len(a) for a in [ttfts, input_lens, output_lens, itls, start_times, errors, gen_texts] if a]
    n = max(_array_lens) if _array_lens else _coerce_int(data.get("num_prompts")) or 0
    if n == 0:
        return []

    base = {k: run_meta[k] for k in [
        "file_path", "file_name", "parent_dir", "source_subdir",
        "experiment_id", "server_config_label", "model_name", "date",
        "offload_gb", "run_concurrency", "run_id",
        "condition", "role",
    ]}

    rows = []
    for i in range(n):
        ttft_s = ttfts[i] if i < len(ttfts) else None
        err    = errors[i] if i < len(errors) else ""
        itl_list = itls[i] if i < len(itls) else []
        gen_text = gen_texts[i] if i < len(gen_texts) else ""

        itl_stats = _itl_stats(itl_list if isinstance(itl_list, list) else [])

        row = {
            **base,
            "request_idx":          i,
            "input_len":            input_lens[i] if i < len(input_lens) else None,
            "target_output_len":    target_output_len,
            "actual_output_len":    output_lens[i] if i < len(output_lens) else None,
            "ttft_s":               ttft_s,
            "ttft_ms":              (ttft_s * 1000.0) if ttft_s is not None else None,
            "request_success":      (err == "" or err is None),
            "error_text":           err if err else "",
            "start_time":           start_times[i] if i < len(start_times) else None,
            "generated_text_preview": str(gen_text)[:120] if gen_text else "",
            **itl_stats,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Sanity Checks
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

    # completed + failed vs. num_prompts
    mask = df[["completed", "failed", "num_prompts"]].notna().all(axis=1)
    sub  = df[mask].copy()
    sub["_total"] = sub["completed"].astype(int) + sub["failed"].astype(int)
    sub["_np"]    = sub["num_prompts"].astype(int)
    bad = sub[sub["_total"] != sub["_np"]]
    if len(bad) > 0:
        print(f"  [WARN] {len(bad)} Runs: completed+failed != num_prompts")
        for _, r in bad.iterrows():
            label = f"{r['file_name']} [{r.get('source_subdir', r['parent_dir'])}]"
            print(f"         {label}  completed={r['completed']} failed={r['failed']} num_prompts={r['num_prompts']}")
    else:
        print(f"  [OK]   completed+failed == num_prompts für alle {len(sub)} Runs mit vollständigen Daten")


def sanity_check_requests(df_runs: pd.DataFrame, df_req: pd.DataFrame) -> None:
    print("\n[SANITY CHECK] Request-Level Tabelle")
    # Compound-Key: file_name + source_subdir — verhindert falsche Aggregation
    # bei gleichen Dateinamen in verschiedenen Bedingungsunterordnern.
    group_keys = ["file_name", "source_subdir"]
    counts = df_req.groupby(group_keys)["request_idx"].count().rename("req_rows")
    ref = df_runs.set_index(group_keys)[["num_prompts", "completed"]].copy()
    ref["num_prompts"] = ref["num_prompts"].apply(_coerce_int)
    ref["completed"]   = ref["completed"].apply(_coerce_int)
    merged = ref.join(counts, how="left")
    merged["req_rows"] = merged["req_rows"].fillna(0).astype(int)

    issues = 0
    for idx, row in merged.iterrows():
        label   = f"{idx[0]} [{idx[1]}]"
        np_val  = row["num_prompts"]
        comp    = row["completed"]
        nreq    = row["req_rows"]
        if np_val is not None and nreq < np_val:
            print(f"  [WARN] Zu wenige Request-Zeilen: {label}  req_rows={nreq} < num_prompts={np_val}")
            issues += 1
        if np_val is not None and nreq > np_val:
            print(f"  [WARN] Zu viele Request-Zeilen:  {label}  req_rows={nreq} > num_prompts={np_val}")
            issues += 1
        if comp is not None and nreq > 0 and nreq < comp:
            print(f"  [WARN] req_rows={nreq} < completed={comp} in {label}")
            issues += 1

    if issues == 0:
        print(f"  [OK]   Request-Zeilenzahl plausibel für alle {len(merged)} Runs")


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
        description="Extrahiert Run- und Request-Level-Tabellen aus vLLM-Re-Run-JSONs."
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
        if _og_stem is not None:
            og = _og_stem
        elif _og_dir is not None:
            og = _og_dir
        else:
            og = 10**9

        _rc = extract_run_concurrency(p.stem)
        rc = _rc if _rc is not None else 10**9

        _ri = extract_run_id(p.stem)
        ri = _ri if _ri is not None else 10**9

        return (og, rc, ri)

    json_files = sorted(args.input_dir.rglob("*.json"), key=sort_key)

    if not json_files:
        print(f"[INFO] Keine JSON-Dateien unter: {args.input_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"[INFO] {len(json_files)} JSON-Dateien gefunden.")

    # --- Verarbeiten ---
    run_rows = []
    req_rows = []
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
    df_reqs = sort_df(pd.DataFrame(req_rows).reindex(columns=REQ_COLUMNS), extra_cols=["request_idx"]) if req_rows else pd.DataFrame(columns=REQ_COLUMNS)

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
