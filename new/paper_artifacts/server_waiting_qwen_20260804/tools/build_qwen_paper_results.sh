#!/usr/bin/env bash
# Build paper-ready Qwen tables and figures from the completed re-audit.
# Read-only with respect to the campaign and re-audit inputs.
set -Eeuo pipefail
umask 022

VERSION="build-qwen-paper-results-v1"
DEFAULT_AUDIT="$HOME/Downloads/qwen_full_reaudit_20260804T124059Z"
AUDIT_DIR="${1:-$DEFAULT_AUDIT}"

usage() {
    cat <<EOF
$VERSION

Verwendung:
  $0
  $0 /pfad/zu/qwen_full_reaudit_<ZEITSTEMPEL>

Eingabe:
  blocks.csv
  cells.csv
  full_reaudit.json

Ausgabe:
  ~/Downloads/qwen_paper_results_<ZEITSTEMPEL>/

Erzeugt:
  table_qwen_results.csv
  table_qwen_results.tex
  figure_class_counts.png/.pdf
  figure_waiting_intervals.png/.pdf
  figure_tpot_deviation.png/.pdf
  figure_itl_deviation.png/.pdf
  paper_claims.txt
  README.txt
  SHA256SUMS.txt

Es startet keine Messung und verändert keine Eingabedatei.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

AUDIT_DIR="$(readlink -f "$AUDIT_DIR")"
test -d "$AUDIT_DIR" || {
    echo "FEHLER: Audit-Verzeichnis fehlt: $AUDIT_DIR" >&2
    exit 2
}

for f in blocks.csv cells.csv full_reaudit.json; do
    test -r "$AUDIT_DIR/$f" || {
        echo "FEHLER: Eingabedatei fehlt: $AUDIT_DIR/$f" >&2
        exit 2
    }
done

choose_python() {
    local candidate
    for candidate in \
        "${VENV_PATH:-$HOME/projects/porto/.venv-py313-repair}/bin/python" \
        "$(command -v python3 2>/dev/null || true)"; do
        if test -n "$candidate" && test -x "$candidate"; then
            if "$candidate" - <<'PY' >/dev/null 2>&1
import matplotlib
PY
            then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON="$(choose_python)" || {
    echo "FEHLER: Kein Python mit matplotlib gefunden." >&2
    echo "Installiere bei Bedarf: python3 -m pip install matplotlib" >&2
    exit 2
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/Downloads/qwen_paper_results_${STAMP}"
mkdir -p "$OUT"

"$PYTHON" - "$AUDIT_DIR" "$OUT" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

audit_dir = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def parse_int(value):
    if value is None or value == "":
        return None
    return int(float(value))

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

cells_raw = read_csv(audit_dir / "cells.csv")
blocks_raw = read_csv(audit_dir / "blocks.csv")
reaudit = json.loads((audit_dir / "full_reaudit.json").read_text(encoding="utf-8"))

cells = []
for row in cells_raw:
    classes = json.loads(row["classes"]) if row.get("classes") else {}
    cells.append({
        "offload": parse_int(row.get("offload")),
        "k": parse_int(row.get("k")),
        "classes": classes,
        "tpot_ratio": parse_float(row.get("tpot_ratio_median")),
        "itl_ratio": parse_float(row.get("max_itl_ratio_median")),
        "completion_delta_ms": parse_float(row.get("completion_delta_median_ms")),
        "ttft_delta_ms": parse_float(row.get("ttft_delta_median_ms")),
        "token_span_delta_ms": parse_float(row.get("token_span_delta_median_ms")),
        "tail_delta_ms": parse_float(row.get("post_token_tail_delta_median_ms")),
        "after_active_s": parse_float(row.get("burst_after_active_median_s")),
        "remaining_all_s": parse_float(row.get("remaining_until_all_median_s")),
    })

cells.sort(key=lambda r: (r["offload"], r["k"]))

labels = [f"off{r['offload']}/k{r['k']}" for r in cells]

# ---------------------------------------------------------------------
# Paper-ready CSV and LaTeX table
# ---------------------------------------------------------------------
csv_path = out / "table_qwen_results.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "Offload_GiB",
        "max_num_seqs",
        "Class_distribution",
        "Median_TPOT_ratio",
        "Median_max_ITL_ratio",
        "Median_completion_delta_ms",
        "Median_TTFT_delta_ms",
        "Median_token_span_delta_ms",
        "Median_tail_delta_ms",
        "Median_burst_after_active_s",
        "Median_remaining_until_all_s",
    ])
    for r in cells:
        class_text = "/".join(f"{k}{v}" for k, v in sorted(r["classes"].items()))
        writer.writerow([
            r["offload"], r["k"], class_text,
            f"{r['tpot_ratio']:.6f}",
            f"{r['itl_ratio']:.6f}",
            f"{r['completion_delta_ms']:.2f}",
            f"{r['ttft_delta_ms']:.2f}",
            f"{r['token_span_delta_ms']:.2f}",
            f"{r['tail_delta_ms']:.2f}",
            f"{r['after_active_s']:.2f}",
            f"{r['remaining_all_s']:.2f}",
        ])

tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Client-visible burst-output regimes and active-cohort effects for Qwen2.5-7B-Instruct. Values are medians across four paired episodes per cell.}",
    r"\label{tab:qwen-server-waiting}",
    r"\begin{tabular}{rrlrrrr}",
    r"\toprule",
    r"Offload & $k$ & Class & TPOT ratio & Max-ITL ratio & Burst after active (s) & Remaining to all (s) \\",
    r"\midrule",
]
for r in cells:
    class_text = "/".join(f"{k}{v}" for k, v in sorted(r["classes"].items()))
    tex_lines.append(
        f"{r['offload']} & {r['k']} & {class_text} & "
        f"{r['tpot_ratio']:.6f} & {r['itl_ratio']:.6f} & "
        f"{r['after_active_s']:.2f} & {r['remaining_all_s']:.2f} \\\\"
    )
tex_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(out / "table_qwen_results.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------
# Figure 1: class counts
# ---------------------------------------------------------------------
b_counts = [r["classes"].get("B", 0) for r in cells]
c_counts = [r["classes"].get("C", 0) for r in cells]
x = list(range(len(cells)))

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.bar(x, b_counts, label="Class B")
ax.bar(x, c_counts, bottom=b_counts, label="Class C")
ax.set_xticks(x, labels)
ax.set_ylabel("Blocks")
ax.set_xlabel("Configuration")
ax.set_ylim(0, 4.5)
ax.legend()
fig.tight_layout()
fig.savefig(out / "figure_class_counts.png", dpi=300)
fig.savefig(out / "figure_class_counts.pdf")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: waiting interval after active cohort
# ---------------------------------------------------------------------
after_active = [r["after_active_s"] for r in cells]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.bar(x, after_active)
ax.set_xticks(x, labels)
ax.set_ylabel("Seconds")
ax.set_xlabel("Configuration")
ax.set_yscale("log")
ax.set_title("First burst output after active-cohort completion")
fig.tight_layout()
fig.savefig(out / "figure_waiting_intervals.png", dpi=300)
fig.savefig(out / "figure_waiting_intervals.pdf")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: TPOT deviation from control
# ---------------------------------------------------------------------
tpot_deviation_pct = [(r["tpot_ratio"] - 1.0) * 100.0 for r in cells]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.bar(x, tpot_deviation_pct)
ax.axhline(0.0, linewidth=1)
ax.set_xticks(x, labels)
ax.set_ylabel("Median TPOT deviation (%)")
ax.set_xlabel("Configuration")
fig.tight_layout()
fig.savefig(out / "figure_tpot_deviation.png", dpi=300)
fig.savefig(out / "figure_tpot_deviation.pdf")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 4: max-ITL deviation from control
# ---------------------------------------------------------------------
itl_deviation_pct = [(r["itl_ratio"] - 1.0) * 100.0 for r in cells]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.bar(x, itl_deviation_pct)
ax.axhline(0.0, linewidth=1)
ax.set_xticks(x, labels)
ax.set_ylabel("Median max-ITL deviation (%)")
ax.set_xlabel("Configuration")
fig.tight_layout()
fig.savefig(out / "figure_itl_deviation.png", dpi=300)
fig.savefig(out / "figure_itl_deviation.pdf")
plt.close(fig)

# ---------------------------------------------------------------------
# Claims and interpretation
# ---------------------------------------------------------------------
max_tpot_dev = max(abs(v) for v in tpot_deviation_pct)
factor_k4 = cells[2]["after_active_s"] / cells[0]["after_active_s"]
factor_k8 = cells[3]["after_active_s"] / cells[1]["after_active_s"]

claims = f"""Qwen campaign: paper-safe result summary

Primary finding
---------------
Across four repeats per cell, all max_num_seqs=4 blocks were classified
as C and all max_num_seqs=8 blocks as B, at both 0 and 12 GiB CPU
offload. Thus, in this experiment, the client-visible overlap class was
associated with sequence capacity rather than offload state.

Temporal amplification by offload
---------------------------------
At k=4, the median interval from completion of the initially active cohort
to first burst output increased from {cells[0]['after_active_s']:.2f} s to
{cells[2]['after_active_s']:.2f} s ({factor_k4:.1f}x).

At k=8, the same interval increased from {cells[1]['after_active_s']:.2f} s
to {cells[3]['after_active_s']:.2f} s ({factor_k8:.1f}x).

Active-cohort decode behavior
-----------------------------
The largest absolute deviation of a cell-median TPOT ratio from 1 was
{max_tpot_dev:.3f}%. These values are descriptive and do not establish a
universal null effect.

Outlier
-------
qwen_off12_k8_rep01 showed a completion-median increase of approximately
2083 ms, almost entirely attributable to TTFT/start-phase displacement
rather than token-span, TPOT, or post-token stream tail.

Required scope boundary
-----------------------
A/B/C describe client-visible output timing only. They do not identify
internal scheduler admission or internal prefill-start time.

The active cohorts were dynamically identified and differed in concrete
request indices across paired episodes. Effect values therefore compare
active-cohort medians within paired episodes; they are not identity-matched
request-level causal effects.
"""
(out / "paper_claims.txt").write_text(claims, encoding="utf-8")

readme = f"""Qwen paper-result bundle

Source re-audit:
{audit_dir}

Generated files:
- table_qwen_results.csv
- table_qwen_results.tex
- figure_class_counts.png/.pdf
- figure_waiting_intervals.png/.pdf
- figure_tpot_deviation.png/.pdf
- figure_itl_deviation.png/.pdf
- paper_claims.txt

The figures use matplotlib's default rendering and contain no hidden
statistical inference. All values are descriptive medians from the
completed four-repeat cells.
"""
(out / "README.txt").write_text(readme, encoding="utf-8")

print("PAPER_RESULTS_CREATED")
print(f"OUTPUT_DIR={out}")
print(f"CELLS={len(cells)}")
print(f"BLOCKS={len(blocks_raw)}")
print(f"MAX_ABS_TPOT_DEVIATION_PERCENT={max_tpot_dev:.3f}")
print(f"OFFLOAD_FACTOR_K4={factor_k4:.2f}")
print(f"OFFLOAD_FACTOR_K8={factor_k8:.2f}")
PY

(
    cd "$OUT"
    find . -type f ! -name SHA256SUMS.txt -print0 |
        sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt >/dev/null
)

ARCHIVE="${OUT}.tar.gz"
tar -C "$(dirname "$OUT")" -czf "$ARCHIVE" "$(basename "$OUT")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo
echo "PAPER_RESULTS_PASS"
echo "OUTPUT_DIR=$OUT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_SHA256=$(awk '{print $1}' "${ARCHIVE}.sha256")"
echo "COMMAND_RC=0"
echo "TERMINAL_STILL_ALIVE"
