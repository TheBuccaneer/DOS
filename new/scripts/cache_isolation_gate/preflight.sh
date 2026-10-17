#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash preflight.sh DOS_ROOT [PYTHON_EXECUTABLE]' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DOS_ROOT="$1"
TASK_PYTHON="${2:-python3}"
[[ -d "$TASK_DOS_ROOT/new/scripts" ]] || { echo 'DOS/new/scripts fehlt' >&2; exit 2; }
TASK_PYTHON_ABS="$(command -v "$TASK_PYTHON")"
TASK_DOS_ROOT="$(cd "$TASK_DOS_ROOT" && pwd)"
TASK_EVIDENCE="$(mktemp -d "${TMPDIR:-/tmp}/dos-icissp-preflight.XXXXXXXX")"
"$TASK_PYTHON_ABS" -B "$SCRIPT_DIR/verify_sources.py" --root "$TASK_DOS_ROOT" > "$TASK_EVIDENCE/source_verification.json"
"$TASK_PYTHON_ABS" -B "$SCRIPT_DIR/inspect_environment.py" --root "$TASK_DOS_ROOT" > "$TASK_EVIDENCE/environment.json"
TASK_AUDIT_STATUS=0
"$TASK_PYTHON_ABS" -B "$SCRIPT_DIR/audit_existing.py" --root "$TASK_DOS_ROOT" > "$TASK_EVIDENCE/historical_audit.json" || TASK_AUDIT_STATUS=$?
"$TASK_PYTHON_ABS" -B -m unittest discover -s "$SCRIPT_DIR/tests" -v
echo "Pruefberichte: $TASK_EVIDENCE"
echo 'Offline checks do not establish a GPU/runtime gate pass.'
exit "$TASK_AUDIT_STATUS"
