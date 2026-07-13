#!/usr/bin/env bash
set -euo pipefail

# Thin environment wrapper around run_phase_a.py. All scheduling, server
# lifecycle management (~20 restarts across the frozen Phase A design),
# request execution, and metrics collection live in run_phase_a.py --
# see that file's module docstring for why this can't be plain shell
# like the profiling stage's run_server.sh /
# run_client_profile_grid_v2.sh. This script only sets up the
# environment (venv activation, GPU visibility, VLLM_API_KEY, logging)
# and then execs into the Python runner, forwarding all CLI arguments
# unchanged.
#
# Usage (mirrors run_phase_a.py's own CLI exactly -- see its --help):
#   bash run_phase_a.sh --self-test
#   bash run_phase_a.sh --dry-run --schedule /path/to/phase_a_schedule.csv
#   VENV_PATH=/path/to/.venv GPU_DEVICE=0 bash run_phase_a.sh \
#       --schedule /path/to/phase_a_schedule.csv \
#       --output-dir /path/to/results \
#       --resume

# --- 1. Script- und Projektpfade --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- 2. Pfade zu run_phase_a.py und run_server.sh ---------------------------
PYTHON_RUNNER="${SCRIPT_DIR}/run_phase_a.py"
SERVER_SCRIPT="${SCRIPT_DIR}/run_server.sh"

# --- 3. VENV_PATH bestimmen --------------------------------------------------
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

# --- 4. Venv-Activate pruefen ------------------------------------------------
VENV_ACTIVATE="${VENV_PATH}/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" || ! -r "$VENV_ACTIVATE" ]]; then
  echo "Fehler: Venv-Activate-Datei '${VENV_ACTIVATE}' existiert nicht als reguläre lesbare Datei." >&2
  echo "Bitte VENV_PATH auf ein gueltiges Python-Environment setzen, z.B.:" >&2
  echo "  VENV_PATH=/pfad/zu/.venv bash run_phase_a.sh ..." >&2
  exit 1
fi

# --- 5. Venv aktivieren -------------------------------------------------------
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# --- 6. run_phase_a.py und run_server.sh pruefen -----------------------------
if [[ ! -f "$PYTHON_RUNNER" || ! -r "$PYTHON_RUNNER" ]]; then
  echo "Fehler: '${PYTHON_RUNNER}' existiert nicht als reguläre lesbare Datei." >&2
  exit 1
fi
if [[ ! -f "$SERVER_SCRIPT" || ! -r "$SERVER_SCRIPT" ]]; then
  echo "Fehler: '${SERVER_SCRIPT}' existiert nicht als reguläre lesbare Datei." >&2
  exit 1
fi

# --- 7. python3 und vllm im aktivierten Environment pruefen ------------------
if ! PYTHON_PATH="$(command -v python3)"; then
  echo "Fehler: 'python3' wurde nach Aktivierung des Venv nicht im PATH gefunden." >&2
  exit 1
fi
if ! VLLM_PATH="$(command -v vllm)"; then
  echo "Fehler: 'vllm' wurde nach Aktivierung des Venv nicht im PATH gefunden." >&2
  exit 1
fi

VENV_BIN_ABS="$(cd "${VENV_PATH}/bin" && pwd)"

PYTHON_DIR_ABS="$(cd "$(dirname "$PYTHON_PATH")" && pwd)"
if [[ "$PYTHON_DIR_ABS" != "$VENV_BIN_ABS" ]]; then
  echo "Fehler: 'python3' wird nicht aus '${VENV_BIN_ABS}' aufgeloest (gefunden: '${PYTHON_PATH}')." >&2
  exit 1
fi

VLLM_DIR_ABS="$(cd "$(dirname "$VLLM_PATH")" && pwd)"
if [[ "$VLLM_DIR_ABS" != "$VENV_BIN_ABS" ]]; then
  echo "Fehler: 'vllm' wird nicht aus '${VENV_BIN_ABS}' aufgeloest (gefunden: '${VLLM_PATH}')." >&2
  exit 1
fi

PYTHON_VERSION="$(python3 --version 2>&1 || true)"
if [[ -z "$PYTHON_VERSION" ]]; then
  PYTHON_VERSION="<unbekannt>"
fi

VLLM_VERSION="$(vllm --version 2>/dev/null || true)"
if [[ -z "$VLLM_VERSION" ]]; then
  VLLM_VERSION="<unbekannt>"
fi

# --- 8. GPU-Sichtbarkeit konfigurieren ----------------------------------------
if [[ -z "${GPU_DEVICE+x}" ]]; then
  : # GPU_DEVICE nicht gesetzt: CUDA_VISIBLE_DEVICES bleibt unveraendert erhalten
elif [[ -z "${GPU_DEVICE}" ]]; then
  echo "Fehler: GPU_DEVICE wurde gesetzt, ist aber leer. Bitte eine gueltige GPU-ID setzen oder GPU_DEVICE nicht setzen." >&2
  exit 1
else
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && "${CUDA_VISIBLE_DEVICES}" != "${GPU_DEVICE}" ]]; then
    echo "Warnung: GPU_DEVICE='${GPU_DEVICE}' unterscheidet sich von bereits gesetztem CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'. GPU_DEVICE gewinnt." >&2
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
fi

# --- 9. VLLM_API_KEY konfigurieren --------------------------------------------
VLLM_API_KEY="${VLLM_API_KEY:-pilotkey}"
export VLLM_API_KEY

# --- 10. Logverzeichnis anlegen -----------------------------------------------
LOGDIR="${PROJECT_ROOT}/new/logs/phase_a"
if ! mkdir -p "$LOGDIR"; then
  echo "Fehler: Logverzeichnis '${LOGDIR}' konnte nicht angelegt werden." >&2
  exit 1
fi

# --- 11. Konkrete Logdatei anlegen bzw. auf Schreibbarkeit pruefen -----------
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/run_phase_a_${TIMESTAMP}_pid${BASHPID}.log"
if ! : > "$LOGFILE"; then
  echo "Fehler: Logdatei '${LOGFILE}' konnte nicht angelegt oder beschrieben werden." >&2
  exit 1
fi

# --- 12. Globales Logging aktivieren ------------------------------------------
exec > >(tee -a "$LOGFILE") 2>&1

# --- 13. Log-Header -----------------------------------------------------------
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "VENV_PATH: ${VENV_PATH}"
echo "PYTHON_RUNNER: ${PYTHON_RUNNER}"
echo "SERVER_SCRIPT: ${SERVER_SCRIPT}"
echo "Python-Pfad: ${PYTHON_PATH}"
echo "Python-Version: ${PYTHON_VERSION}"
echo "vLLM-Pfad: ${VLLM_PATH}"
echo "vLLM-Version: ${VLLM_VERSION}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<nicht gesetzt>}"
echo "LOGFILE: ${LOGFILE}"
echo "UTC-Startzeit: ${TIMESTAMP}"
echo "Argumentanzahl: $#"
printf 'Args:'
printf ' %q' "$@"
printf '\n'
echo

# --- 14. Runner starten (letztes Kommando, exec) ------------------------------
exec "${PYTHON_PATH}" "${PYTHON_RUNNER}" "$@"
