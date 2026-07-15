#!/usr/bin/env bash
set -euo pipefail

# Thin environment wrapper around run_prefill_screen.py. The Python runner
# owns scheduling, server lifecycle, request execution, resume, and integrity.
# This wrapper only validates/activates the venv, maps GPU_DEVICE to
# CUDA_VISIBLE_DEVICES, configures the API-key environment, enables logging,
# and forwards all CLI arguments unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/run_prefill_screen.py"
SERVER_SCRIPT="${SCRIPT_DIR}/run_server.sh"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
VENV_ACTIVATE="${VENV_PATH}/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" || ! -r "$VENV_ACTIVATE" ]]; then
  echo "Fehler: Venv-Activate-Datei '${VENV_ACTIVATE}' existiert nicht als reguläre lesbare Datei." >&2
  echo "Bitte VENV_PATH setzen, z.B.:" >&2
  echo "  VENV_PATH=/pfad/zu/.venv bash run_prefill_screen.sh ..." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

if [[ ! -f "$PYTHON_RUNNER" || ! -r "$PYTHON_RUNNER" ]]; then
  echo "Fehler: '${PYTHON_RUNNER}' existiert nicht als reguläre lesbare Datei." >&2
  exit 1
fi
if [[ ! -f "$SERVER_SCRIPT" || ! -r "$SERVER_SCRIPT" ]]; then
  echo "Fehler: '${SERVER_SCRIPT}' existiert nicht als reguläre lesbare Datei." >&2
  exit 1
fi

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
VLLM_DIR_ABS="$(cd "$(dirname "$VLLM_PATH")" && pwd)"

if [[ "$PYTHON_DIR_ABS" != "$VENV_BIN_ABS" ]]; then
  echo "Fehler: 'python3' wird nicht aus '${VENV_BIN_ABS}' aufgeloest (gefunden: '${PYTHON_PATH}')." >&2
  exit 1
fi
if [[ "$VLLM_DIR_ABS" != "$VENV_BIN_ABS" ]]; then
  echo "Fehler: 'vllm' wird nicht aus '${VENV_BIN_ABS}' aufgeloest (gefunden: '${VLLM_PATH}')." >&2
  exit 1
fi

PYTHON_VERSION="$(python3 --version 2>&1 || true)"
VLLM_VERSION="$(vllm --version 2>/dev/null || true)"
[[ -n "$PYTHON_VERSION" ]] || PYTHON_VERSION="<unbekannt>"
[[ -n "$VLLM_VERSION" ]] || VLLM_VERSION="<unbekannt>"

if [[ -z "${GPU_DEVICE+x}" ]]; then
  :
elif [[ -z "${GPU_DEVICE}" ]]; then
  echo "Fehler: GPU_DEVICE wurde gesetzt, ist aber leer." >&2
  exit 1
else
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && "${CUDA_VISIBLE_DEVICES}" != "${GPU_DEVICE}" ]]; then
    echo "Warnung: GPU_DEVICE='${GPU_DEVICE}' überschreibt CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'." >&2
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
fi

VLLM_API_KEY="${VLLM_API_KEY:-pilotkey}"
export VLLM_API_KEY

LOGDIR="${PROJECT_ROOT}/new/logs/prefill_screen"
if ! mkdir -p "$LOGDIR"; then
  echo "Fehler: Logverzeichnis '${LOGDIR}' konnte nicht angelegt werden." >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/run_prefill_screen_${TIMESTAMP}_pid${BASHPID}.log"
if ! : > "$LOGFILE"; then
  echo "Fehler: Logdatei '${LOGFILE}' konnte nicht angelegt oder beschrieben werden." >&2
  exit 1
fi

exec > >(tee -a "$LOGFILE") 2>&1

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
printf '\n\n'

exec "${PYTHON_PATH}" "${PYTHON_RUNNER}" "$@"
