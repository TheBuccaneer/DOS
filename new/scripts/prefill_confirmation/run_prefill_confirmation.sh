#!/usr/bin/env bash
set -euo pipefail

# Thin environment wrapper. The Python runner owns scheduling, lifecycle,
# requests, resume, and integrity; this script only selects the venv/GPU,
# enables logging, and forwards arguments unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_RUNNER="${SCRIPT_DIR}/run_prefill_confirmation.py"
SERVER_SCRIPT="${SCRIPT_DIR}/run_server.sh"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
VENV_ACTIVATE="${VENV_PATH}/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" || ! -r "$VENV_ACTIVATE" ]]; then
  echo "Fehler: Venv-Activate-Datei '${VENV_ACTIVATE}' fehlt oder ist nicht lesbar." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

for file in "$PYTHON_RUNNER" "$SERVER_SCRIPT"; do
  if [[ ! -f "$file" || ! -r "$file" ]]; then
    echo "Fehler: '${file}' fehlt oder ist nicht lesbar." >&2
    exit 1
  fi
done

PYTHON_PATH="$(command -v python3 || true)"
if [[ -z "$PYTHON_PATH" ]]; then
  echo "Fehler: python3 ist im aktivierten Venv nicht verfügbar." >&2
  exit 1
fi
VENV_BIN_ABS="$(cd "${VENV_PATH}/bin" && pwd)"
[[ "$(cd "$(dirname "$PYTHON_PATH")" && pwd)" == "$VENV_BIN_ABS" ]] || {
  echo "Fehler: python3 stammt nicht aus '${VENV_BIN_ABS}'." >&2; exit 1;
}

REAL_RUN=0
for arg in "$@"; do
  case "$arg" in
    --official-run|--smoke-test) REAL_RUN=1 ;;
  esac
done

VLLM_PATH="$(command -v vllm || true)"
if (( REAL_RUN )); then
  if [[ -z "$VLLM_PATH" ]]; then
    echo "Fehler: vllm ist für Smoke-/Official-Runs im aktivierten Venv erforderlich." >&2
    exit 1
  fi
  [[ "$(cd "$(dirname "$VLLM_PATH")" && pwd)" == "$VENV_BIN_ABS" ]] || {
    echo "Fehler: vllm stammt nicht aus '${VENV_BIN_ABS}'." >&2; exit 1;
  }
  if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "Fehler: VLLM_API_KEY muss für Smoke-/Official-Runs explizit gesetzt sein." >&2
    exit 1
  fi
  export VLLM_API_KEY
fi

# Accept the project's historical GPU_DEVICE convenience variable, but
# fingerprint and expose the selected physical device through the standard
# CUDA_VISIBLE_DEVICES variable used by CUDA/vLLM and the Python runner.
if [[ -n "${GPU_DEVICE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${GPU_DEVICE}" ]]; then
    echo "Fehler: GPU_DEVICE und CUDA_VISIBLE_DEVICES widersprechen sich." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
fi

# Reale Kampagnen dürfen nie still Modelle/Tokenizer aus dem Netz laden.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOGDIR="${PROJECT_ROOT}/new/logs/prefill_confirmation"
mkdir -p "$LOGDIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/run_prefill_confirmation_${TIMESTAMP}_pid${BASHPID}.log"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1

printf 'PROJECT_ROOT: %s\nPYTHON_RUNNER: %s\nSERVER_SCRIPT: %s\n' "$PROJECT_ROOT" "$PYTHON_RUNNER" "$SERVER_SCRIPT"
VLLM_VERSION="<not required>"
if [[ -n "$VLLM_PATH" ]]; then
  VLLM_VERSION="$($VLLM_PATH --version 2>&1 || true)"
fi
printf 'Python: %s\nvLLM: %s\nCUDA_VISIBLE_DEVICES: %s\nLOGFILE: %s\n' \
  "$($PYTHON_PATH --version 2>&1)" "$VLLM_VERSION" \
  "${CUDA_VISIBLE_DEVICES:-<nicht gesetzt>}" "$LOGFILE"
printf 'Args:'; printf ' %q' "$@"; printf '\n\n'

exec "$PYTHON_PATH" "$PYTHON_RUNNER" "$@"
