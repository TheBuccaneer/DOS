#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Nutzung:
#   bash run_server.sh <model> <offload_gb> [host] [port]
#
# <model>:      qwen oder llama
# <offload_gb>: nicht-negative Ganzzahl
# [host]:       optional, Default 127.0.0.1
# [port]:       optional, Default 8000
#
# Beispiele:
#   bash run_server.sh llama 0
#   bash run_server.sh llama 12
#   bash run_server.sh qwen 0
#   bash run_server.sh qwen 12
#   bash run_server.sh qwen 12 127.0.0.2 8123

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Fehler: Erwartet werden 2 bis 4 Argumente (model offload_gb [host] [port]), erhalten: $#" >&2
  echo "Nutzung: bash run_server.sh <model> <offload_gb> [host] [port]" >&2
  exit 1
fi

MODEL_CHOICE="$1"
OFFLOAD_GB="$2"
HOST="${3-127.0.0.1}"
PORT="${4-8000}"

case "$MODEL_CHOICE" in
  llama)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    LOGDIR="${PROJECT_ROOT}/new/logs/server"
    LOGPREFIX="server_llama"
    ;;
  qwen)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    LOGDIR="${PROJECT_ROOT}/new/logs/server"
    LOGPREFIX="server_qwen"
    ;;
  *)
    echo "Fehler: Unbekanntes model '${MODEL_CHOICE}'. Erlaubt sind: qwen, llama" >&2
    exit 1
    ;;
esac

if ! [[ "$OFFLOAD_GB" =~ ^[0-9]+$ ]]; then
  echo "Fehler: offload_gb muss eine nicht-negative Ganzzahl sein, erhalten: '${OFFLOAD_GB}'" >&2
  exit 1
fi

if [[ -z "$HOST" ]]; then
  echo "Fehler: host darf nicht leer sein" >&2
  exit 1
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Fehler: port muss eine Ganzzahl zwischen 1 und 65535 sein, erhalten: '${PORT}'" >&2
  exit 1
fi
PORT_DEC=$((10#$PORT))
if (( PORT_DEC < 1 || PORT_DEC > 65535 )); then
  echo "Fehler: port muss eine Ganzzahl zwischen 1 und 65535 sein, erhalten: '${PORT}'" >&2
  exit 1
fi
PORT="$PORT_DEC"

if ! command -v vllm >/dev/null 2>&1; then
  echo "Fehler: 'vllm' wurde nicht im PATH gefunden." >&2
  exit 1
fi
VLLM_PATH="$(command -v vllm)"

VLLM_VERSION="$(vllm --version 2>/dev/null || true)"
if [[ -z "$VLLM_VERSION" ]]; then
  VLLM_VERSION="<unbekannt>"
fi

GPU_MEM_UTIL="0.90"
TP_SIZE="1"
MAX_MODEL_LEN="8192"
SERVER_SEED="20260711"

if ! mkdir -p "$LOGDIR"; then
  echo "Fehler: Logverzeichnis '${LOGDIR}' konnte nicht angelegt werden." >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/${LOGPREFIX}_offload${OFFLOAD_GB}_${TIMESTAMP}.log"

VLLM_API_KEY="${VLLM_API_KEY:-pilotkey}"
export VLLM_API_KEY

CUDA_DISPLAY="${CUDA_VISIBLE_DEVICES:-<nicht gesetzt>}"

{
  echo "Project root: ${PROJECT_ROOT}"
  echo "Starte vLLM-Server mit ${MODEL_CHOICE} und cpu-offload-gb=${OFFLOAD_GB}"
  echo "Modellkürzel: ${MODEL_CHOICE}"
  echo "Model: ${MODEL}"
  echo "Offload-Wert: ${OFFLOAD_GB}"
  echo "Host: ${HOST}"
  echo "Port: ${PORT}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_DISPLAY}"
  echo "vLLM-Pfad: ${VLLM_PATH}"
  echo "vLLM-Version: ${VLLM_VERSION}"
  echo "Server-Seed: ${SERVER_SEED}"
  echo "Prefix-Caching: disabled"
  echo "Logdatei: ${LOGFILE}"
  echo "UTC-Startzeit: ${TIMESTAMP}"
  echo
} | tee "$LOGFILE"

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --generation-config vllm \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --cpu-offload-gb "$OFFLOAD_GB" \
  --no-enable-prefix-caching \
  --seed "$SERVER_SEED" \
  2>&1 | tee -a "$LOGFILE"
