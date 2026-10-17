#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   MAX_NUM_BATCHED_TOKENS=<budget> bash run_server.sh <model> <offload_gb> [host] [port]
# Frozen Chunk-Budget-Screen model set: llama only.
#
# MAX_NUM_BATCHED_TOKENS is REQUIRED (no default) and must be a positive
# integer -- it is the chunked-prefill budget under test and must never
# silently fall back to a vLLM default.

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Fehler: Erwartet werden 2 bis 4 Argumente (model offload_gb [host] [port]), erhalten: $#" >&2
  echo "Nutzung: MAX_NUM_BATCHED_TOKENS=<budget> bash run_server.sh <model> <offload_gb> [host] [port]" >&2
  exit 1
fi

if [[ -z "${MAX_NUM_BATCHED_TOKENS+x}" ]]; then
  echo "Fehler: MAX_NUM_BATCHED_TOKENS ist nicht gesetzt. Es muss explizit als" >&2
  echo "positiver Integer übergeben werden, z.B.:" >&2
  echo "  MAX_NUM_BATCHED_TOKENS=1024 bash run_server.sh llama 0" >&2
  exit 1
fi
if [[ -z "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  echo "Fehler: MAX_NUM_BATCHED_TOKENS wurde gesetzt, ist aber leer." >&2
  exit 1
fi
if ! [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]] || [[ "$MAX_NUM_BATCHED_TOKENS" -eq 0 ]]; then
  echo "Fehler: MAX_NUM_BATCHED_TOKENS muss eine positive Ganzzahl sein, erhalten: '${MAX_NUM_BATCHED_TOKENS}'" >&2
  exit 1
fi

MODEL_CHOICE="$1"
OFFLOAD_GB="$2"
HOST="${3-127.0.0.1}"
PORT="${4-8000}"

case "$MODEL_CHOICE" in
  llama)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    LOGDIR="${PROJECT_ROOT}/new/logs/chunk_budget_screen/server"
    LOGPREFIX="server_llama"
    ;;
  *)
    echo "Fehler: Unbekanntes model '${MODEL_CHOICE}'. Erlaubt ist ausschließlich: llama" >&2
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
# Deliberately preserved from the Phase-A/Prefill-Screen server
# configuration so the chunked-prefill budget is the intended
# experimental change, not the server RNG seed.
SERVER_SEED="20260711"
# Maximum active sequences in the frozen screen are 4 victims + 4 bursts.
# 16 preserves the already-tested Prefill-Screen/Phase-A startup setting
# and avoids the excessive default sampler warm-up allocation.
MAX_NUM_SEQS="16"

if ! mkdir -p "$LOGDIR"; then
  echo "Fehler: Logverzeichnis '${LOGDIR}' konnte nicht angelegt werden." >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/${LOGPREFIX}_offload${OFFLOAD_GB}_budget${MAX_NUM_BATCHED_TOKENS}_${TIMESTAMP}.log"

VLLM_API_KEY="${VLLM_API_KEY:-pilotkey}"
export VLLM_API_KEY
CUDA_DISPLAY="${CUDA_VISIBLE_DEVICES:-<nicht gesetzt>}"

{
  echo "Project root: ${PROJECT_ROOT}"
  echo "Chunk-Budget-Screen vLLM server"
  echo "Modellkürzel: ${MODEL_CHOICE}"
  echo "Model: ${MODEL}"
  echo "Offload-Wert: ${OFFLOAD_GB}"
  echo "Max num batched tokens (chunked-prefill budget): ${MAX_NUM_BATCHED_TOKENS}"
  echo "Host: ${HOST}"
  echo "Port: ${PORT}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_DISPLAY}"
  echo "vLLM-Pfad: ${VLLM_PATH}"
  echo "vLLM-Version: ${VLLM_VERSION}"
  echo "Server-Seed: ${SERVER_SEED}"
  echo "Max num seqs: ${MAX_NUM_SEQS}"
  echo "Chunked-Prefill: enabled"
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
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --seed "$SERVER_SEED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  2>&1 | tee -a "$LOGFILE"
