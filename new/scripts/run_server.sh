#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Nutzung:
#   bash run_server.sh <model> <offload_gb>
#
# <model>: qwen oder llama
#
# Beispiel:
#   bash run_server.sh llama 0
#   bash run_server.sh llama 8
#   bash run_server.sh qwen 0
#   bash run_server.sh qwen 8
#   bash run_server.sh qwen 12
#   bash run_server.sh qwen 16

if [[ $# -lt 2 ]]; then
  echo "Fehler: Bitte model und cpu-offload-gb angeben, z.B. llama 8"
  echo "Nutzung: bash run_server.sh <model> <offload_gb>"
  exit 1
fi

MODEL_CHOICE="$1"
OFFLOAD_GB="$2"

echo "Project root: ${PROJECT_ROOT}"


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
    echo "Fehler: Unbekanntes model '${MODEL_CHOICE}'. Erlaubt sind: qwen, llama"
    exit 1
    ;;
esac

API_KEY="pilotkey"
GPU_MEM_UTIL="0.90"
TP_SIZE="1"
MAX_MODEL_LEN="8192"

mkdir -p "$LOGDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGDIR}/${LOGPREFIX}_offload${OFFLOAD_GB}_${TIMESTAMP}.log"

export OPENAI_API_KEY="$API_KEY"

echo "Starte vLLM-Server mit ${MODEL_CHOICE} und cpu-offload-gb=${OFFLOAD_GB}"
echo "Model: ${MODEL}"
echo "Logdatei: ${LOGFILE}"
echo

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$API_KEY" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --cpu-offload-gb "$OFFLOAD_GB" \
  2>&1 | tee "$LOGFILE"
