#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Nutzung: bash run_server.sh <model_key> <offload_gb> [host] [port]" >&2
  exit 1
fi
MODEL_KEY="$1"
OFFLOAD_GB="$2"
HOST="${3-127.0.0.1}"
PORT="${4-8000}"

case "$MODEL_KEY" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  *) echo "Fehler: model_key muss llama oder qwen sein, erhalten: '${MODEL_KEY}'." >&2; exit 1 ;;
esac

[[ "$OFFLOAD_GB" =~ ^(0|8|12)$ ]] || {
  echo "Fehler: offload_gb muss im eingefrorenen Design 0, 8 oder 12 sein." >&2; exit 1;
}
[[ -n "$HOST" ]] || { echo "Fehler: host darf nicht leer sein." >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Fehler: ungültiger Port '${PORT}'." >&2; exit 1; }
PORT=$((10#$PORT))
(( PORT >= 1 && PORT <= 65535 )) || { echo "Fehler: Port außerhalb 1..65535." >&2; exit 1; }
command -v vllm >/dev/null 2>&1 || { echo "Fehler: vllm fehlt im PATH." >&2; exit 1; }

# Do not turn a frozen measurement into an implicit model download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPU_MEM_UTIL="0.90"
TP_SIZE="1"
MAX_MODEL_LEN="8192"
SERVER_SEED="20260711"
MAX_NUM_SEQS="16"
MAX_NUM_BATCHED_TOKENS="2048"

LOGDIR="${PROJECT_ROOT}/new/logs/prefill_confirmation/server"
mkdir -p "$LOGDIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/server_${MODEL_KEY}_offload${OFFLOAD_GB}_${TIMESTAMP}.log"

{
  echo "Prefill-Confirmation vLLM server"
  echo "Model key: ${MODEL_KEY}"
  echo "Model: ${MODEL}"
  echo "CPU offload GB: ${OFFLOAD_GB}"
  echo "Chunked prefill: enabled"
  echo "Max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
  echo "Host: ${HOST}"
  echo "Port: ${PORT}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<nicht gesetzt>}"
  echo "UTC start: ${TIMESTAMP}"
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
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-enable-prefix-caching \
  --seed "$SERVER_SEED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  2>&1 | tee -a "$LOGFILE"
