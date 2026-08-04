#!/usr/bin/env bash
set -euo pipefail

# Server launcher for the server-side WAITING confirmation extension.
# Separate from, and never invoked by, the original prefill_confirmation
# run_server.sh. Keeps the main-study serving configuration unchanged
# (--enable-chunked-prefill, --max-num-batched-tokens 2048,
# --no-enable-prefix-caching, --max-model-len 8192, same dtype/
# generation-config/seed/GPU memory utilization) but additionally
# accepts and validates server_max_num_seqs, and passes it explicitly
# to vLLM as --max-num-seqs (the original launcher hardcoded this to
# 16 and never varied it or recorded it as an experimental factor).
#
# Frozen extension design: model is Qwen only, offload is restricted to
# {0, 12} (not the original {0, 8, 12}).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Nutzung: bash run_server_waiting_server.sh <model_key> <offload_gb> <server_max_num_seqs> [host] [port]" >&2
  exit 1
fi
MODEL_KEY="$1"
OFFLOAD_GB="$2"
SERVER_MAX_NUM_SEQS="$3"
HOST="${4-127.0.0.1}"
PORT="${5-8000}"

case "$MODEL_KEY" in
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  *) echo "Fehler: model_key muss qwen sein (frozen extension design ist Qwen-only), erhalten: '${MODEL_KEY}'." >&2; exit 1 ;;
esac

[[ "$OFFLOAD_GB" =~ ^(0|12)$ ]] || {
  echo "Fehler: offload_gb muss im eingefrorenen Extension-Design 0 oder 12 sein (nicht 8), erhalten: '${OFFLOAD_GB}'." >&2; exit 1;
}
[[ "$SERVER_MAX_NUM_SEQS" =~ ^(4|8)$ ]] || {
  echo "Fehler: server_max_num_seqs muss im eingefrorenen Extension-Design 4 oder 8 sein, erhalten: '${SERVER_MAX_NUM_SEQS}'." >&2; exit 1;
}
[[ -n "$HOST" ]] || { echo "Fehler: host darf nicht leer sein." >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Fehler: ungültiger Port '${PORT}'." >&2; exit 1; }
PORT=$((10#$PORT))
(( PORT >= 1 && PORT <= 65535 )) || { echo "Fehler: Port außerhalb 1..65535." >&2; exit 1; }
command -v vllm >/dev/null 2>&1 || { echo "Fehler: vllm fehlt im PATH." >&2; exit 1; }

# Do not turn a frozen measurement into an implicit model download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Kept identical to the main-study server configuration.
GPU_MEM_UTIL="0.90"
TP_SIZE="1"
MAX_MODEL_LEN="8192"
SERVER_SEED="20260711"
MAX_NUM_BATCHED_TOKENS="2048"

LOGDIR="${PROJECT_ROOT}/new/logs/server_waiting_confirmation/server"
mkdir -p "$LOGDIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOGDIR}/server_${MODEL_KEY}_offload${OFFLOAD_GB}_maxseqs${SERVER_MAX_NUM_SEQS}_${TIMESTAMP}.log"

{
  echo "Server-Waiting-Confirmation vLLM server"
  echo "Model key: ${MODEL_KEY}"
  echo "Model: ${MODEL}"
  echo "CPU offload GB: ${OFFLOAD_GB}"
  echo "server_max_num_seqs (--max-num-seqs): ${SERVER_MAX_NUM_SEQS}"
  echo "Chunked prefill: enabled"
  echo "Max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
  echo "Max model len: ${MAX_MODEL_LEN}"
  echo "Prefix caching: disabled (--no-enable-prefix-caching)"
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
  --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  2>&1 | tee -a "$LOGFILE"
