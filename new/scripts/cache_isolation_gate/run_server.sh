#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   MAX_NUM_BATCHED_TOKENS=<budget> bash run_server.sh <model> <offload_gb> [host] [port]
# Derived ICISSP function-check starter; source and delta documented in reference/.
#
# MAX_NUM_BATCHED_TOKENS is REQUIRED (no default) and must be a positive
# integer -- it is the chunked-prefill budget under test and must never
# silently fall back to a vLLM default.
# Optional MAX_MODEL_LEN and NUM_GPU_BLOCKS_OVERRIDE are explicit control-run
# settings. The original max-model-len default remains 8192.

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
if [[ "$OFFLOAD_GB" != "0" ]]; then
  echo "Fehler: Diese Studie verlangt CPU-Weight-Offload 0." >&2
  exit 1
fi
HOST="${3-127.0.0.1}"
PORT="${4-8000}"

case "$MODEL_CHOICE" in
  llama)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    LOGDIR="${PROJECT_ROOT}/new/logs/cache_isolation_gate/server"
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

if [[ "$HOST" != "127.0.0.1" ]]; then
  echo "Fehler: Funktionsstarter bindet nur an 127.0.0.1." >&2
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

if [[ -z "${MAX_MODEL_LEN+x}" ]]; then
  MAX_MODEL_LEN="8192"
fi
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || (( ${#MAX_MODEL_LEN} > 9 )); then
  echo "Fehler: MAX_MODEL_LEN muss eine positive Dezimalzahl mit hoechstens 9 Stellen sein." >&2
  exit 1
fi

if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE+x}" ]]; then
  if ! [[ "$NUM_GPU_BLOCKS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] ||
     (( ${#NUM_GPU_BLOCKS_OVERRIDE} > 9 )) ||
     (( NUM_GPU_BLOCKS_OVERRIDE < 2 )); then
    echo "Fehler: NUM_GPU_BLOCKS_OVERRIDE muss eine Dezimalzahl >= 2 mit hoechstens 9 Stellen sein." >&2
    exit 1
  fi
fi
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE+x}" && -n "${KV_CACHE_MEMORY_BYTES+x}" ]]; then
  echo "Fehler: KV_CACHE_MEMORY_BYTES und NUM_GPU_BLOCKS_OVERRIDE duerfen nicht gemeinsam gesetzt sein." >&2
  exit 1
fi

# Optional explicit allocation; no pressure calibration or campaign driver.
CAPACITY_ARGS=()
if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
  if ! [[ "$KV_CACHE_MEMORY_BYTES" =~ ^[0-9]+$ ]] || [[ "$KV_CACHE_MEMORY_BYTES" -eq 0 ]]; then
    echo "Fehler: KV_CACHE_MEMORY_BYTES muss positiv sein." >&2
    exit 1
  fi
  CAPACITY_ARGS=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE+x}" ]]; then
  CAPACITY_ARGS=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi
if ! [[ "$VLLM_VERSION" =~ (^|[^0-9])0\.17\.1([^0-9]|$) ]]; then
  echo "Fehler: Starter wurde fuer vLLM 0.17.1 geprueft; erhalten: $VLLM_VERSION" >&2
  exit 1
fi
GPU_MEM_UTIL="0.90"
TP_SIZE="1"
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
if ! (set -o noclobber; : > "$LOGFILE") 2>/dev/null; then
  echo "Fehler: Logdatei existiert bereits oder kann nicht reserviert werden: ${LOGFILE}" >&2
  exit 1
fi

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
  echo "Max model len requested: ${MAX_MODEL_LEN}"
  echo "Chunked-Prefill: enabled"
  echo "Prefix-Caching: enabled"
  echo "KV_CACHE_MEMORY_BYTES requested: ${KV_CACHE_MEMORY_BYTES:-automatic; NOT a constant-capacity proof}"
  echo "NUM_GPU_BLOCKS_OVERRIDE requested: ${NUM_GPU_BLOCKS_OVERRIDE:-automatic; verify effective blocks and block size in vLLM startup logs and cache_config_info}"
  echo "Partial prefills: 1 / long partial prefills: 1 / threshold: 0"
  echo "Iteration details: enabled; timing overhead must be evaluated"
  echo "Prompt token details: enabled"
  echo "Logdatei: ${LOGFILE}"
  echo "UTC-Startzeit: ${TIMESTAMP}"
  echo
} | tee -a "$LOGFILE"

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --generation-config vllm \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --cpu-offload-gb "$OFFLOAD_GB" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --long-prefill-token-threshold 0 \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --enable-logging-iteration-details \
  --enable-prompt-tokens-details \
  "${CAPACITY_ARGS[@]}" \
  --seed "$SERVER_SEED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  2>&1 | tee -a "$LOGFILE"
