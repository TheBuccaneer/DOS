#!/usr/bin/env bash
set -euo pipefail

# Nutzung:
#   MODEL_KEY=llama OFFLOAD_GB=0 bash new/scripts/run_client_profile_grid_v2.sh
#   MODEL_KEY=qwen  OFFLOAD_GB=12 bash new/scripts/run_client_profile_grid_v2.sh
#
# oder mit Positionsparametern (haben Vorrang vor Env-Variablen):
#   bash new/scripts/run_client_profile_grid_v2.sh llama 0
#   bash new/scripts/run_client_profile_grid_v2.sh qwen 12
#
# Voraussetzung:
# - vLLM-Server läuft bereits auf http://127.0.0.1:8000
# - Server wurde mit genau diesem model_key und offload_gb gestartet
# - dieses Skript startet nur den Client-Benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Positionsparameter haben Vorrang vor Env-Variablen
MODEL_KEY="${1:-${MODEL_KEY:-}}"
OFFLOAD_GB="${2:-${OFFLOAD_GB:-}}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Fehler: Bitte MODEL_KEY setzen (llama oder qwen)."
  echo "Nutzung: bash new/scripts/run_client_profile_grid_v2.sh <model_key> <offload_gb>"
  echo "     oder: MODEL_KEY=llama OFFLOAD_GB=0 bash new/scripts/run_client_profile_grid_v2.sh"
  exit 1
fi

OFFLOAD_GB="${OFFLOAD_GB:?Bitte OFFLOAD_GB setzen, z.B. OFFLOAD_GB=8}"

case "$MODEL_KEY" in
  llama)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    ;;
  qwen)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    ;;
  *)
    echo "Fehler: Unbekanntes MODEL_KEY '${MODEL_KEY}'. Erlaubt sind: llama, qwen"
    exit 1
    ;;
esac

BASE_URL="http://127.0.0.1:8000"
ENDPOINT="/v1/chat/completions"
API_KEY="pilotkey"

EXPERIMENT_ID="profile_grid_v2"
SERVER_CONFIG_LABEL="${MODEL_KEY}_offload${OFFLOAD_GB}"

NUM_PROMPTS=20
NUM_WARMUPS=1
TEMP=0

CONCURRENCIES=(1 2 4 8 12 16)
INPUT_LENS=(256)
OUTPUT_LENS=(32 64 128)
RUNS_PER_CELL=5

OUTDIR="${PROJECT_ROOT}/new/runs/profile_grid_v2/${MODEL_KEY}/offload${OFFLOAD_GB}"
mkdir -p "$OUTDIR"

LOGDIR="${PROJECT_ROOT}/new/logs/client"
mkdir -p "$LOGDIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGDIR}/client_${MODEL_KEY}_offload${OFFLOAD_GB}_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

export OPENAI_API_KEY="$API_KEY"

TOTAL_CELLS=$(( ${#CONCURRENCIES[@]} * ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * RUNS_PER_CELL ))

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_KEY: ${MODEL_KEY}"
echo "MODEL: ${MODEL}"
echo "OFFLOAD_GB: ${OFFLOAD_GB}"
echo "OUTDIR: ${OUTDIR}"
echo "LOGFILE: ${LOGFILE}"
echo "Gesamte Matrixgroesse: ${TOTAL_CELLS} JSON-Dateien"
echo

echo "Pruefe Erreichbarkeit des Servers unter ${BASE_URL}..."
if ! curl -fsS -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/v1/models" >/dev/null; then
  echo "Fehler: Server unter ${BASE_URL} ist nicht erreichbar."
  echo "Bitte zuerst den vLLM-Server mit model_key=${MODEL_KEY} und offload_gb=${OFFLOAD_GB} starten."
  exit 1
fi
echo "Server ist erreichbar."
echo

#-------------------------------------------------------------------
PREHEAT="${PREHEAT:-1}"

if [[ "$PREHEAT" == "1" ]]; then
  PREHEAT_DIR="${PROJECT_ROOT}/new/status/preheat"
  mkdir -p "$PREHEAT_DIR"

  PREHEAT_TAG="preheat_${MODEL_KEY}_offload${OFFLOAD_GB}_$(date +%Y%m%d_%H%M%S)"

  echo "Starte Throwaway-Preheat-Benchmark: ${PREHEAT_TAG}"

  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --num-prompts 20 \
    --num-warmups 5 \
    --random-input-len 128 \
    --random-output-len 32 \
    --max-concurrency 1 \
    --temperature "$TEMP" \
    --save-result \
    --save-detailed \
    --result-dir "$PREHEAT_DIR" \
    --result-filename "${PREHEAT_TAG}.json" \
    --metadata \
      experiment_id="profile_grid_v2_preheat" \
      server_config_label="${SERVER_CONFIG_LABEL}" \
      model_key="${MODEL_KEY}" \
      model_name="${MODEL}" \
      offload_gb="${OFFLOAD_GB}" \
      role="throwaway_preheat" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99

  echo "Preheat abgeschlossen. Ergebnis wird NICHT für Analyse verwendet."
  echo
fi

#-------------------------------------------------------------------

run_bench() {
  local conc="$1"
  local input_len="$2"
  local output_len="$3"
  local run_no="$4"
  local tag="${MODEL_KEY}_offload${OFFLOAD_GB}_conc${conc}_in${input_len}_out${output_len}_run${run_no}"
  local outfile="${OUTDIR}/${tag}.json"

  if [[ -s "$outfile" ]]; then
    echo "=== SKIP (bereits vorhanden): ${outfile} ==="
    return
  fi

  echo "=== START: model=${MODEL_KEY}, offload=${OFFLOAD_GB}, concurrency=${conc}, input_len=${input_len}, output_len=${output_len}, run=${run_no} ==="

  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --max-concurrency "$conc" \
    --temperature "$TEMP" \
    --save-result \
    --save-detailed \
    --result-dir "$OUTDIR" \
    --result-filename "${tag}.json" \
    --metadata \
      experiment_id="${EXPERIMENT_ID}" \
      server_config_label="${SERVER_CONFIG_LABEL}" \
      model_key="${MODEL_KEY}" \
      model_name="${MODEL}" \
      offload_gb="${OFFLOAD_GB}" \
      concurrency="${conc}" \
      input_len="${input_len}" \
      output_len="${output_len}" \
      num_prompts="${NUM_PROMPTS}" \
      num_warmups="${NUM_WARMUPS}" \
      temperature="${TEMP}" \
      run_no="${run_no}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99

  echo "=== DONE: ${outfile} ==="
  echo
}

for conc in "${CONCURRENCIES[@]}"; do
  for input_len in "${INPUT_LENS[@]}"; do
    for output_len in "${OUTPUT_LENS[@]}"; do
      for run_no in $(seq 1 "$RUNS_PER_CELL"); do
        run_bench "$conc" "$input_len" "$output_len" "$run_no"
      done
    done
  done
done

echo "Alle Benchmarks abgeschlossen."
echo "Ergebnisse liegen in: ${OUTDIR}"
