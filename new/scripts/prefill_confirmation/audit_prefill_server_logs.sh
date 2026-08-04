#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/DOS}"
LLAMA_DIR="${LLAMA_DIR:-$PROJECT_ROOT/new/runs/prefill_confirmation/llama/results/official_initial8_20260717T083031Z}"
QWEN_DIR="${QWEN_DIR:-$PROJECT_ROOT/new/runs/prefill_confirmation/qwen/results/official_initial8_20260718T052207Z}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/new/runs/prefill_confirmation/log_audit_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/SUMMARY.txt"
STRICT_HITS="$OUT_DIR/strict_preemption_hits.txt"
BROAD_HITS="$OUT_DIR/broad_scheduler_kv_hits.txt"
ERROR_HITS="$OUT_DIR/error_warning_hits.txt"
LOG_INDEX="$OUT_DIR/log_index.tsv"

: > "$STRICT_HITS"
: > "$BROAD_HITS"
: > "$ERROR_HITS"
: > "$LOG_INDEX"

roots=("$LLAMA_DIR" "$QWEN_DIR")

strict_re='is preempted by|PreemptionMode\.(RECOMPUTE|SWAP)|cumulative[^[:cntrl:]]*preemption|not enough KV cache space|No available memory for the cache blocks|KV cache is full|out of KV cache|preemption count[^[:digit:]]*[1-9][0-9]*|num_preemptions[^[:digit:]]*[1-9][0-9]*'
broad_re='preempt|recomput|swap|KV cache|cache usage|GPU blocks|CPU blocks|running[^[:alpha:]]|waiting[^[:alpha:]]|pending[^[:alpha:]]|scheduler'
error_re='Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|ERROR|CRITICAL|Killed|segmentation fault|NCCL.*(error|failed)|engine.*dead|worker.*dead'

missing=0
total_logs=0

{
  echo "Prefill-confirmation server-log audit"
  echo "Generated UTC: $(date -u --iso-8601=seconds)"
  echo "Host: $(hostname)"
  echo
} > "$SUMMARY"

for root in "${roots[@]}"; do
  label="unknown"
  [[ "$root" == *"/llama/"* ]] && label="llama"
  [[ "$root" == *"/qwen/"*  ]] && label="qwen"

  logdir="$root/server_logs"

  {
    echo "================================================================"
    echo "MODEL: $label"
    echo "ROOT:  $root"
    echo "LOGDIR: $logdir"
  } >> "$SUMMARY"

  if [[ ! -d "$logdir" ]]; then
    echo "STATUS: MISSING server_logs directory" >> "$SUMMARY"
    echo >> "$SUMMARY"
    missing=$((missing + 1))
    continue
  fi

  mapfile -d '' logs < <(find "$logdir" -type f -name '*.log' -print0 | sort -z)
  count="${#logs[@]}"
  total_logs=$((total_logs + count))

  echo "LOG_FILES: $count" >> "$SUMMARY"

  for f in "${logs[@]}"; do
    bytes=$(stat -c '%s' "$f")
    printf '%s\t%s\t%s\n' "$label" "$bytes" "$f" >> "$LOG_INDEX"
  done

  if (( count == 0 )); then
    echo "STATUS: NO .log files found" >> "$SUMMARY"
    echo >> "$SUMMARY"
    missing=$((missing + 1))
    continue
  fi

  strict_before=$(wc -l < "$STRICT_HITS")
  broad_before=$(wc -l < "$BROAD_HITS")
  error_before=$(wc -l < "$ERROR_HITS")

  # -a: treat logs as text; -nH: include line and filename.
  grep -aHniE "$strict_re" "${logs[@]}" >> "$STRICT_HITS" || true
  grep -aHniE "$broad_re"  "${logs[@]}" >> "$BROAD_HITS"  || true
  grep -aHniE "$error_re"  "${logs[@]}" >> "$ERROR_HITS"  || true

  strict_after=$(wc -l < "$STRICT_HITS")
  broad_after=$(wc -l < "$BROAD_HITS")
  error_after=$(wc -l < "$ERROR_HITS")

  strict_count=$((strict_after - strict_before))
  broad_count=$((broad_after - broad_before))
  error_count=$((error_after - error_before))

  echo "STRICT_PREEMPTION_OR_KV_FAILURE_HITS: $strict_count" >> "$SUMMARY"
  echo "BROAD_SCHEDULER_OR_KV_CONTEXT_HITS:   $broad_count" >> "$SUMMARY"
  echo "ERROR_OR_FATAL_HITS:                 $error_count" >> "$SUMMARY"

  if (( strict_count == 0 && error_count == 0 )); then
    echo "MODEL_AUDIT: PASS_NO_STRICT_INDICATOR"
  else
    echo "MODEL_AUDIT: REVIEW_REQUIRED"
  fi >> "$SUMMARY"
  echo >> "$SUMMARY"
done

strict_total=$(wc -l < "$STRICT_HITS")
broad_total=$(wc -l < "$BROAD_HITS")
error_total=$(wc -l < "$ERROR_HITS")

{
  echo "================================================================"
  echo "OVERALL"
  echo "TOTAL_LOG_FILES: $total_logs"
  echo "MISSING_LOG_DIRECTORIES_OR_EMPTY_SETS: $missing"
  echo "STRICT_PREEMPTION_OR_KV_FAILURE_HITS: $strict_total"
  echo "BROAD_SCHEDULER_OR_KV_CONTEXT_HITS:   $broad_total"
  echo "ERROR_OR_FATAL_HITS:                 $error_total"
  echo
  if (( missing == 0 && strict_total == 0 && error_total == 0 )); then
    echo "AUDIT_RESULT: PASS_NO_LOGGED_PREEMPTION_OR_FATAL_FAILURE"
    echo
    echo "Interpretation:"
    echo "- No strict runtime preemption/KV-capacity failure indicator was found."
    echo "- This supports the statement 'no logged preemption event occurred'."
    echo "- It does NOT by itself prove that KV-cache utilization was always far from capacity."
  else
    echo "AUDIT_RESULT: REVIEW_REQUIRED"
    echo
    echo "Inspect:"
    echo "- $STRICT_HITS"
    echo "- $ERROR_HITS"
  fi
  echo
  echo "Artifacts:"
  echo "- Summary:      $SUMMARY"
  echo "- Strict hits:  $STRICT_HITS"
  echo "- Broad hits:   $BROAD_HITS"
  echo "- Error hits:   $ERROR_HITS"
  echo "- Log index:    $LOG_INDEX"
} >> "$SUMMARY"

cat "$SUMMARY"

echo
echo "Audit directory:"
realpath "$OUT_DIR"
