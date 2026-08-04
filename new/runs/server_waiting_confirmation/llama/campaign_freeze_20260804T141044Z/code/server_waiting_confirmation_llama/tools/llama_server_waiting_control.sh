#!/usr/bin/env bash
# Safe control plane for the Llama Server-WAITING port.
set -Eeuo pipefail
umask 022

VERSION="llama-server-waiting-control-v1"
PROJECT="${PROJECT:-$HOME/projects/DOS}"
VENV="${VENV_PATH:-$HOME/projects/porto/.venv-py313-repair}"
SCRIPT_DIR="$PROJECT/new/scripts/server_waiting_confirmation_llama"
TOOLS_DIR="$SCRIPT_DIR/tools"
RUNNER="$SCRIPT_DIR/run_server_waiting_confirmation.py"
WRAPPER="$SCRIPT_DIR/run_server_waiting_confirmation.sh"
SERVER="$SCRIPT_DIR/run_server_waiting_server.sh"
GENERATOR="$SCRIPT_DIR/generate_server_waiting_schedule.py"
COHORT="$SCRIPT_DIR/_active_cohort.py"
AUDITOR="$SCRIPT_DIR/audit_server_waiting_diagnostic.py"
BASE="$PROJECT/new/scripts/prefill_confirmation/run_prefill_confirmation.py"
SCHEDULE="$PROJECT/new/runs/server_waiting_confirmation/llama/schedule"
RESULT_ROOT="$PROJECT/new/runs/server_waiting_confirmation/llama"
LOG_ROOT="$PROJECT/new/logs/server_waiting_confirmation_llama"
FINALIZER="$TOOLS_DIR/campaign_finalize.py"
SELF="$(readlink -f "$0")"

EXPECTED_RUNNER="d057513ca3517692fced40de4094414ca3cd515ae10234552108a03e72749cd1"
EXPECTED_GENERATOR="13dc5a197bf07250e885757c279893dab98bbc2cc6896c3b39579780a61296b7"
EXPECTED_COHORT="57c88a8410a16d7a85432b3a3684842e813d75b4f9388392279e9bacfbc26c8e"
EXPECTED_AUDITOR="2aeaed134840c3566525d71895b185aa482c1a3fff6fbd7b8b10978bbd5af9f8"
EXPECTED_WRAPPER="06eeb038da93a37f0a29207911e835547986c566ade8edafe48a764e58e1eb25"
EXPECTED_SERVER="6518b10a2e6a49c36a4b57ebe34c59023c1dba6135c3b0cef5c47e44b0d7d207"
EXPECTED_BASE="981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75"
EXPECTED_SCHEDULE_JSON="93f4827859cbe96e8d27515ae836974c1608a5ff9a3f4aed774a558e86d09cb9"
EXPECTED_SCHEDULE_CSV="a0f7eca4fcedaa3b0ccaeaeca39475abc70babf2169a6eb505491e0c2f1b8068"
EXPECTED_SCHEDULE_AUDIT="70a719a8ef477c752030d81adc623f136ed0a2d04c98d3524200ae7de6f35ee8"
EXPECTED_FINGERPRINT="sha256:eaf7d27c0e1d5d19cae5fa5ea7e4c5495cd1391692e382ecf4dd0c757c1754e0"

fail(){ echo "FEHLER: $*" >&2; exit 1; }
sha(){ sha256sum "$1" | awk '{print $1}'; }
assert_sha(){ local p="$1" e="$2" a; test -f "$p" || fail "Fehlt: $p"; a="$(sha "$p")"; printf '%s=%s\n' "$(basename "$p")" "$a"; test "$a" = "$e" || fail "Hashabweichung: $p"; }

usage(){ cat <<EOF2
$VERSION

Nur prüfen, keine Messung:
  bash "$SELF" --check

Reproduktionspaket vor der offiziellen Kampagne einfrieren:
  bash "$SELF" --freeze

REALE diagnostische Paarmessung (2 Episoden) und Raw-Trace-Audit:
  bash "$SELF" --diagnostic

REALE vollständige Llama-Kampagne (32 Episoden / 16 Blöcke):
  bash "$SELF" --campaign

Vollkampagne im Hintergrund:
  bash "$SELF" --background-campaign

Read-only Re-Audit des neuesten final_capture:
  bash "$SELF" --reaudit

Ohne eine dieser Optionen startet nichts.
EOF2
}

idle_check(){
  if ss -H -ltnp 'sport = :8000' 2>/dev/null | grep -q .; then
    ss -H -ltnp 'sport = :8000' 2>/dev/null || true
    fail "Port 8000 ist belegt"
  fi
  echo "PORT_8000_FREE"
  local related gpu
  related="$(pgrep -af '(^|/)(vllm|run_server_waiting_confirmation|run_server_waiting_server)([[:space:]]|$)' 2>/dev/null || true)"
  test -z "$related" || { echo "$related"; fail "Verwandter Prozess läuft"; }
  echo "NO_RELATED_PROCESSES"
  gpu="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  test -z "$gpu" || { echo "$gpu"; fail "GPU-Compute-Prozess läuft"; }
  echo "NO_GPU_COMPUTE_PROCESSES"
}

core_hash_check(){
  echo "=== LLAMA PORT IDENTITIES ==="
  assert_sha "$RUNNER" "$EXPECTED_RUNNER"
  assert_sha "$GENERATOR" "$EXPECTED_GENERATOR"
  assert_sha "$COHORT" "$EXPECTED_COHORT"
  assert_sha "$AUDITOR" "$EXPECTED_AUDITOR"
  assert_sha "$WRAPPER" "$EXPECTED_WRAPPER"
  assert_sha "$SERVER" "$EXPECTED_SERVER"
  assert_sha "$BASE" "$EXPECTED_BASE"
  assert_sha "$SCHEDULE/server_waiting_confirmation_schedule.json" "$EXPECTED_SCHEDULE_JSON"
  assert_sha "$SCHEDULE/server_waiting_confirmation_schedule.csv" "$EXPECTED_SCHEDULE_CSV"
  assert_sha "$SCHEDULE/server_waiting_confirmation_schedule_audit.txt" "$EXPECTED_SCHEDULE_AUDIT"
}

model_cache_check(){
  echo "=== LOCAL LLAMA CACHE ==="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" - <<'PY'
from transformers import AutoTokenizer
model="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(model, local_files_only=True)
print("LOCAL_MODEL_TOKENIZER_PASS")
print(f"tokenizer_class={type(tok).__name__}")
PY
}

preflight(){
  echo "=== LLAMA SERVER-WAITING PREFLIGHT ==="
  echo "version=$VERSION"
  for cmd in bash sha256sum git ss pgrep nvidia-smi find sort xargs awk cp tar gzip readlink mktemp; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Kommando fehlt: $cmd"
  done
  test -x "$VENV/bin/python" || fail "Venv-Python fehlt: $VENV/bin/python"
  test -x "$WRAPPER" || fail "Wrapper fehlt/nicht ausführbar: $WRAPPER"
  test -x "$FINALIZER" || fail "Finalizer fehlt/nicht ausführbar: $FINALIZER"
  test -d "$SCHEDULE" || fail "Schedule fehlt: $SCHEDULE"
  core_hash_check
  test -f "$SCRIPT_DIR/SHA256SUMS.txt" || fail "SHA256SUMS.txt fehlt"
  (cd "$PROJECT" && sha256sum -c new/scripts/server_waiting_confirmation_llama/SHA256SUMS.txt)
  echo "=== GIT TRACKED STATE ==="
  git -C "$PROJECT" diff --quiet || fail "Getrackte Änderungen vorhanden"
  git -C "$PROJECT" diff --cached --quiet || fail "Getrackte Stage-Änderungen vorhanden"
  echo "TRACKED_WORKTREE_CLEAN=yes"
  model_cache_check
  idle_check
  echo "=== OFFLINE SELF-TEST ==="
  (cd "$SCRIPT_DIR" && "$VENV/bin/python" run_server_waiting_confirmation.py --self-test)
  echo "=== OFFICIAL DRY RUN ==="
  (
    cd "$SCRIPT_DIR"
    VENV_PATH="$VENV" PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" \
    GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ./run_server_waiting_confirmation.sh --dry-run --model-key llama --schedule-dir "$SCHEDULE"
  )
  "$VENV/bin/python" - "$SCHEDULE/server_waiting_confirmation_schedule.json" "$EXPECTED_FINGERPRINT" <<'PY'
import json,sys
p,expected=sys.argv[1:]
actual=json.load(open(p,encoding='utf-8'))['schedule_fingerprint']
print(f"schedule_fingerprint={actual}")
if actual != expected: raise SystemExit("fingerprint mismatch")
PY
  echo "LLAMA_PREFLIGHT_PASS"
  echo "NO_SERVER_STARTED"
  echo "NO_REQUESTS_SENT"
}

hash_tree(){
  local root="$1"
  (cd "$root" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
}

freeze_campaign(){
  preflight
  local stamp freeze archive
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  freeze="$RESULT_ROOT/campaign_freeze_${stamp}"
  archive="$freeze.tar.gz"
  test ! -e "$freeze" || fail "Freeze existiert: $freeze"
  mkdir -p "$freeze/code/server_waiting_confirmation_llama" "$freeze/code/prefill_confirmation" "$freeze/schedule" "$freeze/context"
  cp -a "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/SHA256SUMS.txt "$freeze/code/server_waiting_confirmation_llama/"
  cp -a "$SCRIPT_DIR/tools" "$freeze/code/server_waiting_confirmation_llama/"
  cp -a "$BASE" "$freeze/code/prefill_confirmation/"
  cp -a "$SCHEDULE"/. "$freeze/schedule/"
  {
    echo "freeze_created_utc=$(date -u -Is)"
    echo "git_commit=$(git -C "$PROJECT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$PROJECT" branch --show-current)"
    echo "git_status_begin"
    git -C "$PROJECT" status --short
    echo "git_status_end"
    echo "python=$($VENV/bin/python --version 2>&1)"
    echo "vllm=$($VENV/bin/python -c 'import importlib.metadata as m; print(m.version("vllm"))')"
    echo "torch=$($VENV/bin/python -c 'import torch; print(torch.__version__)')"
    echo "transformers=$($VENV/bin/python -c 'import transformers; print(transformers.__version__)')"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
  } > "$freeze/context/ENVIRONMENT_AND_GIT.txt"
  {
    echo "FREEZE_VERSION=llama-server-waiting-freeze-v1"
    echo "RUNNER_SHA256=$EXPECTED_RUNNER"
    echo "GENERATOR_SHA256=$EXPECTED_GENERATOR"
    echo "COHORT_SHA256=$EXPECTED_COHORT"
    echo "AUDITOR_SHA256=$EXPECTED_AUDITOR"
    echo "WRAPPER_SHA256=$EXPECTED_WRAPPER"
    echo "SERVER_SHA256=$EXPECTED_SERVER"
    echo "BASE_SHA256=$EXPECTED_BASE"
    echo "SCHEDULE_FINGERPRINT=$EXPECTED_FINGERPRINT"
  } > "$freeze/FREEZE_READY.env"
  (
    cd "$freeze"
    find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt >/dev/null
  )
  tar --sort=name --owner=0 --group=0 --numeric-owner -C "$(dirname "$freeze")" -cf - "$(basename "$freeze")" | gzip -n > "$archive"
  sha256sum "$archive" > "$archive.sha256"
  chmod -R a-w "$freeze"
  chmod a-w "$archive" "$archive.sha256"
  printf '%s\n' "$freeze" > "$RESULT_ROOT/LATEST_FREEZE_PATH.txt"
  printf '%s\n' "$archive" > "$RESULT_ROOT/LATEST_FREEZE_ARCHIVE.txt"
  echo "LLAMA_CAMPAIGN_FREEZE_PASS"
  echo "FREEZE_DIR=$freeze"
  echo "FREEZE_ARCHIVE=$archive"
  echo "FREEZE_ARCHIVE_SHA256=$(awk '{print $1}' "$archive.sha256")"
}

require_freeze(){
  local marker freeze envfile
  marker="$RESULT_ROOT/LATEST_FREEZE_PATH.txt"
  test -f "$marker" || fail "Kein Llama-Freeze vorhanden. Zuerst --freeze ausführen."
  freeze="$(cat "$marker")"
  envfile="$freeze/FREEZE_READY.env"
  test -r "$envfile" || fail "Freeze-Metadaten fehlen: $envfile"
  # shellcheck disable=SC1090
  source "$envfile"
  test "${RUNNER_SHA256:-}" = "$EXPECTED_RUNNER" || fail "Freeze passt nicht zum Runner"
  test "${GENERATOR_SHA256:-}" = "$EXPECTED_GENERATOR" || fail "Freeze passt nicht zum Generator"
  test "${COHORT_SHA256:-}" = "$EXPECTED_COHORT" || fail "Freeze passt nicht zum Kohortenmodul"
  test "${AUDITOR_SHA256:-}" = "$EXPECTED_AUDITOR" || fail "Freeze passt nicht zum Auditor"
  test "${BASE_SHA256:-}" = "$EXPECTED_BASE" || fail "Freeze passt nicht zum Base-Runner"
  test "${SCHEDULE_FINGERPRINT:-}" = "$EXPECTED_FINGERPRINT" || fail "Freeze passt nicht zum Schedule"
  (cd "$freeze" && sha256sum -c SHA256SUMS.txt >/dev/null)
  echo "FREEZE_VERIFIED=$freeze"
}

real_env(){
  export VENV_PATH="$VENV" PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"
  export GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export VLLM_API_KEY
  VLLM_API_KEY="$($VENV/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
}

run_diagnostic(){
  preflight
  local stamp output audit rc arc
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output="$RESULT_ROOT/diagnostic_pair_${stamp}"
  audit="$RESULT_ROOT/raw_trace_audit_${stamp}"
  test ! -e "$output" || fail "Output existiert: $output"
  test ! -e "$audit" || fail "Audit existiert: $audit"
  real_env
  echo "=== REALE LLAMA-DIAGNOSE STARTET ==="
  echo "THIS_IS_A_REAL_RUN=yes"
  echo "episodes=2"
  echo "block=llama_off12_k8_rep01"
  echo "output=$output"
  cd "$SCRIPT_DIR"
  set +e
  ./run_server_waiting_confirmation.sh --diagnostic-pair-only --model-key llama --schedule-dir "$SCHEDULE" --output-dir "$output" --host 127.0.0.1 --port 8000
  rc=$?
  set -e
  unset VLLM_API_KEY
  echo "DIAGNOSTIC_RUN_RC=$rc"
  if test "$rc" -ne 0 && test "$rc" -ne 2; then
    echo "DIAGNOSTIC_RUN_FAILED"
    echo "DO_NOT_RERUN_AUTOMATICALLY"
    exit "$rc"
  fi
  idle_check
  set +e
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" "$AUDITOR" \
    --diagnostic-dir "$output" --schedule-dir "$SCHEDULE" --audit-output-dir "$audit"
  arc=$?
  set -e
  echo "RAW_TRACE_AUDIT_RC=$arc"
  echo "DIAGNOSTIC_DIR=$output"
  echo "AUDIT_DIR=$audit"
  if test "$arc" -ne 0; then
    echo "LLAMA_DIAGNOSTIC_AUDIT_FAILED"
    exit "$arc"
  fi
  "$VENV/bin/python" - "$audit/auditor_result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"overall_audit_status={r.get('overall_audit_status')}")
print(f"scientifically_evaluable={r.get('scientifically_evaluable')}")
print(f"stored_semantic_classification={r.get('stored_semantic_classification')}")
print(f"raw_output_overlap_classification={r.get('raw_output_overlap_classification')}")
if r.get('overall_audit_status') != 'PASS' or r.get('scientifically_evaluable') is not True:
    raise SystemExit(2)
PY
  echo "LLAMA_DIAGNOSTIC_PASS"
}

verify_counts(){
  "$VENV/bin/python" - "$1" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
for name,expected,pat in [('episodes',32,'*.json'),('block_summaries',16,'*.json'),('stabilization',16,'*.json'),('server_logs',16,'*.log')]:
    n=len(list((r/name).glob(pat))); print(f'{name}_count={n}')
    if n != expected: raise SystemExit(f'{name}: {n} != {expected}')
s=json.load(open(r/'official_run_summary.json',encoding='utf-8'))
for k in ('overall_status','valid_complete_episodes','completed_blocks','missing_episodes'):
    print(f'{k}={s.get(k)}')
if not (s.get('overall_status')=='complete' and s.get('valid_complete_episodes')==32 and s.get('completed_blocks')==16 and s.get('missing_episodes')==0):
    raise SystemExit('official summary incomplete')
PY
}

finalize_campaign(){
  local source="$1" stamp="$2" capture archive before after
  capture="$RESULT_ROOT/final_capture_${stamp}"
  archive="$capture.tar.gz"
  test ! -e "$capture" || fail "Capture existiert: $capture"
  mkdir -p "$capture/provenance"
  before="$(mktemp)"; after="$(mktemp)"
  hash_tree "$source" > "$before"
  cp -a "$before" "$capture/provenance/source_campaign_before.sha256"
  cp -a "$source" "$capture/source_campaign"
  hash_tree "$source" > "$after"
  cmp "$before" "$after" || fail "Quellkampagne änderte sich beim Kopieren"
  rm -f "$before" "$after"
  echo "SOURCE_TREE_UNCHANGED_PASS"
  "$VENV/bin/python" "$FINALIZER" "$capture/source_campaign" "$capture"
  cat > "$capture/README.md" <<'EOF2'
# Finalized Llama Server-WAITING Campaign Capture

The source campaign is copied byte-for-byte. The runner's known inner-manifest
ordering gap is preserved transparently and neutralized by the outer
BUNDLE_SHA256SUMS.txt, which is generated last over copied and derived files.

Scope: client-visible output timing only; no claim about internal admission or
internal prefill-start time.
EOF2
  {
    echo "control_version=$VERSION"
    echo "control_sha256=$(sha "$SELF")"
    echo "runner_sha256=$(sha "$RUNNER")"
    echo "base_sha256=$(sha "$BASE")"
    echo "cohort_sha256=$(sha "$COHORT")"
    echo "generator_sha256=$(sha "$GENERATOR")"
    echo "auditor_sha256=$(sha "$AUDITOR")"
    echo "schedule_fingerprint=$EXPECTED_FINGERPRINT"
    echo "git_commit=$(git -C "$PROJECT" rev-parse HEAD)"
    echo "capture_utc=$(date -u -Is)"
    echo "source_campaign=$source"
    echo "freeze_path=$(cat "$RESULT_ROOT/LATEST_FREEZE_PATH.txt")"
  } > "$capture/provenance/capture_identity.txt"
  (
    cd "$capture"
    find . -type f ! -name BUNDLE_SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > BUNDLE_SHA256SUMS.txt
    sha256sum -c BUNDLE_SHA256SUMS.txt >/dev/null
  )
  echo "OUTER_MANIFEST_VERIFICATION_PASS"
  tar --sort=name --owner=0 --group=0 --numeric-owner -C "$(dirname "$capture")" -cf - "$(basename "$capture")" | gzip -n > "$archive"
  sha256sum "$archive" > "$archive.sha256"
  chmod -R a-w "$capture"
  chmod a-w "$archive" "$archive.sha256"
  sync
  echo "FINAL_CAPTURE_COMPLETE"
  echo "FINAL_CAPTURE_DIR=$capture"
  echo "FINAL_CAPTURE_ARCHIVE=$archive"
  echo "FINAL_CAPTURE_ARCHIVE_SHA256=$(awk '{print $1}' "$archive.sha256")"
}

run_campaign(){
  preflight
  require_freeze
  local stamp output rc
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output="$RESULT_ROOT/official_campaign_${stamp}"
  test ! -e "$output" || fail "Output existiert: $output"
  real_env
  echo "=== REALE LLAMA-VOLLKAMPAGNE STARTET ==="
  echo "THIS_IS_A_REAL_RUN=yes"
  echo "episodes=32"
  echo "blocks=16"
  echo "output=$output"
  cd "$SCRIPT_DIR"
  set +e
  ./run_server_waiting_confirmation.sh --official-run --model-key llama --schedule-dir "$SCHEDULE" --output-dir "$output" --host 127.0.0.1 --port 8000
  rc=$?
  set -e
  unset VLLM_API_KEY
  echo "RUN_RC=$rc"
  echo "OUTPUT_DIR=$output"
  if test "$rc" -ne 0; then
    echo "CAMPAIGN_FAILED"
    echo "DO_NOT_RERUN_AUTOMATICALLY"
    exit "$rc"
  fi
  idle_check
  verify_counts "$output"
  finalize_campaign "$output" "$stamp"
  echo "LLAMA_CAMPAIGN_PASS"
  echo "SOURCE_CAMPAIGN_DIR=$output"
}

background_campaign(){
  mkdir -p "$LOG_ROOT"
  local stamp log pid
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$LOG_ROOT/llama_campaign_${stamp}.log"
  nohup bash "$SELF" --campaign > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "BACKGROUND_CAMPAIGN_STARTED"
  echo "PID=$pid"
  echo "LOG=$log"
  echo "FOLLOW_LOG=tail -f \"$log\""
}

run_reaudit(){
  exec bash "$TOOLS_DIR/llama_full_campaign_reaudit.sh" "${2:-}"
}

case "${1:---help}" in
  --help|-h) usage ;;
  --check) preflight ;;
  --freeze) freeze_campaign ;;
  --diagnostic) run_diagnostic ;;
  --campaign) run_campaign ;;
  --background-campaign) background_campaign ;;
  --reaudit) bash "$TOOLS_DIR/llama_full_campaign_reaudit.sh" ;;
  *) usage >&2; fail "Unbekannte Option: ${1:-}" ;;
esac
