#!/usr/bin/env bash
set -euo pipefail

# Thin environment wrapper around run_phase_a.py. All scheduling, server
# lifecycle management (~20 restarts across the frozen Phase A design),
# request execution, and metrics collection live in run_phase_a.py --
# see that file's module docstring for why this can't be plain shell
# like the profiling stage's run_server.sh /
# run_client_profile_grid_v2.sh. This script only sets up the
# environment (venv, GPU visibility, logging) and forwards all
# arguments through unchanged.
#
# Usage (mirrors run_phase_a.py's own CLI exactly -- see its --help):
#   bash run_phase_a.sh --self-test
#   bash run_phase_a.sh --dry-run --schedule /path/to/phase_a_schedule.csv
#   GPU_DEVICE=0 bash run_phase_a.sh \
#       --schedule /path/to/phase_a_schedule.csv \
#       --output-dir /path/to/results \
#       --resume

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ -n "${GPU_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
fi

LOGDIR="${PROJECT_ROOT}/new/logs/phase_a"
mkdir -p "$LOGDIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGDIR}/run_phase_a_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "LOGFILE: ${LOGFILE}"
echo "Args: $*"
echo

python3 "${SCRIPT_DIR}/run_phase_a.py" "$@"
