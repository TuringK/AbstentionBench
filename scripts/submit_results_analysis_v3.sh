#!/usr/bin/env bash
#
# Submit Slurm array for scripts/run_results_analysis_v3.sh.
#
# From repo root (after: source env.sh):
#   MAX_PARALLEL=20 ./scripts/submit_results_analysis_v3.sh
#
# Uses PROJECT_ROOT for paths (no cd). --export=ALL passes env to tasks.
#
set -euo pipefail

: "${PROJECT_ROOT:?source env.sh first - PROJECT_ROOT must be set}"

MAX_PARALLEL="${MAX_PARALLEL:-20}"
JOB_SCRIPT="${PROJECT_ROOT}/scripts/run_results_analysis_v3.sh"
LOGDIR="${PROJECT_ROOT}/scripts/logs/batch_results_analysis"

if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "error: MAX_PARALLEL must be a positive integer, got '${MAX_PARALLEL}'" >&2
  exit 1
fi

NUM_TASKS="$("${JOB_SCRIPT}" --print-task-count)"
if ! [[ "${NUM_TASKS}" =~ ^[0-9]+$ ]] || [[ "${NUM_TASKS}" -lt 1 ]]; then
  echo "error: computed invalid task count '${NUM_TASKS}'" >&2
  exit 1
fi

# Concurrency cap cannot exceed total number of tasks.
if [[ "${MAX_PARALLEL}" -gt "${NUM_TASKS}" ]]; then
  MAX_PARALLEL="${NUM_TASKS}"
fi

mkdir -p "${LOGDIR}"

echo "Tasks: ${NUM_TASKS}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Submitting array: 1-${NUM_TASKS}%${MAX_PARALLEL}"
exec sbatch \
  --export=ALL \
  --output="${LOGDIR}/%x_%A_%a.out" \
  --error="${LOGDIR}/%x_%A_%a.err" \
  --array="1-${NUM_TASKS}%${MAX_PARALLEL}" \
  "${JOB_SCRIPT}"
