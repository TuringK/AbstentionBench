#!/usr/bin/env bash
#
# Submit Slurm array for scripts/run_results_analysis_v3.sh.
#
# Usage:
#   MAX_PARALLEL=20 ./scripts/submit_results_analysis_v3.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_PARALLEL="${MAX_PARALLEL:-20}"
JOB_SCRIPT="scripts/run_results_analysis_v3.sh"

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

mkdir -p scripts/logs/batch_results_analysis

echo "Tasks: ${NUM_TASKS}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Submitting array: 1-${NUM_TASKS}%${MAX_PARALLEL}"
exec sbatch --array="1-${NUM_TASKS}%${MAX_PARALLEL}" "${JOB_SCRIPT}"
