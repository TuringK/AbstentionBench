#!/usr/bin/env bash
#
# Minimal results analysis runner:
# - loop over model prefixes
# - loop over coeffs
# - run one pass excluding rulebreakers
#
# Edit MODELS and COEFFS below as needed.
#
# Usage (after: source env.sh - PROJECT_ROOT must be set):
#   ./scripts/run_results_analysis_v3.sh
#
# Optional env overrides (absolute paths, or under PROJECT_ROOT):
#   PYTHON_CMD=python3
#   RESULTS_PARENT   default ${PROJECT_ROOT}/data/new_dataset_exps
#   OUTPUT_ROOT      default ${PROJECT_ROOT}/data/v3_csv
#   TRAINING_DATA    default ${PROJECT_ROOT}/data/abstention_training_dataset.json
#   DATASET=rulebreakers
#   DRY_RUN=1
#
# Slurm array mode:
#   ./scripts/submit_results_analysis_v3.sh   # passes absolute log paths + array script path
#   Uses exported PROJECT_ROOT (submit script uses --export=ALL).
#
#SBATCH --output=scripts/logs/batch_results_analysis/%x_%A_%a.out
#SBATCH --error=scripts/logs/batch_results_analysis/%x_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${PROJECT_ROOT:?source env.sh first - PROJECT_ROOT must be exported}"

PYTHON_CMD="${PYTHON_CMD:-${PYTHON_BIN:-python}}"
RESULTS_PARENT="${RESULTS_PARENT:-${PROJECT_ROOT}/data/new_dataset_exps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/v3_csv}"
TRAINING_DATA="${TRAINING_DATA:-${PROJECT_ROOT}/data/abstention_training_dataset.json}"
DATASET="${DATASET:-rulebreakers}"
DRY_RUN="${DRY_RUN:-0}"

# Format per entry: "<DIR_PREFIX>|<OUT_PREFIX>"
# DIR_PREFIX: part before "_coeff_<x>_v3_sweep" in results dir name
# OUT_PREFIX: short prefix used in output file/dir names
MODELS=(
  "Llama3_1_Tulu_3_1_8B_DecayCPFSim_judge_CAA|tulu_8"
  "Qwen2_5_0_5B_Instruct_DecayCPFSim_judge_CAA|qwen_0_5"
  "Qwen2_5_1_5B_Instruct_DecayCPFSim_judge_CAA|qwen_1_5"
  "Qwen2_5_3B_Instruct_DecayCPFSim_judge_CAA|qwen_3"
  "Qwen2_5_7B_Instruct_DecayCPFSim_judge_CAA|qwen_7"
)

# Use underscore format to match directory naming: 1_0, 2_0, ... 10_0
COEFFS=(1_0 2_0 3_0 4_0 5_0 6_0 7_0 8_0 9_0 10_0)

total_tasks() {
  echo $(( ${#MODELS[@]} * ${#COEFFS[@]} ))
}

if [[ "${1:-}" == "--print-task-count" ]]; then
  total_tasks
  exit 0
fi

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_one_combo() {
  local model_spec="$1"
  local coeff="$2"

  dir_prefix="${model_spec%%|*}"
  out_prefix="${model_spec##*|}"
  out_dir="${OUTPUT_ROOT}/${out_prefix}_sweep"
  mkdir -p "${out_dir}"

  steering_dir="${RESULTS_PARENT}/${dir_prefix}_coeff_${coeff}_v3_sweep"

  if [[ ! -d "${steering_dir}" ]]; then
    echo "warning: missing directory, skipping: ${steering_dir}" >&2
    return 0
  fi

  # Single pass: everything except rulebreakers
  out_exclude="${out_dir}/${out_prefix}_v3_sweep_${coeff}.csv"
  run_cmd "${PYTHON_CMD}" "${PROJECT_ROOT}/analysis/results_analysis.py" \
    --steering-dir "${steering_dir}" \
    --filter-training \
    --training-data "${TRAINING_DATA}" \
    --exclude-datasets "${DATASET}" \
    --output "${out_exclude}"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  model_count="${#MODELS[@]}"
  coeff_count="${#COEFFS[@]}"
  n_tasks="$(total_tasks)"

  task_id="${SLURM_ARRAY_TASK_ID}"
  if (( task_id < 1 || task_id > n_tasks )); then
    echo "error: SLURM_ARRAY_TASK_ID=${task_id} out of range 1..${n_tasks}" >&2
    exit 1
  fi

  zero_based=$((task_id - 1))
  model_idx=$((zero_based / coeff_count))
  coeff_idx=$((zero_based % coeff_count))

  model_spec="${MODELS[model_idx]}"
  coeff="${COEFFS[coeff_idx]}"

  echo "slurm task ${task_id}/${n_tasks}: model_idx=${model_idx} coeff_idx=${coeff_idx}"
  run_one_combo "${model_spec}" "${coeff}"
else
  for model_spec in "${MODELS[@]}"; do
    for coeff in "${COEFFS[@]}"; do
      run_one_combo "${model_spec}" "${coeff}"
    done
  done
fi

echo "done"
