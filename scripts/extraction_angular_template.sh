#!/bin/bash

#SBATCH --output=scripts/logs/extract/%x_%A.out
#SBATCH --error=scripts/logs/extract/%x_%A.err

module load GCC
module load CUDA/12.4

echo "=== Angular Extraction ==="
echo "Model:  ${ANG_MODEL_NAME}"
echo "Output: ${ANG_OUTPUT_PATH}"
echo "Data:   ${ANG_DATA_PATH}"
echo "=========================="

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p "$(dirname "${ANG_OUTPUT_PATH}")"

EXTRA_ARGS=""
if [[ "${ANG_USE_SYSTEM_PROMPT}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_system_prompt"
fi
if [[ -n "${ANG_MAX_SAMPLES}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max_samples ${ANG_MAX_SAMPLES}"
fi
if [[ -n "${ANG_BATCH_SIZE}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --batch_size ${ANG_BATCH_SIZE}"
fi
if [[ -n "${ANG_NORM_FLOOR}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --norm_floor ${ANG_NORM_FLOOR}"
fi
if [[ -n "${ANG_EXCLUDE_TASKS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --exclude_tasks ${ANG_EXCLUDE_TASKS}"
fi
if [[ -n "${ANG_SEED}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --seed ${ANG_SEED}"
fi
if [[ "${ANG_DEDUPE}" == "0" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no_dedupe"
fi
if [[ "${ANG_STRATIFIED}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --stratified"
fi
if [[ -n "${ANG_SUFFIX_POOL}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --suffix_pool ${ANG_SUFFIX_POOL}"
fi
if [[ "${ANG_SAVE_NOTEBOOK_CONFIG}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save_notebook_config"
fi
if [[ -n "${ANG_LOG_LEVEL}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --log_level ${ANG_LOG_LEVEL}"
fi

"${ANG_PYTHON_BIN}" angular/extract_angular.py \
    --model_name "${ANG_MODEL_NAME}" \
    --data_path "${ANG_DATA_PATH}" \
    --output_path "${ANG_OUTPUT_PATH}" \
    ${EXTRA_ARGS}

echo "Done extracting angular vectors."

