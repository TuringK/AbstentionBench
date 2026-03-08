#!/bin/bash

#SBATCH --output=scripts/logs/extract/%x_%A_%a.out
#SBATCH --error=scripts/logs/extract/%x_%A_%a.err

# Generic SLURM job template for CAA vector extraction.
# All values are passed via environment variables (EXT_*)
# set by caa/run_extraction.py.

module load GCC
module load CUDA/12.4

LAYER_IDX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_FILE="${EXT_OUTPUT_DIR}/vec_layer_${LAYER_IDX}.pt"

echo "=== CAA Extraction ==="
echo "Model:  ${EXT_MODEL_NAME}"
echo "Layer:  ${LAYER_IDX}"
echo "Output: ${OUTPUT_FILE}"
echo "Data:   ${EXT_DATA_PATH}"
echo "======================"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# build optional flags
EXTRA_ARGS=""
if [[ "${EXT_USE_SYSTEM_PROMPT}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_system_prompt"
fi
if [[ "${EXT_WEIGHTED}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --weighted"
fi
if [[ -n "${EXT_EXCLUDE_SCENARIOS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --exclude_scenarios ${EXT_EXCLUDE_SCENARIOS}"
fi
if [[ "${EXT_NORMALIZE}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --normalize"
fi
if [[ -n "${EXT_RESPONSE_TOKENS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --response_tokens ${EXT_RESPONSE_TOKENS}"
fi

"${EXT_PYTHON_BIN}" caa/extract_caa_vectors.py \
    --model_name "${EXT_MODEL_NAME}" \
    --data_path "${EXT_DATA_PATH}" \
    --output_path "${OUTPUT_FILE}" \
    --layer_idx ${LAYER_IDX} \
    ${EXTRA_ARGS}

echo "Done extracting layer ${LAYER_IDX}"
