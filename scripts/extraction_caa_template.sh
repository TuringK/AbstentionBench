#!/bin/bash

#SBATCH --output=scripts/logs/extract_caa/%x_%A_%a.out
#SBATCH --error=scripts/logs/extract_caa/%x_%A_%a.err

# Generic SLURM job template for CAA vector extraction.
# All values are passed via environment variables (CAA_EXT_*)
# set by caa/run_extraction.py.

module load GCC
module load CUDA/12.4

LAYER_IDX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_FILE="${CAA_EXT_OUTPUT_DIR}/vec_layer_${LAYER_IDX}.pt"

echo "=== CAA Extraction ==="
echo "Model:  ${CAA_EXT_MODEL_NAME}"
echo "Layer:  ${LAYER_IDX}"
echo "Output: ${OUTPUT_FILE}"
echo "Data:   ${CAA_EXT_DATA_PATH}"
echo "======================"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# build optional flags
EXTRA_ARGS=""
if [[ "${CAA_EXT_USE_SYSTEM_PROMPT}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_system_prompt"
fi
if [[ "${CAA_EXT_WEIGHTED}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --weighted"
fi
if [[ -n "${CAA_EXT_EXCLUDE_SCENARIOS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --exclude_scenarios ${CAA_EXT_EXCLUDE_SCENARIOS}"
fi
if [[ "${CAA_EXT_NORMALIZE}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --normalize"
fi
if [[ -n "${CAA_EXT_RESPONSE_TOKENS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --response_tokens ${CAA_EXT_RESPONSE_TOKENS}"
fi
if [[ -n "${CAA_EXT_DATA_FORMAT}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --data_format ${CAA_EXT_DATA_FORMAT}"
fi

"${CAA_EXT_PYTHON_BIN}" caa/extract_caa_vectors.py \
    --model_name "${CAA_EXT_MODEL_NAME}" \
    --data_path "${CAA_EXT_DATA_PATH}" \
    --output_path "${OUTPUT_FILE}" \
    --layer_idx ${LAYER_IDX} \
    ${EXTRA_ARGS}

echo "Done extracting layer ${LAYER_IDX}"
