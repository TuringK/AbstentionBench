#!/bin/bash

#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err

# Generic SLURM job template for CAA experiments.
# All experiment-specific values are passed via environment
# variables (EXP_*) set by run_experiment.py.

module load GCC
module load CUDA/12.4

export VLLM_USE_V1=0

# Resolve paths for this array task
LAYER_IDX="${SLURM_ARRAY_TASK_ID}"
VECTOR_PATH="${EXP_VECTOR_DIR}/vec_layer_${LAYER_IDX}.pt"

if [[ ! -f "$VECTOR_PATH" ]]; then
    echo "Vector file not found: $VECTOR_PATH : skipping layer ${LAYER_IDX}"
    exit 0
fi

COMMON_DIR="${EXP_COMMON_DIR_BASE}/${LAYER_IDX}/"

echo "=== CAA Job ==="
echo "Model:  ${EXP_MODEL_ID}"
echo "Layer:  ${LAYER_IDX}"
echo "Vector: ${VECTOR_PATH}"
echo "Coeff:  ${EXP_COEFF}"
echo "==============="

# Build optional email args
EMAIL_ARGS=""
if [[ -n "${EXP_USER_EMAIL}" ]]; then
    EMAIL_ARGS="+hydra.launcher.additional_parameters.mail-type=ALL +hydra.launcher.additional_parameters.mail-user=${EXP_USER_EMAIL}"
fi

# Run main.py
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  "${EXP_PYTHON_BIN}" -u main.py -m \
    mode="${EXP_MODE}" \
    dataset="'${EXP_DATASETS}'" \
    model="${EXP_MODEL_ID}" \
    abstention_detector="${EXP_JUDGE}" \
    run_single_job_for_inference_and_judge="${EXP_SINGLE_JOB}" \
    common_dir="${COMMON_DIR}" \
    module.steering_vector_path="${VECTOR_PATH}" \
    module.steering_layer_idx="${LAYER_IDX}" \
    module.steering_coeff="${EXP_COEFF}" \
    ${EMAIL_ARGS}
