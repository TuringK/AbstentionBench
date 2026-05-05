#!/bin/bash

#SBATCH --output=scripts/logs/run_caa/%x_%A_%a.out
#SBATCH --error=scripts/logs/run_caa/%x_%A_%a.err

# Generic SLURM job template for CAA experiments.
# All experiment-specific values are passed via environment
# variables (CAA_EXP_*) set by run_experiment.py.

module load GCC
module load CUDA/12.4

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
# Avoid PyTorch Inductor -> Triton -> host gcc when profiling Gemma (and similar) in vLLM:
# batch nodes with `module load GCC` can hit Assembler/Illegal-instruction or linker failures.
export TORCH_COMPILE_DISABLE=1

# Resolve paths for this array task
LAYER_IDX="${SLURM_ARRAY_TASK_ID}"
VECTOR_PATH="${CAA_EXP_VECTOR_DIR}/vec_layer_${LAYER_IDX}.pt"

if [[ ! -f "$VECTOR_PATH" ]]; then
    echo "Vector file not found: $VECTOR_PATH : skipping layer ${LAYER_IDX}"
    exit 0
fi

COMMON_DIR="${CAA_EXP_COMMON_DIR_BASE}/${LAYER_IDX}/"

echo "=== CAA Job ==="
echo "Model:  ${CAA_EXP_MODEL_ID}"
echo "Layer:  ${LAYER_IDX}"
echo "Vector: ${VECTOR_PATH}"
echo "Coeff:  ${CAA_EXP_COEFF}"
echo "==============="

# Build optional email args (only for cluster mode; BasicLauncher doesn't support additional_parameters)
EMAIL_ARGS=""
if [[ -n "${CAA_EXP_USER_EMAIL}" ]] && [[ "${CAA_EXP_MODE}" != "local" ]]; then
    EMAIL_ARGS="+hydra.launcher.additional_parameters.mail-type=ALL +hydra.launcher.additional_parameters.mail-user=${CAA_EXP_USER_EMAIL}"
fi

# Run main.py
env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  "${CAA_EXP_PYTHON_BIN}" -u main.py -m \
    mode="${CAA_EXP_MODE}" \
    dataset="${CAA_EXP_DATASETS}" \
    model="${CAA_EXP_MODEL_ID}" \
    abstention_detector="${CAA_EXP_JUDGE}" \
    run_single_job_for_inference_and_judge="${CAA_EXP_SINGLE_JOB}" \
    common_dir="${COMMON_DIR}" \
    module.steering_vector_path="${VECTOR_PATH}" \
    module.steering_layer_idx="${LAYER_IDX}" \
    module.steering_coeff="${CAA_EXP_COEFF}" \
    ${EMAIL_ARGS}
