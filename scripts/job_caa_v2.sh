#!/bin/bash

#SBATCH --job-name=abstention_caa
#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err
#SBATCH --partition=gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=82G
#SBATCH --time=04:00:00

module load GCC
module load CUDA/12.4

export VLLM_USE_V1=0

# Ensure args are set
if [[ -z "$TARGET_MODEL_ID" ]] || [[ -z "$TARGET_MODEL_DIR" ]]; then
  echo "Error: TARGET_MODEL_ID or TARGET_MODEL_DIR variables not set."
  exit 1
fi

DATASETS='glob(*,exclude=dummy)'
# DATASETS=rulebreakers
JUDGE=contains_abstention_keyword
SINGLE_JOB=True

STEERING_VECTOR_IDX="${SLURM_ARRAY_TASK_ID}"
STEERING_VECTOR_COEFF=1.0

# paths
STEERING_VECTOR_PATH="/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/vectors/${TARGET_MODEL_DIR}/vec_layer_${STEERING_VECTOR_IDX}.pt"
COMMON_DIR_NAME="${TARGET_MODEL_DIR}_Keywords_judge_CAA_coeff_1_0_rulebreakers/${STEERING_VECTOR_IDX}/"
COMMON_DIR="/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/${COMMON_DIR_NAME}"
PYTHON_BIN=/mnt/parscratch/users/${USER_NAME}/private/mamba/envs/abstention-bench/bin/python

if [[ ! -f "$STEERING_VECTOR_PATH" ]]; then
    echo "Vector file not found at: $STEERING_VECTOR_PATH"
    echo "Skipping task ${STEERING_VECTOR_IDX}."
    exit 0
fi

echo "Running Job for Model: $TARGET_MODEL_ID | Layer: $STEERING_VECTOR_IDX"
echo "Vector Path: $STEERING_VECTOR_PATH"

env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  "$PYTHON_BIN" -u main.py -m \
    mode=local \
    dataset="${DATASETS}" \
    model="${TARGET_MODEL_ID}" \
    abstention_detector="${JUDGE}" \
    run_single_job_for_inference_and_judge="${SINGLE_JOB}" \
    common_dir="${COMMON_DIR}" \
    module.steering_vector_path="${STEERING_VECTOR_PATH}" \
    module.steering_layer_idx="${STEERING_VECTOR_IDX}" \
    module.steering_coeff="${STEERING_VECTOR_COEFF}"
    # $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-type=ALL ) \
    # $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-user="${USER_EMAIL}" )