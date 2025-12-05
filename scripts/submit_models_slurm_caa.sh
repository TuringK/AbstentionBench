#!/bin/bash

#SBATCH --job-name=abstention_master_caa
#SBATCH --output=scripts/logs/%j/%j.out
#SBATCH --error=scripts/logs/%j/%j.err

# Optional: if you want to set email and username in activate.sh
# Warning: mamba env activation will fail, we use python bin directly for that
source ./activate.sh

# CAA vars
STEERING_VECTOR_PATH=/mnt/parscratch/users/acb22av/private/projects/AbstentionBench/data/Qwen/Qwen2_5_1_5B_Instruct/8667158/vec_layer_14.pt
STEERING_VECTOR_IDX=14
STEERING_VECTOR_COEFF=1.0

# Set vars
DATASETS='glob(*,exclude=dummy)'
JUDGE=contains_abstention_keyword
SINGLE_JOB=True
COMMON_DIR_NAME="Qwen2_5_1_5B_Instruct_Keywards_judge_CAA_idx_${STEERING_VECTOR_IDX}"

# Change to match your path
PYTHON_BIN=/mnt/parscratch/users/${USER_NAME}/private/mamba/envs/abstention-bench/bin/python
COMMON_DIR=/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/${COMMON_DIR_NAME}

# Models to iterate
MODELS=(
#   "gemma_3_1B"
  "qwen2_5_1_5B_instruct"
#   "allenai_llama_3_1_tulu_3_1_8B"
)

# Check if email was set in activate.sh
if [[ -z "$USER_EMAIL" ]]; then
  echo "Warning: USER_EMAIL is not set. Mail notifications will be disabled."
fi

for model in "${MODELS[@]}"; do

  # We need to unset children jobs' vars to not run into memory config conflict
  # e.g. "srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE 
  #       are mutually exclusive."

  env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
    "$PYTHON_BIN" -u main.py -m \
      mode=cluster \
      dataset="${DATASETS}" \
      model="${model}" \
      abstention_detector="${JUDGE}" \
      run_single_job_for_inference_and_judge="${SINGLE_JOB}" \
      common_dir="${COMMON_DIR}" \
      module.steering_vector_path="${STEERING_VECTOR_PATH}" \
      module.steering_layer_idx="${STEERING_VECTOR_IDX}" \
      module.steering_coeff="${STEERING_VECTOR_COEFF}"
      $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-type=ALL ) \
      $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-user="${USER_EMAIL}" )
done