#!/bin/bash

#SBATCH --job-name=abstention_master
#SBATCH --output=scripts/logs/%j/%j.out
#SBATCH --error=scripts/logs/%j/%j.err

# Optional: if you want to set email and username in activate.sh
# Warning: mamba env activation will fail, we use python bin directly for that
source ./activate.sh

# Set vars
DATASETS='glob(*,exclude=dummy)'
JUDGE=llm_judge_llama_3_1_8B_instruct
SINGLE_JOB=False
COMMON_DIR_NAME=All_models_Llama8B_judge

# Change to match your path
PYTHON_BIN=/mnt/parscratch/users/${USER_NAME}/private/mamba/envs/abstention-bench/bin/python
COMMON_DIR=/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/${COMMON_DIR_NAME}

# Models to iterate
MODELS=(
  "gemma_3_1B"
  "qwen2_5_1_5B_instruct"
  "allenai_llama_3_1_tulu_3_1_8B"
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
      $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-type=ALL ) \
      $( [[ -n "$USER_EMAIL" ]] && echo +hydra.launcher.additional_parameters.mail-user="${USER_EMAIL}" )
done