#!/bin/bash

#SBATCH --job-name=eval_soft_prompt
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================
# Configuration
# ============================================================
PYTHON_BIN=/mnt/parscratch/users/acb20df/private/mamba/envs/abstention-bench/bin/python
ABSTENTION_BENCH_DIR=/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench
COMMON_DIR=/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench/PromptTuning

# Model and dataset configuration (use spaces, not commas)
MODELS=('qwen2_5_1_5B_soft_prompt') #"olmo_3_7b_soft_prompt") # "gemma_3_1b_soft_prompt" )
DATASET='glob(*,exclude=dummy)' 
JUDGE=contains_abstention_keyword #llm_judge_llama_3_1_8B_instruct

# ============================================================
# Setup
# ============================================================
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================"

mkdir -p logs

cd ${ABSTENTION_BENCH_DIR}

if [[ -z "$USER_EMAIL" ]]; then
  echo "Warning: USER_EMAIL is not set. Mail notifications will be disabled."
fi

# ============================================================
# Run Evaluation for each model
# ============================================================
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running AbstentionBench evaluation..."
    echo "Model: ${MODEL}"
    echo "Dataset: ${DATASET}"
    echo "Judge: ${JUDGE}"
    echo "============================================================"
    echo ""

    env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
      $PYTHON_BIN -u main.py \
          --multirun \
          mode=cluster \
          model="${MODEL}" \
          dataset="${DATASET}" \
          abstention_detector="${JUDGE}" \
          common_dir="${COMMON_DIR}" \
          run_single_job_for_inference_and_judge=True

    echo ""
    echo "Finished ${MODEL}: $(date)"
done

echo ""
echo "============================================================"
echo "All evaluations complete: $(date)"
echo "============================================================"