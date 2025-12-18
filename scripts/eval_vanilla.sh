#!/bin/bash

#SBATCH --job-name=eval_vanilla_gemma
#SBATCH --output=logs/eval_vanilla_%j.out
#SBATCH --error=logs/eval_vanilla_%j.err
#SBATCH --time=02:00:00
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

# Model and dataset configuration
MODEL=qwen2_5_1_5B_vanilla
DATASET=balanced_eval
JUDGE=contains_abstention_keywords

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

#nvidia-smi

# ============================================================
# Run Evaluation
# ============================================================
echo ""
echo "Running AbstentionBench evaluation (VANILLA MODEL)..."
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Judge: ${JUDGE}"
echo ""

$PYTHON_BIN -u main.py \
    mode=local \
    model="${MODEL}" \
    dataset="${DATASET}" \
    abstention_detector="${JUDGE}" \
    common_dir="${COMMON_DIR}" \
    run_single_job_for_inference_and_judge=True

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"