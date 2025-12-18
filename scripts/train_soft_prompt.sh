#!/bin/bash

#SBATCH --job-name=train_soft_prompt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================
# Configuration - Edit these paths as needed
# ============================================================
BASE_DIR=/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench/PromptTuning
PYTHON_BIN=/mnt/parscratch/users/acb20df/private/mamba/envs/abstention-bench/bin/python

# Model to train
MODEL_NAMES=("Qwen/Qwen2.5-1.5B-Instruct") #"allenai/Olmo-3-7B-Instruct" "google/gemma-3-1b-it")
OUTPUT_NAMES=("qwen2_5_1_5B_soft_prompt") #"olmo_3_7b_soft_prompt" "gemma_3_1b_soft_prompt")

# Training hyperparameters
NUM_VIRTUAL_TOKENS=50
NUM_EPOCHS=3
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=0.3
MAX_SEQ_LENGTH=512
SEED=42

# Data paths
TRAIN_CSV="${BASE_DIR}/data/sample_pairs.csv"

# ============================================================
# Setup
# ============================================================
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================"

# Create log directory
mkdir -p logs

# Print GPU info
nvidia-smi

# Loop over both arrays using index
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    OUTPUT_NAME="${OUTPUT_NAMES[$i]}"
    OUTPUT_DIR="${BASE_DIR}/trained_models/${OUTPUT_NAME}-${NUM_VIRTUAL_TOKENS}tok-FT"
    
    # ============================================================
    # Run Training
    # ============================================================
    echo ""
    echo "Starting training..."
    echo "Model: ${MODEL_NAME}"
    echo "Output: ${OUTPUT_DIR}"
    echo ""

    $PYTHON_BIN PromptTuning/train_soft_prompt.py \
        --model_name "${MODEL_NAME}" \
        --train_csv "${TRAIN_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_virtual_tokens ${NUM_VIRTUAL_TOKENS} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --seed ${SEED}

    echo ""
    echo "============================================================"
    echo "Finished training ${MODEL_NAME}: $(date)"
    echo "============================================================"
done

echo ""
echo "All models trained: $(date)"