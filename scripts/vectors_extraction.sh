#!/bin/bash
#SBATCH --job-name=caa_extract
#SBATCH --output=scripts/logs/extract/%j/extract_%A_%a.out
#SBATCH --error=scripts/logs/extract/%j/extract_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=82G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=10-28 # layers to extract

source ./activate.sh

PYTHON_BIN="/mnt/parscratch/users/${USER_NAME}/private/mamba/envs/abstention-bench/bin/python"
DATA_PATH="/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/sample_pairs.csv"

OUTPUT_DIR_NAME="Qwen/Qwen2_5_1_5B_Instruct/${SLURM_JOB_ID}"
OUTPUT_PATH="/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/${OUTPUT_DIR_NAME}"

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# $SLURM_ARRAY_TASK_ID serves as the layer_idx
export PYTHONPATH=$PYTHONPATH:$(pwd)

"${PYTHON_BIN}" vectors_extraction/extract_caa_vectors.py \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}/vec_layer_${SLURM_ARRAY_TASK_ID}.pt" \
    --layer_idx ${SLURM_ARRAY_TASK_ID} \
    --use_system_prompt

echo "Done extracting layer ${SLURM_ARRAY_TASK_ID}"