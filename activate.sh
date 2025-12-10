#!/bin/bash
# activate.sh

module load GCC
module load CUDA/12.4

# Note: We don't activate mamba here because the script uses the Python binary directly
# This avoids issues with mamba activation in non-interactive SLURM contexts

export USER_EMAIL="your.email@sheffield.ac.uk"
export USER_NAME="your_username"

# ALSO EXPORT HF CACHE IF YOU ARE USING GATED MODELS SUCH AS GEMMA
export HF_HOME="/mnt/parscratch/users/${USER}/private/.tmp/huggingface_cache"
export HF_DATASETS_CACHE="/mnt/parscratch/users/${USER}/private/.tmp/huggingface_cache/datasets"
