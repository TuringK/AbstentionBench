#!/bin/bash
# activate.sh

module load GCC
module load CUDA/12.4

# Note: We don't activate mamba here because the script uses the Python binary directly
# This avoids issues with mamba activation in non-interactive SLURM contexts

export USER_EMAIL="your.email@sheffield.ac.uk"
export USER_NAME="your_username"