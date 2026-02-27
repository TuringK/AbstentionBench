# activate.sh

# Load per-user environment (paths, username, email)
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/env.sh" ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
else
    echo "Warning: env.sh not found. Copy env.sh.example to env.sh and fill in your values."
fi

module load GCC
module load CUDA/12.4

mamba activate abstention-bench