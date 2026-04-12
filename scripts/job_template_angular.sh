#!/bin/bash

#SBATCH --output=scripts/logs/run_angular/%x_%A.out
#SBATCH --error=scripts/logs/run_angular/%x_%A.err

# SLURM job template for Angular steering benchmark runs (mirrors scripts/job_template.sh for CAA).
# Environment variables ANG_EXP_* are set by angular/run_experiment.py.

module load GCC
module load CUDA/12.4

export HYDRA_FULL_ERROR=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Default: fail fast if prefill/decode cannot be inferred (safe for sweeps)
ANG_EXP_PROMPT_ONLY_STRICT="${ANG_EXP_PROMPT_ONLY_STRICT:-true}"

echo "=== Angular benchmark ==="
echo "Model:   ${ANG_EXP_MODEL_ID}"
echo "Config:  ${ANG_EXP_STEERING_CONFIG}"
echo "Degree:  ${ANG_EXP_DEGREE}"
echo "Adapt:   ${ANG_EXP_ADAPTIVE_MODE}"
echo "Prompt-only: ${ANG_EXP_PROMPT_ONLY}"
echo "Prompt-only strict: ${ANG_EXP_PROMPT_ONLY_STRICT}"
echo "========================="

COMMON_DIR="${ANG_EXP_COMMON_DIR_BASE}/"

env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_MEM_PER_GPU \
  "${ANG_EXP_PYTHON_BIN}" -u main.py -m \
    mode="${ANG_EXP_MODE}" \
    dataset="${ANG_EXP_DATASETS}" \
    model="${ANG_EXP_MODEL_ID}" \
    abstention_detector="${ANG_EXP_JUDGE}" \
    run_single_job_for_inference_and_judge="${ANG_EXP_SINGLE_JOB}" \
    common_dir="${COMMON_DIR}" \
    module.angular_steering_config_path="${ANG_EXP_STEERING_CONFIG}" \
    module.angular_degree="${ANG_EXP_DEGREE}" \
    module.angular_adaptive_mode="${ANG_EXP_ADAPTIVE_MODE}" \
    module.angular_prompt_only="${ANG_EXP_PROMPT_ONLY}" \
    module.angular_prompt_only_strict="${ANG_EXP_PROMPT_ONLY_STRICT}" \
    ${ANG_EXP_EMAIL_ARGS}
