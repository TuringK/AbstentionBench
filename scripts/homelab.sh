#!/bin/bash

PROJECT_ROOT="/workspace/AbstentionBench"
PYTHON_BIN="/workspace/mamba/envs/abstention-bench/bin/python"
VECTOR_BASE_PATH="${PROJECT_ROOT}/data/vectors"
LOG_DIR="scripts/logs_local"

mkdir -p "$LOG_DIR"

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export DATASETS=rulebreakers
export JUDGE=contains_abstention_keyword
export SINGLE_JOB=True

declare -A MODEL_DIRS
# MODEL_DIRS["qwen2_5_1_5B_instruct"]="Qwen2_5_1_5B_Instruct"
# MODEL_DIRS["qwen2_5_0_5B_instruct"]="Qwen2_5_0_5B_Instruct"
# MODEL_DIRS["qwen2_5_3B_instruct"]="Qwen2_5_3B_Instruct"
# MODEL_DIRS["qwen2_5_7B_instruct"]="Qwen2_5_7B_Instruct"
# MODEL_DIRS["gemma_3_1B_instruct"]="Gemma3_1B_Instruct"
MODEL_DIRS["allenai_llama_3_1_tulu_3_1_8B"]="Llama3_1_Tulu_3_1_8B"

declare -A MODEL_LAYERS
# MODEL_LAYERS["qwen2_5_1_5B_instruct"]=14
# MODEL_LAYERS["qwen2_5_0_5B_instruct"]=23
# MODEL_LAYERS["qwen2_5_3B_instruct"]=23
# MODEL_LAYERS["qwen2_5_7B_instruct"]=15
# MODEL_LAYERS["gemma_3_1B_instruct"]=12
MODEL_LAYERS["allenai_llama_3_1_tulu_3_1_8B"]=17

for model_id in "${!MODEL_DIRS[@]}"; do
    dir_name="${MODEL_DIRS[$model_id]}"
    target_layer="${MODEL_LAYERS[$model_id]}"

    TARGET_MODEL_ID="$model_id"
    TARGET_MODEL_DIR="$dir_name"
    STEERING_VECTOR_IDX="$target_layer"
    STEERING_VECTOR_COEFF=1.0

    STEERING_VECTOR_PATH="${VECTOR_BASE_PATH}/${TARGET_MODEL_DIR}/vec_layer_${STEERING_VECTOR_IDX}.pt"
    COMMON_DIR_NAME="${TARGET_MODEL_DIR}_Keywords_judge_CAA_coeff_1_0_rulebreakers/${STEERING_VECTOR_IDX}/"
    COMMON_DIR="${PROJECT_ROOT}/data/${COMMON_DIR_NAME}"

    echo "=================================================="
    echo "Processing Model: $TARGET_MODEL_ID"
    echo "Steering Layer: $STEERING_VECTOR_IDX"
    echo "Vector Path: $STEERING_VECTOR_PATH"
    echo "=================================================="

    if [[ ! -f "$STEERING_VECTOR_PATH" ]]; then
        echo "Error: Vector file not found at $STEERING_VECTOR_PATH"
        echo "Skipping..."
        continue
    fi

    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="${LOG_DIR}/caa_${TARGET_MODEL_ID}_layer_${STEERING_VECTOR_IDX}_${TIMESTAMP}.log"

    "$PYTHON_BIN" -u main.py -m \
        mode=local \
        dataset="${DATASETS}" \
        model="${TARGET_MODEL_ID}" \
        abstention_detector="${JUDGE}" \
        run_single_job_for_inference_and_judge="${SINGLE_JOB}" \
        common_dir="${COMMON_DIR}" \
        module.steering_vector_path="${STEERING_VECTOR_PATH}" \
        module.steering_layer_idx="${STEERING_VECTOR_IDX}" \
        module.steering_coeff="${STEERING_VECTOR_COEFF}" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "Finished $TARGET_MODEL_ID. Log saved to $LOG_FILE"
    echo "--------------------------------------------------"
    
    # pause to let GPU cool or memory clear if needed
    sleep 2 
done

echo "All jobs completed."