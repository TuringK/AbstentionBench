#!/bin/bash

source ./activate.sh

VECTOR_BASE_PATH="/mnt/parscratch/users/${USER_NAME}/private/projects/AbstentionBench/data/vectors"

declare -A MODEL_DIRS

# MODEL_DIRS["qwen2_5_1_5B_instruct"]="Qwen2_5_1_5B_Instruct"
# MODEL_DIRS["qwen2_5_0_5B_instruct"]="Qwen2_5_0_5B_Instruct"
# MODEL_DIRS["qwen2_5_3B_instruct"]="Qwen2_5_3B_Instruct"
# MODEL_DIRS["qwen2_5_7B_instruct"]="Qwen2_5_7B_Instruct"
# MODEL_DIRS["gemma_3_1B_instruct"]="Gemma3_1B_Instruct" 
# MODEL_DIRS["olmo_3_7B_instruct"]="Olmo3_7B_Instruct"
MODEL_DIRS["allenai_llama_3_1_tulu_3_1_8B"]="Llama3_1_Tulu_3_1_8B"

# loop through the keys of the map
for model_id in "${!MODEL_DIRS[@]}"; do
    dir_name="${MODEL_DIRS[$model_id]}"

    echo "Processing Model ID: $model_id"
    echo "Looking in Directory: $dir_name"
    
    FULL_VEC_DIR="${VECTOR_BASE_PATH}/${dir_name}"

    # determine min and max layers
    # look for vec_layer_X.pt inside the directory
    layers=$(ls "${FULL_VEC_DIR}"/vec_layer_*.pt 2>/dev/null | grep -oP '(?<=vec_layer_)\d+(?=.pt)' | sort -n)

    min_layer=$(echo "$layers" | head -n1)
    max_layer=$(echo "$layers" | tail -n1)

    echo "Found vectors ranging from layer $min_layer to $max_layer"
    
    JOB_NAME="caa_${model_id}"
    
    sbatch \
        --job-name="${JOB_NAME}" \
        --array=${min_layer}-${max_layer} \
        --export=ALL,TARGET_MODEL_ID="${model_id}",TARGET_MODEL_DIR="${dir_name}" \
        scripts/job_caa.sh
        
    echo "Submitted batch job for $model_id"
    echo ""
done