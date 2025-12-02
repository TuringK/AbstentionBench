#!/bin/bash
# Script to run all datasets sequentially on the current GPU node

# List of datasets (excluding dummy)
datasets=("alcuna" "bbq" "big_bench_disambiguate" "big_bench_known_unknowns" "coconot" "falseqa" "freshqa" "gpqa" "gsm8k" "kuq" "mediq" "mmlu_history" "mmlu_math" "moralchoice" "musique" "qaqa" "qasper" "self_aware" "situated_qa" "squad2" "umwp" "worldsense")
models=("gemma_3_1b" "qwen2_5_1_5B_instruct" "allenai_llama_3_1_tulu_3_1_8B")

# Check if models exists
for model in "${models[@]}"; do
    if [ ! -f "$(pwd)/configs/model/${model}.yaml" ]; then
        echo "ERROR: ${model} doesn't exist"
    else
        echo "Found model ${model}"
    fi
done

# Run each model and each dataset sequentially
model_iteration=1

for model in "${models[@]}"; do
    dataset_iteration=1
    for dataset in "${datasets[@]}"; do
        echo "======================================"
        echo "Model: $model ($model_iteration/${#models[@]})" 
        echo "Dataset: $dataset ($dataset_iteration/${#datasets[@]})"
        echo "======================================"

        python main.py mode=local dataset=${dataset} model=${model} common_dir=TestAbstention abstention_detector=contains_abstention_keyword run_single_job_for_inference_and_judge=True

        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: Dataset $dataset failed with exit code $exit_code"
            echo "!! Continuing to next dataset !!"
        else
            echo "SUCCESS: Dataset $dataset completed"
        fi
        echo ""
        dataset_iteration=$((dataset_iteration + 1))
        
    done
    model_iteration=$((model_iteration + 1))
done

echo "Completed running ${models} on all datasets"