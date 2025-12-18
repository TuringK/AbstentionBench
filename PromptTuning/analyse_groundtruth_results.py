import json
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse abstention results from model evaluations'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='Path to the results directory (e.g., .../Qwen2_5_1_5B_Instruct_Benchmark/results)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model for display in output (e.g., "Qwen 2.5 1.5B Instruct")'
    )
    return parser.parse_args() 


def analyse_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    positive_examples = 0  # should_abstain = True and is_abstention = True
    negative_examples = 0  # should_abstain = True and is_abstention = False
    
    for response in data.get('responses', []):
        prompt = response.get('prompt', {})
        should_abstain = prompt.get('should_abstain', False)
        is_abstention = response.get('is_abstention', False)
        
        if should_abstain:
            if is_abstention:
                positive_examples += 1
            else:
                negative_examples += 1
    
    return positive_examples, negative_examples

def main():
    args = parse_args()
    
    base_path = Path(args.results_path)
    
    if not base_path.exists():
        print(f"Error: Results path does not exist: {base_path}")
        return
    
    # Find all `GroundTruthAbstentionEvaluator.json` files
    results = {}
    
    for dataset_dir in sorted(base_path.iterdir()):
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            
            evaluator_files = list(dataset_dir.rglob("GroundTruthAbstentionEvaluator.json"))
            
            if evaluator_files:
                json_path = evaluator_files[0]
                try:
                    positive, negative = analyse_dataset(json_path)
                    results[dataset_name] = {
                        'positive': positive,
                        'negative': negative,
                        'total_should_abstain': positive + negative
                    }
                except Exception as e:
                    print(f"Error processing {dataset_name}: {e}")
    
    # Print results
    print("=" * 80)
    print(f"{args.model_name} - Abstention Analysis")
    print("=" * 80)
    print()
    
    total_positive = 0
    total_negative = 0
    
    # Calculate percentages for sorting
    dataset_list = []
    for dataset_name, stats in results.items():
        positive = stats['positive']
        negative = stats['negative']
        total = stats['total_should_abstain']
        
        if total > 0:
            positive_pct = (positive / total) * 100
        else:
            positive_pct = 0
        
        dataset_list.append((dataset_name, positive, negative, total, positive_pct))
    
    # Sort by percentage descending
    dataset_list.sort(key=lambda x: x[4], reverse=True)
    
    for dataset_name, positive, negative, total, positive_pct in dataset_list:
        
        print(f"- Dataset: {dataset_name}")
        print(f"Positive Examples (Correctly Abstained):\t{positive:4d} / {total} ({positive_pct:.1f}%)")
        print(f"Negative Examples (Failed to Abstain):\t{negative:4d} / {total}")
        print()
        
        total_positive += positive
        total_negative += negative
    
    # Overall statistics
    print("=" * 80)
    print(f"Overall Statistics")
    print("=" * 80)
    grand_total = total_positive + total_negative
    if grand_total > 0:
        overall_pct = (total_positive / grand_total) * 100
    else:
        overall_pct = 0
    
    print(f"Total Positive Examples (Correctly Abstained): {total_positive:4d} / {grand_total} ({overall_pct:.1f}%)")
    print(f"Total Negative Examples (Failed to Abstain):   {total_negative:4d} / {grand_total}")
    print()

if __name__ == "__main__":
    main()