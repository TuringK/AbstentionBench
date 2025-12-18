"""
Analyse abstention results from a single evaluation JSON file.

Usage:
    python analyse_results.py --json-path /path/to/GroundTruthAbstentionEvaluator.json
    python analyse_results.py --json-path /path/to/GroundTruthAbstentionEvaluator.json --model-name "Gemma 3 1B Soft Prompt"
"""

import json
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse abstention results from a single evaluation JSON file'
    )
    parser.add_argument(
        '--json-path',
        type=str,
        required=True,
        help='Path to the GroundTruthAbstentionEvaluator.json file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default="Model",
        help='Name of the model for display in output (default: "Model")'
    )
    return parser.parse_args()


def analyse_results(json_path):
    """Analyse a single evaluation JSON file and return detailed metrics."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Confusion matrix components
    true_positive = 0   # should_abstain=True, is_abstention=True (correct abstention)
    false_negative = 0  # should_abstain=True, is_abstention=False (missed abstention)
    false_positive = 0  # should_abstain=False, is_abstention=True (unnecessary abstention)
    true_negative = 0   # should_abstain=False, is_abstention=False (correct answer)
    
    # For detailed breakdown
    results_by_dataset = {}
    
    for response in data.get('responses', []):
        prompt = response.get('prompt', {})
        should_abstain = prompt.get('should_abstain', False)
        is_abstention = response.get('is_abstention', False)
        
        # Get dataset name from metadata if available
        metadata = prompt.get('metadata', {})
        dataset_name = metadata.get('dataset', 'unknown')
        
        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        
        if should_abstain and is_abstention:
            true_positive += 1
            results_by_dataset[dataset_name]['tp'] += 1
        elif should_abstain and not is_abstention:
            false_negative += 1
            results_by_dataset[dataset_name]['fn'] += 1
        elif not should_abstain and is_abstention:
            false_positive += 1
            results_by_dataset[dataset_name]['fp'] += 1
        else:  # not should_abstain and not is_abstention
            true_negative += 1
            results_by_dataset[dataset_name]['tn'] += 1
    
    return {
        'tp': true_positive,
        'fn': false_negative,
        'fp': false_positive,
        'tn': true_negative,
        'by_dataset': results_by_dataset
    }


def compute_metrics(tp, fn, fp, tn):
    """Compute precision, recall, F1, and accuracy."""
    total = tp + fn + fp + tn
    
    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Precision: of all predicted abstentions, how many were correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: of all cases that should abstain, how many did we catch?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = parse_args()
    
    json_path = Path(args.json_path)
    
    if not json_path.exists():
        print(f"Error: JSON file does not exist: {json_path}")
        return
    
    # Analyse results
    results = analyse_results(json_path)
    
    tp = results['tp']
    fn = results['fn']
    fp = results['fp']
    tn = results['tn']
    
    metrics = compute_metrics(tp, fn, fp, tn)
    
    # Print results
    print("=" * 70)
    print(f"{args.model_name} - Abstention Analysis")
    print("=" * 70)
    print()
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print("-" * 50)
    print(f"                      Predicted")
    print(f"                   Abstain    Answer")
    print(f"Actual  Should      {tp:4d}      {fn:4d}    (Total: {tp+fn})")
    print(f"        Abstain")
    print(f"        Should      {fp:4d}      {tn:4d}    (Total: {fp+tn})")
    print(f"        Answer")
    print()
    
    # Metrics
    print("Metrics:")
    print("-" * 50)
    print(f"Accuracy:  {metrics['accuracy']*100:6.2f}%  (Overall correctness)")
    print(f"Precision: {metrics['precision']*100:6.2f}%  (When it abstains, is it right?)")
    print(f"Recall:    {metrics['recall']*100:6.2f}%  (Does it catch cases needing abstention?)")
    print(f"F1 Score:  {metrics['f1']*100:6.2f}%  (Harmonic mean of precision & recall)")
    print()
    
    # Breakdown by category
    print("Breakdown:")
    print("-" * 50)
    print(f"True Positives:  {tp:4d}  (Correctly abstained)")
    print(f"False Negatives: {fn:4d}  (Should have abstained but didn't)")
    print(f"False Positives: {fp:4d}  (Abstained unnecessarily)")
    print(f"True Negatives:  {tn:4d}  (Correctly answered)")
    print(f"Total:           {tp+fn+fp+tn:4d}")
    print()
    
    # Per-dataset breakdown if available
    if len(results['by_dataset']) > 1 or 'unknown' not in results['by_dataset']:
        print("Per-Dataset Breakdown:")
        print("-" * 50)
        for dataset_name, stats in sorted(results['by_dataset'].items()):
            d_tp, d_fn, d_fp, d_tn = stats['tp'], stats['fn'], stats['fp'], stats['tn']
            d_metrics = compute_metrics(d_tp, d_fn, d_fp, d_tn)
            d_total = d_tp + d_fn + d_fp + d_tn
            print(f"\n{dataset_name} ({d_total} samples):")
            print(f"  Accuracy: {d_metrics['accuracy']*100:.1f}% | "
                  f"Precision: {d_metrics['precision']*100:.1f}% | "
                  f"Recall: {d_metrics['recall']*100:.1f}% | "
                  f"F1: {d_metrics['f1']*100:.1f}%")
            print(f"  TP: {d_tp} | FN: {d_fn} | FP: {d_fp} | TN: {d_tn}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()