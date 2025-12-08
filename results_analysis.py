import os
import argparse
import pandas as pd
from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process abstention benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single results directory
  python results_analysis.py --results-dir data/results --output results.csv

  # Steering sweep mode
  python results_analysis.py --steering-dir data/vectors --vector-indices 10 11 12 --output sweep.xlsx

  # Filter training data
  python results_analysis.py --results-dir data/results --filter-training --training-data data/sample_pairs.csv

  # Exclude specific datasets
  python results_analysis.py --results-dir data/results --exclude-datasets WorldSense MoralChoice
        """
    )
    
    # mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--results-dir",
        type=str,
        help="Path to a single results directory."
    )
    mode_group.add_argument(
        "--steering-dir",
        type=str,
        help="Base directory for steering sweep (contains subdirectories for each vector index)."
    )
    
    # steering-specific args
    parser.add_argument(
        "--vector-indices",
        type=int,
        nargs="+",
        help="List of vector indices to process (required for steering mode)."
    )
    
    # filtering options
    parser.add_argument(
        "--filter-training",
        action="store_true",
        help="Filter out questions present in training data."
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/sample_pairs.csv",
        help="Path to training data CSV file (default: data/sample_pairs.csv)."
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        nargs="+",
        help="List of dataset names to exclude from results."
    )
    
    # output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path. Use .csv or .xlsx extension to specify format."
    )
    parser.add_argument(
        "--save-per-vector",
        action="store_true",
        help="Save results for each vector index in a separate file (steering mode only)."
    )
    parser.add_argument(
        "--find-best",
        action="store_true",
        help="Find and print the best performing vector across all metrics (steering mode only)."
    )
    
    args = parser.parse_args()
    
    # validate steering mode requires vector indices
    if args.steering_dir and not args.vector_indices:
        parser.error("--vector-indices is required when using --steering-dir")
    
    return args


def normalise_text(text):
    """
    Normalises text by removing all whitespace characters.
    """
    if not isinstance(text, str):
        return str(text)
    return "".join(text.split())

def filter_training_data(df: pd.DataFrame, training_data_path: str) -> pd.DataFrame:
    """
    Filters out questions from the dataframe that are present in the training data.
    
    Args:
        df: DataFrame containing the results.
        training_data_path: Path to the CSV file containing training data.
        
    Returns:
        DataFrame with training data removed.
    """
    training_df = pd.read_csv(training_data_path)
    
    # normalise
    training_data_normalised = set(training_df['question'].apply(normalise_text))
    
    # filter
    df['prompt_question_normalised'] = df['prompt_question'].apply(normalise_text)
    initial_len = len(df)
    filtered_df = df[~df['prompt_question_normalised'].isin(training_data_normalised)].copy()
    filtered_df = filtered_df.drop(columns=['prompt_question_normalised'])
    
    print(f"Filtered out {initial_len - len(filtered_df)} rows. Remaining rows: {len(filtered_df)}")
    return filtered_df

def print_abstention_stats(df: pd.DataFrame):
    """
    Prints detailed abstention statistics for the given dataframe.
    """
    # filter for cases where abstention was required (Ground Truth = Should Abstain)
    should_abstain_mask = df['prompt_should_abstain'] == True
    subset = df[should_abstain_mask]
    
    total_should_abstain = len(subset)
    
    correctly_abstained = subset[subset['is_abstention'] == True].shape[0]
    failed_to_abstain = subset[subset['is_abstention'] == False].shape[0]
    percentage = (correctly_abstained / total_should_abstain) * 100
    
    print(f"Total Positive Examples (Correctly Abstained): {correctly_abstained} / {total_should_abstain} ({percentage:.1f}%)")
    print(f"Total Negative Examples (Failed to Abstain):   {failed_to_abstain} / {total_should_abstain}")

def process_results_dir(
    results_dir: str,
    filter_training: bool = False,
    training_data_path: str = "data/sample_pairs.csv",
    excluded_datasets: list[str] | None = None
) -> pd.DataFrame:
    """
    Processes results for a single results directory.
    
    Args:
        results_dir: Directory containing the results.
        filter_training: Whether to filter out training data.
        training_data_path: Path to the training data CSV file.
        excluded_datasets: List of datasets to exclude.
        
    Returns:
        DataFrame containing the results table.
    """
    print(f"\nProcessing results in {results_dir}...\n")

    # r = Results(
    #     base_results_dir=results_dir,
    #     filter_indeterminate_abstentions=False,
    #     sweep_dir=""
    # )
    
    # manually find result paths to bypass JobManager if it fails to find files with empty sweep_dir
    final_file = "GroundTruthAbstentionEvaluator.json"
    result_path_names = []
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            if final_file in files:
                rel_path = os.path.relpath(root, results_dir)
                result_path_names.append(rel_path)

    r = Results(
        base_results_dir=results_dir,
        filter_indeterminate_abstentions=False,
        sweep_dir="",
        result_path_names=result_path_names if result_path_names else None
    )

    if r.df.empty:
        print(f"No results found in {results_dir} (DataFrame is empty).")
        return pd.DataFrame()

    if filter_training:
        r.df = filter_training_data(r.df, training_data_path)
    
    if excluded_datasets:
        mask = pd.Series(False, index=r.df.index)
        if 'dataset_name' in r.df.columns:
            mask |= r.df['dataset_name'].isin(excluded_datasets)
        if 'dataset_name_formatted' in r.df.columns:
            mask |= r.df['dataset_name_formatted'].isin(excluded_datasets)
        
        if mask.any():
            initial_len_ds = len(r.df)
            r.df = r.df[~mask]
            print(f"Filtered out {initial_len_ds - len(r.df)} rows belonging to datasets: {excluded_datasets}")

    table = AbstentionF1ScoreTable(results=r)
    table_df = table.table_df
    
    print(f"\nResults table:")
    print(table_df.to_string())

    print("\nAbstention Statistics:")
    print_abstention_stats(r.df)
    print("-" * 40)
    
    return table_df

def process_steering_results(
    base_dir: str,
    vector_indices: list[int],
    filter_training: bool,
    training_data_path: str = "data/sample_pairs.csv",
    excluded_datasets: list[str] | None = None,
    output_path: str | None = None,
    save_per_vector: bool = False
) -> pd.DataFrame:
    """
    Processes results for each steering vector index.
    
    Args:
        base_dir: Base directory containing subdirectories for each vector index.
        vector_indices: List of vector indices to process.
        filter_training: Whether to filter out training data.
        training_data_path: Path to the training data CSV file.
        excluded_datasets: List of datasets to exclude.
        output_path: Output file path for saving results.
        save_per_vector: Whether to save each vector's results in a separate file.
        
    Returns:
        DataFrame containing aggregated results for all vectors.
    """
    all_results = []
    
    for idx in vector_indices:
        results_dir = os.path.join(base_dir, str(idx), "results")
        
        print(f"\nProcessing vector index {idx}...")
        table_df = process_results_dir(results_dir, filter_training, training_data_path, excluded_datasets)
        
        if not table_df.empty:
            # update model_name_formatted to include steered suffix
            if 'model_name_formatted' in table_df.columns:
                table_df['model_name_formatted'] = table_df['model_name_formatted'].apply(
                    lambda x: f"{x}_steered_{idx}"
                )
            
            table_df['vector_index'] = idx
            all_results.append(table_df)
            
            # Save per-vector file if requested
            if save_per_vector and output_path:
                vector_output_path = _add_suffix_to_path(output_path, f"_vector_{idx}")
                save_results(table_df, vector_output_path)
            
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def _add_suffix_to_path(path: str, suffix: str) -> str:
    """
    Adds a suffix to a file path before the extension.
    
    Args:
        path: Original file path.
        suffix: Suffix to add.
        
    Returns:
        Modified file path with suffix.
    """
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def find_best_vector_overall(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    aggregation: str = "mean"
) -> dict:
    """
    Finds the best vector across multiple metrics using average rank.
    
    Args:
        df: DataFrame with 'vector_index' column.
        metrics: List of metrics to consider (default: f1_score, precision, recall).
        aggregation: How to aggregate across datasets per vector ('mean', 'median', 'min').
        
    Returns:
        Dict with best vector and ranking details.
    """
    if df.empty or 'vector_index' not in df.columns:
        return {"error": "No vector results found"}
    
    if metrics is None:
        metrics = ["f1_score", "precision", "recall"]
    
    # filter to available metrics
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        available = [c for c in df.columns if c not in ['vector_index', 'model_name_formatted', 'dataset_name_formatted']]
        return {"error": f"No valid metrics found. Available columns: {available}"}
    
    # aggregate each metric per vector
    agg_func = {'mean': 'mean', 'median': 'median', 'min': 'min'}[aggregation]
    vector_scores = df.groupby('vector_index')[metrics].agg(agg_func)
    
    # rank each metric (higher is better)
    ranks = vector_scores.rank(ascending=False)
    
    # average rank across metrics
    avg_rank = ranks.mean(axis=1).sort_values()
    
    best_idx = avg_rank.index[0]
    
    return {
        "best_vector_index": int(best_idx),
        "average_rank": float(avg_rank[best_idx]),
        "metrics_used": metrics,
        "aggregation": aggregation,
        "scores": vector_scores.loc[best_idx].to_dict(),
        "all_rankings": {int(k): float(v) for k, v in avg_rank.to_dict().items()},
        "all_scores": vector_scores.to_dict()
    }


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the results DataFrame to a file (CSV or Excel).
    
    Args:
        df: DataFrame to save.
        output_path: Output file path. Extension determines format (.csv or .xlsx).
    """
    if df.empty:
        print("No results to save.")
        return
    
    if output_path.endswith(".xlsx"):
        try:
            df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")
        except ImportError:
            print("Error: Excel export requires 'openpyxl' package. Install with: pip install openpyxl")
    else:
        if not output_path.endswith(".csv"):
            output_path = output_path + ".csv"
            
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.steering_dir:
        # steering sweep mode
        results_df = process_steering_results(
            base_dir=args.steering_dir,
            vector_indices=args.vector_indices,
            filter_training=args.filter_training,
            training_data_path=args.training_data,
            excluded_datasets=args.exclude_datasets,
            output_path=args.output,
            save_per_vector=args.save_per_vector,
        )
    else:
        # single results directory mode
        results_df = process_results_dir(
            results_dir=args.results_dir,
            filter_training=args.filter_training,
            training_data_path=args.training_data,
            excluded_datasets=args.exclude_datasets,
        )
    
    if args.output:
        save_results(results_df, args.output)
    
    # find and print best vector if requested (steering mode only)
    if args.steering_dir and args.find_best:
        best = find_best_vector_overall(results_df)
        if "error" in best:
            print(f"\nError finding best vector: {best['error']}")
        else:
            print("\n\n" + "=" * 50)
            print("BEST VECTOR ANALYSIS")
            print("=" * 50)
            print(f"Best vector index: {best['best_vector_index']}")
            print(f"Average rank: {best['average_rank']:.2f}")
            print(f"Metrics used: {', '.join(best['metrics_used'])}")
            print(f"Aggregation: {best['aggregation']}")
            print(f"\nScores for best vector ({best['best_vector_index']}):")
            
            for metric, score in best['scores'].items():
                print(f"  {metric}: {score:.4f}")
                
            print(f"\nAll vector rankings (lower is better):")
            
            for vec_idx, rank in sorted(best['all_rankings'].items(), key=lambda x: x[1]):
                marker = " <-- BEST" if vec_idx == best['best_vector_index'] else ""
                print(f"  Vector {vec_idx}: rank {rank:.2f}{marker}")
                
            print("=" * 50)