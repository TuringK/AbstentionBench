import os
import argparse
import pandas as pd
import contextlib
import json
import sys

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    # NOTE: these imports can pull in heavyweight optional dependencies (e.g. vllm).
    # keep them lazy so utilities like training-data filtering can be unit-tested
    # and used in lightweight environments.
    Results = None
    AbstentionF1ScoreTable = None


def _lazy_import_analysis_components():
    global Results, AbstentionF1ScoreTable
    if Results is None:
        from analysis.load_results import Results as _Results

        Results = _Results
    if AbstentionF1ScoreTable is None:
        from analysis.tables import AbstentionF1ScoreTable as _Table

        AbstentionF1ScoreTable = _Table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process abstention benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single results directory
  python results_analysis.py --results-dir data/results --output results.csv

  # Steering sweep mode (auto-detect vectors in steering-dir)
  python results_analysis.py --steering-dir data/vectors --output sweep.xlsx

  # Steering sweep mode (with ranges and individual vectors)
  python results_analysis.py --steering-dir data/vectors --vector-indices 1 2 5-10 --output sweep.xlsx

  # Filter training overlap CSV or JSON array training file
  python results_analysis.py --results-dir data/results --filter-training --training-data data/sample_pairs.csv
  python results_analysis.py --results-dir data/results --filter-training --training-data data/abstention_training_dataset.json

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
        type=str,
        nargs="*",
        help="List of vector indices or ranges (e.g., 1 2 5-10) to process. If not provided, will auto-detect from steering-dir."
    )
    
    # filtering options
    parser.add_argument(
        "--filter-training",
        action="store_true",
        help=(
            "Drop benchmark rows whose prompt matches training text after whitespace "
            "normalisation. JSON training counts every row with a non-empty question "
            "field. Labels such as should_abstain do not change overlap."
        ),
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/sample_pairs.csv",
        help=(
            "Training file path CSV with a question column or JSON array (.json / .jsonl). "
            "Default data/sample_pairs.csv."
        ),
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        nargs="+",
        help="List of dataset names to exclude from results."
    )
    parser.add_argument(
        "--include-datasets",
        type=str,
        nargs="+",
        help="List of dataset names to specifically include (ignores all others)."
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
        type=str,
        nargs="*",
        default=None,
        help="Find and print the best performing vector across a list of metrics (steering mode only)."
             " If provided without metrics (just --find-best) the script will use the default"
             " metrics: ['f1_score', 'precision', 'recall'].",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information during filtering."
    )
    
    args = parser.parse_args()
    
    return args

def parse_vector_indices(indices_args, steering_dir=None):
    """
    Parses a list of vector index strings, supporting ranges like '1-5'.
    If no indices are provided, attempts to auto-detect integer subdirectories in steering_dir.
    """
    indices = set()
    if not indices_args:
        if steering_dir and os.path.exists(steering_dir):
            for d in os.listdir(steering_dir):
                if os.path.isdir(os.path.join(steering_dir, d)) and d.isdigit():
                    indices.add(int(d))
            if not indices:
                print(f"Warning: No valid vector directories found in {steering_dir}")
        return sorted(list(indices))

    for arg in indices_args:
        if isinstance(arg, int):
            indices.add(arg)
        elif isinstance(arg, str):
            if '-' in arg:
                parts = arg.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start, end = int(parts[0]), int(parts[1])
                    indices.update(range(start, end + 1))
                else:
                    raise ValueError(f"Invalid range format: {arg}")
            elif arg.isdigit():
                indices.add(int(arg))
            else:
                raise ValueError(f"Invalid vector index: {arg}")
    return sorted(list(indices))


def normalise_text(text):
    """Collapse internal whitespace for stable overlap checks between prompts.

    Values that are not strings become empty strings so missing cells never match
    string forms such as None or nan from coercion bugs.
    """
    if not isinstance(text, str):
        return ""
    return "".join(text.split())


def _load_training_questions(training_data_path: str) -> tuple[list[str], str]:
    """Load question strings from CSV pairs or from the JSON training array format.

    CSV files must include a question column. Missing cells are dropped. Other columns
    are ignored.

    JSON uses json.load on the whole file. The root must be a list of objects. Objects
    without a question key are skipped. Whitespace-only strings are skipped.

    Extensions .json and .jsonl both use this path and match the extractor convention
    for a single JSON array document.

    Returns:
        Tuple of question list and format tag csv or json.
    """
    _, ext = os.path.splitext(training_data_path)
    ext = ext.lower()

    if ext in {".json", ".jsonl"}:
        with open(training_data_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Training JSON must be a list of examples, got {type(data)}"
            )
        questions = [d.get("question") for d in data if isinstance(d, dict)]
        questions = [q for q in questions if isinstance(q, str) and q.strip()]
        return questions, "json"

    # default: CSV
    training_df = pd.read_csv(training_data_path)
    if "question" not in training_df.columns:
        raise ValueError(
            f"Training CSV must contain a 'question' column. Columns: {list(training_df.columns)}"
        )
    questions = training_df["question"].dropna().astype(str).tolist()
    return questions, "csv"


def _get_results_question_column(df: pd.DataFrame) -> str:
    """Return the first present prompt column name in a fixed preference order."""
    for candidate in ["prompt_question", "question", "prompt", "prompt_text"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Results dataframe does not contain a recognisable question column. "
        f"Tried: prompt_question/question/prompt/prompt_text. Columns: {list(df.columns)}"
    )

def filter_training_data(df: pd.DataFrame, training_data_path: str, debug: bool = False) -> pd.DataFrame:
    """Drop rows whose prompt matches any normalised training question string.

    Prompt column detection follows prompt_question then question then prompt then
    prompt_text.

    Matching uses normalise_text on both sides so spacing differences alone do not miss
    overlap.

    Rows whose prompt cell is missing or not a string normalise to empty and stay unless
    empty appears among normalised training strings.

    Args:
        df: Results table before aggregation output.
        training_data_path: CSV or JSON array training file same formats as --training-data.
        debug: Print row counts and duplicate hints when True.

    Returns:
        Copy of df without overlapping prompt rows.
    """
    questions, fmt = _load_training_questions(training_data_path)
    training_questions_raw = pd.Series(questions, dtype="string")

    print(
        f"Training data: {training_data_path} ({fmt}), "
        f"{len(questions)} rows, {training_questions_raw.nunique()} unique questions"
    )

    # normalise
    training_data_normalised = set(training_questions_raw.apply(normalise_text))
    print(
        f"Training data after normalisation: {len(training_data_normalised)} unique normalised questions"
    )

    # check if normalisation reduced unique count (indicates collisions)
    if len(training_data_normalised) < training_questions_raw.nunique():
        print(
            "WARNING: Normalisation reduced unique questions by "
            f"{training_questions_raw.nunique() - len(training_data_normalised)}"
        )

    # filter
    results_q_col = _get_results_question_column(df)
    df["prompt_question_normalised"] = df[results_q_col].apply(normalise_text)
    
    # debug: check for duplicates in benchmark data
    if debug:
        benchmark_questions = df['prompt_question_normalised'].value_counts()
        duplicates = benchmark_questions[benchmark_questions > 1]
        if not duplicates.empty:
            print(f"\nDuplicate questions in benchmark data:")
            print(f"  Total unique questions: {len(benchmark_questions)}")
            print(f"  Questions appearing >1 time: {len(duplicates)}")
            print(f"  Total duplicate rows: {duplicates.sum() - len(duplicates)}")
    
    # which questions will be filtered
    matches_mask = df['prompt_question_normalised'].isin(training_data_normalised)
    matched_rows = df[matches_mask]
    
    if debug:
        # check if matched questions appear multiple times
        matched_question_counts = matched_rows['prompt_question_normalised'].value_counts()
        multi_match = matched_question_counts[matched_question_counts > 1]
        if not multi_match.empty:
            print(f"\nTraining questions matching multiple benchmark rows:")
            print(f"  Unique training questions matched: {len(matched_question_counts)}")
            print(f"  Training questions matching >1 row: {len(multi_match)}")
            print(f"  Extra rows from duplicates: {multi_match.sum() - len(multi_match)}")
        
        # breakdown by dataset
        print(f"\nFiltered rows by dataset:")
        # 'dataset_name' exists on Results, but guard for other tables
        group_col = 'dataset_name' if 'dataset_name' in matched_rows.columns else None
        if group_col:
            print(matched_rows.groupby(group_col)[results_q_col].count().to_string())
        else:
            print(matched_rows[results_q_col].count())
    
    initial_len = len(df)
    filtered_df = df[~matches_mask].copy()
    filtered_df = filtered_df.drop(columns=['prompt_question_normalised'])
    
    unique_questions_filtered = matched_rows['prompt_question_normalised'].nunique()
    print(f"\nFiltered out {initial_len - len(filtered_df)} rows ({unique_questions_filtered} unique questions). Remaining rows: {len(filtered_df)}")
    
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
    excluded_datasets: list[str] | None = None,
    included_datasets: list[str] | None = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Processes results for a single results directory.
    
    Args:
        results_dir: Directory containing the results.
        filter_training: Whether to drop training overlaps via filter_training_data.
        training_data_path: CSV or JSON training file path for overlap filtering.
        excluded_datasets: List of datasets to exclude.
        included_datasets: List of datasets to specifically include.
        
    Returns:
        DataFrame containing the results table.
    """
    print(f"\nProcessing results in {results_dir}...\n")

    _lazy_import_analysis_components()

    # r = Results(
    #     base_results_dir=results_dir,
    #     filter_indeterminate_abstentions=False,
    #     sweep_dir=""
    # )
    
    # manually find result paths to bypass JobManager if it fails to find files with empty sweep_dir
    final_file = "GroundTruthAbstentionEvaluator.json"
    result_path_names = []
    missing_datasets = []
    
    if os.path.exists(results_dir):
        for item in sorted(os.listdir(results_dir)):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                found_final_file = False
                for root, dirs, files in os.walk(item_path):
                    if final_file in files:
                        rel_path = os.path.relpath(root, results_dir)
                        result_path_names.append(rel_path)
                        found_final_file = True
                
                if not found_final_file:
                    missing_datasets.append(item)
                    
    if missing_datasets:
        print(f"WARNING: The following {len(missing_datasets)} datasets are missing {final_file}:")
        for ds in missing_datasets:
            print(f"  - {ds}")

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
        r.df = filter_training_data(r.df, training_data_path, debug=debug)
    
    if excluded_datasets:
        mask = pd.Series(False, index=r.df.index)
        for ds in excluded_datasets:
            if 'dataset_name' in r.df.columns:
                mask |= r.df['dataset_name'].str.contains(ds, case=False, na=False)
            if 'dataset_name_formatted' in r.df.columns:
                mask |= r.df['dataset_name_formatted'].str.contains(ds, case=False, na=False)
        
        if mask.any():
            initial_len_ds = len(r.df)
            r.df = r.df[~mask]
            print(f"Filtered out {initial_len_ds - len(r.df)} rows belonging to datasets matching: {excluded_datasets}")

    if included_datasets:
        mask = pd.Series(False, index=r.df.index)
        for ds in included_datasets:
            if 'dataset_name' in r.df.columns:
                mask |= r.df['dataset_name'].str.contains(ds, case=False, na=False)
            if 'dataset_name_formatted' in r.df.columns:
                mask |= r.df['dataset_name_formatted'].str.contains(ds, case=False, na=False)
        
        initial_len_ds = len(r.df)
        r.df = r.df[mask]
        print(f"Kept only {len(r.df)} rows belonging to included datasets matching: {included_datasets} (filtered out {initial_len_ds - len(r.df)})")

    # Stop if DataFrame is empty after filtering
    if r.df.empty:
        print(f"No results left after filtering datasets.")
        return pd.DataFrame()

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
    included_datasets: list[str] | None = None,
    output_path: str | None = None,
    save_per_vector: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Processes results for each steering vector index.
    
    Args:
        base_dir: Base directory containing subdirectories for each vector index.
        vector_indices: List of vector indices to process.
        filter_training: Whether to drop training overlaps via filter_training_data.
        training_data_path: CSV or JSON training file path for overlap filtering.
        excluded_datasets: List of datasets to exclude.
        included_datasets: List of datasets to specifically include.
        output_path: Output file path for saving results.
        save_per_vector: Whether to save each vector's results in a separate file.
        
    Returns:
        DataFrame containing aggregated results for all vectors.
    """
    all_results = []
    
    for idx in vector_indices:
        results_dir = os.path.join(base_dir, str(idx), "results")
        
        print(f"\nProcessing vector index {idx}...")
        table_df = process_results_dir(
            results_dir=results_dir, 
            filter_training=filter_training, 
            training_data_path=training_data_path, 
            excluded_datasets=excluded_datasets, 
            included_datasets=included_datasets,
            debug=debug
        )
        
        if not table_df.empty:
            # update model_name_formatted to include steered suffix
            if 'model_name_formatted' in table_df.columns:
                table_df['model_name_formatted'] = table_df['model_name_formatted'].apply(
                    lambda x: f"{x}_steered_{idx}"
                )
            
            table_df['vector_index'] = idx
            all_results.append(table_df)
            
            # save per-vector file if requested
            if save_per_vector and output_path:
                vector_output_path = _add_suffix_to_path(output_path, f"_{idx}")
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
        vector_indices_parsed = parse_vector_indices(args.vector_indices, args.steering_dir)
        if not vector_indices_parsed:
            print(f"Error: No vector indices provided and none found in {args.steering_dir}")
            sys.exit(1)
            
        results_df = process_steering_results(
            base_dir=args.steering_dir,
            vector_indices=vector_indices_parsed,
            filter_training=args.filter_training,
            training_data_path=args.training_data,
            excluded_datasets=args.exclude_datasets,
            included_datasets=args.include_datasets,
            output_path=args.output,
            save_per_vector=args.save_per_vector,
            debug=args.debug,
        )
    else:
        # single results directory mode
        results_df = process_results_dir(
            results_dir=args.results_dir,
            filter_training=args.filter_training,
            training_data_path=args.training_data,
            excluded_datasets=args.exclude_datasets,
            included_datasets=args.include_datasets,
            debug=args.debug,
        )
    
    if args.output:
        save_results(results_df, args.output)
    
    # find and print best vector if requested (steering mode only)
    if args.steering_dir and args.find_best is not None:
        if args.find_best:
            best = find_best_vector_overall(results_df, metrics=args.find_best)
        else:
            # flag present but no metrics -> use default metrics in function
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