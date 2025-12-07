import os
import pandas as pd
from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable

def normalise_text(text):
    """
    Normalises text by removing all whitespace characters.
    """
    if not isinstance(text, str):
        return str(text)
    return "".join(text.split())

def filter_training_data(df: pd.DataFrame, training_data_path: str = "data/sample_pairs.csv") -> pd.DataFrame:
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

def process_results_dir(results_dir: str, filter_training: bool = False, excluded_datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Processes results for a single results directory.
    
    Args:
        results_dir: Directory containing the results.
        filter_training: Whether to filter out training data.
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
        r.df = filter_training_data(r.df)
    
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

def process_steering_results(base_dir: str, vector_indices: list[int], filter_training: bool, excluded_datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Processes results for each steering vector index.
    
    Args:
        base_dir: Base directory containing subdirectories for each vector index.
        vector_indices: List of vector indices to process.
        
    Returns:
        DataFrame containing aggregated results for all vectors.
    """
    all_results = []
    
    for idx in vector_indices:
        results_dir = os.path.join(base_dir, str(idx), "results")
        
        print(f"\nProcessing vector index {idx}...")
        table_df = process_results_dir(results_dir, filter_training, excluded_datasets)
        
        if not table_df.empty:
            table_df['vector_index'] = idx
            all_results.append(table_df)
            
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


if __name__ == "__main__":
    # base_dir = "data/Qwen2_5_1_5B_Instruct_Keywords_judge_CAA_idx_10-27_coeff_1_0"
    # # vector_idx = range(10, 28) # 10 to 27 inclusive
    # vector_idx = [14]
    # # exclude_datasets = ["WorldSense", "MoralChoice"]

    # results_df = process_steering_results(
    #     base_dir=BASE_DIR, 
    #     vector_indices=vector_idx, 
    #     filter_training=False,
    #     # excluded_datasets=exclude_datasets,
    # )
    
    results_dir = "data/original_abstention/Qwen2_5_1_5B_Instruct_Benchmark/results"
    filter_training = True
    
    results_df = process_results_dir(results_dir=results_dir, filter_training=filter_training)
    results_df.to_excel("data/vanila_qwen2_5_1_5B_instruct_keywords_judge_filtered.xlsx")
    
    
    