import pandas as pd

REQUIRED_DATASETS = [
    "ALCUNA", "BB/Disambiguate", "BB/Known unknowns", "BBQ", 
    "CoCoNot/False presumptions", "CoCoNot/Humanizing", "CoCoNot/Incomprehensible", 
    "CoCoNot/Subjective", "CoCoNot/Temporal", "CoCoNot/Unknowns", 
    "CoCoNot/Unsupported", "FalseQA", "FreshQA", "GPQA-Diamond", "GSM8K", 
    "KUQ/Ambiguous", "KUQ/Controversial", "KUQ/False assumptions", 
    "KUQ/Future unknowns", "KUQ/Unsolved problems", "MMLU History", 
    "MMLU Math", "MediQ", "MoralChoice", "Musique", "QASPER", "QAQA", 
    "SQuAD 2.0", "SituatedQA/Geo", "UMWP", "WorldSense"
]

def filter_incomplete_layers(df: pd.DataFrame) -> pd.DataFrame:
    required = set(REQUIRED_DATASETS)
    counts = df[df['dataset_name_formatted'].isin(required)].groupby('vector_index')['dataset_name_formatted'].nunique()
    complete_layers = counts[counts == len(required)].index
    
    all_layers = set(df['vector_index'].unique())
    filtered_layers = all_layers - set(complete_layers)
    if filtered_layers:
        print(f"Filtered out layers missing required datasets: {sorted(list(filtered_layers))}")
        
    return df[df['vector_index'].isin(complete_layers)]

def summarize_layer_metrics(csv_path: str, model_name: str = "Qwen 2.5 0.5B Instruct", filter_incomplete: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    if filter_incomplete:
        df = filter_incomplete_layers(df)
    
    summary = df.groupby('vector_index')[['precision', 'recall', 'f1_score']].mean().reset_index()
    
    summary = summary.rename(columns={
        'vector_index': f'{model_name} layers',
        'precision': 'Avg Precision',
        'recall': 'Avg Recall',
        'f1_score': 'Avg F1 Score'
    })
    
    summary['Avg Precision'] = summary['Avg Precision'].round(4)
    summary['Avg Recall'] = summary['Avg Recall'].round(4)
    summary['Avg F1 Score'] = summary['Avg F1 Score'].round(4)
    
    summary = summary.sort_values(by='Avg F1 Score', ascending=False)
    
    return summary
