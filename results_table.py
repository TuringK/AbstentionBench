from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable

r1 = Results(
    base_results_dir="All_models_Llama8B_judge/results",
    filter_indeterminate_abstentions=True,
    sweep_dir=""
)

table_df = AbstentionF1ScoreTable(results=r1).table_df
table_df.to_csv("analysis/all_models_llama_3_1_8B_judge.csv", index=False)

unique_models = table_df['model_name_formatted'].unique()
for model in unique_models:
    model_df = table_df[table_df['model_name_formatted'] == model]
    print(f"\n\n{'='*80}\nModel: {model}\n{'='*80}\n")
    print(model_df)