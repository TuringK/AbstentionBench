# %%
from pathlib import Path
from notebook_scripts.best_combinations import best_combinations
from notebook_scripts.generate_paper_plots import generate_plots

# %%
results_dir = Path("../data/v3_csv")
output_results_dir = Path("./analysis_outputs/v3")
output_rulebreakers_dir = Path("./analysis_outputs/v3_rulebreakers")
models = {
    "qwen_0_5": "Qwen 2.5 0.5B",
    "qwen_1_5": "Qwen 2.5 1.5B",
    "qwen_3": "Qwen 2.5 3B",
    "qwen_7": "Qwen 2.5 7B",
    "tulu_8": "Tulu 3.1 8B",
    "gemma_1": "Gemma 3 1B",
}
model_params = {
    "Qwen 2.5 0.5B": 0.5,
    "Qwen 2.5 1.5B": 1.5,
    "Qwen 2.5 3B": 3.0,
    "Qwen 2.5 7B": 7.0,
    "Tulu 3.1 8B": 8.0,
    "Gemma 3 1B": 1.0,
}
coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# %%
best_combinations(
    models, 
    coeffs, 
    results_dir, 
    output_dir=output_results_dir, 
    save=True, 
    print_table=False
)

# %%
generate_plots(
    models,
    model_params,
    coeffs,
    results_dir,
    output_dir=output_results_dir,
    save=True,
    rulebreakers_subdir="rulenreakers",
    rulebreakers_output_dir=output_rulebreakers_dir,
)


