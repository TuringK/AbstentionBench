# %%
from pathlib import Path

from notebook_scripts_v2.generate_qa_plots import generate_qa_plots

# %%
OUTPUT_DIR = Path("./general_cap_outputs/qa")

# %%
generate_qa_plots(
    output_dir=OUTPUT_DIR,
    save=True,
    show=False,
    log_scale=False,
    horizontal=False,
    mean_plot=True,
    per_dataset=True,
    per_method=True,
    per_method_dataset=True,
)

# %%
