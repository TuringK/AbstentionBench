# %%
from pathlib import Path

from notebook_scripts_v2.generate_ppl_plots import generate_ppl_plots

# %%
OUTPUT_DIR = Path("./general_cap_outputs/ppl")

# %%
generate_ppl_plots(
    output_dir=OUTPUT_DIR,
    save=True,
    show=False,
    log_scale=False,
    horizontal=True,
)

# %%

