# %%
from pathlib import Path

from notebook_scripts_v2.generate_caa_plots import generate_caa_plots
from plot_configs.methods_config import CAA_MODELS, MODEL_PARAMS

# %%
OUTPUT_ROOT = Path("./notebook_scripts_v2_outputs")
output_dir = OUTPUT_ROOT / "v3"
output_rulebreakers_dir = OUTPUT_ROOT / "v3_rulebreakers"

# %%
generate_caa_plots(
    CAA_MODELS,
    MODEL_PARAMS,
    output_dir=output_dir,
    rulebreakers_output_dir=output_rulebreakers_dir,
    save=True,
    show=True,
)

# %%
