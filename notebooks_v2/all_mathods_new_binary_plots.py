# %%
from pathlib import Path

from notebook_scripts_v2.generate_method_plots import generate_method_plots
from plot_configs.methods_config import (
    COMBINED_PLOT_DPI,
    MACRO_F1_BASELINE_METHOD,
    METHOD_ORDER,
    METHODS,
    MODEL_ORDER,
    MODEL_PARAMS
)

# %%
OUTPUT_ROOT = Path("./turing_abstention_bench_outputs")
output_dir = OUTPUT_ROOT / "abstention_bench_plots"
output_rulebreakers_dir = OUTPUT_ROOT / "rulebreakers_plots"

# %%
generate_method_plots(
    methods=METHODS,
    model_params=MODEL_PARAMS,
    output_dir=output_dir,
    rulebreakers_output_dir=output_rulebreakers_dir,
    save=True,
    show=False,
    combined=False,
    combined_horizontal=False,
    model_order=MODEL_ORDER,
    method_order=METHOD_ORDER,
    macro_f1_baseline_method=MACRO_F1_BASELINE_METHOD,
    combined_plot_dpi=COMBINED_PLOT_DPI,
)

# %%
