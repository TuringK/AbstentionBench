from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .export_summary_tables import export_method_summary_csvs
from .generate_caa_plots import _plot_method_split
from .loaders.caa_sweep import load_caa_sweep_data
from .loaders.flat_csv import load_flat_csv_method
from .metrics_schema import to_plot_dataframe
from .plots.bar_charts import configure_plot_style, plot_combined_method_bars

MethodSpec = Dict[str, object]
MethodRegistry = Dict[str, MethodSpec]

_LOADERS = {
    "caa_sweep": lambda method, models: load_caa_sweep_data(models)[0],
    "flat_csv": lambda method, models: load_flat_csv_method(models, method),
}


def _load_method_rows(method: str, spec: MethodSpec) -> list:
    models = spec.get("models")
    if not models:
        return []

    loader = spec["loader"]
    try:
        load_fn = _LOADERS[loader]
    except KeyError as exc:
        known = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unknown loader {loader!r} for method {method!r}. Known loaders: {known}"
        ) from exc

    return load_fn(method, models)


def generate_method_plots(
    *,
    methods: MethodRegistry,
    model_params: Dict[str, float],
    output_dir: Path,
    rulebreakers_output_dir: Optional[Path] = None,
    save: bool = True,
    show: bool = True,
    combined: bool = False,
    export_summary_tables: bool = True,
) -> pd.DataFrame:
    """Cross-method: per-method bar pairs and optional combined comparison charts."""
    configure_plot_style()
    output_dir = Path(output_dir)
    rb_out = Path(rulebreakers_output_dir or output_dir.parent / "v3_rulebreakers")

    all_rows = []
    for method, spec in methods.items():
        all_rows.extend(_load_method_rows(method, spec))

    if not all_rows:
        print("No method data loaded.")
        return pd.DataFrame()

    plot_df = to_plot_dataframe(all_rows)
    methods_found = sorted(plot_df["Method"].unique())

    if export_summary_tables:
        summary_dir = output_dir / "summary_tables"
        export_method_summary_csvs(
            plot_df,
            summary_dir,
            model_params=model_params,
        )

    for method in methods_found:
        _plot_method_split(
            plot_df,
            model_params,
            method=method,
            split="in_domain",
            id_output_dir=output_dir / method,
            rb_output_dir=rb_out / method,
            save=save,
            show=show,
        )
        _plot_method_split(
            plot_df,
            model_params,
            method=method,
            split="rulebreakers",
            id_output_dir=output_dir / method,
            rb_output_dir=rb_out / method,
            save=save,
            show=show,
        )

    if combined:
        combined_dir = output_dir / "combined"
        if save:
            combined_dir.mkdir(parents=True, exist_ok=True)

        model_order = sorted(
            model_params.keys(),
            key=lambda m: model_params[m],
        )
        for split, f1_name, pr_name in [
            (
                "in_domain",
                "combined_macro_f1_vs_model_scale.png",
                "combined_precision_recall.png",
            ),
            (
                "rulebreakers",
                "combined_rulebreakers_f1.png",
                "combined_rulebreakers_precision_recall.png",
            ),
        ]:
            subset = plot_df[plot_df["Split"] == split]
            if subset.empty:
                continue
            plot_combined_method_bars(
                subset,
                [m for m in model_order if m in subset["Model"].values],
                save=save,
                output_dir=combined_dir,
                split=split,
                f1_filename=f1_name,
                pr_filename=pr_name,
                show=show,
            )

    return plot_df
