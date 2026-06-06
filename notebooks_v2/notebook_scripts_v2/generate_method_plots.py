from pathlib import Path
from typing import Dict, Optional, Sequence

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
    model_order: Optional[Sequence[str]] = None,
    method_order: Optional[Sequence[str]] = None,
    macro_f1_baseline_method: Optional[str] = None,
    combined_plot_dpi: Optional[float] = None,
    combined_horizontal: bool = False,
) -> pd.DataFrame:
    """Cross-method: per-method bar pairs and optional combined comparison charts."""
    from plot_configs.methods_config import (
        COMBINED_PLOT_DPI,
        MACRO_F1_BASELINE_METHOD,
        METHOD_ORDER,
        MODEL_ORDER,
        resolve_method_order,
        resolve_model_order,
    )

    configure_plot_style()
    output_dir = Path(output_dir)
    rb_out = Path(rulebreakers_output_dir or output_dir.parent / "v3_rulebreakers")

    resolved_model_order = resolve_model_order(model_params, model_order or MODEL_ORDER)
    resolved_method_order = method_order or METHOD_ORDER

    all_rows = []
    for method, spec in methods.items():
        all_rows.extend(_load_method_rows(method, spec))

    if not all_rows:
        print("No method data loaded.")
        return pd.DataFrame()

    plot_df = to_plot_dataframe(all_rows)
    methods_found = resolve_method_order(
        plot_df["Method"].unique(),
        resolved_method_order,
    )
    baseline_method = (
        macro_f1_baseline_method
        if macro_f1_baseline_method is not None
        else MACRO_F1_BASELINE_METHOD
    )
    resolved_combined_dpi = (
        combined_plot_dpi if combined_plot_dpi is not None else COMBINED_PLOT_DPI
    )

    if export_summary_tables:
        summary_dir = output_dir / "summary_tables"
        export_method_summary_csvs(
            plot_df,
            summary_dir,
            model_order=resolved_model_order,
            method_order=methods_found,
        )

    plot_kwargs = {
        "model_order": resolved_model_order,
        "macro_f1_baseline_method": baseline_method,
    }
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
            **plot_kwargs,
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
            **plot_kwargs,
        )

    if combined:
        combined_dir = output_dir / "combined"
        if save:
            combined_dir.mkdir(parents=True, exist_ok=True)

        chart_model_order = [
            model
            for model in resolved_model_order
            if model in plot_df["Model"].values
        ]
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
                chart_model_order,
                method_order=methods_found,
                save=save,
                output_dir=combined_dir,
                split=split,
                f1_filename=f1_name,
                pr_filename=pr_name,
                show=show,
                save_dpi=resolved_combined_dpi,
                horizontal=combined_horizontal,
            )

    return plot_df
