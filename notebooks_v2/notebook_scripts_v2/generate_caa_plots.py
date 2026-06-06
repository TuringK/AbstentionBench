from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .loaders.caa_sweep import load_caa_sweep_data
from .loaders.flat_csv import load_flat_csv_method
from .metrics_schema import best_combinations_dataframe, to_plot_dataframe
from .plots.bar_charts import (
    configure_plot_style,
    plot_combined_method_bars,
    plot_f1_baseline_comparison_bars,
    plot_optimal_config_bars,
    prepare_bar_plot_base,
)
from .plots.caa_diagnostics import generate_caa_diagnostics


def _plot_f1_chart(
    *,
    plot_df: pd.DataFrame,
    plot_base: pd.DataFrame,
    method: str,
    split: str,
    resolved_model_order: Sequence[str],
    out_dir: Path,
    save: bool,
    show: bool,
    macro_f1_baseline_method: Optional[str],
) -> None:
    use_baseline = (
        macro_f1_baseline_method is not None
        and method != macro_f1_baseline_method
    )
    if use_baseline:
        comparison_df = plot_df[
            (plot_df["Method"].isin([macro_f1_baseline_method, method]))
            & (plot_df["Split"] == split)
        ]
        if macro_f1_baseline_method not in comparison_df["Method"].values:
            print(
                f"No baseline data for {macro_f1_baseline_method} ({split}); "
                f"falling back to single-bar plot for {method}."
            )
            use_baseline = False

    if split == "in_domain":
        f1_title = f"{method}: F1"
        f1_ylabel = "Macro-average F1"
        f1_filename = "macro_f1_vs_model_scale.png"
    else:
        f1_title = f"{method}: Rulebreakers F1"
        f1_ylabel = "F1 score"
        f1_filename = "rulebreakers_f1_vs_model_scale.png"

    if use_baseline:
        plot_f1_baseline_comparison_bars(
            comparison_df,
            list(resolved_model_order),
            hue_order=[macro_f1_baseline_method, method],
            save=save,
            output_dir=out_dir,
            title=f"{f1_title} vs {macro_f1_baseline_method}",
            ylabel=f1_ylabel,
            filename=f1_filename,
            show=show,
        )
        return

    plot_optimal_config_bars(
        plot_base,
        list(resolved_model_order),
        save=save,
        output_dir=out_dir,
        f1_title=f1_title,
        f1_ylabel=f1_ylabel,
        pr_title="",
        pr_ylabel="",
        f1_filename=f1_filename,
        pr_filename="",
        show=show,
        plot_pr=False,
    )


def _plot_method_split(
    plot_df: pd.DataFrame,
    model_params: Dict[str, float],
    *,
    method: str,
    split: str,
    id_output_dir: Path,
    rb_output_dir: Path,
    save: bool,
    show: bool,
    model_order: Optional[Sequence[str]] = None,
    macro_f1_baseline_method: Optional[str] = None,
) -> None:
    subset = plot_df[(plot_df["Method"] == method) & (plot_df["Split"] == split)]
    if subset.empty:
        print(f"No {split} data for {method}; skipping bar plots.")
        return

    prepared = prepare_bar_plot_base(subset, model_params, model_order=model_order)
    if prepared is None:
        print(f"Could not prepare bar plot for {method} ({split}).")
        return

    plot_base, resolved_model_order = prepared
    out_dir = id_output_dir if split == "in_domain" else rb_output_dir
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    _plot_f1_chart(
        plot_df=plot_df,
        plot_base=plot_base,
        method=method,
        split=split,
        resolved_model_order=resolved_model_order,
        out_dir=out_dir,
        save=save,
        show=show,
        macro_f1_baseline_method=macro_f1_baseline_method,
    )

    if split == "in_domain":
        pr_title = f"{method}: Precision and Recall"
        pr_ylabel = "Macro-average Precision and Recall"
        pr_filename = "precision_recall_best_config.png"
    else:
        pr_title = f"{method}: Rulebreakers Precision and Recall"
        pr_ylabel = "Score on Rulebreakers"
        pr_filename = "rulebreakers_precision_recall_best_config.png"

    plot_optimal_config_bars(
        plot_base,
        list(resolved_model_order),
        save=save,
        output_dir=out_dir,
        f1_title="",
        f1_ylabel="",
        pr_title=pr_title,
        pr_ylabel=pr_ylabel,
        f1_filename="",
        pr_filename=pr_filename,
        show=show,
        plot_f1=False,
    )


def generate_caa_plots(
    caa_models: dict,
    model_params: Dict[str, float],
    *,
    output_dir: Path,
    rulebreakers_output_dir: Optional[Path] = None,
    save: bool = True,
    show: bool = True,
    model_order: Optional[Sequence[str]] = None,
    macro_f1_baseline_method: Optional[str] = None,
) -> pd.DataFrame:
    """CAA-only: diagnostics, best_combinations.csv, and CAA bar charts."""
    configure_plot_style()
    output_dir = Path(output_dir)
    rb_out = Path(rulebreakers_output_dir or output_dir.parent / "v3_rulebreakers")

    rows, df_all = load_caa_sweep_data(caa_models)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    generate_caa_diagnostics(
        df_all, output_dir=output_dir / "caa_diagnostics", save=save, show=show
    )

    bc_df = best_combinations_dataframe(rows)
    if save and not bc_df.empty:
        bc_df.to_csv(output_dir / "best_combinations.csv", index=False)
        print(f"Saved best combinations to {output_dir / 'best_combinations.csv'}")

    plot_df = to_plot_dataframe(rows)
    plot_kwargs = {
        "model_order": model_order,
        "macro_f1_baseline_method": macro_f1_baseline_method,
    }
    _plot_method_split(
        plot_df,
        model_params,
        method="CAA",
        split="in_domain",
        id_output_dir=output_dir / "CAA",
        rb_output_dir=rb_out / "CAA",
        save=save,
        show=show,
        **plot_kwargs,
    )
    _plot_method_split(
        plot_df,
        model_params,
        method="CAA",
        split="rulebreakers",
        id_output_dir=output_dir / "CAA",
        rb_output_dir=rb_out / "CAA",
        save=save,
        show=show,
        **plot_kwargs,
    )

    return bc_df
