import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, List, Dict

from .best_combinations import best_combinations
from .summarize_metrics import (
    filter_incomplete_layers,
    load_rulebreakers_metrics,
    steering_results_csv_path,
)


def _effective_savefig_dpi(default: float = 300.0) -> float:
    """Numeric DPI for savefig when rcParams uses the string sentinel ``'figure'``."""
    dpi = plt.rcParams.get("savefig.dpi", default)
    if dpi == "figure":
        return float(plt.rcParams["figure.dpi"])
    return float(dpi)


def process_csv(path: Path, model_name: str, coeff: float, all_data: List):
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = filter_incomplete_layers(df)

    if df.empty:
        return

    summary = (
        df.groupby("vector_index")[["precision", "recall", "f1_score"]]
        .mean()
        .reset_index()
    )
    summary["model"] = model_name
    summary["coeff"] = coeff
    all_data.append(summary)


def plot_optimal_config_bars(
    plot_base: pd.DataFrame,
    model_order: List[str],
    *,
    save: bool,
    output_dir: Optional[Path],
    f1_title: str,
    pr_title: str,
    pr_ylabel: str,
    f1_filename: str,
    pr_filename: str,
) -> None:
    _save_kw = dict(
        bbox_inches="tight",
        dpi=_effective_savefig_dpi(),
        pad_inches=0.18,
        facecolor=plt.rcParams.get("savefig.facecolor", "white"),
    )

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    x = np.arange(len(model_order), dtype=float)
    heights = plot_base["F1"].astype(float).tolist()
    bar_width = 0.48
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(model_order)))
    ax.bar(
        x,
        heights,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=0, ha="center")
    ax.set_title(f1_title, pad=15)
    ax.set_xlabel("Models", labelpad=15)
    ax.set_ylabel("Max achieved F1 score")
    ymax_f1 = float(max(heights))
    ax.set_ylim(0, min(0.85, ymax_f1 * 1.2))

    for xi, h in zip(x, heights):
        ax.annotate(
            f"{h:.2f}",
            (xi, h),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2c3e50",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.tight_layout()
    if save and output_dir:
        plt.savefig(output_dir / f1_filename, **_save_kw)

    plt.show()

    pr_colors = {"Precision": "#2E7BC8", "Recall": "#18A878"}
    melted_pr = plot_base.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(9, 6))
    ax_pr = sns.barplot(
        data=melted_pr,
        x="Model",
        y="value",
        hue="metric",
        hue_order=["Precision", "Recall"],
        order=model_order,
        palette=pr_colors,
        width=0.55,
        gap=0.08,
        edgecolor="white",
        linewidth=0.8,
    )

    ax_pr.set_title(pr_title, pad=15)
    ax_pr.set_xlabel("Models", labelpad=15)
    ax_pr.set_ylabel(pr_ylabel)
    ymax_pr = float(melted_pr["value"].max())
    # Headroom above max bar height for annotations (scores are in ~[0, 1])
    ax_pr.set_ylim(0, ymax_pr + 0.14)
    plt.setp(ax_pr.get_xticklabels(), rotation=0, ha="center")
    sns.move_legend(
        ax_pr,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        title="",
    )

    for p in ax_pr.patches:
        val = p.get_height()

        if val <= 0 or np.isnan(val):
            continue

        ax_pr.annotate(
            f"{val:.2f}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2c3e50",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.tight_layout()
    if save and output_dir:
        plt.savefig(output_dir / pr_filename, **_save_kw)
    plt.show()


def prepare_bar_plot_base(
    bc_df: pd.DataFrame, model_params: Dict[str, float]
) -> Optional[tuple[pd.DataFrame, List[str]]]:
    if (
        bc_df is None
        or bc_df.empty
        or not model_params
        or not set(bc_df["Model"]).intersection(model_params.keys())
    ):
        return None

    plot_base = bc_df[bc_df["Model"].isin(model_params.keys())].copy()
    plot_base["params"] = plot_base["Model"].map(model_params)
    plot_base = plot_base.sort_values("params")
    return plot_base, plot_base["Model"].tolist()


def generate_plots(
    models: Dict[str, str],
    model_params: Dict[str, float],
    coeffs: List[float],
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False,
    *,
    csv_version: str = "v3",
    baseline_csv_at_root: bool = False,
    rulebreakers_subdir: Optional[str] = None,
    rulebreakers_output_dir: Optional[Union[str, Path]] = None,
):
    """Generate and display the performance plots. Optionally save to PNG."""
    results_dir = Path(results_dir)

    # config
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
        }
    )

    sns.set_theme(
        style="whitegrid", rc={"axes.edgecolor": "0.15", "axes.linewidth": 1.25}
    )

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # data
    all_data = []

    for model_key, model_name in models.items():
        for coeff in coeffs:
            csv_path = steering_results_csv_path(
                results_dir,
                model_key,
                coeff,
                csv_version=csv_version,
                baseline_csv_at_root=baseline_csv_at_root,
            )
            process_csv(csv_path, model_name, coeff, all_data)

    if not all_data:
        print("No valid CSV data found to plot.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"Data loaded successfully. Found {len(df_all)} data point configurations.")

    # heatmaps
    print("Generating Heatmaps...")
    for model in df_all["model"].unique():
        subset = df_all[df_all["model"] == model]
        pivot = subset.pivot(index="coeff", columns="vector_index", values="f1_score")
        pivot = pivot.sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            pivot, cmap="inferno", annot=False, cbar_kws={"label": "F1 Score"}
        )

        plt.title(f"F1 Score Landscape: {model}", pad=15)
        plt.ylabel("Steering Coefficient")
        plt.xlabel("Layer Index")
        plt.tight_layout()

        if save and output_dir:
            plt.savefig(
                output_dir / f"heatmap_{model.replace(' ', '_').replace('.', '_')}.png",
                bbox_inches="tight",
            )

        plt.show()

    # f1 trajectory
    print("Generating F1 Trajectory Plot...")
    plt.figure(figsize=(9, 6))

    best_layers = {}
    for model in df_all["model"].unique():
        subset = df_all[df_all["model"] == model]
        best_overall_row = subset.loc[subset["f1_score"].idxmax()]
        best_layers[model] = best_overall_row["vector_index"]

    markers = ["o", "s", "^", "D", "v"]
    palette = sns.color_palette("muted", len(df_all["model"].unique()))

    for i, model in enumerate(df_all["model"].unique()):
        best_layer = best_layers[model]
        subset = df_all[
            (df_all["model"] == model) & (df_all["vector_index"] == best_layer)
        ]
        subset = subset.sort_values("coeff")

        plt.plot(
            subset["coeff"],
            subset["f1_score"],
            marker=markers[i % len(markers)],
            color=palette[i % len(palette)],
            label=f"{model} (Layer {int(best_layer)})",
            linewidth=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1,
        )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # adjust x ticks depending on available coeffs
    actual_coeffs = sorted(df_all["coeff"].unique())
    ax.set_xticks(actual_coeffs)

    plt.title("Performance Trajectory at Optimal Steered Layer", pad=15)
    plt.xlabel("Steering Coefficient")
    plt.ylabel("F1 Score")
    plt.legend(
        title="Model & Setup",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        edgecolor="black",
    )

    plt.tight_layout()

    if save and output_dir:
        plt.savefig(output_dir / "f1_trajectory_best_layer.png", bbox_inches="tight")

    plt.show()

    # pr trade-off
    print("Generating Precision-Recall Pareto Frontiers...")
    plt.figure(figsize=(9, 7))

    for idx, model in enumerate(df_all["model"].unique()):
        subset = df_all[df_all["model"] == model]

        # faintly plot all tested permutations for context
        plt.scatter(
            subset["recall"],
            subset["precision"],
            color=palette[idx % len(palette)],
            alpha=0.15,
            s=25,
            edgecolor=None,
        )

        # compute Pareto frontier
        pareto = []
        for i, row1 in subset.iterrows():
            dominated = False

            for j, row2 in subset.iterrows():
                if (
                    row2["recall"] >= row1["recall"]
                    and row2["precision"] >= row1["precision"]
                ) and (
                    row2["recall"] > row1["recall"]
                    or row2["precision"] > row1["precision"]
                ):
                    dominated = True
                    break

            if not dominated:
                pareto.append(row1)

        if pareto:
            pareto_df = pd.DataFrame(pareto).sort_values("recall")
            plt.plot(
                pareto_df["recall"],
                pareto_df["precision"],
                marker=markers[idx % len(markers)],
                color=palette[idx % len(palette)],
                label=model,
                linewidth=2,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1,
            )

    plt.title("Precision-Recall Pareto Frontiers", pad=15)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(
        title="Model",
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    if save and output_dir:
        plt.savefig(output_dir / "pr_tradeoff_pareto.png", bbox_inches="tight")
    plt.show()

    # optimal config bar charts (same layer-coeff row as best_combinations)
    print(
        "Generating optimal F1 vs. scale + precision/recall bar plots (best config per model)..."
    )
    bc_df = best_combinations(
        models,
        coeffs,
        results_dir,
        save=False,
        print_table=False,
        csv_version=csv_version,
        baseline_csv_at_root=baseline_csv_at_root,
    )

    prepared = prepare_bar_plot_base(bc_df, model_params)
    if prepared is not None:
        plot_base, model_order = prepared
        plot_optimal_config_bars(
            plot_base,
            model_order,
            save=save,
            output_dir=output_dir,
            f1_title="Optimal Steered F1 Score vs. Model Scale",
            pr_title="Precision and recall at best steering configuration",
            pr_ylabel="Score (mean across benchmark)",
            f1_filename="max_f1_vs_model_scale.png",
            pr_filename="precision_recall_best_config.png",
        )
    else:
        print(
            "Best combinations unavailable or model_params mismatch; skipping best-config bar plots."
        )

    if rulebreakers_subdir is not None:
        print("Generating Rulebreakers bar plots from pre-computed CSVs...")
        rb_out = (
            Path(rulebreakers_output_dir)
            if rulebreakers_output_dir is not None
            else output_dir
        )
        if save and rb_out is not None:
            rb_out.mkdir(parents=True, exist_ok=True)

        rb_df = load_rulebreakers_metrics(
            models,
            results_dir,
            csv_version=csv_version,
            rulebreakers_subdir=rulebreakers_subdir,
        )
        rb_prepared = prepare_bar_plot_base(rb_df, model_params)
        if rb_prepared is not None:
            rb_plot_base, rb_model_order = rb_prepared
            plot_optimal_config_bars(
                rb_plot_base,
                rb_model_order,
                save=save,
                output_dir=rb_out,
                f1_title="Rulebreakers F1 at Optimal Steering Configuration",
                pr_title="Rulebreakers precision and recall at best steering configuration",
                pr_ylabel="Score on Rulebreakers",
                f1_filename="rulebreakers_max_f1_vs_model_scale.png",
                pr_filename="rulebreakers_precision_recall_best_config.png",
            )
        else:
            print(
                "Rulebreakers metrics unavailable or model_params mismatch; skipping rulebreakers bar plots."
            )
