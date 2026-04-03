import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, List, Dict

from .summarize_metrics import filter_incomplete_layers


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


def generate_plots(
    models: Dict[str, str],
    model_params: Dict[str, float],
    coeffs: List[float],
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False,
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
            if coeff == 1.0:
                csv_path = results_dir / f"{model_key}_v1.csv"
            else:
                coeff_str = f"{coeff:.1f}".replace(".", "_")
                csv_path = (
                    results_dir
                    / f"{model_key}_sweep"
                    / f"{model_key}_v1_sweep_{coeff_str}.csv"
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
                output_dir / f"heatmap_{model.replace(' ', '_')}.png",
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

    # Optional: adjust x ticks depending on available coeffs
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
    plt.legend(title="Model", loc="upper right", frameon=True, edgecolor="black")
    plt.tight_layout()
    if save and output_dir:
        plt.savefig(output_dir / "pr_tradeoff_pareto.png", bbox_inches="tight")
    plt.show()

    # best f1 vs scale
    print("Generating F1 vs Scale Plot...")
    plt.figure(figsize=(9, 6))

    best_f1_pts = []
    for model in df_all["model"].unique():
        subset = df_all[df_all["model"] == model]
        if model in model_params:
            best_f1 = subset["f1_score"].max()
            params = model_params[model]
            best_f1_pts.append({"model": model, "params": params, "best_f1": best_f1})

    if best_f1_pts:
        best_f1_df = pd.DataFrame(best_f1_pts).sort_values("params")

        # bar chart
        ax = sns.barplot(
            x="model",
            y="best_f1",
            data=best_f1_df,
            hue="model",
            palette="viridis",
            legend=False,
        )

        plt.title("Optimal Steered F1 Score vs. Model Scale", pad=15)
        plt.xlabel("Model Structure")
        plt.ylabel("Max Achieved F1 Score")

        # explicit F1 score values above bars
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(
                    f"{val:.3f}",
                    (p.get_x() + p.get_width() / 2.0, val),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#2c3e50",
                    xytext=(0, 6),
                    textcoords="offset points",
                    fontweight="bold",
                )

        plt.ylim(0, best_f1_df["best_f1"].max() * 1.20)
        plt.xticks(rotation=15)

        plt.tight_layout()
        if save and output_dir:
            plt.savefig(output_dir / "max_f1_vs_model_scale.png", bbox_inches="tight")
        plt.show()
    else:
        print(
            "Model parameters for scaling plot not provided or matched. Skipping F1 vs Scale plot."
        )
