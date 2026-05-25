from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from .bar_charts import place_legend_below_xaxis


def generate_caa_diagnostics(
    df_all: pd.DataFrame,
    *,
    output_dir: Optional[Path] = None,
    save: bool = False,
    show: bool = True,
) -> None:
    """Heatmaps, F1 trajectory, and Pareto plots from ID-only sweep summary."""
    if df_all.empty:
        print("No CAA sweep data for diagnostics.")
        return

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Heatmaps...")
    for model in df_all["model"].unique():
        subset = df_all[df_all["model"] == model]
        pivot = subset.pivot(index="coeff", columns="vector_index", values="f1_score")
        pivot = pivot.sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(
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
        if show:
            plt.show()
        else:
            plt.close()

    print("Generating F1 Trajectory Plot...")
    plt.figure(figsize=(9, 8))

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
        ].sort_values("coeff")

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
    ax.set_xticks(sorted(df_all["coeff"].unique()))
    ymax_f1 = float(df_all["f1_score"].max()) if not df_all.empty else 0.0
    ax.set_ylim(0, max(1.0, ymax_f1 * 1.2 + 0.01))
    plt.title("Performance Trajectory at Optimal Steered Layer", pad=15)
    plt.xlabel("Steering Coefficient")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    place_legend_below_xaxis(
        ax,
        ncol=3,
        title="Model & Setup",
        edgecolor="black",
    )
    if save and output_dir:
        plt.savefig(output_dir / "f1_trajectory_best_layer.png", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print("Generating Precision-Recall Pareto Frontiers...")
    plt.figure(figsize=(9, 8))

    for idx, model in enumerate(df_all["model"].unique()):
        subset = df_all[df_all["model"] == model]
        plt.scatter(
            subset["recall"],
            subset["precision"],
            color=palette[idx % len(palette)],
            alpha=0.15,
            s=25,
            edgecolor=None,
        )

        pareto = []
        for _, row1 in subset.iterrows():
            dominated = False
            for _, row2 in subset.iterrows():
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
    ax_pareto = plt.gca()
    xmax = float(df_all["recall"].max()) if not df_all.empty else 0.0
    ymax = float(df_all["precision"].max()) if not df_all.empty else 0.0
    ax_pareto.set_xlim(0, max(1.0, xmax + 0.05))
    ax_pareto.set_ylim(0, max(1.0, ymax + 0.05))
    plt.tight_layout()
    place_legend_below_xaxis(
        ax_pareto,
        ncol=3,
        title="Model",
        edgecolor="black",
    )
    if save and output_dir:
        plt.savefig(output_dir / "pr_tradeoff_pareto.png", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
