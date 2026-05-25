from math import ceil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _effective_savefig_dpi(default: float = 300.0) -> float:
    dpi = plt.rcParams.get("savefig.dpi", default)
    if dpi == "figure":
        return float(plt.rcParams["figure.dpi"])
    return float(dpi)


def _score_axis_top(max_val: float, *, scaled: float, floor: float = 1.0) -> float:
    """Return y-axis upper bound: at least ``floor``, or ``scaled`` when data exceeds it."""
    return max(floor, scaled)


def _default_legend_ncol(n_entries: int) -> int:
    if n_entries <= 2:
        return n_entries
    if n_entries <= 4:
        return n_entries
    if n_entries <= 6:
        return 3
    return 4


def place_legend_below_xaxis(
    ax,
    *,
    ncol: Optional[int] = None,
    title: Optional[str] = None,
    frameon: bool = True,
    edgecolor: Optional[str] = None,
) -> None:
    """Place legend in a horizontal row centered under the x-axis label."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    n = len(handles)
    ncol = ncol if ncol is not None else _default_legend_ncol(n)
    legend_kw: dict = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.14),
        "ncol": ncol,
        "frameon": frameon,
    }
    if title is not None:
        legend_kw["title"] = title
    if edgecolor is not None:
        legend_kw["edgecolor"] = edgecolor

    if ax.get_legend() is not None:
        sns.move_legend(ax, **legend_kw)
    else:
        ax.legend(handles, labels, **legend_kw)

    rows = ceil(n / ncol)
    bottom = 0.20 + 0.05 * rows
    if ax.get_xlabel():
        bottom += 0.02
    ax.figure.subplots_adjust(bottom=bottom)


def configure_plot_style() -> None:
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


def prepare_bar_plot_base(
    plot_df: pd.DataFrame, model_params: Dict[str, float]
) -> Optional[tuple[pd.DataFrame, List[str]]]:
    if (
        plot_df is None
        or plot_df.empty
        or not model_params
        or not set(plot_df["Model"]).intersection(model_params.keys())
    ):
        return None

    plot_base = plot_df[plot_df["Model"].isin(model_params.keys())].copy()
    plot_base["params"] = plot_base["Model"].map(model_params)
    plot_base = plot_base.sort_values("params")
    return plot_base, plot_base["Model"].tolist()


def plot_optimal_config_bars(
    plot_base: pd.DataFrame,
    model_order: List[str],
    *,
    save: bool,
    output_dir: Optional[Path] = None,
    f1_title: str,
    f1_ylabel: str,
    pr_title: str,
    pr_ylabel: str,
    f1_filename: str,
    pr_filename: str,
    show: bool = True,
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
    ax.set_ylabel(f1_ylabel)
    ymax_f1 = float(max(heights)) if heights else 0.0
    ax.set_ylim(0, _score_axis_top(ymax_f1, scaled=ymax_f1 * 1.2 + 0.01))

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
    if show:
        plt.show()
    else:
        plt.close()

    pr_colors = {"Precision": "#2E7BC8", "Recall": "#18A878"}
    melted_pr = plot_base.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(9, 7))
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
    ymax_pr = float(melted_pr["value"].max()) if not melted_pr.empty else 0.0
    ax_pr.set_ylim(0, _score_axis_top(ymax_pr, scaled=ymax_pr + 0.14))
    plt.setp(ax_pr.get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    place_legend_below_xaxis(ax_pr, ncol=2, title="")

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

    if save and output_dir:
        plt.savefig(output_dir / pr_filename, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_method_bars(
    plot_df: pd.DataFrame,
    model_order: List[str],
    *,
    save: bool,
    output_dir: Optional[Path],
    split: str,
    f1_filename: str,
    pr_filename: str,
    show: bool = True,
) -> None:
    """Grouped bars with hue=Method for cross-method comparison."""
    if plot_df.empty:
        print(f"No data for combined {split} plot.")
        return

    methods = sorted(plot_df["Method"].unique())
    _save_kw = dict(
        bbox_inches="tight",
        dpi=_effective_savefig_dpi(),
        pad_inches=0.18,
        facecolor=plt.rcParams.get("savefig.facecolor", "white"),
    )

    n_models = len(model_order)
    n_methods = len(methods)
    fig_w = max(12, n_models * max(2.8, n_methods * 0.75))

    plt.figure(figsize=(fig_w, 7))
    ax = sns.barplot(
        data=plot_df,
        x="Model",
        y="F1",
        hue="Method",
        hue_order=methods,
        order=model_order,
        palette="muted",
        width=0.72,
        gap=0.12,
        edgecolor="white",
    )
    title = (
        "F1 across alignment methods"
        if split == "in_domain"
        else "F1 across alignment methods: Rulebreakers"
    )
    ax.set_title(title, pad=15)
    ax.set_xlabel("Models", labelpad=15)
    ax.set_ylabel("Macro-average F1" if split == "in_domain" else "F1 score")
    ymax_f1 = float(plot_df["F1"].max()) if not plot_df.empty else 0.0
    ax.set_ylim(0, _score_axis_top(ymax_f1, scaled=ymax_f1 * 1.2 + 0.01))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    place_legend_below_xaxis(ax, ncol=min(len(methods), 4), title="")

    for p in ax.patches:
        val = p.get_height()
        if val <= 0 or np.isnan(val):
            continue
        ax.annotate(
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

    if save and output_dir:
        plt.savefig(output_dir / f1_filename, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close()

    # Precision / Recall comparison (currently meaningless)
    # melted = plot_df.melt(
    #     id_vars=["Model", "Method"],
    #     value_vars=["Precision", "Recall"],
    #     var_name="metric",
    #     value_name="value",
    # )
    # plt.figure(figsize=(max(9, len(model_order) * 1.5), 6))
    # sns.barplot(
    #     data=melted,
    #     x="Model",
    #     y="value",
    #     hue="metric",
    #     order=model_order,
    #     palette={"Precision": "#2E7BC8", "Recall": "#18A878"},
    #     edgecolor="white",
    # )
    # plt.title(f"Precision / Recall comparison ({split.replace('_', ' ')})", pad=15)
    # plt.xlabel("Models", labelpad=15)
    # plt.tight_layout()
    # if save and output_dir:
    #     plt.savefig(output_dir / pr_filename, **_save_kw)
    # if show:
    #     plt.show()
    # else:
    #     plt.close()
