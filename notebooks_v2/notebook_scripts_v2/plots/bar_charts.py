from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _effective_savefig_dpi(default: float = 300.0) -> float:
    dpi = plt.rcParams.get("savefig.dpi", default)
    if dpi == "figure":
        return float(plt.rcParams["figure.dpi"])
    return float(dpi)


def _savefig_kwargs(*, dpi: Optional[float] = None) -> dict:
    return dict(
        bbox_inches="tight",
        dpi=dpi if dpi is not None else _effective_savefig_dpi(),
        pad_inches=0.18,
        facecolor=plt.rcParams.get("savefig.facecolor", "white"),
    )


def _annotate_bar_values(ax) -> None:
    for patch in ax.patches:
        val = patch.get_height()
        if val <= 0 or np.isnan(val):
            continue
        ax.annotate(
            f"{val:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2c3e50",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
        )


def _annotate_horizontal_bar_values(ax) -> None:
    for patch in ax.patches:
        if patch.get_height() <= 0:
            continue
        val = patch.get_width()
        if val <= 0 or np.isnan(val):
            continue
        ax.annotate(
            f"{val:.2f}",
            (val, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=10,
            color="#2c3e50",
            xytext=(4, 0),
            textcoords="offset points",
            fontweight="bold",
        )


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
    legend_bbox_y: float = -0.14,
    bottom_base: float = 0.20,
    bottom_per_row: float = 0.05,
    xlabel_bottom_extra: float = 0.02,
) -> None:
    """Place legend in a horizontal row centered under the x-axis label."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    n = len(handles)
    ncol = ncol if ncol is not None else _default_legend_ncol(n)
    legend_kw: dict = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, legend_bbox_y),
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
    bottom = bottom_base + bottom_per_row * rows
    if ax.get_xlabel():
        bottom += xlabel_bottom_extra
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
    plot_df: pd.DataFrame,
    model_params: Dict[str, float],
    *,
    model_order: Optional[Sequence[str]] = None,
) -> Optional[tuple[pd.DataFrame, List[str]]]:
    if (
        plot_df is None
        or plot_df.empty
        or not model_params
        or not set(plot_df["Model"]).intersection(model_params.keys())
    ):
        return None

    plot_base = plot_df[plot_df["Model"].isin(model_params.keys())].copy()
    if model_order is not None:
        order = [model for model in model_order if model in plot_base["Model"].values]
    else:
        plot_base["params"] = plot_base["Model"].map(model_params)
        plot_base = plot_base.sort_values("params")
        order = plot_base["Model"].tolist()
        return plot_base, order

    plot_base = plot_base.set_index("Model").loc[order].reset_index()
    return plot_base, order


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
    plot_f1: bool = True,
    plot_pr: bool = True,
) -> None:
    _save_kw = _savefig_kwargs()

    if plot_f1:
        f1_by_model = plot_base.set_index("Model")["F1"]
        heights = [float(f1_by_model[model]) for model in model_order]
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        x = np.arange(len(model_order), dtype=float)
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

    if not plot_pr:
        return

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


def plot_f1_baseline_comparison_bars(
    plot_df: pd.DataFrame,
    model_order: List[str],
    *,
    hue_order: List[str],
    save: bool,
    output_dir: Optional[Path],
    title: str,
    ylabel: str,
    filename: str,
    show: bool = True,
) -> None:
    """Grouped F1 bars comparing a baseline method with the target method."""
    if plot_df.empty:
        print(f"No data for {filename}; skipping baseline comparison plot.")
        return

    _save_kw = _savefig_kwargs()
    n_models = len(model_order)
    fig_w = max(9, n_models * 1.5)

    plt.figure(figsize=(fig_w, 7))
    ax = sns.barplot(
        data=plot_df,
        x="Model",
        y="F1",
        hue="Method",
        hue_order=hue_order,
        order=model_order,
        palette="muted",
        width=0.55,
        gap=0.08,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title(title, pad=15)
    ax.set_xlabel("Models", labelpad=15)
    ax.set_ylabel(ylabel)
    ymax_f1 = float(plot_df["F1"].max()) if not plot_df.empty else 0.0
    ax.set_ylim(0, _score_axis_top(ymax_f1, scaled=ymax_f1 * 1.2 + 0.01))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    place_legend_below_xaxis(ax, ncol=len(hue_order), title="")
    _annotate_bar_values(ax)

    if save and output_dir:
        plt.savefig(output_dir / filename, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_method_bars(
    plot_df: pd.DataFrame,
    model_order: List[str],
    *,
    method_order: List[str],
    save: bool,
    output_dir: Optional[Path],
    split: str,
    f1_filename: str,
    pr_filename: str,
    show: bool = True,
    save_dpi: Optional[float] = None,
    horizontal: bool = False,
) -> None:
    """Grouped bars with hue=Method for cross-method comparison."""
    if plot_df.empty:
        print(f"No data for combined {split} plot.")
        return

    _save_kw = _savefig_kwargs(dpi=save_dpi)

    n_models = len(model_order)
    n_methods = len(method_order)
    value_label = "Macro-average F1" if split == "in_domain" else "F1 score"
    title = (
        "F1 across alignment methods"
        if split == "in_domain"
        else "F1 across alignment methods: Rulebreakers"
    )

    if horizontal:
        fig_w = max(10, 8 + n_methods * 0.4)
        fig_h = max(9, n_models * 2.5)
        bar_width = 0.88
        bar_gap = 0.06
    else:
        fig_w = max(12, n_models * max(2.8, n_methods * 0.75))
        fig_h = 7
        bar_width = 0.72
        bar_gap = 0.12

    plt.figure(figsize=(fig_w, fig_h))
    barplot_kw = dict(
        data=plot_df,
        hue="Method",
        hue_order=method_order,
        order=model_order,
        palette="muted",
        width=bar_width,
        gap=bar_gap,
        edgecolor="white",
    )
    if horizontal:
        ax = sns.barplot(y="Model", x="F1", **barplot_kw)
        ax.set_xlabel(value_label, labelpad=8)
        ax.set_ylabel("Models", labelpad=15)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    else:
        ax = sns.barplot(x="Model", y="F1", **barplot_kw)
        ax.set_xlabel("Models", labelpad=15)
        ax.set_ylabel(value_label)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.set_title(title, pad=15)
    ymax_f1 = float(plot_df["F1"].max()) if not plot_df.empty else 0.0
    upper = _score_axis_top(ymax_f1, scaled=ymax_f1 * 1.2 + 0.01)
    if horizontal:
        ax.set_xlim(0, upper)
    else:
        ax.set_ylim(0, upper)
    plt.tight_layout()
    legend_kw = (
        dict(
            legend_bbox_y=-0.08,
            bottom_base=0.12,
            bottom_per_row=0.035,
            xlabel_bottom_extra=0.008,
        )
        if horizontal
        else {}
    )
    place_legend_below_xaxis(ax, ncol=len(method_order), title="", **legend_kw)
    if horizontal:
        _annotate_horizontal_bar_values(ax)
    else:
        _annotate_bar_values(ax)

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
