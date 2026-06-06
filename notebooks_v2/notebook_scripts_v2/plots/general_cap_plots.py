from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_configs.general_cap_config import (
    COMPARISON_METHOD_COLOR,
    METHOD_COLORS,
    METHOD_ORDER,
    MODEL_COLORS,
    MODEL_SHORT_LABELS,
    QA_BENCHMARK_ORDER,
    VANILLA_BASELINE_METHOD,
    benchmark_slug,
    method_slug,
)

from .bar_charts import (
    _savefig_kwargs,
    _score_axis_top,
    configure_plot_style,
    place_legend_below_xaxis,
)


def _method_palette(method_order: Sequence[str]) -> List[str]:
    return [METHOD_COLORS.get(method, "#333333") for method in method_order]


def _annotate_vertical_bar_values(ax, *, fmt: str = ".2f", fontsize: int = 10) -> None:
    for patch in ax.patches:
        if patch.get_width() <= 0:
            continue
        val = patch.get_height()
        if np.isnan(val):
            continue
        ax.annotate(
            format(val, fmt),
            (patch.get_x() + patch.get_width() / 2.0, val),
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=fontsize,
            color="#2c3e50",
            xytext=(0, 4 if val >= 0 else -4),
            textcoords="offset points",
            fontweight="bold",
        )


def _annotate_horizontal_bar_values(ax, *, fmt: str = ".2f") -> None:
    for patch in ax.patches:
        if patch.get_height() <= 0:
            continue
        val = patch.get_width()
        if np.isnan(val):
            continue
        ax.annotate(
            format(val, fmt),
            (val, patch.get_y() + patch.get_height() / 2.0),
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=10,
            color="#2c3e50",
            xytext=(4 if val >= 0 else -4, 0),
            textcoords="offset points",
            fontweight="bold",
        )


def _apply_value_axis_limits(
    ax,
    values: pd.Series,
    *,
    log_scale: bool,
    horizontal: bool,
    linear_floor: float = 1.0,
) -> None:
    if log_scale:
        positive = values[values > 0]
        vmin = float(positive.min()) if not positive.empty else 1.0
        vmax = float(values.max()) if not values.empty else vmin
        low, high = vmin * 0.85, vmax * 1.25
        if horizontal:
            ax.set_xscale("log")
            ax.set_xlim(low, high)
        else:
            ax.set_yscale("log")
            ax.set_ylim(low, high)
        return

    vmax = float(values.max()) if not values.empty else 0.0
    upper = _score_axis_top(vmax, scaled=vmax * 1.2 + 0.01, floor=linear_floor)
    if horizontal:
        ax.set_xlim(0, upper)
    else:
        ax.set_ylim(0, upper)


def plot_mean_general_cap_bars(
    plot_df: pd.DataFrame,
    model_order: List[str],
    *,
    value_col: str,
    title: str,
    value_label: str,
    method_order: Optional[Sequence[str]] = None,
    save: bool,
    output_dir: Optional[Path] = None,
    filename: str = "mean_metric_by_method.png",
    show: bool = False,
    log_scale: bool = False,
    horizontal: bool = False,
    linear_floor: float = 1.0,
    value_fmt: str = ".2f",
) -> None:
    """Grouped mean-metric bars; models on x-axis (vertical) or y-axis (horizontal)."""
    if plot_df.empty:
        print(f"No data for {filename}.")
        return

    method_order = list(method_order or METHOD_ORDER)
    configure_plot_style()
    _save_kw = _savefig_kwargs()

    n_models = len(model_order)
    n_methods = len(method_order)
    axis_label = f"{value_label} (log scale)" if log_scale else value_label

    if horizontal:
        fig_w = max(10, 8 + n_methods * 0.4)
        fig_h = max(9, n_models * 2.5)
    else:
        fig_w = max(12, n_models * max(2.8, n_methods * 0.75))
        fig_h = 7

    plt.figure(figsize=(fig_w, fig_h))
    barplot_kw = dict(
        data=plot_df,
        hue="Method",
        hue_order=method_order,
        order=model_order,
        palette=_method_palette(method_order),
        width=0.72,
        gap=0.12,
        edgecolor="white",
        linewidth=0.8,
    )
    if horizontal:
        ax = sns.barplot(y="Model", x=value_col, **barplot_kw)
        ax.set_xlabel(axis_label, labelpad=15)
        ax.set_ylabel("Models", labelpad=15)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    else:
        ax = sns.barplot(x="Model", y=value_col, **barplot_kw)
        ax.set_xlabel("Models", labelpad=15)
        ax.set_ylabel(axis_label)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.set_title(title, pad=15)
    _apply_value_axis_limits(
        ax,
        plot_df[value_col],
        log_scale=log_scale,
        horizontal=horizontal,
        linear_floor=linear_floor,
    )

    plt.tight_layout()
    place_legend_below_xaxis(ax, ncol=len(method_order), title="")
    if horizontal:
        _annotate_horizontal_bar_values(ax, fmt=value_fmt)
    else:
        _annotate_vertical_bar_values(ax, fmt=value_fmt)

    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / filename, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_ppl_bars(plot_df: pd.DataFrame, model_order: List[str], **kwargs) -> None:
    plot_mean_general_cap_bars(
        plot_df,
        model_order,
        value_col="mean_ppl",
        title="Mean perplexity across languages",
        value_label="Mean PPL",
        **kwargs,
    )


def plot_mean_qa_bars(plot_df: pd.DataFrame, model_order: List[str], **kwargs) -> None:
    plot_mean_general_cap_bars(
        plot_df,
        model_order,
        value_col="mean_accuracy",
        title="Mean accuracy across QA benchmarks",
        value_label="Mean accuracy",
        linear_floor=1.0,
        **kwargs,
    )


def _model_palette(model_order: Sequence[str]) -> List[str]:
    return [MODEL_COLORS.get(model, "#333333") for model in model_order]


def _comparison_hue_order(method: str) -> List[str]:
    if method == VANILLA_BASELINE_METHOD:
        return [VANILLA_BASELINE_METHOD]
    return [VANILLA_BASELINE_METHOD, method]


def _comparison_palette(method: str) -> List[str]:
    if method == VANILLA_BASELINE_METHOD:
        return [METHOD_COLORS[VANILLA_BASELINE_METHOD]]
    return [METHOD_COLORS[VANILLA_BASELINE_METHOD], COMPARISON_METHOD_COLOR]


def _build_vanilla_comparison_df(
    plot_df: pd.DataFrame,
    method: str,
    benchmark: str,
    model_order: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for model in model_order:
        base = plot_df[(plot_df["Model"] == model) & (plot_df["Benchmark"] == benchmark)]
        vanilla_rows = base[base["Method"] == VANILLA_BASELINE_METHOD]
        if not vanilla_rows.empty:
            rows.append(
                {
                    "Model": model,
                    "ModelShort": MODEL_SHORT_LABELS.get(model, model),
                    "Series": VANILLA_BASELINE_METHOD,
                    "accuracy": float(vanilla_rows.iloc[0]["accuracy"]),
                }
            )
        if method != VANILLA_BASELINE_METHOD:
            method_rows = base[base["Method"] == method]
            if not method_rows.empty:
                rows.append(
                    {
                        "Model": model,
                        "ModelShort": MODEL_SHORT_LABELS.get(model, model),
                        "Series": method,
                        "accuracy": float(method_rows.iloc[0]["accuracy"]),
                    }
                )
    return pd.DataFrame(rows)


def _short_model_order(model_order: Sequence[str]) -> List[str]:
    return [MODEL_SHORT_LABELS.get(model, model) for model in model_order]


def _configure_comparison_grid_axes(
    ax,
    idx: int,
    *,
    n_cols: int,
    n_rows: int,
    ylabel: str,
    xlabel: str = "Model scale",
) -> None:
    """Shared y-axis on left column; x-axis ticks/label only on bottom row."""
    row = idx // n_cols
    col = idx % n_cols
    is_bottom_row = row == n_rows - 1
    is_left_col = col == 0

    if is_bottom_row:
        ax.set_xlabel(xlabel, labelpad=4)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    if is_left_col:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")


LEGEND_BOTTOM_MARGIN = 0.07


def _place_vanilla_comparison_legends(
    fig,
    *,
    legend_handles,
    legend_labels,
    hue_order: Sequence[str],
    legend_y: float,
    fontsize: float = 11,
) -> None:
    """Place the method comparison legend in the margin below plots."""
    series_legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        ncol=len(hue_order),
        frameon=True,
        fontsize=fontsize,
    )
    fig.add_artist(series_legend)


def _apply_axis_font_sizes(
    ax,
    *,
    label_fontsize: float,
    tick_fontsize: float,
) -> None:
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)


def _draw_grouped_bar_ax(
    ax,
    plot_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    x_order: List[str],
    hue_order: List[str],
    palette: List[str],
    ylabel: str,
    title: Optional[str] = "",
    log_scale: bool = False,
    annotate: bool = True,
    horizontal: bool = False,
    linear_floor: float = 1.0,
    annotation_fontsize: int = 10,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
) -> None:
    barplot_kw = dict(
        data=plot_df,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        width=0.72,
        gap=0.12,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    if horizontal:
        sns.barplot(y=x, x=y, order=x_order, **barplot_kw)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    else:
        sns.barplot(x=x, y=y, order=x_order, **barplot_kw)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    if title:
        ax.set_title(title, pad=12, fontsize=12, fontweight="bold")
    if label_fontsize is not None or tick_fontsize is not None:
        _apply_axis_font_sizes(
            ax,
            label_fontsize=label_fontsize or 14,
            tick_fontsize=tick_fontsize or 12,
        )
    _apply_value_axis_limits(
        ax,
        plot_df[y],
        log_scale=log_scale,
        horizontal=horizontal,
        linear_floor=linear_floor,
    )
    if annotate:
        if horizontal:
            _annotate_horizontal_bar_values(ax)
        else:
            _annotate_vertical_bar_values(ax, fontsize=annotation_fontsize)
    return ax


def plot_qa_by_dataset_bars(
    plot_df: pd.DataFrame,
    benchmark: str,
    model_order: List[str],
    *,
    method_order: Optional[Sequence[str]] = None,
    save: bool,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
    log_scale: bool = False,
    horizontal: bool = False,
    annotate: bool = True,
) -> None:
    """One QA benchmark: models on x-axis, methods as grouped bars."""
    method_order = list(method_order or METHOD_ORDER)
    subset = plot_df[plot_df["Benchmark"] == benchmark].copy()
    if subset.empty:
        print(f"No QA data for benchmark {benchmark!r}.")
        return

    configure_plot_style()
    _save_kw = _savefig_kwargs()
    n_models = len(model_order)
    n_methods = len(method_order)

    if horizontal:
        fig_w = max(10, 8 + n_methods * 0.4)
        fig_h = max(9, n_models * 2.5)
    else:
        fig_w = max(12, n_models * max(2.8, n_methods * 0.75))
        fig_h = 7

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ylabel = "Accuracy (log scale)" if log_scale else "Accuracy"
    _draw_grouped_bar_ax(
        ax,
        subset,
        x="Model",
        y="accuracy",
        hue="Method",
        x_order=model_order,
        hue_order=method_order,
        palette=_method_palette(method_order),
        ylabel=ylabel,
        title=benchmark,
        log_scale=log_scale,
        annotate=annotate,
        horizontal=horizontal,
        linear_floor=1.0,
    )
    ax.set_xlabel("Models", labelpad=15)
    if horizontal:
        ax.set_ylabel("Models", labelpad=15)
    plt.tight_layout()
    place_legend_below_xaxis(ax, ncol=len(method_order), title="")

    out_name = filename or f"{benchmark_slug(benchmark)}.png"
    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / out_name, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_qa_by_dataset_grid(
    plot_df: pd.DataFrame,
    model_order: List[str],
    benchmark_order: Sequence[str],
    *,
    method_order: Optional[Sequence[str]] = None,
    save: bool,
    output_dir: Optional[Path] = None,
    filename: str = "combined_by_dataset.png",
    show: bool = False,
    log_scale: bool = False,
) -> None:
    """Grid of per-benchmark charts with shared method legend."""
    method_order = list(method_order or METHOD_ORDER)
    benchmarks = [b for b in benchmark_order if b in plot_df["Benchmark"].unique()]
    if not benchmarks:
        return

    configure_plot_style()
    _save_kw = _savefig_kwargs()
    n_cols = 4
    n_rows = int(np.ceil(len(benchmarks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.8 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    legend_handles = None
    legend_labels = None
    ylabel = "Accuracy (log scale)" if log_scale else "Accuracy"
    for idx, benchmark in enumerate(benchmarks):
        subset = plot_df[plot_df["Benchmark"] == benchmark]
        ax = axes_flat[idx]
        _draw_grouped_bar_ax(
            ax,
            subset,
            x="Model",
            y="accuracy",
            hue="Method",
            x_order=model_order,
            hue_order=method_order,
            palette=_method_palette(method_order),
            ylabel=ylabel,
            title=benchmark,
            log_scale=log_scale,
            annotate=False,
            linear_floor=1.0,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes_flat[len(benchmarks) :]:
        ax.set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=len(method_order),
            frameon=True,
            fontsize=10,
        )
    fig.suptitle("QA accuracy by benchmark", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])

    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_qa_by_method_bars(
    plot_df: pd.DataFrame,
    method: str,
    model_order: List[str],
    benchmark_order: Sequence[str],
    *,
    save: bool,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
    log_scale: bool = False,
) -> None:
    """One method vs Vanilla: grid of benchmarks with two bars per model."""
    if method == VANILLA_BASELINE_METHOD:
        return

    benchmarks = [b for b in benchmark_order if b in plot_df["Benchmark"].unique()]
    if not benchmarks:
        return

    configure_plot_style()
    _save_kw = _savefig_kwargs()
    hue_order = _comparison_hue_order(method)
    palette = _comparison_palette(method)
    n_cols = 4
    n_rows = int(np.ceil(len(benchmarks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    legend_handles = None
    legend_labels = None
    ylabel = "Accuracy (log scale)" if log_scale else "Accuracy"
    all_values: list[float] = []
    for idx, benchmark in enumerate(benchmarks):
        comparison_df = _build_vanilla_comparison_df(
            plot_df,
            method,
            benchmark,
            model_order,
        )
        if comparison_df.empty:
            continue
        all_values.extend(comparison_df["accuracy"].tolist())
        ax = axes_flat[idx]
        _draw_grouped_bar_ax(
            ax,
            comparison_df,
            x="ModelShort",
            y="accuracy",
            hue="Series",
            x_order=_short_model_order(model_order),
            hue_order=hue_order,
            palette=palette,
            ylabel=ylabel,
            title=benchmark,
            log_scale=log_scale,
            annotate=True,
            annotation_fontsize=8,
            linear_floor=1.0,
        )
        _configure_comparison_grid_axes(
            ax,
            idx,
            n_cols=n_cols,
            n_rows=n_rows,
            ylabel=ylabel,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes_flat[len(benchmarks) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"QA accuracy: {method} vs {VANILLA_BASELINE_METHOD}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, LEGEND_BOTTOM_MARGIN, 1.0, 0.975])

    if all_values and not log_scale:
        upper = _score_axis_top(float(max(all_values)), scaled=float(max(all_values)) * 1.2 + 0.01)
        for ax in axes_flat[: len(benchmarks)]:
            if ax.get_visible():
                ax.set_ylim(0, upper)

    if legend_handles:
        _place_vanilla_comparison_legends(
            fig,
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            hue_order=hue_order,
            legend_y=LEGEND_BOTTOM_MARGIN * 0.42,
        )

    out_name = filename or f"{method_slug(method)}.png"
    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / out_name, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close(fig)


_COMPACT_LABEL_FONTSIZE = 10
_COMPACT_TICK_FONTSIZE = 9
_COMPACT_SUPTITLE_FONTSIZE = 12
_COMPACT_LEGEND_FONTSIZE = 9


def plot_qa_by_method_dataset_bars(
    plot_df: pd.DataFrame,
    method: str,
    benchmark: str,
    model_order: List[str],
    *,
    save: bool,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
    log_scale: bool = False,
) -> None:
    """Single method and benchmark panel, styled like one cell in plot_qa_by_method_bars."""
    if method == VANILLA_BASELINE_METHOD:
        return

    comparison_df = _build_vanilla_comparison_df(
        plot_df,
        method,
        benchmark,
        model_order,
    )
    if comparison_df.empty:
        print(f"No QA data for method={method!r}, benchmark={benchmark!r}.")
        return

    configure_plot_style()
    _save_kw = _savefig_kwargs()
    hue_order = _comparison_hue_order(method)
    palette = _comparison_palette(method)
    all_values = comparison_df["accuracy"].tolist()

    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    ylabel = "Accuracy (log scale)" if log_scale else "Accuracy"
    _draw_grouped_bar_ax(
        ax,
        comparison_df,
        x="ModelShort",
        y="accuracy",
        hue="Series",
        x_order=_short_model_order(model_order),
        hue_order=hue_order,
        palette=palette,
        ylabel=ylabel,
        title=None,
        log_scale=log_scale,
        annotate=True,
        annotation_fontsize=8,
        linear_floor=1.0,
        label_fontsize=_COMPACT_LABEL_FONTSIZE,
        tick_fontsize=_COMPACT_TICK_FONTSIZE,
    )
    ax.set_xlabel("Model scale", labelpad=2, fontsize=_COMPACT_LABEL_FONTSIZE)

    fig.suptitle(
        f"{benchmark}: {method} vs {VANILLA_BASELINE_METHOD}",
        fontsize=_COMPACT_SUPTITLE_FONTSIZE,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, LEGEND_BOTTOM_MARGIN, 1.0, 0.99])

    if all_values and not log_scale:
        upper = _score_axis_top(
            float(max(all_values)),
            scaled=float(max(all_values)) * 1.2 + 0.01,
        )
        ax.set_ylim(0, upper)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if legend_handles:
        _place_vanilla_comparison_legends(
            fig,
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            hue_order=hue_order,
            legend_y=LEGEND_BOTTOM_MARGIN * 0.42,
            fontsize=_COMPACT_LEGEND_FONTSIZE,
        )

    out_name = filename or f"{method_slug(method)}_{benchmark_slug(benchmark)}.png"
    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / out_name, **_save_kw)
    if show:
        plt.show()
    else:
        plt.close(fig)

