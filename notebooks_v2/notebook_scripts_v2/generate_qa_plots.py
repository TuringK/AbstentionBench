from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from plot_configs.general_cap_config import (
    DEFAULT_QA_OUTPUT_DIR,
    METHOD_ORDER,
    MODEL_FILES,
    MODEL_ORDER,
    QA_BENCHMARK_ORDER,
    QA_DATA_DIR,
    VANILLA_BASELINE_METHOD,
    benchmark_slug,
    method_slug,
)

from .general_cap_loaders import resolve_method_value
from .plots.general_cap_plots import (
    plot_mean_qa_bars,
    plot_qa_by_dataset_bars,
    plot_qa_by_dataset_grid,
    plot_qa_by_method_bars,
    plot_qa_by_method_dataset_bars,
)


def load_qa_data(
    data_dir: Path | None = None,
    *,
    model_files: Dict[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load all QA CSVs into a long DataFrame."""
    data_dir = Path(data_dir or QA_DATA_DIR)
    model_files = model_files or MODEL_FILES
    model_order = list(model_order or MODEL_ORDER)
    method_order = list(method_order or METHOD_ORDER)

    rows: list[dict] = []
    for model_key in model_order:
        if model_key not in model_files:
            continue
        csv_path = data_dir / f"{model_key}_qa.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing QA CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        for _, record in df.iterrows():
            benchmark = str(record["Dataset"])
            for method in method_order:
                value = resolve_method_value(record, method)
                if value is None:
                    continue
                rows.append(
                    {
                        "model_key": model_key,
                        "model": model_files[model_key],
                        "benchmark": benchmark,
                        "method": method,
                        "accuracy": value,
                    }
                )

    if not rows:
        raise ValueError(f"No QA rows loaded from {data_dir}")

    return pd.DataFrame(rows)


def compute_mean_accuracy(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Average accuracy across QA benchmarks for each model and method."""
    return (
        raw_df.groupby(["model_key", "model", "method"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mean_accuracy"})
    )


def export_qa_csv(raw_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(path, index=False)


def export_mean_accuracy_csv(mean_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(path, index=False)


def _plot_qa_breakdowns(
    plot_df: pd.DataFrame,
    *,
    model_display_order: list[str],
    benchmark_order: Sequence[str],
    method_order: Sequence[str],
    output_dir: Path,
    save: bool,
    show: bool,
    log_scale: bool,
    horizontal: bool,
    per_dataset: bool,
    per_method: bool,
    per_method_dataset: bool,
) -> None:
    if per_dataset:
        dataset_dir = output_dir / "by_dataset"
        for benchmark in benchmark_order:
            if benchmark not in plot_df["Benchmark"].unique():
                continue
            plot_qa_by_dataset_bars(
                plot_df,
                benchmark,
                model_display_order,
                method_order=method_order,
                save=save,
                output_dir=dataset_dir,
                show=show,
                log_scale=log_scale,
                horizontal=horizontal,
            )
        plot_qa_by_dataset_grid(
            plot_df,
            model_display_order,
            benchmark_order,
            method_order=method_order,
            save=save,
            output_dir=dataset_dir,
            show=show,
            log_scale=log_scale,
        )

    if per_method:
        method_dir = output_dir / "by_method"
        for method in method_order:
            if method == VANILLA_BASELINE_METHOD:
                continue
            if method not in plot_df["Method"].unique():
                continue
            plot_qa_by_method_bars(
                plot_df,
                method,
                model_display_order,
                benchmark_order,
                save=save,
                output_dir=method_dir,
                show=show,
                log_scale=log_scale,
            )

    if per_method_dataset:
        pair_dir = output_dir / "by_method_dataset"
        for method in method_order:
            if method == VANILLA_BASELINE_METHOD:
                continue
            for benchmark in benchmark_order:
                subset = plot_df[
                    (plot_df["Method"] == method) & (plot_df["Benchmark"] == benchmark)
                ]
                if subset.empty:
                    continue
                method_dir = pair_dir / method_slug(method)
                plot_qa_by_method_dataset_bars(
                    plot_df,
                    method,
                    benchmark,
                    model_display_order,
                    save=save,
                    output_dir=method_dir,
                    filename=f"{benchmark_slug(benchmark)}.png",
                    show=show,
                    log_scale=log_scale,
                )


def generate_qa_plots(
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    save: bool = True,
    show: bool = False,
    log_scale: bool = False,
    horizontal: bool = False,
    mean_plot: bool = True,
    per_dataset: bool = True,
    per_method: bool = True,
    per_method_dataset: bool = True,
    model_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
    benchmark_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load QA CSVs and save mean, per-dataset, per-method, and per-pair bar charts."""
    output_dir = Path(output_dir or DEFAULT_QA_OUTPUT_DIR)
    model_order = list(model_order or MODEL_ORDER)
    method_order = list(method_order or METHOD_ORDER)
    benchmark_order = list(benchmark_order or QA_BENCHMARK_ORDER)
    model_display_order = [MODEL_FILES[key] for key in model_order if key in MODEL_FILES]

    raw_df = load_qa_data(
        data_dir,
        model_order=model_order,
        method_order=method_order,
    )
    plot_df = raw_df.rename(
        columns={"model": "Model", "method": "Method", "benchmark": "Benchmark"}
    )

    if save:
        export_qa_csv(raw_df, output_dir / "qa_scores.csv")

    if mean_plot:
        mean_df = compute_mean_accuracy(raw_df)
        mean_plot_df = mean_df.rename(columns={"model": "Model", "method": "Method"})
        if save:
            export_mean_accuracy_csv(mean_df, output_dir / "mean_accuracy.csv")
        filename = (
            "mean_accuracy_by_method_horizontal.png"
            if horizontal
            else "mean_accuracy_by_method.png"
        )
        plot_mean_qa_bars(
            mean_plot_df,
            model_display_order,
            method_order=method_order,
            save=save,
            output_dir=output_dir,
            filename=filename,
            show=show,
            log_scale=log_scale,
            horizontal=horizontal,
        )

    _plot_qa_breakdowns(
        plot_df,
        model_display_order=model_display_order,
        benchmark_order=benchmark_order,
        method_order=method_order,
        output_dir=output_dir,
        save=save,
        show=show,
        log_scale=log_scale,
        horizontal=horizontal,
        per_dataset=per_dataset,
        per_method=per_method,
        per_method_dataset=per_method_dataset,
    )

    return raw_df
