from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from plot_configs.general_cap_config import (
    DATASET_TO_LANGUAGE,
    DEFAULT_PPL_OUTPUT_DIR,
    METHOD_ORDER,
    MODEL_FILES,
    MODEL_ORDER,
    PPL_DATA_DIR,
)

from .general_cap_loaders import resolve_method_value
from .plots.general_cap_plots import plot_mean_ppl_bars


def _parse_language(dataset: str) -> str:
    language = DATASET_TO_LANGUAGE.get(dataset)
    if language is None:
        raise ValueError(f"Unknown PPL dataset label: {dataset!r}")
    return language


def load_ppl_data(
    data_dir: Path | None = None,
    *,
    model_files: Dict[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load all PPL CSVs into a long DataFrame."""
    data_dir = Path(data_dir or PPL_DATA_DIR)
    model_files = model_files or MODEL_FILES
    model_order = list(model_order or MODEL_ORDER)
    method_order = list(method_order or METHOD_ORDER)

    rows: list[dict] = []
    for model_key in model_order:
        if model_key not in model_files:
            continue
        csv_path = data_dir / f"{model_key}_ppl.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing PPL CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        for _, record in df.iterrows():
            language = _parse_language(str(record["Dataset"]))
            for method in method_order:
                value = resolve_method_value(record, method)
                if value is None:
                    continue
                rows.append(
                    {
                        "model_key": model_key,
                        "model": model_files[model_key],
                        "language": language,
                        "method": method,
                        "raw_ppl": value,
                    }
                )

    if not rows:
        raise ValueError(f"No PPL rows loaded from {data_dir}")

    return pd.DataFrame(rows)


def compute_mean_ppl(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Average raw PPL across languages for each model and method."""
    return (
        raw_df.groupby(["model_key", "model", "method"], as_index=False)["raw_ppl"]
        .mean()
        .rename(columns={"raw_ppl": "mean_ppl"})
    )


def export_mean_ppl_csv(mean_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(path, index=False)


def generate_ppl_plots(
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    save: bool = True,
    show: bool = False,
    log_scale: bool = False,
    horizontal: bool = False,
    model_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load PPL CSVs, compute mean PPL across languages, and save bar chart."""
    output_dir = Path(output_dir or DEFAULT_PPL_OUTPUT_DIR)
    model_order = list(model_order or MODEL_ORDER)
    method_order = list(method_order or METHOD_ORDER)
    model_display_order = [MODEL_FILES[key] for key in model_order if key in MODEL_FILES]

    raw_df = load_ppl_data(
        data_dir,
        model_order=model_order,
        method_order=method_order,
    )
    mean_df = compute_mean_ppl(raw_df)
    plot_df = mean_df.rename(columns={"model": "Model", "method": "Method"})

    if save:
        export_mean_ppl_csv(mean_df, output_dir / "mean_ppl.csv")

    filename = (
        "mean_ppl_by_method_horizontal.png"
        if horizontal
        else "mean_ppl_by_method.png"
    )
    plot_mean_ppl_bars(
        plot_df,
        model_display_order,
        method_order=method_order,
        save=save,
        output_dir=output_dir,
        filename=filename,
        show=show,
        log_scale=log_scale,
        horizontal=horizontal,
    )

    return mean_df


def generate_ppl_radar_plots(**kwargs) -> pd.DataFrame:
    """Backward-compatible alias for ``generate_ppl_plots``."""
    return generate_ppl_plots(**kwargs)
