"""Export wide CSV summary tables (Model x Method) for paper-style metric tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from tabulate import tabulate

from .metrics_schema import Split

METRIC_COLUMNS: dict[str, str] = {
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}


def build_method_summary_table(
    plot_df: pd.DataFrame,
    *,
    split: Split,
    metric: str,
    model_order: Sequence[str],
    method_order: Sequence[str],
) -> pd.DataFrame:
    """Pivot loaded plot data into a wide table: rows=models, columns=methods."""
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(METRIC_COLUMNS)}")

    metric_col = METRIC_COLUMNS[metric]
    subset = plot_df[plot_df["Split"] == split]
    if subset.empty:
        return pd.DataFrame({"Model": list(model_order)})

    wide = subset.pivot(index="Model", columns="Method", values=metric_col)
    wide = wide.reindex(index=list(model_order), columns=list(method_order))
    return wide.reset_index()


def export_method_summary_csvs(
    plot_df: pd.DataFrame,
    output_dir: Path,
    *,
    model_order: Sequence[str],
    method_order: Sequence[str],
    splits: Sequence[Split] = ("in_domain", "rulebreakers"),
    log_tables: bool = True,
) -> dict[str, Path]:
    """Write F1 / precision / recall summary CSVs and log their contents."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    for split in splits:
        split_subset = plot_df[plot_df["Split"] == split]
        if split_subset.empty:
            print(f"Skipping {split} summary tables (no rows loaded).")
            continue

        for metric in METRIC_COLUMNS:
            table = build_method_summary_table(
                plot_df,
                split=split,
                metric=metric,
                model_order=model_order,
                method_order=method_order,
            )
            path = output_dir / f"{split}_{metric}.csv"
            table.to_csv(path, index=False)
            written[f"{split}_{metric}"] = path

            print(f"Saved {split} {metric} summary: {path}")
            if log_tables:
                display = table.copy()
                for col in method_order:
                    if col in display.columns:
                        display[col] = display[col].map(
                            lambda value: f"{value:.2f}" if pd.notna(value) else ""
                        )
                print(tabulate(display, headers="keys", tablefmt="simple", showindex=False))
                print()

    return written
